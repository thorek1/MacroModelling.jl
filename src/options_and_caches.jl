@stable default_mode = "disable" begin


# ============================================================================
# Doubling power-cache helpers
# ============================================================================
# These are shared by the Lyapunov and Sylvester `:doubling` solvers and their
# adjoint helpers. Capture is gated by `workspace.pow_capture`; the rrule
# wrapper is responsible for setting/clearing the flag and bumping the stamp.
# Slot type is AbstractMatrix so the same vector can hold dense or sparse
# A^(2^k) / B^(2^k) entries (sparse-aware).



"""
    cache_set!(vec, k, M)

Place `M` into slot `k` of the doubling power cache vector. If the slot does
not yet exist, the vector is extended by one (sequential extension).
The slot is reused in-place when its concrete type and shape match `M`,
otherwise it is replaced with `copy(M)` to preserve the input's concrete type.

When `transposed=true`, dense matrices are stored as `copy(M)'` (an `Adjoint`
wrapping a sequential copy of the data). This avoids the expensive element
rearrangement of `permutedims` — the solver dispatch handles `Adjoint` matrices
natively. Subsequent writes to the same slot reuse the existing parent buffer
(zero allocation after the first write). Sparse matrices are still materialised
as `SparseMatrixCSC(M')` since CSC format is required for efficient sparse ops.
"""
@inline function cache_set!(vec::Vector{<:AbstractMatrix}, k::Int, M::AbstractMatrix, transposed::Bool = false)
    if transposed
        if M isa AbstractSparseMatrix
            Mt = SparseMatrixCSC(M')
            if length(vec) < k
                push!(vec, Mt)
            else
                slot = vec[k]
                if typeof(slot) === typeof(Mt) && size(slot) == size(Mt)
                    copyto!(slot, Mt)
                else
                    vec[k] = Mt
                end
            end
        else
            # Dense: store as Adjoint view — sequential copy + zero-cost wrapper.
            # On subsequent writes, reuse the parent buffer to avoid allocation.
            if length(vec) < k
                push!(vec, copy(M)')
            else
                slot = vec[k]
                if slot isa ℒ.Adjoint && typeof(parent(slot)) === typeof(M) && size(parent(slot)) == size(M)
                    copyto!(parent(slot), M)
                else
                    vec[k] = copy(M)'
                end
            end
        end
    else
        if length(vec) < k
            push!(vec, copy(M))
        else
            slot = vec[k]
            if typeof(slot) === typeof(M) && size(slot) == size(M)
                copyto!(slot, M)
            else
                vec[k] = copy(M)
            end
        end
    end
    return nothing
end








"""
Create an empty `second_order_indices` struct with all fields initialized to empty/zero values.
These will be lazily populated by various ensure_*! functions as needed.

See [`second_order_indices`](@ref) for field documentation.
"""
function Second_order_indices()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    empty_sparse_bool = spzeros(Bool, 0, 0)
    empty_sparse_float = spzeros(Float64, 0, 0)
    empty_matrix_float = Matrix{Float64}(undef, 0, 0)
    return second_order_indices(
        # Auxiliary matrices (𝛔, 𝛔_sym, 𝛔c₂, 𝛔𝐂₂, 𝐂₂, 𝐔₂, 𝐔∇₂, 𝐈ₙ₊, 𝐈ₙ₋)
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        Int[],               # ∇₂_nonempty_col_as_kron_rowmask
        Int[],               # 𝛔𝐂₂_nonempty_row_as_kron_colmask
        # Pre-transposed constants for rrule pullback
        empty_sparse_int,    # 𝛔ᵀ
        empty_sparse_int,    # 𝐂₂ᵀ
        empty_sparse_int,    # 𝐔₂ᵀ
        empty_sparse_int,    # 𝐔∇₂ᵀ
        # Computational index caches (BitVectors)
        BitVector(),         # s_in_s⁺
        BitVector(),         # s_in_s
        BitVector(),         # kron_s⁺_s⁺
        BitVector(),         # kron_s⁺_s
        BitVector(),         # e_in_s⁺
        BitVector(),         # v_in_s⁺
        BitVector(),         # kron_s_s
        BitVector(),         # kron_e_e
        BitVector(),         # kron_v_v
        BitVector(),         # kron_s_e
        BitVector(),         # kron_e_s
        BitVector(),         # kron_s⁺_s⁺_s⁺
        BitVector(),         # kron_s_s⁺_s⁺
        # Index arrays
        Int[],               # shockvar_idxs
        Int[],               # shock_idxs
        Int[],               # shock_idxs2
        Int[],               # shock²_idxs
        Int[],               # var_vol²_idxs
        # Conditional forecast indices
        Int[],               # var²_idxs
        Int[],               # shockvar²_idxs
        # Moment computation caches
        BitVector(),         # kron_states
        empty_sparse_float,  # I_plus_s_s
        Float64[],           # e4
        Float64[],           # vec_Iₑ
        empty_matrix_float,  # e4_nᵉ²_nᵉ²
        empty_matrix_float,  # e4_nᵉ_nᵉ³
        empty_matrix_float,  # e4_minus_vecIₑ_outer
    )
end

"""
    Third_order_indices()

Create an empty `third_order_indices` struct with all fields initialized to empty/zero values.
These will be lazily populated by various ensure_*! functions as needed.

See [`third_order_indices`](@ref) for field documentation.
"""
function Third_order_indices()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    empty_matrix_float = Matrix{Float64}(undef, 0, 0)
    return third_order_indices(
        # Auxiliary matrices (𝐂₃, 𝐔₃, 𝐈₃, 𝐂∇₃, 𝐔∇₃, 𝐏, 𝐏𝐂₃, 𝐏₁ₗ, 𝐏₁ᵣ, ...)
        empty_sparse_int,    # 𝐂₃
        empty_sparse_int,    # 𝐔₃
        Dict{Vector{Int}, Int}(),  # 𝐈₃
        empty_sparse_int,    # 𝐂∇₃
        empty_sparse_int,    # 𝐔∇₃
        Int[],               # ∇₃_rowmask
        empty_sparse_int,    # 𝐏
        empty_sparse_int,    # 𝐏𝐂₃
        empty_sparse_int,    # 𝐏₁ₗ
        empty_sparse_int,    # 𝐏₁ᵣ
        empty_sparse_int,    # 𝐏₁ₗ̂
        empty_sparse_int,    # 𝐏₂ₗ̂
        empty_sparse_int,    # 𝐏₁ₗ̄
        empty_sparse_int,    # 𝐏₂ₗ̄
        empty_sparse_int,    # 𝐏₁ᵣ̃
        empty_sparse_int,    # 𝐏₂ᵣ̃
        empty_sparse_int,    # 𝐒𝐏
        # Pre-transposed constants for rrule pullback
        empty_sparse_int,    # 𝐂₃ᵀ
        empty_sparse_int,    # 𝐔₃ᵀ
        empty_sparse_int,    # 𝐏𝐂₃ᵀ
        empty_sparse_int,    # 𝐏₁ₗᵀ
        empty_sparse_int,    # 𝐏₁ᵣᵀ
        empty_sparse_int,    # 𝐏₁ₗ̄ᵀ
        empty_sparse_int,    # 𝐏₂ₗ̄ᵀ
        empty_sparse_int,    # 𝐏₁ᵣ̃ᵀ
        empty_sparse_int,    # 𝐏₂ᵣ̃ᵀ
        # Conditional forecast index caches
        Int[],               # var_vol³_idxs
        Int[],               # shock_idxs2
        Int[],               # shock_idxs3
        Int[],               # shock³_idxs
        Int[],               # shockvar1_idxs
        Int[],               # shockvar2_idxs
        Int[],               # shockvar3_idxs
        Int[],               # shockvar³2_idxs
        Int[],               # shockvar³_idxs
        # Moment computation caches
        Float64[],           # e6
        BitVector(),         # kron_e_v
        empty_matrix_float,  # e6_nᵉ³_nᵉ³
        # Substate index Dict caches
        Dict{Int, moments_substate_indices}(),
        Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_indices}(),
    )
end


"""
Create a workspace for nonlinear solvers (Levenberg-Marquardt and Newton).

# Arguments
- `func_buffer::Vector{T}`: Pre-allocated buffer for function evaluation
- `jac_buffer::AbstractMatrix{T}`: Pre-allocated buffer for Jacobian
- `chol_buffer::LinearCache`: Pre-allocated Cholesky factorization cache
- `lu_buffer::LinearCache`: Pre-allocated LU factorization cache
"""
@unstable function Nonlinear_solver_workspace(func_buffer::Vector{T}, jac_buffer::AbstractMatrix{T}, 
                                    chol_buffer::𝒮.LinearCache, lu_buffer::𝒮.LinearCache) where T <: Real
    n = length(func_buffer)
    nonlinear_solver_workspace(
        func_buffer,          # func_buffer
        jac_buffer,           # jac_buffer
        chol_buffer,          # chol_buffer
        lu_buffer,            # lu_buffer
        zeros(T, n),          # current_guess
        zeros(T, n),          # previous_guess
        zeros(T, n),          # guess_update
        zeros(T, n),          # current_guess_untransformed
        zeros(T, n),          # previous_guess_untransformed
        zeros(T, n),          # best_previous_guess
        zeros(T, n),          # best_current_guess
        zeros(T, n),          # factor
        zeros(T, n),          # u_bounds
        zeros(T, n),          # l_bounds
    )
end


@unstable function Krylov_workspace(;S::Type = Float64)
    krylov_workspace(  GmresWorkspace(0,0,Vector{S}),
                    DqgmresWorkspace(0,0,Vector{S}),
                    BicgstabWorkspace(0,0,Vector{S}))
end

@unstable function Sylvester_workspace(;S::Type = Float64, T::Type = Float64)
    sylvester_workspace(
        0, 0,                   # n, m dimensions
        zeros(S,0,0),           # tmp (Krylov)
        zeros(S,0,0),           # 𝐗 (Krylov)
        zeros(S,0,0),           # 𝐂 (Krylov)
        zeros(S,0,0),           # 𝐀 (doubling)
        zeros(S,0,0),           # 𝐀¹ (doubling)
        zeros(S,0,0),           # 𝐁 (doubling)
        zeros(S,0,0),           # 𝐁¹ (doubling)
        zeros(S,0,0),           # 𝐂_dbl (doubling)
        zeros(S,0,0),           # 𝐂¹ (doubling)
        zeros(S,0,0),           # 𝐂B (doubling)
        Krylov_workspace(S = S),
        zeros(S,0,0),           # P (stable primal cache)
        # Doubling power cache
        Vector{AbstractMatrix{S}}(),    # 𝐀_pow
        Vector{AbstractMatrix{S}}(),    # 𝐁_pow
        0,                      # pow_iters
        false,                  # pow_capture
        false,                  # pow_transposed
        zeros(T,0,0),           # P̃
        zeros(T,0,0),           # Ã_fd
        zeros(T,0,0),           # B̃_fd
        zeros(T,0,0))           # C̃_fd
end

"""
    Find_shocks_workspace(;T::Type = Float64)

Create a workspace for find_shocks conditional forecast with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_find_shocks_buffers!.
"""
@unstable function Find_shocks_workspace(;T::Type = Float64)
    find_shocks_workspace{T}(
        0,                      # n_exo dimension
        zeros(T,0),             # kron_buffer (n_exo^2)
        zeros(T,0,0),           # kron_buffer2 (n_exo × n_exo)
        zeros(T,0),             # kron_buffer² (n_exo^3)
        zeros(T,0,0),           # kron_buffer3 (n_exo × n_exo^2)
        zeros(T,0,0),           # kron_buffer4 (n_exo^2 × n_exo)
        0,                      # n_past dimension
        zeros(T,0),             # kron_state_vol
        zeros(T,0),             # kron_state_vol3
        zeros(T,0),             # kron_state₁₂
        zeros(T,0,0),           # kron_I_state
        zeros(T,0,0),           # kron_I_state₂
        zeros(T,0,0))           # kron_I_state_state
end

@unstable function Higher_order_workspace(;T::Type = Float64, S::Type = Float64)
    empty_dx_prob = 𝒮.LinearProblem(zeros(Float64, 0, 0), zeros(Float64, 0))
    empty_dx_lu_buffer = 𝒮.init(empty_dx_prob,
                                𝒮.FastLUFactorization(),
                                verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
    empty_lu_factors = zeros(Float64, 0, 0)
    empty_lu_ws = FastLapackInterface.LUWs(empty_lu_factors)
    higher_order_workspace(spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        zeros(T,0,0),  # 𝐒₁
                        zeros(T,0,0),  # 𝐒₁₋╱𝟏ₑ
                        zeros(T,0,0),
                        Sylvester_workspace(S = S),
                        zeros(T,0),    # ∂∇_vec
                        # Second order pullback gradient buffers (lazily allocated)
                        zeros(T,0,0),  # ∂∇₂
                        zeros(T,0,0),  # ∂∇₁
                        zeros(T,0,0),  # ∂𝐒₁
                        zeros(T,0,0),  # ∂spinv
                        zeros(T,0,0),  # ∂𝐒₁₋╱𝟏ₑ
                        zeros(T,0,0),  # ∂𝐒₁₊╱𝟎
                        zeros(T,0,0),  # ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋
                        # Third order pullback gradient buffers (only dense matrices)
                        zeros(T,0,0),  # ∂∇₁_3rd
                        zeros(T,0,0),  # ∂𝐒₁_3rd
                        zeros(T,0,0),  # ∂spinv_3rd
                        zeros(T,0,0),  # ∂∇₂_3rd
                        zeros(T,0,0),  # ∂∇₃_3rd
                        zeros(T,0,0),  # ∂𝐒₂_3rd
                        zeros(T,0,0),  # ∂𝐒₁₋╱𝟏ₑ_3rd
                        zeros(T,0,0),  # ∂𝐒₁₊╱𝟎_3rd
                        zeros(T,0,0),  # ∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋_3rd
                        # Third order pullback temporary buffers
                        zeros(T,0,0),  # ∂𝐒₂₊╱𝟎_3rd
                        zeros(T,0,0),  # ∂R_c_3rd
                        zeros(T,0,0),  # ∂L_c_3rd
                        zeros(T,0,0),  # ∂L_d_3rd
                        zeros(T,0,0),  # ∂R_d_3rd
                        zeros(T,0,0),  # ∂𝐒₂₋╱𝟎_3rd
                        zeros(T,0,0),  # ∂𝐒₁₋╱𝟏ₑ_t8_3rd
                        zeros(T,0,0),  # ∂𝐒₁₊╱𝟎_tmp_3rd
                        zeros(T,0,0),  # ∂𝐒₁₊╱𝟎_tk0_3rd
                        zeros(T,0,0),  # ∂tmpkron0_σ_3rd
                        zeros(T,0,0),  # ∂aux_3rd
                        zeros(T,0,0),  # ∂S1S1_from_ck_3rd
                        zeros(T,0,0),  # ∂S1p0_kron_sigma_3rd
                        zeros(T,0,0),  # ∂S1p0_left_3rd
                        zeros(T,0,0),  # ∂S1p0_right_3rd
                        # Third order pullback intermediate product buffers (for mul!)
                        zeros(T,0,0),  # ∂A_3rd
                        zeros(T,0,0),  # ∂B_sylv_3rd
                        zeros(T,0,0),  # ∂𝐗₃_3rd
                        zeros(T,0,0),  # ∂𝐗₃_pre_3rd
                        zeros(T,0,0),  # ∂out2_3rd
                        zeros(T,0,0),  # ∂∇₁₊_3rd
                        zeros(T,0,0),  # ∂∇₁₊𝐒₁➕∇₁₀_3rd
                        zeros(T,0,0),  # ∇₂t_∂out2_3rd
                        zeros(T,0,0),  # mul_tmp_3rd
                        # LinearSolve cache (FastLUFactorization) for SSS Newton iter ∂x \ Δx
                        empty_dx_lu_buffer,                              # dx_lu_buffer
                        # LinearSolve cache (FastLUFactorization) for SSS common-block tmp \ vec
                        let p = 𝒮.LinearProblem(zeros(Float64, 0, 0), zeros(Float64, 0))
                            𝒮.init(p,
                                   𝒮.FastLUFactorization(),
                                   verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
                        end,                                             # sss_tmp_lu_buffer
                        # Dedicated FastLapackInterface LU workspace for SSS pullback transpose solves
                        empty_lu_ws,                                      # fast_lu_ws_sss_pullback
                        (0, 0),                                           # fast_lu_dims_sss_pullback
                        # SSS Newton iter kron! buffers
                        zeros(T, 0),    # x_aug_buf
                        zeros(T, 0),    # kron_x_aug_xx
                        zeros(T, 0),    # kron_x_aug_x_kron
                        zeros(T, 0, 0), # kron_x_aug_I
                        zeros(T, 0, 0), # kron_x_kron_I
                        # ForwardDiff partials buffers for stochastic steady state (accessed via model struct)
                        zeros(S,0,0),  # ∂x_second_order
                        zeros(S,0,0))  # ∂x_third_order
end

@unstable function ensure_higher_order_solution_buffers!(ws::higher_order_workspace{S,G,H}, n::Int, nₑ₋::Int) where {S <: Real, G <: AbstractFloat, H <: Real}
    size(ws.𝐒₁) == (n, nₑ₋) || (ws.𝐒₁ = zeros(S, n, nₑ₋))
    size(ws.𝐒₁₋╱𝟏ₑ) == (nₑ₋, nₑ₋) || (ws.𝐒₁₋╱𝟏ₑ = zeros(S, nₑ₋, nₑ₋))
    return ws
end

"""
    ensure_sss_kron_buffers!(ws, nPast; third_order=false)

Lazily (re)allocate kron! buffers used by the stochastic-steady-state Newton iter
on `ws` (a `higher_order_workspace`). `nPast` is `T.nPast_not_future_and_mixed`.
The 3rd-order-only buffers are only sized when `third_order=true`.
"""
@unstable function ensure_sss_kron_buffers!(ws::higher_order_workspace{S,G,H}, nPast::Int; third_order::Bool=false) where {S <: Real, G <: AbstractFloat, H <: Real}
    n_aug = nPast + 1
    length(ws.x_aug_buf) == n_aug || (ws.x_aug_buf = zeros(S, n_aug))
    length(ws.kron_x_aug_xx) == n_aug^2 || (ws.kron_x_aug_xx = zeros(S, n_aug^2))
    size(ws.kron_x_aug_I) == (n_aug * nPast, nPast) || (ws.kron_x_aug_I = zeros(S, n_aug * nPast, nPast))
    if third_order
        length(ws.kron_x_aug_x_kron) == n_aug^3 || (ws.kron_x_aug_x_kron = zeros(S, n_aug^3))
        size(ws.kron_x_kron_I) == (n_aug^2 * nPast, nPast) || (ws.kron_x_kron_I = zeros(S, n_aug^2 * nPast, nPast))
    end
    return ws
end

"""
    ensure_dx_lu_buffer!(ws, ∂x, Δx)

Ensure the LinearSolve cache `ws.dx_lu_buffer` is sized for the SSS Newton iter
linear system `∂x * y = Δx`. If dimensions match, reuse the cache and just rebind
`A` and `b`; otherwise re-`init` the cache (FastLUFactorization backend).
"""
@unstable function ensure_dx_lu_buffer!(ws::higher_order_workspace, ∂x::AbstractMatrix{Float64}, Δx::AbstractVector{Float64})
    cache = ws.dx_lu_buffer
    if size(cache.A) != size(∂x) || length(cache.b) != length(Δx)
        prob = 𝒮.LinearProblem(∂x, Δx)
        ws.dx_lu_buffer = 𝒮.init(prob,
                                 𝒮.FastLUFactorization(),
                                 verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
    else
        cache.A = ∂x
        cache.b = Δx
    end
    return ws.dx_lu_buffer
end

"""
    ensure_sss_tmp_lu_buffer!(ws, tmp, rhs)

Ensure the LinearSolve cache `ws.sss_tmp_lu_buffer` is sized for the SSS common-block
solve `tmp * y = rhs`. If dimensions match, reuse the cache and just rebind `A` and `b`;
otherwise re-`init` the cache (FastLUFactorization backend).
"""
@unstable function ensure_sss_tmp_lu_buffer!(ws::higher_order_workspace, tmp::AbstractMatrix{Float64}, rhs::AbstractVector{Float64})
    cache = ws.sss_tmp_lu_buffer
    if size(cache.A) != size(tmp) || length(cache.b) != length(rhs)
        prob = 𝒮.LinearProblem(tmp, rhs)
        ws.sss_tmp_lu_buffer = 𝒮.init(prob,
                                      𝒮.FastLUFactorization(),
                                      verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
    else
        cache.A = tmp
        cache.b = rhs
    end
    return ws.sss_tmp_lu_buffer
end

@unstable function ensure_sss_pullback_fast_lu_workspace!(ws::higher_order_workspace{T}, tmp::AbstractMatrix{T}) where {T <: Union{Float32, Float64}}
    dims = (size(tmp, 1), size(tmp, 2))
    if ws.fast_lu_dims_sss_pullback != dims
        ws.fast_lu_ws_sss_pullback = FastLapackInterface.LUWs(tmp)
        ws.fast_lu_dims_sss_pullback = dims
    end
    return ws.fast_lu_ws_sss_pullback, ws.fast_lu_dims_sss_pullback
end

"""
    First_order_workspace(; T::Type = Float64, S::Type = Float64)

Create a pre-allocated workspace for first-order perturbation and related AD paths.
"""
@unstable function First_order_workspace(; T::Type{TT} = Float64, S::Type{SS} = Float64) where {TT <: AbstractFloat, SS <: Real}
    empty_qr_factors = zeros(TT, 0, 0)
    empty_qr_ws::FastLapackInterface.QRWs = FastLapackInterface.QRWs(empty_qr_factors)
    empty_qr_rhs = zeros(TT, 0, 0)
    empty_qr_orm_ws::FastLapackInterface.QROrmWs = FastLapackInterface.QROrmWs(empty_qr_ws, 'L', 'T', empty_qr_factors, empty_qr_rhs)
    empty_lu_factors = zeros(TT, 0, 0)
    empty_lu_ws = FastLapackInterface.LUWs(empty_lu_factors)
    empty_sparse = spzeros(TT, 0, 0)
    empty_sparse_rhs = zeros(TT, 0)
    empty_sparse_prob = 𝒮.LinearProblem(empty_sparse, empty_sparse_rhs)
    empty_sparse_lu::𝒮.LinearCache = 𝒮.init(empty_sparse_prob,
                                            𝒮.LUFactorization(),
                                            verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

    first_order_workspace(
                    Sylvester_workspace(S = TT, T = SS),  # sylvester
                    # ForwardDiff partials buffers
                    zeros(SS, 0, 0),  # X̃_first_order
                    zeros(SS, 0, 0),  # p_tmp
                    zeros(SS, 0, 0),  # ∂SS_and_pars
                    zeros(TT, 0),     # ∂∇₁_vec
                    # First-order perturbation workspaces (primal)
                    zeros(TT, 0, 0),  # 𝐧ₚ₋
                    zeros(TT, 0, 0),  # 𝐌
                    zeros(TT, 0, 0),  # 𝐀₊
                    zeros(TT, 0, 0),  # 𝐀₀
                    zeros(TT, 0, 0),  # 𝐀₋
                    zeros(TT, 0, 0),  # 𝐀̃₊
                    zeros(TT, 0, 0),  # 𝐀̃₀
                    zeros(TT, 0, 0),  # 𝐀̃₋
                    zeros(TT, 0, 0),  # 𝐀̄₀ᵤ
                    zeros(TT, 0, 0),  # 𝐀₊ᵤ
                    zeros(TT, 0, 0),  # 𝐀̃₀ᵤ
                    zeros(TT, 0, 0),  # 𝐀₋ᵤ
                    zeros(TT, 0, 0),  # 𝐀
                    zeros(TT, 0, 0),  # ∇₀
                    zeros(TT, 0, 0),  # ∇ₑ
                    # FastLapackInterface QR workspaces
                    empty_qr_factors,
                    empty_qr_ws,
                    empty_qr_orm_ws,
                    (0, 0, 0),
                    empty_qr_orm_ws,
                    (0, 0, 0),
                    empty_qr_orm_ws,
                    (0, 0, 0),
                    # FastLapackInterface LU workspaces
                    empty_lu_ws,
                    (0, 0),
                    empty_lu_ws,
                    (0, 0),
                    # Dedicated FastLapackInterface LU workspace for NSSS implicit derivatives
                    empty_lu_ws,
                    (0, 0),
                    empty_sparse_lu,
                    zeros(TT, 0),
                    zeros(TT, 0, 0))
end

"""
    Qme_doubling_workspace(n::Int; T::Type = Float64, S::Type = Float64)

Create a pre-allocated workspace for the quadratic matrix equation doubling algorithm.
`n` is the dimension of the square matrices (nVars - nPresent_only).
"""
@unstable function Qme_doubling_workspace(n::Int; T::Type = Float64, S::Type = Float64)
    empty_lu_factors = zeros(T, 0, 0)
    empty_lu_ws = FastLapackInterface.LUWs(empty_lu_factors)

    qme_doubling_workspace(
                    zeros(T, n, n),  # E
                    zeros(T, n, n),  # F
                    zeros(T, n, n),  # X
                    zeros(T, n, n),  # Y
                    zeros(T, n, n),  # X_new
                    zeros(T, n, n),  # Y_new
                    zeros(T, n, n),  # E_new
                    zeros(T, n, n),  # F_new
                    zeros(T, n, n),  # temp1
                    zeros(T, n, n),  # temp2
                    zeros(T, n, n),  # temp3
                    zeros(T, n, n),  # B̄
                    zeros(T, n, n),  # AXX
                    Sylvester_workspace(S = T, T = S),  # sylvester
                    # ForwardDiff partials buffers
                    zeros(S, 0, 0),  # X̃
                    # FastLapackInterface LU workspaces
                    empty_lu_ws,
                    (0, 0),
                    empty_lu_ws,
                    (0, 0))
end

@unstable function ensure_first_order_fast_qr_workspace!(ws::first_order_workspace{T}, qr_mat::AbstractMatrix) where {T <: Union{Float32, Float64}}
    if size(ws.fast_qr_factors) != size(qr_mat)
        ws.fast_qr_factors = zeros(T, size(qr_mat, 1), size(qr_mat, 2))
        ws.fast_qr_ws = FastLapackInterface.QRWs(ws.fast_qr_factors)
    end
    copyto!(ws.fast_qr_factors, qr_mat)

    return ws.fast_qr_factors, ws.fast_qr_ws
end

"""
    Schur_workspace(n::Int, nMixed::Int, nPfm::Int, nFnpm::Int; T::Type = Float64)

Create a pre-allocated workspace for the schur-based quadratic matrix equation solver.
Dimensions:
- `n` = nVars - nPresent_only (dynamic variables)
- `nMixed` = number of mixed timing variables
- `nPfm` = nPast_not_future_and_mixed
- `nFnpm` = nFuture_not_past_and_mixed
"""
@unstable function Schur_workspace(n::Int, nMixed::Int, nPfm::Int, nFnpm::Int; T::Type = Float64)
    companion_size = n + nMixed
    nComb = nPfm + nFnpm  # comb = union(future_not_past_and_mixed, past_not_future)
    qz_seed_size = max(companion_size, 1)
    qz_seed = zeros(T, qz_seed_size, qz_seed_size)
    qz_ws = FastLapackInterface.GeneralizedSchurWs(qz_seed)
    lu_seed_size = max(nPfm, 1)
    lu_seed = zeros(T, lu_seed_size, lu_seed_size)
    empty_lu_ws = FastLapackInterface.LUWs(lu_seed)
    schur_workspace(
        zeros(T, companion_size, companion_size),  # D
        zeros(T, companion_size, companion_size),  # E
        zeros(T, n, nPfm),                         # Ã₋
        zeros(T, n, nFnpm),                        # Ã₀₊
        zeros(T, n, nPfm),                         # Ã₀₋
        zeros(T, nPfm, nPfm),                      # Z₁₁
        zeros(T, nFnpm, nPfm),                     # Z₂₁
        zeros(T, nPfm, nPfm),                      # S₁₁
        zeros(T, nPfm, nPfm),                      # T₁₁
        zeros(T, n, nPfm),                         # sol
        zeros(T, n, n),                            # temp_X2
        zeros(T, n, n),                            # AXX
        Vector{Bool}(undef, companion_size),       # eigenselect
        qz_ws,
        (0, 0),
        empty_lu_ws,
        (0, 0),
        empty_lu_ws,
        (0, 0),
        zeros(T, nPfm, nFnpm),                     # fast_lu_rhs_t_z21
        zeros(T, nPfm, nPfm))                      # fast_lu_rhs_t_s11
end

"""
    Lyapunov_workspace(n::Int; T::Type = Float64)

Create a workspace for the Lyapunov equation solver with lazy buffer allocation.
`n` is the dimension of the square matrices.
Buffers are initialized to 0-dimensional objects and resized on-demand when the corresponding algorithm is used.
"""
@unstable function Lyapunov_workspace(n::Int; T::Type = Float64)
    lyapunov_workspace{T, T}(
        n,                      # dimension
        zeros(T, 0, 0),         # 𝐂 (doubling)
        zeros(T, 0, 0),         # 𝐂¹ (doubling)
        zeros(T, 0, 0),         # 𝐀 (doubling)
        zeros(T, 0, 0),         # 𝐂A (doubling)
        zeros(T, 0, 0),         # 𝐀² (doubling)
        zeros(T, 0, 0),         # tmp̄ (Krylov)
        zeros(T, 0, 0),         # 𝐗 (Krylov)
        zeros(T, 0),            # b (Krylov)
        Krylov.BicgstabWorkspace(0, 0, Vector{T}),  # bicgstab
        Krylov.GmresWorkspace(0, 0, Vector{T}; memory = 20),  # gmres
        Krylov.DqgmresWorkspace(0, 0, Vector{T}),  # dqgmres
        zeros(T, 0),            # b_vech (vech-space Krylov)
        Krylov.BicgstabWorkspace(0, 0, Vector{T}),  # bicgstab_vech
        Krylov.GmresWorkspace(0, 0, Vector{T}; memory = 20),  # gmres_vech
        Krylov.DqgmresWorkspace(0, 0, Vector{T}),  # dqgmres_vech
        zeros(T, 0, 0),         # P (stable primal cache)
        # ForwardDiff partials buffers
        zeros(T, 0, 0),         # P̃
        zeros(T, 0, 0),         # Ã_fd
        zeros(T, 0, 0),         # C̃_fd
        # Doubling power cache (sparse-aware)
        Vector{AbstractMatrix{T}}(),  # 𝐀_pow
        0,                      # pow_iters
        false,                  # pow_capture
        false,                  # pow_transposed
        FastLapackInterface.SchurWs(zeros(T, 1, 1))  # schur_ws (lazily resized by gees!)
    )
end

"""
    ensure_lyapunov_doubling_buffers!(ws::lyapunov_workspace{T}) where T

Ensure the doubling algorithm buffers are allocated in the workspace.
"""
@unstable function ensure_lyapunov_doubling_buffers!(ws::lyapunov_workspace{T}) where T
    n = ws.n
    if size(ws.𝐂, 1) != n
        ws.𝐂 = zeros(T, n, n)
    end
    if size(ws.𝐂¹, 1) != n
        ws.𝐂¹ = zeros(T, n, n)
    end
    if size(ws.𝐀, 1) != n
        ws.𝐀 = zeros(T, n, n)
    end
    if size(ws.𝐂A, 1) != n
        ws.𝐂A = zeros(T, n, n)
    end
    if size(ws.𝐀², 1) != n
        ws.𝐀² = zeros(T, n, n)
    end
    return ws
end

"""
    ensure_lyapunov_krylov_buffers!(ws::lyapunov_workspace{T}) where T

Ensure the Krylov method buffers are allocated in the workspace.
"""
@unstable function ensure_lyapunov_krylov_buffers!(ws::lyapunov_workspace{T}) where T
    n = ws.n
    if size(ws.tmp̄, 1) != n
        ws.tmp̄ = zeros(T, n, n)
    end
    if size(ws.𝐗, 1) != n
        ws.𝐗 = zeros(T, n, n)
    end
    if length(ws.b) != n * n
        ws.b = zeros(T, n * n)
    end
    return ws
end

"""
    ensure_lyapunov_krylov_solver!(ws::lyapunov_workspace{T}, algorithm::Symbol) where T

Ensure Krylov method buffers and the requested solver workspace are allocated.
Supported algorithms are `:bicgstab`, `:gmres`, and `:dqgmres`.
"""
@unstable function ensure_lyapunov_krylov_solver!(ws::lyapunov_workspace{T}, algorithm::Symbol) where T
    ensure_lyapunov_krylov_buffers!(ws)
    n = ws.n
    if n == 0
        return ws
    end

    if algorithm == :bicgstab
        if length(ws.bicgstab.x) != n * n
            ws.bicgstab = Krylov.BicgstabWorkspace(n * n, n * n, Vector{T})
        end
    elseif algorithm == :gmres
        if length(ws.gmres.x) != n * n
            ws.gmres = Krylov.GmresWorkspace(n * n, n * n, Vector{T}; memory = 20)
        end
    elseif algorithm == :dqgmres
        if length(ws.dqgmres.x) != n * n
            ws.dqgmres = Krylov.DqgmresWorkspace(n * n, n * n, Vector{T})
        end
    else
        error("Invalid Krylov algorithm: $algorithm. Must be :bicgstab, :gmres, or :dqgmres")
    end

    return ws
end

"""
    ensure_lyapunov_krylov_vech_solver!(ws::lyapunov_workspace{T}, algorithm::Symbol) where T

Ensure vech-space Krylov buffers and solver workspace are allocated for symmetric Lyapunov equations.
The vech dimension is n(n+1)/2 instead of n².
"""
@unstable function ensure_lyapunov_krylov_vech_solver!(ws::lyapunov_workspace{T}, algorithm::Symbol) where T
    ensure_lyapunov_krylov_buffers!(ws)
    n = ws.n
    if n == 0
        return ws
    end
    n_vech = n * (n + 1) ÷ 2

    if length(ws.b_vech) != n_vech
        ws.b_vech = zeros(T, n_vech)
    end

    if algorithm == :bicgstab
        if length(ws.bicgstab_vech.x) != n_vech
            ws.bicgstab_vech = Krylov.BicgstabWorkspace(n_vech, n_vech, Vector{T})
        end
    elseif algorithm == :gmres
        if length(ws.gmres_vech.x) != n_vech
            ws.gmres_vech = Krylov.GmresWorkspace(n_vech, n_vech, Vector{T}; memory = 20)
        end
    elseif algorithm == :dqgmres
        if length(ws.dqgmres_vech.x) != n_vech
            ws.dqgmres_vech = Krylov.DqgmresWorkspace(n_vech, n_vech, Vector{T})
        end
    end

    return ws
end

# ============================================================================
# Sylvester workspace ensure functions
# ============================================================================

"""
    ensure_sylvester_doubling_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T

Ensure the doubling algorithm buffers are allocated in the workspace.
`n` is the row dimension (size of A), `m` is the column dimension (size of B).
"""
@unstable function ensure_sylvester_doubling_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T
    # Update stored dimensions
    ws.n = n
    ws.m = m
    
    # A-related buffers (n×n)
    if size(ws.𝐀, 1) != n || size(ws.𝐀, 2) != n
        ws.𝐀 = zeros(T, n, n)
    end
    if size(ws.𝐀¹, 1) != n || size(ws.𝐀¹, 2) != n
        ws.𝐀¹ = zeros(T, n, n)
    end
    
    # B-related buffers (m×m)
    if size(ws.𝐁, 1) != m || size(ws.𝐁, 2) != m
        ws.𝐁 = zeros(T, m, m)
    end
    if size(ws.𝐁¹, 1) != m || size(ws.𝐁¹, 2) != m
        ws.𝐁¹ = zeros(T, m, m)
    end
    
    # C-related buffers (n×m)
    if size(ws.𝐂_dbl, 1) != n || size(ws.𝐂_dbl, 2) != m
        ws.𝐂_dbl = zeros(T, n, m)
    end
    if size(ws.𝐂¹, 1) != n || size(ws.𝐂¹, 2) != m
        ws.𝐂¹ = zeros(T, n, m)
    end
    if size(ws.𝐂B, 1) != n || size(ws.𝐂B, 2) != m
        ws.𝐂B = zeros(T, n, m)
    end
    
    return ws
end

"""
    ensure_sylvester_krylov_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T

Ensure the Krylov method buffers are allocated in the workspace.
"""
@unstable function ensure_sylvester_krylov_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T
    ws.n = n
    ws.m = m
    
    # All are n×m matrices
    if size(ws.tmp, 1) != n || size(ws.tmp, 2) != m
        ws.tmp = zeros(T, n, m)
    end
    if size(ws.𝐗, 1) != n || size(ws.𝐗, 2) != m
        ws.𝐗 = zeros(T, n, m)
    end
    if size(ws.𝐂, 1) != n || size(ws.𝐂, 2) != m
        ws.𝐂 = zeros(T, n, m)
    end
    
    return ws
end

"""
    ensure_find_shocks_buffers!(ws::find_shocks_workspace{T}, n_exo::Int; third_order::Bool = false) where T

Ensure the find_shocks workspaces are allocated for the given number of shocks.
Only allocates 3rd order buffers if third_order=true.
Buffer sizes: kron_buffer (n_exo^2), kron_buffer2 (n_exo^2 × n_exo), 
              kron_buffer² (n_exo^3), kron_buffer3 (n_exo^3 × n_exo), kron_buffer4 (n_exo^3 × n_exo^2)
"""
@unstable function ensure_find_shocks_buffers!(ws::find_shocks_workspace{T}, n_exo::Int; third_order::Bool = false) where T
    ws.n_exo = n_exo
    
    n_exo² = n_exo^2
    n_exo³ = n_exo^3
    
    # 2nd order buffers (always needed)
    if length(ws.kron_buffer) != n_exo²
        ws.kron_buffer = zeros(T, n_exo²)
    end
    if size(ws.kron_buffer2, 1) != n_exo² || size(ws.kron_buffer2, 2) != n_exo
        ws.kron_buffer2 = zeros(T, n_exo², n_exo)
    end
    
    # 3rd order buffers (only if needed)
    if third_order
        if length(ws.kron_buffer²) != n_exo³
            ws.kron_buffer² = zeros(T, n_exo³)
        end
        if size(ws.kron_buffer3, 1) != n_exo³ || size(ws.kron_buffer3, 2) != n_exo
            ws.kron_buffer3 = zeros(T, n_exo³, n_exo)
        end
        if size(ws.kron_buffer4, 1) != n_exo³ || size(ws.kron_buffer4, 2) != n_exo²
            ws.kron_buffer4 = zeros(T, n_exo³, n_exo²)
        end
    end
    
    return ws
end

"""
    ensure_find_shocks_state_buffers!(ws, n_exo, n_past; third_order=false, third_order_pruning=false)

Ensure state-related kron buffers used by `find_shocks_conditional_forecast` are sized
for the given dimensions. `n_exo` is `T.nExo`; `n_past` is `T.nPast_not_future_and_mixed`.
The `state_vol` vector has length `n_past+1`. 3rd-order-only and pruning-only buffers
are sized only when those flags are set.
"""
@unstable function ensure_find_shocks_state_buffers!(ws::find_shocks_workspace{T}, n_exo::Int, n_past::Int;
                                           third_order::Bool = false,
                                           third_order_pruning::Bool = false) where T
    ws.n_past = n_past
    n_aug = n_past + 1

    if length(ws.kron_state_vol) != n_aug^2
        ws.kron_state_vol = zeros(T, n_aug^2)
    end
    if size(ws.kron_I_state) != (n_exo * n_aug, n_exo)
        ws.kron_I_state = zeros(T, n_exo * n_aug, n_exo)
    end

    if third_order
        if length(ws.kron_state_vol3) != n_aug^3
            ws.kron_state_vol3 = zeros(T, n_aug^3)
        end
        if size(ws.kron_I_state_state) != (n_exo * n_aug^2, n_exo)
            ws.kron_I_state_state = zeros(T, n_exo * n_aug^2, n_exo)
        end
        if third_order_pruning
            if length(ws.kron_state₁₂) != n_past^2
                ws.kron_state₁₂ = zeros(T, n_past^2)
            end
            if size(ws.kron_I_state₂) != (n_exo * n_past, n_exo)
                ws.kron_I_state₂ = zeros(T, n_exo * n_past, n_exo)
            end
        end
    end

    return ws
end


"""
    Inversion_workspace(;T::Type = Float64)

Create a workspace for inversion filter computations with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_inversion_buffers!.
"""
@unstable function Inversion_workspace(;T::Type = Float64)
    inversion_workspace{T}(
        0, 0,                   # n_exo, n_past dimensions
        zeros(T, 0),            # kron_buffer (n_exo^2)
        zeros(T, 0, 0),         # kron_buffer2 (n_exo^2 × n_exo)
        zeros(T, 0),            # kron_buffer² (n_exo^3)
        zeros(T, 0, 0),         # kron_buffer3 (n_exo^3 × n_exo)
        zeros(T, 0, 0),         # kron_buffer4 (n_exo^3 × n_exo^2)
        zeros(T, 0, 0),         # kron_buffer_state (n_exo × n_past+1)
        zeros(T, 0),            # kronstate_vol ((n_past+1)^2)
        zeros(T, 0),            # kronaug_state ((n_past+1+n_exo)^2)
        zeros(T, 0),            # kron_kron_aug_state ((n_past+1+n_exo)^3)
        zeros(T, 0),            # state_vol (n_past+1)
        zeros(T, 0),            # aug_state₁ (n_past+1+n_exo)
        zeros(T, 0),            # aug_state₂ (n_past+1+n_exo)
        # Estimation loop temporaries
        0,                      # n_cond_var
        zeros(T, 0),            # shock_independent (n_cond_var)
        zeros(T, 0),            # init_guess (n_exo)
        zeros(T, 0, 0),         # Si_buffer (n_cond_var × n_exo)
        zeros(T, 0, 0),         # jacc_buffer (n_cond_var × n_exo)
        zeros(T, 0, 0),         # Si2e_buffer (n_cond_var × n_exo^2)
        zeros(T, 0),            # y_obs (n_cond_var)
        zeros(T, 0),            # x_shocks (n_exo)
        zeros(T, 0),            # state_concat (n_past + n_exo)
        zeros(T, 0),            # aug_state₃ (n_past+1+n_exo)
        zeros(T, 0),            # aug_state₁̂ (n_past+1+n_exo)
        zeros(T, 0),            # state²⁻_vol (n_past+1)
        zeros(T, 0),            # kronstate_vol³ ((n_past+1)^3)
        # Pullback buffers (for reverse-mode AD)
        zeros(T, 0, 0),         # ∂_tmp1 (n_exo × n_past+n_exo)
        zeros(T, 0, 0),         # ∂_tmp2 (n_past × n_past+n_exo)
        zeros(T, 0),            # ∂_tmp3 (n_past+n_exo)
        zeros(T, 0, 0),         # ∂𝐒t⁻ (n_past × n_past+n_exo)
        zeros(T, 0, 0),         # ∂data (n_past × n_periods)
        # Pullback buffers for pruned second order
        zeros(T, 0, 0),         # ∂𝐒ⁱ²ᵉtmp (n_exo × n_exo*n_obs)
        zeros(T, 0, 0),         # ∂𝐒ⁱ²ᵉtmp2 (n_obs × n_exo^2)
        zeros(T, 0),            # kronSλ (n_obs * n_exo)
        zeros(T, 0))            # kronxS (n_exo * n_obs)
end


"""
    ensure_inversion_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_past::Int; third_order::Bool = false) where T

Ensure the inversion workspaces are allocated for the given dimensions.
Only allocates 3rd order buffers if third_order=true.
"""
@unstable function ensure_inversion_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_past::Int; third_order::Bool = false) where T
    ws.n_exo = n_exo
    ws.n_past = n_past
    
    n_exo² = n_exo^2
    n_exo³ = n_exo^3
    n_state_vol = n_past + 1
    n_aug = n_past + 1 + n_exo
    
    # Shock-related kron buffers (2nd order)
    if length(ws.kron_buffer) != n_exo²
        ws.kron_buffer = zeros(T, n_exo²)
    end
    if size(ws.kron_buffer2, 1) != n_exo² || size(ws.kron_buffer2, 2) != n_exo
        ws.kron_buffer2 = zeros(T, n_exo², n_exo)
    end
    
    # Shock-related kron buffers (3rd order)
    if third_order
        if length(ws.kron_buffer²) != n_exo³
            ws.kron_buffer² = zeros(T, n_exo³)
        end
        if size(ws.kron_buffer3, 1) != n_exo³ || size(ws.kron_buffer3, 2) != n_exo
            ws.kron_buffer3 = zeros(T, n_exo³, n_exo)
        end
        if size(ws.kron_buffer4, 1) != n_exo³ || size(ws.kron_buffer4, 2) != n_exo²
            ws.kron_buffer4 = zeros(T, n_exo³, n_exo²)
        end
    end
    
    # State-related kron buffers
    # kron_buffer_state holds kron(J, state_vol) where J is (n_exo, n_exo) and state_vol is (n_state_vol,)
    if size(ws.kron_buffer_state, 1) != n_exo * n_state_vol || size(ws.kron_buffer_state, 2) != n_exo
        ws.kron_buffer_state = zeros(T, n_exo * n_state_vol, n_exo)
    end
    if length(ws.kronstate_vol) != n_state_vol^2
        ws.kronstate_vol = zeros(T, n_state_vol^2)
    end
    if length(ws.kronaug_state) != n_aug^2
        ws.kronaug_state = zeros(T, n_aug^2)
    end
    if third_order
        if length(ws.kron_kron_aug_state) != n_aug^3
            ws.kron_kron_aug_state = zeros(T, n_aug^3)
        end
    end
    
    # State vectors
    if length(ws.state_vol) != n_state_vol
        ws.state_vol = zeros(T, n_state_vol)
    end
    if length(ws.aug_state₁) != n_aug
        ws.aug_state₁ = zeros(T, n_aug)
    end
    if length(ws.aug_state₂) != n_aug
        ws.aug_state₂ = zeros(T, n_aug)
    end
    
    # Estimation loop temporaries (init_guess depends only on n_exo)
    if length(ws.init_guess) != n_exo
        ws.init_guess = zeros(T, n_exo)
    end
    if length(ws.x_shocks) != n_exo
        ws.x_shocks = zeros(T, n_exo)
    end
    if length(ws.state_concat) != n_past + n_exo
        ws.state_concat = zeros(T, n_past + n_exo)
    end
    
    # Augmented state buffers for pruned third-order
    if third_order
        if length(ws.aug_state₃) != n_aug
            ws.aug_state₃ = zeros(T, n_aug)
        end
        if length(ws.aug_state₁̂) != n_aug
            ws.aug_state₁̂ = zeros(T, n_aug)
        end
        if length(ws.state²⁻_vol) != n_state_vol
            ws.state²⁻_vol = zeros(T, n_state_vol)
        end
        if length(ws.kronstate_vol³) != n_state_vol^3
            ws.kronstate_vol³ = zeros(T, n_state_vol^3)
        end
    end
    
    return ws
end


"""
    ensure_inversion_estimation_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_cond_var::Int) where T

Ensure observation-dimension-dependent estimation buffers are allocated.
Call after ensure_inversion_buffers! when the number of conditioning variables (observables) is known.
"""
@unstable function ensure_inversion_estimation_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_cond_var::Int; third_order::Bool = false) where T
    if ws.n_cond_var == n_cond_var && length(ws.shock_independent) == n_cond_var && 
       size(ws.Si_buffer) == (n_cond_var, n_exo)
        return ws
    end
    
    ws.n_cond_var = n_cond_var
    
    if length(ws.shock_independent) != n_cond_var
        ws.shock_independent = zeros(T, n_cond_var)
    end
    if length(ws.y_obs) != n_cond_var
        ws.y_obs = zeros(T, n_cond_var)
    end
    if size(ws.Si_buffer) != (n_cond_var, n_exo)
        ws.Si_buffer = zeros(T, n_cond_var, n_exo)
    end
    if size(ws.jacc_buffer) != (n_cond_var, n_exo)
        ws.jacc_buffer = zeros(T, n_cond_var, n_exo)
    end
    if third_order
        n_exo² = n_exo^2
        if size(ws.Si2e_buffer) != (n_cond_var, n_exo²)
            ws.Si2e_buffer = zeros(T, n_cond_var, n_exo²)
        end
    end
    
    return ws
end


"""
    Kalman_workspace(;T::Type = Float64)

Create a workspace for Kalman filter computations with lazy buffer allocation.
All buffers are initialized to 0-dimensional objects and resized on-demand via ensure_kalman_workspaces!.
"""
@unstable function Kalman_workspace(;T::Type = Float64)
    empty_lu_factors = zeros(T, 1, 1)
    empty_lu_ws = FastLapackInterface.LUWs(empty_lu_factors)

    kalman_workspace{T}(
        0, 0,                   # n_obs, n_states dimensions
        zeros(T, 0),            # u (n_states)
        zeros(T, 0),            # z (n_obs)
        zeros(T, 0),            # ztmp (n_obs)
        zeros(T, 0),            # utmp (n_states)
        zeros(T, 0, 0),         # Ctmp (n_obs × n_states)
        zeros(T, 0, 0),         # 𝐁 (n_states × n_states)
        zeros(T, 0, 0),         # F (n_obs × n_obs)
        zeros(T, 0, 0),         # K (n_states × n_obs)
        zeros(T, 0, 0),         # tmp (n_states × n_states)
        zeros(T, 0, 0),         # Ptmp (n_states × n_states)
        empty_lu_ws,
        (0, 0),
        zeros(T, 0, 0))         # fast_lu_rhs_t_k (n_obs × n_states)
end


"""
    ensure_kalman_workspaces!(workspaces::workspaces, n_obs::Int, n_states::Int)

Ensure the Kalman workspace inside `workspaces` is allocated for the given dimensions and return it.
"""
@unstable function ensure_kalman_workspaces!(workspaces::workspaces, n_obs::Int, n_states::Int)
    ws = workspaces.kalman
    T = eltype(ws.u)

    # Check if dimensions changed
    if ws.n_obs == n_obs && ws.n_states == n_states
        return ws
    end
    
    ws.n_obs = n_obs
    ws.n_states = n_states
    
    # State and observation vectors
    if length(ws.u) != n_states
        ws.u = zeros(T, n_states)
    end
    if length(ws.z) != n_obs
        ws.z = zeros(T, n_obs)
    end
    if length(ws.ztmp) != n_obs
        ws.ztmp = zeros(T, n_obs)
    end
    if length(ws.utmp) != n_states
        ws.utmp = zeros(T, n_states)
    end
    
    # Matrix buffers
    if size(ws.Ctmp, 1) != n_obs || size(ws.Ctmp, 2) != n_states
        ws.Ctmp = zeros(T, n_obs, n_states)
    end
    if size(ws.𝐁, 1) != n_states || size(ws.𝐁, 2) != n_states
        ws.𝐁 = zeros(T, n_states, n_states)
    end
    if size(ws.F, 1) != n_obs || size(ws.F, 2) != n_obs
        ws.F = zeros(T, n_obs, n_obs)
    end
    if size(ws.K, 1) != n_states || size(ws.K, 2) != n_obs
        ws.K = zeros(T, n_states, n_obs)
    end
    if size(ws.tmp, 1) != n_states || size(ws.tmp, 2) != n_states
        ws.tmp = zeros(T, n_states, n_states)
    end
    if size(ws.Ptmp, 1) != n_states || size(ws.Ptmp, 2) != n_states
        ws.Ptmp = zeros(T, n_states, n_states)
    end
    if size(ws.fast_lu_rhs_t_k, 1) != n_obs || size(ws.fast_lu_rhs_t_k, 2) != n_states
        ws.fast_lu_rhs_t_k = zeros(T, n_obs, n_states)
    end
    
    return ws
end


@unstable function Workspaces(;T::Type{Float64} = Float64, S::Type{Float64} = Float64)
    workspaces(Higher_order_workspace(T = T, S = S),
                Higher_order_workspace(T = T, S = S),
                Float64[],
                First_order_workspace(T = T, S = S),  # Initialize with size 0, will be resized when needed
                Qme_doubling_workspace(0, T = T, S = S),  # Initialize with size 0, will be resized when needed
                Schur_workspace(0, 0, 0, 0, T = T),  # Initialize with size 0, will be resized when needed
                Lyapunov_workspace(0, T = T),  # 1st order - will be resized
                Lyapunov_workspace(0, T = T),  # 2nd order - will be resized
                Lyapunov_workspace(0, T = T),  # 3rd order - will be resized
                Lyapunov_workspace(0, T = T),  # block-triangular inner - will be resized
                Sylvester_workspace(S = S),  # 1st order sylvester - will be resized
                Sylvester_workspace(S = S),  # block-triangular sylvester - will be resized
                Find_shocks_workspace(T = T),  # conditional forecast - will be resized
                Inversion_workspace(T = T),  # inversion filter - will be resized
                Kalman_workspace(T = T),  # Kalman filter - will be resized
                NSSSSolverWorkspace())  # NSSS solver scratch buffers
end

"""
Create a fresh `Workspaces` instance that preserves the NSSS solver workspace from `orig`.

The NSSS solver workspace contains buffers sized during model compilation that cannot be
lazily re-created, unlike other workspace fields which are lazily allocated via `ensure_*!`.
"""
function fresh_workspaces(orig::workspaces)
    ws = Workspaces()
    ws.nsss_solver = orig.nsss_solver
    return ws
end

function Constants(model_struct; T::Type = Float64, S::Type = Float64)
    constants( model_struct,
            post_parameters_macro(
                Symbol[],
                false,
                :single_equation,
                :ESCH,
                120.0,
                Dict{Symbol, Float64}(),
                Set{Symbol}[],
                Set{Symbol}[],
                # Set{Symbol}[],
                # Set{Symbol}[],
                Dict{Symbol,Tuple{Float64,Float64}}()
                ),
            post_complete_parameters{Symbol}(
                Symbol[],
                Symbol[],
                Int[],
                Int[],
                Int[],
                Int[],
                # Int[],
                ℒ.I(0),
                Symbol[],
                Symbol[],
                Symbol[],
                Symbol[],
                # false,
                # false,
                Symbol[],
                # Symbol[],
                # Symbol[],
                # Int[],
                # Symbol[],
                Symbol[],
                spzeros(Float64, 0, 0),
                spzeros(Float64, 0, 0),
                Symbol[],
                Symbol[],
                Int[],
                Int[],
                Symbol[],
                # Symbol[],
                Int[],
                Int[],
                Int[],
                false,
                1:0,
                Int[],
                Int[],
                Int[],
                Int[],
                ℒ.I(0),
                ℒ.I(0),
                1:0,
                1:0,
                1,
                zeros(Bool, 0, 0),
                zeros(Bool, 0, 0),
                Int[],
                Int[],                      # indices_past_not_future_in_comb
                zeros(Bool, 0, 0),          # I_nPast_not_mixed
                zeros(Bool, 0, 0),          # Ir_past_selector
                zeros(Bool, 0, 0),          # schur_Z₊
                zeros(Bool, 0, 0),          # schur_I₊
                zeros(Bool, 0, 0),          # schur_Z₋
                zeros(Bool, 0, 0),          # schur_I₋
                nothing,
                0,
                Int[],
                0,
                Symbol[],
                Int[],
                Symbol[],
                1),
            Second_order_indices(),
            Third_order_indices(),
            NSSSSolverConstants())
end

function axis_has_string(axis)
    axis === nothing && return false
    T = eltype(axis)
    if T === String
        return true
    elseif T === Symbol
        return false
    elseif T <: Union{Symbol, String}
        return !isempty(axis) && any(x -> x isa String, axis)
    end
    return false
end

function choose_axis_type(var_axis, calib_axis, exo_axis_plain, exo_axis_with_subscript, full_NSSS_display)
    return (axis_has_string(var_axis) ||
            axis_has_string(calib_axis) ||
            axis_has_string(exo_axis_plain) ||
            axis_has_string(exo_axis_with_subscript) ||
            axis_has_string(full_NSSS_display)) ? String : Symbol
end

function convert_axis(axis, ::Type{S}) where {S <: Union{Symbol, String}}
    axis === nothing && return Vector{S}()
    return S === String ? string.(axis) : Symbol.(axis)
end

"""
    update_post_parameters_macro(p::post_parameters_macro; kwargs...)

Return a new `post_parameters_macro` where the fields supplied as keyword
arguments replace the corresponding entries on `p`. All other fields are
preserved. Accepted keyword arguments:

- `parameters_as_function_of_parameters`
- `precompile`
- `ss_symbolic_mode`
- `ss_solver_parameters_algorithm`
- `ss_solver_parameters_maxtime`
- `guess`
- `ss_calib_list`
- `par_calib_list`
- `bounds`
"""
function update_post_parameters_macro(p::post_parameters_macro; kwargs...)
    return post_parameters_macro(
        get(kwargs, :parameters_as_function_of_parameters, p.parameters_as_function_of_parameters),
        get(kwargs, :precompile, p.precompile),
        get(kwargs, :ss_symbolic_mode, p.ss_symbolic_mode),
        get(kwargs, :ss_solver_parameters_algorithm, p.ss_solver_parameters_algorithm),
        get(kwargs, :ss_solver_parameters_maxtime, p.ss_solver_parameters_maxtime),
        get(kwargs, :guess, p.guess),
        get(kwargs, :ss_calib_list, p.ss_calib_list),
        get(kwargs, :par_calib_list, p.par_calib_list),
        get(kwargs, :bounds, p.bounds),
    )
end


function update_post_complete_parameters(p::post_complete_parameters; kwargs...)
    var_axis_in = get(kwargs, :var_axis, p.var_axis)
    calib_axis_in = get(kwargs, :calib_axis, p.calib_axis)
    exo_axis_plain_in = get(kwargs, :exo_axis_plain, p.exo_axis_plain)
    exo_axis_with_subscript_in = get(kwargs, :exo_axis_with_subscript, p.exo_axis_with_subscript)
    full_NSSS_display_in = get(kwargs, :full_NSSS_display, p.full_NSSS_display)
    S = choose_axis_type(var_axis_in, calib_axis_in, exo_axis_plain_in, exo_axis_with_subscript_in, full_NSSS_display_in)
    var_axis = convert_axis(var_axis_in, S)
    calib_axis = convert_axis(calib_axis_in, S)
    exo_axis_plain = convert_axis(exo_axis_plain_in, S)
    exo_axis_with_subscript = convert_axis(exo_axis_with_subscript_in, S)
    full_NSSS_display = convert_axis(full_NSSS_display_in, S)
    return post_complete_parameters{S}(
        get(kwargs, :parameters, p.parameters),
        get(kwargs, :missing_parameters, p.missing_parameters),
        get(kwargs, :dyn_var_future_idx, p.dyn_var_future_idx),
        get(kwargs, :dyn_var_present_idx, p.dyn_var_present_idx),
        get(kwargs, :dyn_var_past_idx, p.dyn_var_past_idx),
        get(kwargs, :dyn_ss_idx, p.dyn_ss_idx),
        # get(kwargs, :shocks_ss, p.shocks_ss),
        get(kwargs, :diag_nVars, p.diag_nVars),
        var_axis,
        calib_axis,
        exo_axis_plain,
        exo_axis_with_subscript,
        # get(kwargs, :var_has_curly, p.var_has_curly),
        # get(kwargs, :exo_has_curly, p.exo_has_curly),
        get(kwargs, :SS_and_pars_names, p.SS_and_pars_names),
        # get(kwargs, :all_variables, p.all_variables),
        # get(kwargs, :NSSS_labels, p.NSSS_labels),
        # get(kwargs, :aux_indices, p.aux_indices),
        # get(kwargs, :processed_all_variables, p.processed_all_variables),
        full_NSSS_display,
        get(kwargs, :steady_state_expand_matrix, p.steady_state_expand_matrix),
        get(kwargs, :custom_ss_expand_matrix, p.custom_ss_expand_matrix),
        get(kwargs, :vars_in_ss_equations, p.vars_in_ss_equations),
        get(kwargs, :vars_in_ss_equations_with_aux, p.vars_in_ss_equations_with_aux),
        get(kwargs, :ss_var_idx_in_var_and_calib, p.ss_var_idx_in_var_and_calib),
        get(kwargs, :calib_idx_in_var_and_calib, p.calib_idx_in_var_and_calib),
        get(kwargs, :SS_and_pars_names_lead_lag, p.SS_and_pars_names_lead_lag),
        # get(kwargs, :SS_and_pars_names_no_exo, p.SS_and_pars_names_no_exo),
        get(kwargs, :SS_and_pars_no_exo_idx, p.SS_and_pars_no_exo_idx),
        get(kwargs, :vars_idx_excluding_aux_obc, p.vars_idx_excluding_aux_obc),
        get(kwargs, :vars_idx_excluding_obc, p.vars_idx_excluding_obc),
        get(kwargs, :initialized, p.initialized),
        get(kwargs, :dyn_index, p.dyn_index),
        get(kwargs, :reverse_dynamic_order, p.reverse_dynamic_order),
        get(kwargs, :comb, p.comb),
        get(kwargs, :future_not_past_and_mixed_in_comb, p.future_not_past_and_mixed_in_comb),
        get(kwargs, :past_not_future_and_mixed_in_comb, p.past_not_future_and_mixed_in_comb),
        get(kwargs, :Ir, p.Ir),
        get(kwargs, :I_n, hasfield(typeof(p), :I_n) ? p.I_n : ℒ.I(0)),
        get(kwargs, :nabla_zero_cols, p.nabla_zero_cols),
        get(kwargs, :nabla_minus_cols, p.nabla_minus_cols),
        get(kwargs, :nabla_e_start, p.nabla_e_start),
        get(kwargs, :expand_future, p.expand_future),
        get(kwargs, :expand_past, p.expand_past),
        get(kwargs, :past_not_future_and_mixed_in_present_but_not_only,
            hasfield(typeof(p), :past_not_future_and_mixed_in_present_but_not_only) ? p.past_not_future_and_mixed_in_present_but_not_only : Int[]),
        get(kwargs, :indices_past_not_future_in_comb, hasfield(typeof(p), :indices_past_not_future_in_comb) ? p.indices_past_not_future_in_comb : Int[]),
        get(kwargs, :I_nPast_not_mixed, hasfield(typeof(p), :I_nPast_not_mixed) ? p.I_nPast_not_mixed : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :Ir_past_selector, hasfield(typeof(p), :Ir_past_selector) ? p.Ir_past_selector : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :schur_Z₊, hasfield(typeof(p), :schur_Z₊) ? p.schur_Z₊ : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :schur_I₊, hasfield(typeof(p), :schur_I₊) ? p.schur_I₊ : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :schur_Z₋, hasfield(typeof(p), :schur_Z₋) ? p.schur_Z₋ : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :schur_I₋, hasfield(typeof(p), :schur_I₋) ? p.schur_I₋ : Matrix{Bool}(undef, 0, 0)),
        get(kwargs, :nsss_dependencies, p.nsss_dependencies),
        get(kwargs, :nsss_n_sol, p.nsss_n_sol),
        get(kwargs, :nsss_output_indices, p.nsss_output_indices),
        get(kwargs, :nsss_n_ext_params, p.nsss_n_ext_params),
        get(kwargs, :nsss_sol_names, p.nsss_sol_names),
        get(kwargs, :nsss_exo_zero_indices, p.nsss_exo_zero_indices),
        get(kwargs, :nsss_param_names_ext, p.nsss_param_names_ext),
        get(kwargs, :nsss_fastest_solver_parameter_idx, p.nsss_fastest_solver_parameter_idx),
    )
end

# Initialize all commonly used constants at once (call at entry points)
# This reduces repeated ensure_*! calls throughout the codebase
function initialise_constants!(𝓂)
    ensure_computational_constants!(𝓂.constants)
    ensure_name_display_constants!(𝓂)
    ensure_first_order_constants!(𝓂.constants)
    return 𝓂.constants
end

@unstable function ensure_name_display_constants!(𝓂)
    constants = 𝓂.constants
    # Use model from constants
    T = constants.post_model_macro
    
    if isempty(constants.post_complete_parameters.var_axis)
        var_has_curly = any(x -> contains(string(x), "◖"), T.var)
        if var_has_curly
            var_decomposed = decompose_name.(T.var)
            var_axis = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in var_decomposed]
        else
            var_axis = T.var
        end

        if var_has_curly
            calib_axis = replace.(string.(𝓂.equations.calibration_parameters), "◖" => "{", "◗" => "}")
        else
            calib_axis = 𝓂.equations.calibration_parameters
        end

        exo_has_curly = any(x -> contains(string(x), "◖"), T.exo)
        if exo_has_curly
            exo_decomposed = decompose_name.(T.exo)
            exo_axis_plain = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in exo_decomposed]
            exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), exo_axis_plain)
        else
            exo_axis_plain = T.exo
            exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), T.exo)
        end

        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            var_axis = var_axis,
            calib_axis = calib_axis,
            exo_axis_plain = exo_axis_plain,
            exo_axis_with_subscript = exo_axis_with_subscript,
            var_has_curly = var_has_curly,
            exo_has_curly = exo_has_curly,
        )
    end

    return constants.post_complete_parameters
end


function set_up_name_display_cache(T::post_model_macro, calibration_equations_parameters)
    var_has_curly = any(x -> contains(string(x), "◖"), T.var)
    if var_has_curly
        var_decomposed = decompose_name.(T.var)
        var_axis = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in var_decomposed]
    else
        var_axis = T.var
    end

    if var_has_curly
        calib_axis = replace.(string.(calibration_equations_parameters), "◖" => "{", "◗" => "}")
    else
        calib_axis = calibration_equations_parameters
    end

    exo_has_curly = any(x -> contains(string(x), "◖"), T.exo)
    if exo_has_curly
        exo_decomposed = decompose_name.(T.exo)
        exo_axis_plain = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in exo_decomposed]
        exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), exo_axis_plain)
    else
        exo_axis_plain = T.exo
        exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), T.exo)
    end

    return (
        var_axis = var_axis,
        calib_axis = calib_axis,
        exo_axis_plain = exo_axis_plain,
        exo_axis_with_subscript = exo_axis_with_subscript,
        var_has_curly = var_has_curly,
        exo_has_curly = exo_has_curly,
    )
end


@unstable function ensure_computational_constants!(constants::constants)
    so = constants.second_order
    if isempty(so.s_in_s⁺)
        # Use timings from constants
        T = constants.post_model_macro
        nᵉ = T.nExo
        nˢ = T.nPast_not_future_and_mixed

        s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ + 1), zeros(Bool, nᵉ)))
        s_in_s = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)

        kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
        kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)

        e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_e_s = ℒ.kron(e_in_s⁺, s_in_s⁺)

        # Compute sparse index patterns for filter operations
        shockvar_idxs = sparse(ℒ.kron(e_in_s⁺, s_in_s⁺)).nzind
        shock_idxs = sparse(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)).nzind
        shock_idxs2 = sparse(ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)).nzind
        shock²_idxs = sparse(ℒ.kron(e_in_s⁺, e_in_s⁺)).nzind
        var_vol²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind

        so.s_in_s⁺ = s_in_s⁺
        so.s_in_s = s_in_s
        so.kron_s⁺_s⁺ = kron_s⁺_s⁺
        so.kron_s⁺_s = kron_s⁺_s
        so.kron_s⁺_s⁺_s⁺ = kron_s⁺_s⁺_s⁺
        so.kron_s_s⁺_s⁺ = kron_s_s⁺_s⁺
        so.e_in_s⁺ = e_in_s⁺
        so.v_in_s⁺ = v_in_s⁺
        so.kron_s_s = kron_s_s
        so.kron_e_e = kron_e_e
        so.kron_v_v = kron_v_v
        so.kron_e_s = kron_e_s
        so.shockvar_idxs = shockvar_idxs
        so.shock_idxs = shock_idxs
        so.shock_idxs2 = shock_idxs2
        so.shock²_idxs = shock²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    return constants.second_order
end

@unstable function ensure_conditional_forecast_constants!(constants::constants; third_order::Bool = false)
    so = ensure_computational_constants!(constants)

    if isempty(so.var²_idxs)
        s_in_s⁺ = so.s_in_s
        e_in_s⁺ = so.e_in_s⁺

        shock_idxs = so.shock_idxs
        shock²_idxs = so.shock²_idxs
        shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)
        var_vol²_idxs = so.var_vol²_idxs
        var²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind
        so.var²_idxs = var²_idxs
        so.shockvar²_idxs = shockvar²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    if third_order
        to = constants.third_order
        if isempty(to.var_vol³_idxs)
            sv_in_s⁺ = so.s_in_s⁺
            e_in_s⁺ = so.e_in_s⁺
            ones_e = zero(e_in_s⁺) .+ 1

            var_vol³_idxs = sparse(ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺))).nzind
            shock_idxs2 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, ones_e), ones_e)).nzind
            shock_idxs3 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), ones_e)).nzind
            shock³_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar1_idxs = sparse(ℒ.kron(ones_e, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar2_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(ones_e, e_in_s⁺))).nzind
            shockvar3_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, ones_e))).nzind
            shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
            shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)

            to.var_vol³_idxs = var_vol³_idxs
            to.shock_idxs2 = shock_idxs2
            to.shock_idxs3 = shock_idxs3
            to.shock³_idxs = shock³_idxs
            to.shockvar1_idxs = shockvar1_idxs
            to.shockvar2_idxs = shockvar2_idxs
            to.shockvar3_idxs = shockvar3_idxs
            to.shockvar³2_idxs = shockvar³2_idxs
            to.shockvar³_idxs = shockvar³_idxs
        end
    end

    return so
end

function build_first_order_index_cache(T, I_nVars)
    dyn_index = T.nPresent_only + 1:T.nVars

    reverse_dynamic_order_tmp = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    if any(isnothing.(reverse_dynamic_order_tmp))
        reverse_dynamic_order = Int[]
    else
        reverse_dynamic_order = Int.(reverse_dynamic_order_tmp)
    end
    
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb_tmp = indexin(T.future_not_past_and_mixed_idx, comb)
    
    if any(isnothing.(future_not_past_and_mixed_in_comb_tmp))
        future_not_past_and_mixed_in_comb = Int[]
    else
        future_not_past_and_mixed_in_comb = Int.(future_not_past_and_mixed_in_comb_tmp)
    end

    past_not_future_and_mixed_in_comb_tmp = indexin(T.past_not_future_and_mixed_idx, comb)
    
    if any(isnothing.(past_not_future_and_mixed_in_comb_tmp))
        past_not_future_and_mixed_in_comb = Int[]
    else
        past_not_future_and_mixed_in_comb = Int.(past_not_future_and_mixed_in_comb_tmp)
    end

    Ir = ℒ.I(length(comb))

    nabla_zero_cols = (T.nFuture_not_past_and_mixed + 1):(T.nFuture_not_past_and_mixed + T.nVars)
    nabla_minus_cols = (T.nFuture_not_past_and_mixed + T.nVars + 1):(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed)
    nabla_e_start = T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1

    expand_future = I_nVars[T.future_not_past_and_mixed_idx,:]
    expand_past = I_nVars[T.past_not_future_and_mixed_idx,:]

    past_not_future_and_mixed_in_present_but_not_only_tmp = indexin(T.past_not_future_and_mixed_idx, T.present_but_not_only_idx)
    if any(isnothing.(past_not_future_and_mixed_in_present_but_not_only_tmp))
        past_not_future_and_mixed_in_present_but_not_only = Int[]
    else
        past_not_future_and_mixed_in_present_but_not_only = Int.(past_not_future_and_mixed_in_present_but_not_only_tmp)
    end

    # Schur QME cached indices and constant matrices
    indices_past_not_future_in_comb_tmp = indexin(T.past_not_future_idx, comb)
    if any(isnothing.(indices_past_not_future_in_comb_tmp))
        indices_past_not_future_in_comb = Int[]
    else
        indices_past_not_future_in_comb = Int.(indices_past_not_future_in_comb_tmp)
    end

    I_nPast = ℒ.I(T.nPast_not_future_and_mixed)
    I_nPast_not_mixed = Matrix{Bool}(I_nPast[T.not_mixed_in_past_idx, :])
    Ir_past_selector = Matrix{Bool}(Ir[past_not_future_and_mixed_in_comb, :])
    I_n = ℒ.I(T.nVars - T.nPresent_only)
    
    schur_Z₊ = zeros(Bool, T.nMixed, T.nFuture_not_past_and_mixed)
    I_nFuture = ℒ.I(T.nFuture_not_past_and_mixed)
    schur_I₊ = Matrix{Bool}(I_nFuture[T.mixed_in_future_idx, :])
    
    schur_Z₋ = zeros(Bool, T.nMixed, T.nPast_not_future_and_mixed)
    schur_I₋ = Matrix{Bool}(I_nPast[T.mixed_in_past_idx, :])

    return (
        initialized = true,
        dyn_index = dyn_index,
        reverse_dynamic_order = reverse_dynamic_order,
        comb = comb,
        future_not_past_and_mixed_in_comb = future_not_past_and_mixed_in_comb,
        past_not_future_and_mixed_in_comb = past_not_future_and_mixed_in_comb,
        Ir = Ir,
        I_n = I_n,
        nabla_zero_cols = nabla_zero_cols,
        nabla_minus_cols = nabla_minus_cols,
        nabla_e_start = nabla_e_start,
        expand_future = expand_future,
        expand_past = expand_past,
        past_not_future_and_mixed_in_present_but_not_only = past_not_future_and_mixed_in_present_but_not_only,
        indices_past_not_future_in_comb = indices_past_not_future_in_comb,
        I_nPast_not_mixed = I_nPast_not_mixed,
        Ir_past_selector = Ir_past_selector,
        schur_Z₊ = schur_Z₊,
        schur_I₊ = schur_I₊,
        schur_Z₋ = schur_Z₋,
        schur_I₋ = schur_I₋,
    )
end

@unstable function ensure_first_order_constants!(constants::constants)
    if !constants.post_complete_parameters.initialized
        # Use timings from constants if available
        T = constants.post_model_macro
        diag_nVars = constants.post_complete_parameters.diag_nVars
        if size(diag_nVars, 1) == 0
            diag_nVars = ℒ.I(T.nVars)
        end
        cache = build_first_order_index_cache(T, diag_nVars)
        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            diag_nVars = diag_nVars,
            initialized = cache.initialized,
            dyn_index = cache.dyn_index,
            reverse_dynamic_order = cache.reverse_dynamic_order,
            comb = cache.comb,
            future_not_past_and_mixed_in_comb = cache.future_not_past_and_mixed_in_comb,
            past_not_future_and_mixed_in_comb = cache.past_not_future_and_mixed_in_comb,
            Ir = cache.Ir,
            I_n = cache.I_n,
            nabla_zero_cols = cache.nabla_zero_cols,
            nabla_minus_cols = cache.nabla_minus_cols,
            nabla_e_start = cache.nabla_e_start,
            expand_future = cache.expand_future,
            expand_past = cache.expand_past,
            past_not_future_and_mixed_in_present_but_not_only = cache.past_not_future_and_mixed_in_present_but_not_only,
            indices_past_not_future_in_comb = cache.indices_past_not_future_in_comb,
            I_nPast_not_mixed = cache.I_nPast_not_mixed,
            Ir_past_selector = cache.Ir_past_selector,
            schur_Z₊ = cache.schur_Z₊,
            schur_I₊ = cache.schur_I₊,
            schur_Z₋ = cache.schur_Z₋,
            schur_I₋ = cache.schur_I₋,
        )
    end
    return constants.post_complete_parameters
end


"""
    ensure_qme_doubling_workspace!(workspaces, n)

Ensure the QME doubling workspace has dimension `n`.
If the workspace is the wrong size, it is reallocated.
"""
@unstable function ensure_qme_doubling_workspace!(workspaces::workspaces, n::Int)
    ws = workspaces.qme_doubling
    if size(ws.E, 1) != n
        workspaces.qme_doubling = Qme_doubling_workspace(n)
    end
    return workspaces.qme_doubling
end

"""
    ensure_third_order_pullback_workspaces!(ℂ, S, T, M₂, M₃)

Ensure workspace buffers for the third-order pullback are allocated with correct dimensions.
Only dense intermediate-product temporaries are workspace-backed; gradient accumulators for
∇₂, ∇₃, 𝐒₂ and "may be sparse" matrices are freshly allocated via `zero()` inside the
pullback to preserve their sparse/dense format.
"""
@unstable function ensure_third_order_pullback_workspaces!(ℂ::higher_order_workspace, ::Type{S}, T, M₂, M₃) where S
    n      = T.nVars
    n₊     = T.nFuture_not_past_and_mixed
    n₋     = T.nPast_not_future_and_mixed
    nₑ     = T.nExo
    nₑ₋    = n₋ + 1 + nₑ
    n_stack = n₊ + n + n₋ + nₑ

    # Structural dimensions from constants
    n_∇₂     = size(M₂.𝐔∇₂, 2)
    n_𝐂₃_r   = size(M₃.𝐂₃, 1)
    n_𝐂₃     = size(M₃.𝐂₃, 2)
    σ_c       = size(M₂.𝛔, 2)
    n_out2_c  = σ_c * nₑ₋

    # Dense workspace: always-dense gradient accumulators (matches main branch)
    size(ℂ.∂spinv_3rd)          == (n, n)             || (ℂ.∂spinv_3rd = zeros(S, n, n))
    size(ℂ.∂∇₁_3rd)            == (n, n_stack)       || (ℂ.∂∇₁_3rd = zeros(S, n, n_stack))
    size(ℂ.∂𝐒₁_3rd)            == (n, nₑ₋)          || (ℂ.∂𝐒₁_3rd = zeros(S, n, nₑ₋))

    # Dense workspace: intermediate-product temporaries (overwritten by mul! each call)
    size(ℂ.∂A_3rd)              == (n, n)             || (ℂ.∂A_3rd = zeros(S, n, n))
    size(ℂ.∂∇₁₊𝐒₁➕∇₁₀_3rd)   == (n, n)             || (ℂ.∂∇₁₊𝐒₁➕∇₁₀_3rd = zeros(S, n, n))
    size(ℂ.mul_tmp_3rd)         == (n, n)             || (ℂ.mul_tmp_3rd = zeros(S, n, n))
    size(ℂ.∂B_sylv_3rd)        == (n_𝐂₃, n_𝐂₃)     || (ℂ.∂B_sylv_3rd = zeros(S, n_𝐂₃, n_𝐂₃))
    size(ℂ.∂𝐗₃_3rd)            == (n, n_𝐂₃)         || (ℂ.∂𝐗₃_3rd = zeros(S, n, n_𝐂₃))
    size(ℂ.∂𝐗₃_pre_3rd)        == (n, n_𝐂₃_r)      || (ℂ.∂𝐗₃_pre_3rd = zeros(S, n, n_𝐂₃_r))
    size(ℂ.∂out2_3rd)          == (n, n_out2_c)      || (ℂ.∂out2_3rd = zeros(S, n, n_out2_c))
    size(ℂ.∇₂t_∂out2_3rd)     == (n_∇₂, n_out2_c)  || (ℂ.∇₂t_∂out2_3rd = zeros(S, n_∇₂, n_out2_c))

    # Pullback gradient accumulator buffers (zeroed at start of each pullback call)
    size(ℂ.∂𝐒₁₊╱𝟎_tmp_3rd)    == (n_stack, nₑ₋)    || (ℂ.∂𝐒₁₊╱𝟎_tmp_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂𝐒₂₊╱𝟎_3rd)        == (n_stack, nₑ₋^2)  || (ℂ.∂𝐒₂₊╱𝟎_3rd = zeros(S, n_stack, nₑ₋^2))
    size(ℂ.∂L_c_3rd)           == (n_stack, nₑ₋)    || (ℂ.∂L_c_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂R_c_3rd)           == (n_stack, nₑ₋^2)  || (ℂ.∂R_c_3rd = zeros(S, n_stack, nₑ₋^2))
    size(ℂ.∂L_d_3rd)           == (n_stack, nₑ₋)    || (ℂ.∂L_d_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂R_d_3rd)           == (n_stack, nₑ₋^2)  || (ℂ.∂R_d_3rd = zeros(S, n_stack, nₑ₋^2))
    size(ℂ.∂𝐒₁₋╱𝟏ₑ_t8_3rd)   == (nₑ₋, nₑ₋)       || (ℂ.∂𝐒₁₋╱𝟏ₑ_t8_3rd = zeros(S, nₑ₋, nₑ₋))
    size(ℂ.∂𝐒₂₋╱𝟎_3rd)        == (nₑ₋, nₑ₋^2)     || (ℂ.∂𝐒₂₋╱𝟎_3rd = zeros(S, nₑ₋, nₑ₋^2))
    size(ℂ.∂𝐒₁₋╱𝟏ₑ_3rd)      == (nₑ₋, nₑ₋)       || (ℂ.∂𝐒₁₋╱𝟏ₑ_3rd = zeros(S, nₑ₋, nₑ₋))
    size(ℂ.∂𝐒₁₊╱𝟎_3rd)        == (n_stack, nₑ₋)    || (ℂ.∂𝐒₁₊╱𝟎_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂𝐒₁₊╱𝟎_tk0_3rd)    == (n_stack, nₑ₋)    || (ℂ.∂𝐒₁₊╱𝟎_tk0_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋_3rd) == (n_stack, nₑ₋) || (ℂ.∂⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂aux_3rd)           == (n_stack, nₑ₋)    || (ℂ.∂aux_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂tmpkron0_σ_3rd)   == (nₑ₋^2, nₑ₋^2)   || (ℂ.∂tmpkron0_σ_3rd = zeros(S, nₑ₋^2, nₑ₋^2))
    size(ℂ.∂∇₁₊_3rd)          == (n, n)             || (ℂ.∂∇₁₊_3rd = zeros(S, n, n))
    size(ℂ.∂S1S1_from_ck_3rd) == (n_stack, nₑ₋)    || (ℂ.∂S1S1_from_ck_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂S1p0_kron_sigma_3rd) == (n_stack^2, σ_c) || (ℂ.∂S1p0_kron_sigma_3rd = zeros(S, n_stack^2, σ_c))
    size(ℂ.∂S1p0_left_3rd)    == (n_stack, nₑ₋)    || (ℂ.∂S1p0_left_3rd = zeros(S, n_stack, nₑ₋))
    size(ℂ.∂S1p0_right_3rd)   == (n_stack, nₑ₋)    || (ℂ.∂S1p0_right_3rd = zeros(S, n_stack, nₑ₋))

    return ℂ
end

"""
    ensure_first_order_workspace_buffers!(ws, T, n_dyn, n_comb)

Ensure all first-order perturbation buffers in `first_order_workspace` are allocated with
the correct dimensions.
"""
@unstable function ensure_first_order_workspace_buffers!(ws::first_order_workspace{R,S}, T, n_dyn::Int, n_comb::Int) where {R <: Real, S <: Real}
    n = T.nVars
    n₊ = T.nFuture_not_past_and_mixed
    n₋ = T.nPast_not_future_and_mixed
    nₑ = T.nExo
    nᵤ = T.nPresent_only
    n₀ᵤ = length(T.present_but_not_only_idx)

    size(ws.𝐀₊) == (n, n₊) || (ws.𝐀₊ = zeros(R, n, n₊))
    size(ws.𝐀₀) == (n, n) || (ws.𝐀₀ = zeros(R, n, n))
    size(ws.𝐀₋) == (n, n₋) || (ws.𝐀₋ = zeros(R, n, n₋))

    size(ws.𝐀̃₊) == (n_dyn, n_comb) || (ws.𝐀̃₊ = zeros(R, n_dyn, n_comb))
    size(ws.𝐀̃₀) == (n_dyn, n_comb) || (ws.𝐀̃₀ = zeros(R, n_dyn, n_comb))
    size(ws.𝐀̃₋) == (n_dyn, n_comb) || (ws.𝐀̃₋ = zeros(R, n_dyn, n_comb))

    size(ws.𝐀̄₀ᵤ) == (nᵤ, nᵤ) || (ws.𝐀̄₀ᵤ = zeros(R, nᵤ, nᵤ))
    size(ws.𝐀₊ᵤ) == (nᵤ, n₊) || (ws.𝐀₊ᵤ = zeros(R, nᵤ, n₊))
    size(ws.𝐀̃₀ᵤ) == (nᵤ, n₀ᵤ) || (ws.𝐀̃₀ᵤ = zeros(R, nᵤ, n₀ᵤ))
    size(ws.𝐀₋ᵤ) == (nᵤ, n₋) || (ws.𝐀₋ᵤ = zeros(R, nᵤ, n₋))

    size(ws.𝐧ₚ₋) == (nᵤ, n₋) || (ws.𝐧ₚ₋ = zeros(R, nᵤ, n₋))
    size(ws.𝐌) == (n₊, n) || (ws.𝐌 = zeros(R, n₊, n))
    size(ws.𝐀) == (n, n₋) || (ws.𝐀 = zeros(R, n, n₋))
    size(ws.∇₀) == (n, n) || (ws.∇₀ = zeros(R, n, n))
    size(ws.∇ₑ) == (n, nₑ) || (ws.∇ₑ = zeros(R, n, nₑ))

    return ws
end

@unstable function ensure_first_order_cotangent_buffer!(ws::first_order_workspace{T}, n::Int) where T <: Real
    if length(ws.∂∇₁_vec) != n
        ws.∂∇₁_vec = zeros(T, n)
    end

    return ws.∂∇₁_vec
end

@unstable function ensure_higher_order_cotangent_buffer!(ws::higher_order_workspace{T}, n::Int) where T <: Real
    if length(ws.∂∇_vec) != n
        ws.∂∇_vec = zeros(T, n)
    end

    return ws.∂∇_vec
end

"""
    ensure_schur_workspace!(workspaces, n, nMixed, nPfm, nFnpm)

Ensure the schur workspace is properly sized for the model.
Dimensions are:
- `n = nVars - nPresent_only` (dynamic variables)
- `nMixed` (mixed timing variables)
- `nPfm = nPast_not_future_and_mixed`
- `nFnpm = nFuture_not_past_and_mixed`

If the workspace is the wrong size, it will be reallocated.
"""
@unstable function ensure_schur_workspace!(workspaces::workspaces, n::Int, nMixed::Int, nPfm::Int, nFnpm::Int)
    workspaces.schur = ensure_schur_workspace!(workspaces.schur, n, nMixed, nPfm, nFnpm)
    return workspaces.schur
end

@unstable function ensure_schur_workspace!(ws::schur_workspace{T}, n::Int, nMixed::Int, nPfm::Int, nFnpm::Int) where T
    companion_size = n + nMixed
    if size(ws.D, 1) != companion_size ||
       size(ws.sol) != (n, nPfm) ||
       size(ws.Z₁₁) != (nPfm, nPfm) ||
       size(ws.Z₂₁) != (nFnpm, nPfm)
        return Schur_workspace(n, nMixed, nPfm, nFnpm, T = T)
    end
    return ws
end

"""
    ensure_lyapunov_workspace!(workspaces, n, order::Symbol)

Ensure the Lyapunov workspace for the specified moment order is properly sized.
`n` is the dimension of the square matrices.
`order` should be `:first_order`, `:second_order`, or `:third_order`.
If the workspace is the wrong size, it will be reallocated.
Note: buffers are still lazily allocated when algorithms are actually used.
"""
@unstable function ensure_lyapunov_workspace!(workspaces::workspaces, n::Int, order::Symbol)
    if order == :first_order
        ws = workspaces.lyapunov_1st_order
        if ws.n != n
            workspaces.lyapunov_1st_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_1st_order
    elseif order == :second_order
        ws = workspaces.lyapunov_2nd_order
        if ws.n != n
            workspaces.lyapunov_2nd_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_2nd_order
    elseif order == :third_order
        ws = workspaces.lyapunov_3rd_order
        if ws.n != n
            workspaces.lyapunov_3rd_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_3rd_order
    elseif order == :block
        ws = workspaces.lyapunov_block
        if ws.n != n
            workspaces.lyapunov_block = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_block
    else
        error("Invalid order: $order. Must be :first_order, :second_order, :third_order, or :block")
    end
end

function create_selector_matrix(target::Vector{Symbol}, source::Vector{Symbol})
    selector = spzeros(Float64, length(target), length(source))
    idx = indexin(target, source)
    for (i, j) in enumerate(idx)
        if !isnothing(j)
            selector[i, j] = 1.0
        end
    end
    return selector
end

@unstable function ensure_model_structure_constants!(constants::constants, calibration_parameters::Vector{Symbol})
    T = constants.post_model_macro
    if isempty(constants.post_complete_parameters.SS_and_pars_names)
        SS_and_pars_names = vcat(
            Symbol.(replace.(string.(sort(union(T.var, T.exo_past, T.exo_future))),
                    r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),
            calibration_parameters,
        )

        all_variables = Symbol.(sort(union(T.var, T.aux, T.exo_present)))

        NSSS_labels = Symbol.(vcat(sort(union(T.exo_present, T.var)), calibration_parameters))

        aux_indices = Int.(indexin(T.aux, all_variables))
        processed_all_variables = copy(all_variables)
        processed_all_variables[aux_indices] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), T.aux)

        full_NSSS = copy(processed_all_variables)
        if any(x -> contains(string(x), "◖"), full_NSSS)
            full_NSSS_decomposed = decompose_name.(full_NSSS)
            full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
        end
        # full_NSSS_display = Vector{Union{Symbol, String}}(full_NSSS)
        full_NSSS_display = copy(full_NSSS)

        steady_state_expand_matrix = create_selector_matrix(processed_all_variables, NSSS_labels)

        vars_in_ss_equations = T.vars_in_ss_equations_no_aux
        vars_in_ss_equations_with_aux = T.vars_in_ss_equations
        vars_and_calib = vcat(T.var, calibration_parameters)
        ss_var_idx_in_var_and_calib = Int.(indexin(vars_in_ss_equations, vars_and_calib))
        calib_idx_in_var_and_calib = Int.(indexin(calibration_parameters, vars_and_calib))
        extended_SS_and_pars = vcat(map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), T.var), calibration_parameters)
        custom_ss_expand_matrix = create_selector_matrix(extended_SS_and_pars, vcat(vars_in_ss_equations, calibration_parameters))

        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(T.var, T.exo_past, T.exo_future)))), calibration_parameters)
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(T.var, T.exo_past, T.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), calibration_parameters)
        SS_and_pars_no_exo_idx = Int.(indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag))

        vars_non_obc = T.var[.!contains.(string.(T.var), "ᵒᵇᶜ")]
        vars_idx_excluding_aux_obc = Int.(indexin(setdiff(vars_non_obc, union(T.aux, T.exo_present)), all_variables))
        vars_idx_excluding_obc = Int.(indexin(vars_non_obc, all_variables))

        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            SS_and_pars_names = SS_and_pars_names,
            # all_variables = all_variables,
            # NSSS_labels = NSSS_labels,
            # aux_indices = aux_indices,
            # processed_all_variables = processed_all_variables,
            full_NSSS_display = full_NSSS_display,
            steady_state_expand_matrix = steady_state_expand_matrix,
            custom_ss_expand_matrix = custom_ss_expand_matrix,
            vars_in_ss_equations = vars_in_ss_equations,
            vars_in_ss_equations_with_aux = vars_in_ss_equations_with_aux,
            ss_var_idx_in_var_and_calib = ss_var_idx_in_var_and_calib,
            calib_idx_in_var_and_calib = calib_idx_in_var_and_calib,
            SS_and_pars_names_lead_lag = SS_and_pars_names_lead_lag,
            # SS_and_pars_names_no_exo = SS_and_pars_names_no_exo,
            SS_and_pars_no_exo_idx = SS_and_pars_no_exo_idx,
            vars_idx_excluding_aux_obc = vars_idx_excluding_aux_obc,
            vars_idx_excluding_obc = vars_idx_excluding_obc,
        )
    end

    return constants.post_complete_parameters
end

function compute_e4(nᵉ::Int)
    if nᵉ == 0
        return Float64[]
    end
    # Isserlis' theorem for i.i.d. standard normal shocks:
    # E[ε_a ε_b ε_c ε_d] = δ_ab δ_cd + δ_ac δ_bd + δ_ad δ_bc
    e4 = zeros(nᵉ^4)
    for d in 1:nᵉ, c in 1:nᵉ, b in 1:nᵉ, a in 1:nᵉ
        e4[a + nᵉ*(b-1) + nᵉ^2*(c-1) + nᵉ^3*(d-1)] = Float64((a==b)*(c==d) + (a==c)*(b==d) + (a==d)*(b==c))
    end
    return e4
end

function compute_e6(nᵉ::Int)
    if nᵉ == 0
        return Float64[]
    end
    # Isserlis' theorem for i.i.d. standard normal shocks:
    # E[ε_a ε_b ε_c ε_d ε_e ε_f] = sum over all 15 perfect matchings
    e6 = zeros(nᵉ^6)
    for f in 1:nᵉ, e in 1:nᵉ, d in 1:nᵉ, c in 1:nᵉ, b in 1:nᵉ, a in 1:nᵉ
        e6[a + nᵉ*(b-1) + nᵉ^2*(c-1) + nᵉ^3*(d-1) + nᵉ^4*(e-1) + nᵉ^5*(f-1)] = Float64(
            (a==b)*((c==d)*(e==f) + (c==e)*(d==f) + (c==f)*(d==e)) +
            (a==c)*((b==d)*(e==f) + (b==e)*(d==f) + (b==f)*(d==e)) +
            (a==d)*((b==c)*(e==f) + (b==e)*(c==f) + (b==f)*(c==e)) +
            (a==e)*((b==c)*(d==f) + (b==d)*(c==f) + (b==f)*(c==d)) +
            (a==f)*((b==c)*(d==e) + (b==d)*(c==e) + (b==e)*(c==d)))
    end
    return e6
end

@unstable function ensure_moments_constants!(constants::constants)
    so = ensure_computational_constants!(constants)
    to = constants.third_order
    # Use timings from constants
    T = constants.post_model_macro
    nᵉ = T.nExo
    
    if isempty(so.kron_states)
        so.kron_states = ℒ.kron(so.s_in_s, so.s_in_s)
    end
    if isempty(so.kron_s_e)
        so.kron_s_e = ℒ.kron(so.s_in_s, so.e_in_s⁺)
    end
    if size(so.I_plus_s_s, 1) == 0
        nˢ = T.nPast_not_future_and_mixed
        so.I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
    end
    if isempty(so.e4)
        so.e4 = compute_e4(nᵉ)
    end
    # Cache vec(I(nᵉ)) - used repeatedly in moments calculations
    if isempty(so.vec_Iₑ)
        so.vec_Iₑ = vec(collect(Float64, ℒ.I(nᵉ)))
    end
    # Cache reshaped e4 matrices - used repeatedly in moments calculations
    if isempty(so.e4_nᵉ²_nᵉ²)
        so.e4_nᵉ²_nᵉ² = reshape(so.e4, nᵉ^2, nᵉ^2)
    end
    if isempty(so.e4_nᵉ_nᵉ³)
        so.e4_nᵉ_nᵉ³ = reshape(so.e4, nᵉ, nᵉ^3)
    end
    # Cache e4 minus outer product of vec(I) - used in Γ₂ and Γ₃ matrices
    if isempty(so.e4_minus_vecIₑ_outer)
        so.e4_minus_vecIₑ_outer = so.e4_nᵉ²_nᵉ² - so.vec_Iₑ * so.vec_Iₑ'
    end
    if isempty(to.e6)
        to.e6 = compute_e6(nᵉ)
    end
    if isempty(to.kron_e_v)
        to.kron_e_v = ℒ.kron(so.e_in_s⁺, so.v_in_s⁺)
    end
    # Cache reshaped e6 matrix - used in third order moments calculations
    if isempty(to.e6_nᵉ³_nᵉ³)
        to.e6_nᵉ³_nᵉ³ = reshape(to.e6, nᵉ^3, nᵉ^3)
    end
    return so
end

@unstable function ensure_moments_substate_indices!(𝓂, nˢ::Int)
    constants = 𝓂.constants
    to = constants.third_order
    if !haskey(to.substate_indices, nˢ)
        # Use timings from constants if available, otherwise from model
        T = constants.post_model_macro
        nᵉ = T.nExo
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ * nˢ)), nˢ * nᵉ^2, nˢ * nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ * nˢ^2, nᵉ * nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

        # Second-order duplication/elimination matrices (D₂ˢ: nˢ² × nˢ(nˢ+1)/2, L₂ˢ: nˢ(nˢ+1)/2 × nˢ²)
        # D₂ˢ * vech(M) = vec(M) for symmetric M; L₂ˢ * vec(M) = vech(M)
        # vech ordering: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3), ... (upper triangle, col-major)
        canonical2 = [nˢ * (i-1) + k for i in 1:nˢ for k in 1:i]  # canonical vec positions
        rows2 = Int[]; cols2 = Int[]
        col_idx = 0
        for i in 1:nˢ
            for k in 1:i
                col_idx += 1
                push!(rows2, nˢ * (i-1) + k)  # M_{k,i} position
                push!(cols2, col_idx)
                if i != k
                    push!(rows2, nˢ * (k-1) + i)  # M_{i,k} symmetric duplicate
                    push!(cols2, col_idx)
                end
            end
        end
        D₂ˢ = sparse(rows2, cols2, 1.0, nˢ^2, col_idx)
        L₂ˢ = sparse(1:length(canonical2), canonical2, 1.0, length(canonical2), nˢ^2)

        # Third-order duplication/elimination matrices (D₃ˢ: nˢ³ × nˢ(nˢ+1)(nˢ+2)/6, L₃ˢ: inverse)
        # D₃ˢ * vech₃(T) = vec(T) for symmetric 3-tensor T; L₃ˢ * vec(T) = vech₃(T)
        canonical3 = [nˢ^2 * (i-1) + nˢ * (k-1) + l for i in 1:nˢ for k in 1:i for l in 1:k]
        rows3 = Int[]; cols3 = Int[]
        col_idx = 0
        for i in 1:nˢ
            for k in 1:i
                for l in 1:k
                    col_idx += 1
                    perms = Set{Tuple{Int,Int,Int}}()
                    for p in ((i,k,l), (i,l,k), (k,i,l), (k,l,i), (l,i,k), (l,k,i))
                        push!(perms, p)
                    end
                    for (a, b, c) in perms
                        push!(rows3, nˢ^2 * (a-1) + nˢ * (b-1) + c)
                        push!(cols3, col_idx)
                    end
                end
            end
        end
        D₃ˢ = sparse(rows3, cols3, 1.0, nˢ^3, col_idx)
        L₃ˢ = sparse(1:length(canonical3), canonical3, 1.0, length(canonical3), nˢ^3)

        to.substate_indices[nˢ] = moments_substate_indices(I_plus_s_s, e_es, e_ss, ss_s, s_s, D₂ˢ, L₂ˢ, D₃ˢ, L₃ˢ)
    end
    return to.substate_indices[nˢ]
end

@unstable function ensure_moments_dependency_kron_indices!(𝓂, dependencies::Vector{Symbol}, s_in_s⁺::BitVector)
    constants = 𝓂.constants
    to = constants.third_order
    key = Tuple(dependencies)
    if !haskey(to.dependency_kron_indices, key)
        so = ensure_computational_constants!(constants)
        to.dependency_kron_indices[key] = moments_dependency_kron_indices(
            ℒ.kron(s_in_s⁺, s_in_s⁺),
            ℒ.kron(s_in_s⁺, so.e_in_s⁺),
            ℒ.kron(s_in_s⁺, so.v_in_s⁺),
        )
    end
    return to.dependency_kron_indices[key]
end


"""
    SolverTolerances

Tolerance settings for a single numerical equation solver (Sylvester, Lyapunov, or QME).

# Fields
- `atol::Float64`: absolute convergence tolerance (used by Krylov solvers).
- `rtol::Float64`: relative convergence tolerance (used by iterative stopping checks).
- `initial_guess_acceptance_tol::Float64`: if an initial guess achieves a relative
  residual below this threshold it is accepted immediately, skipping the full solve.
- `acceptance_tol::Float64`: result is accepted when the relative residual falls below
  this threshold; otherwise the dispatcher retries with a fallback algorithm.

Construct via `SolverTolerances(; atol, rtol, initial_guess_acceptance_tol, acceptance_tol)`.
Default values differ by solver type and are set by the enclosing tolerance hierarchy;
see [`Tolerances`](@ref) and [`FirstOrderTolerances`](@ref) / [`HigherOrderTolerances`](@ref).
"""
struct SolverTolerances
    atol::Float64
    rtol::Float64
    initial_guess_acceptance_tol::Float64
    acceptance_tol::Float64
end

function SolverTolerances(; atol::Float64 = 1e-14,
                          rtol::Float64 = 1e-14,
                          initial_guess_acceptance_tol::Float64 = 1e-10,
                          acceptance_tol::Float64 = 1e-10)
    return SolverTolerances(atol, rtol, initial_guess_acceptance_tol, acceptance_tol)
end

# Generic SolverTolerances keyword-constructor defaults, used by resolve_tol to detect
# which fields were left at their generic value and should be replaced by context defaults.
const GENERIC_SOLVER_TOL = SolverTolerances(1e-14, 1e-14, 1e-10, 1e-10)

# Merge a SolverTolerances with context-specific defaults.  Fields that still sit at the
# generic SolverTolerances() defaults are replaced by the context base; explicitly changed
# fields are kept.
function resolve_tol(override::SolverTolerances, base::SolverTolerances)
    SolverTolerances(
        override.atol != GENERIC_SOLVER_TOL.atol ? override.atol : base.atol,
        override.rtol != GENERIC_SOLVER_TOL.rtol ? override.rtol : base.rtol,
        override.initial_guess_acceptance_tol != GENERIC_SOLVER_TOL.initial_guess_acceptance_tol ? override.initial_guess_acceptance_tol : base.initial_guess_acceptance_tol,
        override.acceptance_tol != GENERIC_SOLVER_TOL.acceptance_tol ? override.acceptance_tol : base.acceptance_tol)
end

# Merge a NamedTuple of partial overrides with a SolverTolerances base.
function resolve_tol(nt::NamedTuple, base::SolverTolerances)
    SolverTolerances(
        Float64(get(nt, :atol, base.atol)),
        Float64(get(nt, :rtol, base.rtol)),
        Float64(get(nt, :initial_guess_acceptance_tol, base.initial_guess_acceptance_tol)),
        Float64(get(nt, :acceptance_tol, base.acceptance_tol)))
end

"""
    NsssTolerances

Tolerance settings for the non-stochastic steady state (NSSS) solver.

# Fields
- `acceptance_tol::Float64` [Default: `1e-12`]: solution is accepted when the residual
  norm falls below this value.
- `initial_guess_acceptance_tol::Float64` [Default: `1e-12`]: an initial guess is reused
  when its residual is below this threshold.
- `xtol::Float64` [Default: `1e-12`]: absolute step-size tolerance.
- `ftol::Float64` [Default: `1e-14`]: absolute function-value tolerance.
- `rel_xtol::Float64` [Default: `eps()`]: relative step-size tolerance.

Construct via `NsssTolerances(; acceptance_tol, initial_guess_acceptance_tol, xtol, ftol, rel_xtol)`.
"""
struct NsssTolerances
    acceptance_tol::Float64
    initial_guess_acceptance_tol::Float64
    xtol::Float64
    ftol::Float64
    rel_xtol::Float64
end

function NsssTolerances(; acceptance_tol::Float64 = 1e-12,
                        initial_guess_acceptance_tol::Float64 = 1e-12,
                        xtol::Float64 = 1e-12,
                        ftol::Float64 = 1e-14,
                        rel_xtol::Float64 = eps())
    return NsssTolerances(acceptance_tol, initial_guess_acceptance_tol, xtol, ftol, rel_xtol)
end

resolve_tol(s::NsssTolerances, ::NsssTolerances) = s
function resolve_tol(nt::NamedTuple, base::NsssTolerances)
    NsssTolerances(
        Float64(get(nt, :acceptance_tol, base.acceptance_tol)),
        Float64(get(nt, :initial_guess_acceptance_tol, base.initial_guess_acceptance_tol)),
        Float64(get(nt, :xtol, base.xtol)),
        Float64(get(nt, :ftol, base.ftol)),
        Float64(get(nt, :rel_xtol, base.rel_xtol)))
end

"""
    AdTolerances

Tolerance settings passed to the automatic differentiation (AD) paths of each equation
solver. Each field is a [`SolverTolerances`](@ref) that controls the corresponding solver
when it is called inside a ForwardDiff dual-number overload or a ChainRulesCore rrule.

# Fields
- `qme::SolverTolerances`: tolerances for the quadratic matrix equation (QME) derivative solve.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-10`, `acceptance_tol=1e-6`.
- `sylvester::SolverTolerances`: tolerances for the Sylvester equation derivative solve.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-10`, `acceptance_tol=1e-10`.
- `lyapunov::SolverTolerances`: tolerances for the Lyapunov equation derivative solve.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-12`, `acceptance_tol=1e-12`.

Construct via `AdTolerances(; qme, sylvester, lyapunov)`.
"""
struct AdTolerances
    qme::SolverTolerances
    sylvester::SolverTolerances
    lyapunov::SolverTolerances
end

function AdTolerances(; qme = (;), sylvester = (;), lyapunov = (;))
    _base_qme  = SolverTolerances(1e-14, 1e-14, 1e-10, 1e-6)
    _base_sylv = SolverTolerances(1e-14, 1e-14, 1e-10, 1e-10)
    _base_lyap = SolverTolerances(1e-14, 1e-14, 1e-12, 1e-12)
    return AdTolerances(
        resolve_tol(qme, _base_qme),
        resolve_tol(sylvester, _base_sylv),
        resolve_tol(lyapunov, _base_lyap))
end

resolve_tol(s::AdTolerances, ::AdTolerances) = s
function resolve_tol(nt::NamedTuple, base::AdTolerances)
    AdTolerances(
        resolve_tol(get(nt, :qme, (;)), base.qme),
        resolve_tol(get(nt, :sylvester, (;)), base.sylvester),
        resolve_tol(get(nt, :lyapunov, (;)), base.lyapunov))
end

"""
    FirstOrderTolerances

Tolerance settings for the first-order perturbation solution and its AD pathways.

# Fields
- `qme::SolverTolerances`: tolerances for the quadratic matrix equation solver.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-10`, `acceptance_tol=1e-6`.
- `lyapunov::SolverTolerances`: tolerances for the Lyapunov equation solver used to
  compute first-order covariance matrices.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-12`, `acceptance_tol=1e-12`.
- `droptol::Float64` [Default: `1e-14`]: entries smaller than this threshold in solution
  matrices are dropped (set to zero) to reduce sparsity fill-in.
- `dependencies_tol::Float64` [Default: `1e-12`]: threshold for determining variable
  dependencies when isolating subsystems for covariance statistics.
- `ad::AdTolerances`: tolerances used in the AD derivative evaluation paths.

Construct via `FirstOrderTolerances(; qme, lyapunov, droptol, dependencies_tol, ad)`.
"""
struct FirstOrderTolerances
    qme::SolverTolerances
    lyapunov::SolverTolerances
    droptol::Float64
    dependencies_tol::Float64
    ad::AdTolerances
end

function FirstOrderTolerances(; qme = (;),
                              lyapunov = (;),
                              droptol::Float64 = 1e-14,
                              dependencies_tol::Float64 = 1e-12,
                              ad = (;))
    _base_qme  = SolverTolerances(1e-14, 1e-14, 1e-10, 1e-6)
    _base_lyap = SolverTolerances(1e-14, 1e-14, 1e-12, 1e-12)
    _base_ad   = AdTolerances()
    return FirstOrderTolerances(
        resolve_tol(qme, _base_qme),
        resolve_tol(lyapunov, _base_lyap),
        droptol, dependencies_tol,
        resolve_tol(ad, _base_ad))
end

resolve_tol(s::FirstOrderTolerances, ::FirstOrderTolerances) = s
function resolve_tol(nt::NamedTuple, base::FirstOrderTolerances)
    FirstOrderTolerances(
        resolve_tol(get(nt, :qme, (;)), base.qme),
        resolve_tol(get(nt, :lyapunov, (;)), base.lyapunov),
        Float64(get(nt, :droptol, base.droptol)),
        Float64(get(nt, :dependencies_tol, base.dependencies_tol)),
        resolve_tol(get(nt, :ad, (;)), base.ad))
end

"""
    HigherOrderTolerances

Tolerance settings for second- and third-order perturbation solutions and their AD pathways.

# Fields
- `sylvester::SolverTolerances`: tolerances for the Sylvester equation solver.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-10`, `acceptance_tol=1e-10`.
- `lyapunov::SolverTolerances`: tolerances for the Lyapunov equation solver used to
  compute higher-order covariance matrices.
    Default: `atol=1e-14`, `rtol=1e-14`, `initial_guess_acceptance_tol=1e-12`, `acceptance_tol=1e-12`.
- `droptol::Float64` [Default: `1e-14`]: entries smaller than this threshold in solution
  matrices are dropped (set to zero) to reduce sparsity fill-in.
- `dependencies_tol::Float64` [Default: `1e-12`]: threshold for determining variable
  dependencies when isolating subsystems for covariance statistics.
- `ad::AdTolerances`: tolerances used in the AD derivative evaluation paths.

Construct via `HigherOrderTolerances(; sylvester, lyapunov, droptol, dependencies_tol, ad)`.
"""
struct HigherOrderTolerances
    sylvester::SolverTolerances
    lyapunov::SolverTolerances
    droptol::Float64
    dependencies_tol::Float64
    ad::AdTolerances
end

function HigherOrderTolerances(; sylvester = (;),
                               lyapunov = (;),
                               droptol::Float64 = 1e-14,
                               dependencies_tol::Float64 = 1e-12,
                               ad = (;))
    _base_sylv = SolverTolerances(1e-14, 1e-14, 1e-10, 1e-10)
    _base_lyap = SolverTolerances(1e-14, 1e-14, 1e-12, 1e-12)
    _base_ad   = AdTolerances()
    return HigherOrderTolerances(
        resolve_tol(sylvester, _base_sylv),
        resolve_tol(lyapunov, _base_lyap),
        droptol, dependencies_tol,
        resolve_tol(ad, _base_ad))
end

resolve_tol(s::HigherOrderTolerances, ::HigherOrderTolerances) = s
function resolve_tol(nt::NamedTuple, base::HigherOrderTolerances)
    HigherOrderTolerances(
        resolve_tol(get(nt, :sylvester, (;)), base.sylvester),
        resolve_tol(get(nt, :lyapunov, (;)), base.lyapunov),
        Float64(get(nt, :droptol, base.droptol)),
        Float64(get(nt, :dependencies_tol, base.dependencies_tol)),
        resolve_tol(get(nt, :ad, (;)), base.ad))
end

struct Tolerances
    nsss::NsssTolerances
    first_order::FirstOrderTolerances
    second_order::HigherOrderTolerances
    third_order::HigherOrderTolerances
end

struct CalculationOptions
    quadratic_matrix_equation_algorithm::Symbol
    
    sylvester_algorithm²::Symbol
    sylvester_algorithm³::Symbol
    
    sylvester_preconditioner::Symbol
    
    lyapunov_algorithm::Symbol
    
    tol::Tolerances
    verbose::Bool
end

"""
$(SIGNATURES)

Define tolerances for the numerical solvers used throughout model solution and estimation.
Tolerances are organised in a two-level hierarchy:

```
Tolerances
├── nsss          :: NsssTolerances          — non-stochastic steady state solver
├── first_order   :: FirstOrderTolerances    — first-order perturbation solution
│   ├── qme       :: SolverTolerances        — quadratic matrix equation (QME)
│   ├── lyapunov  :: SolverTolerances        — Lyapunov equation
│   ├── droptol                              — zero-threshold for solution matrices
│   ├── dependencies_tol                     — subsystem isolation threshold
│   └── ad        :: AdTolerances            — AD derivative paths
│       ├── qme      :: SolverTolerances
│       ├── sylvester:: SolverTolerances
│       └── lyapunov :: SolverTolerances
├── second_order  :: HigherOrderTolerances   — second-order perturbation solution
│   ├── sylvester :: SolverTolerances        — Sylvester equation
│   ├── lyapunov  :: SolverTolerances        — Lyapunov equation
│   ├── droptol / dependencies_tol
│   └── ad        :: AdTolerances
└── third_order   :: HigherOrderTolerances   — third-order perturbation solution
└── (same structure as second_order, except sylvester and lyapunov acceptance_tol default to 1e-8 on primal and AD paths)
```

Each [`SolverTolerances`](@ref) carries four values:
- `atol`: absolute convergence tolerance used by Krylov solvers.
- `rtol`: relative convergence tolerance used by iterative stopping checks.
- `initial_guess_acceptance_tol`: accept an initial guess without re-solving if its
  residual is already below this threshold.
- `acceptance_tol`: accept the final result when the residual falls below this threshold;
  otherwise the dispatcher retries with a fallback algorithm.

# Keyword Arguments
- `nsss` [Default: `NsssTolerances()`]: tolerances for the non-stochastic steady state
  solver. See [`NsssTolerances`](@ref).
- `first_order` [Default: `FirstOrderTolerances()`]: tolerances for the first-order
  solution and its AD paths. See [`FirstOrderTolerances`](@ref).
- `second_order` [Default: `HigherOrderTolerances()`]: tolerances for the second-order
  solution and its AD paths. See [`HigherOrderTolerances`](@ref).
- `third_order` [Default: `HigherOrderTolerances()`]: tolerances for the third-order
  solution and its AD paths. See [`HigherOrderTolerances`](@ref).

# Examples
```julia
# use defaults
tol = Tolerances()

# tighten the NSSS solver
tol = Tolerances(nsss = NsssTolerances(xtol = 1e-14))

# tighten first-order Lyapunov acceptance_tol while preserving context defaults
tol = Tolerances(first_order = FirstOrderTolerances(lyapunov = SolverTolerances(acceptance_tol = 1e-14)))

# concise NamedTuple syntax — equivalent to the above
tol = Tolerances(first_order = (lyapunov = (acceptance_tol = 1e-14,),))

# tighten second- and third-order Sylvester/Lyapunov solvers
tol = Tolerances(
    second_order = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14),
                                         lyapunov  = SolverTolerances(acceptance_tol = 1e-14)),
    third_order  = HigherOrderTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-14),
                                         lyapunov  = SolverTolerances(acceptance_tol = 1e-14)),
)
```
"""
function Tolerances(; nsss = (;),
                    first_order = (;),
                    second_order = (;),
                    third_order = (;))
    _base = Tolerances(NsssTolerances(), FirstOrderTolerances(), HigherOrderTolerances(),
                       HigherOrderTolerances(
                           sylvester = SolverTolerances(acceptance_tol = 1e-8),
                           lyapunov  = SolverTolerances(acceptance_tol = 1e-8),
                           ad = AdTolerances(sylvester = SolverTolerances(acceptance_tol = 1e-8),
                                             lyapunov  = SolverTolerances(acceptance_tol = 1e-8))))
    return Tolerances(
        resolve_tol(nsss, _base.nsss),
        resolve_tol(first_order, _base.first_order),
        resolve_tol(second_order, _base.second_order),
        resolve_tol(third_order, _base.third_order))
end

resolve_tol(s::Tolerances, ::Tolerances) = s
function resolve_tol(nt::NamedTuple, base::Tolerances)
    Tolerances(
        resolve_tol(get(nt, :nsss, (;)), base.nsss),
        resolve_tol(get(nt, :first_order, (;)), base.first_order),
        resolve_tol(get(nt, :second_order, (;)), base.second_order),
        resolve_tol(get(nt, :third_order, (;)), base.third_order))
end


function merge_calculation_options(;quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                    sylvester_algorithm²::Symbol = :doubling,
                                    sylvester_algorithm³::Symbol = :bicgstab,
                                    sylvester_preconditioner::Symbol = :ilu,
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::Tolerances = Tolerances(),
                                    verbose::Bool = false)
                                    
    return CalculationOptions(quadratic_matrix_equation_algorithm, 
                                sylvester_algorithm², 
                                sylvester_algorithm³, 
                                sylvester_preconditioner, 
                                lyapunov_algorithm, 
                                tol, 
                                verbose)
end


end # @stable
