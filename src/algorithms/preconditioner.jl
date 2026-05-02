
# ─── Incomplete LU (ILU) preconditioner for Krylov Sylvester solvers ─────────
#
# Crout ILU(τ) factorisation and Sylvester block-diagonal preconditioner builder.
# Adapted from KrylovPreconditioners.jl (MPL-2.0, © 2023 Alexis Montoison)
# which incorporated IncompleteLU.jl.  Only the subset needed here is retained.

# ─── Sparse-vector accumulator (Gustavson's O(1)-reset technique) ────────────
#
# Dense-length vectors with a generation counter avoid O(n) zeroing each step.
# An entry is "live" when occupied[i] == gen.

mutable struct SparseAccum{Tv,Ti}
    occupied::Vector{Ti}
    nzind::Vector{Ti}
    nzval::Vector{Tv}
    nnz::Ti
    gen::Ti

    SparseAccum{Tv,Ti}(n::Integer) where {Tv,Ti} =
        new(zeros(Ti, n), Vector{Ti}(undef, n), Vector{Tv}(undef, n), zero(Ti), one(Ti))
end

@inline function _scatter!(v::SparseAccum, a, idx)
    @inbounds if v.occupied[idx] == v.gen
        v.nzval[idx] += a
    else
        v.nnz += 1
        v.occupied[idx] = v.gen
        v.nzval[idx] = a
        v.nzind[v.nnz] = idx
    end
    nothing
end

# Drop entries below threshold, sort survivors, append as column j of A, reset.
function flush_column!(A::SparseMatrixCSC, v::SparseAccum, j::Integer,
                        drop, scale = one(eltype(A)))
    total = 0
    @inbounds for i = 1:v.nnz
        row = v.nzind[i]
        if abs(v.nzval[row]) >= drop || row == j
            total += 1
            v.nzind[total] = row
        end
    end
    sort!(v.nzind, 1, total, Base.Sort.QuickSort, Base.Order.Forward)
    @inbounds for i = 1:total
        row = v.nzind[i]
        push!(A.rowval, row)
        push!(A.nzval, scale * v.nzval[row])
    end
    @inbounds A.colptr[j + 1] = A.colptr[j] + total
    v.gen += 1;  v.nnz = 0
    nothing
end

# ─── ILUFactorization ───────────────────────────────────────────────────────

struct ILUFactorization{Tv,Ti} <: ℒ.Factorization{Tv}
    L::SparseMatrixCSC{Tv,Ti}
    U::SparseMatrixCSC{Tv,Ti}
end

function ℒ.ldiv!(F::ILUFactorization, y::AbstractVecOrMat)
    # Forward substitution (unit lower-triangular L, no stored diagonal)
    L = F.L
    @inbounds for col = 1 : L.n - 1
        for idx = L.colptr[col] : L.colptr[col + 1] - 1
            y[L.rowval[idx]] -= L.nzval[idx] * y[col]
        end
    end
    # Backward substitution (upper-triangular U with stored diagonal)
    U = F.U
    @inbounds for col = U.n : -1 : 1
        for idx = U.colptr[col + 1] - 1 : -1 : U.colptr[col] + 1
            y[col] -= U.nzval[idx] * y[U.rowval[idx]]
        end
        y[col] /= U.nzval[U.colptr[col]]
    end
    y
end

function ℒ.ldiv!(y::AbstractVector, F::ILUFactorization, x::AbstractVector)
    y .= x
    ℒ.ldiv!(F, y)
end

# ─── Crout ILU(τ) factorisation ─────────────────────────────────────────────
#
# Row access to all matrices (A, L, U) uses linked-list indices over the
# CSC structure.  Direct .colptr/.rowval/.nzval field access is used
# throughout for minimal overhead in tight loops.

function ilu(A::SparseMatrixCSC{ATv,Ti}; τ = 1e-3) where {ATv,Ti}
    n  = size(A, 1)
    Tv = typeof(oneunit(ATv) / (oneunit(ATv) + zero(ATv)))

    L  = spzeros(Tv, Ti, n, n)
    U  = spzeros(Tv, Ti, n, n)
    Ur = SparseAccum{Tv,Ti}(n)
    Lc = SparseAccum{Tv,Ti}(n)

    # Linked-list row index for A (pre-populated)
    A_nxt = A.colptr[1:n]
    A_head = zeros(Ti, n);  A_rnxt = zeros(Ti, n)
    @inbounds for i = Ti(1):Ti(n)
        row = A.rowval[A.colptr[i]]
        A_head[row], A_rnxt[i] = i, A_head[row]
    end

    # Linked-list row index for L and U (start empty, built incrementally)
    L_nxt = zeros(Ti, n);  L_head = zeros(Ti, n);  L_rnxt = zeros(Ti, n)
    U_nxt = zeros(Ti, n);  U_head = zeros(Ti, n);  U_rnxt = zeros(Ti, n)

    @inbounds for k = Ti(1):Ti(n)

        # --- Scatter row k of A into Ur, column k of A into Lc ---
        c = A_head[k]
        while c != 0
            _scatter!(Ur, A.nzval[A_nxt[c]], c)
            nc = A_rnxt[c]
            A_nxt[c] += 1
            if A_nxt[c] < A.colptr[c + 1] && A.rowval[A_nxt[c]] <= c
                row = A.rowval[A_nxt[c]]
                A_head[row], A_rnxt[c] = c, A_head[row]
            end
            c = nc
        end
        for idx = A_nxt[k] : A.colptr[k + 1] - 1
            _scatter!(Lc, A.nzval[idx], A.rowval[idx])
        end

        # --- Ur[k:n] -= L[k,i] * U[i, k:n]  for i < k ---
        c = L_head[k]
        while c != 0
            a = -L.nzval[L_nxt[c]]
            for idx = U_nxt[c] : U.colptr[c + 1] - 1
                _scatter!(Ur, a * U.nzval[idx], U.rowval[idx])
            end
            nc = L_rnxt[c]
            L_nxt[c] += 1
            if L_nxt[c] < L.colptr[c + 1]
                row = L.rowval[L_nxt[c]]
                L_head[row], L_rnxt[c] = c, L_head[row]
            end
            c = nc
        end

        # --- Lc[k+1:n] -= U[i,k] * L[i, k+1:n]  for i < k ---
        if k < n
            c = U_head[k]
            while c != 0
                a = -U.nzval[U_nxt[c]]
                for idx = L_nxt[c] : L.colptr[c + 1] - 1
                    _scatter!(Lc, a * L.nzval[idx], L.rowval[idx])
                end
                nc = U_rnxt[c]
                U_nxt[c] += 1
                if U_nxt[c] < U.colptr[c + 1]
                    row = U.rowval[U_nxt[c]]
                    U_head[row], U_rnxt[c] = c, U_head[row]
                end
                c = nc
            end
        end

        # --- Drop small entries, store columns of U and L ---
        flush_column!(U, Ur, k, τ)
        flush_column!(L, Lc, k, τ, inv(Ur.nzval[k]))

        # Register new entries in row-traversal index
        U_nxt[k] = U.colptr[k] + 1
        if U.colptr[k] < U.colptr[k + 1] - 1
            row = U.rowval[U_nxt[k]]
            U_head[row], U_rnxt[k] = k, U_head[row]
        end

        L_nxt[k] = L.colptr[k]
        if L.colptr[k] < L.colptr[k + 1]
            row = L.rowval[L_nxt[k]]
            L_head[row], L_rnxt[k] = k, L_head[row]
        end
    end

    ILUFactorization(L, U)
end

# ─── Sylvester preconditioner builder ────────────────────────────────────────
#
# Approximates the block-diagonal of the vectorised Sylvester operator
#   L(X) = X − AXB
# When B has diagonal entries dⱼ the j-th n×n block is (I − dⱼ A).
# An ILU(τ) factorisation of each unique block serves as a right preconditioner
# for the Krylov solver (bicgstab / dqgmres / gmres).

const DEFAULT_ILU_TAU = 1e-3

to_sparse(B::SparseMatrixCSC) = B
to_sparse(B::ThreadedSparseArrays.ThreadedSparseMatrixCSC) = B.A
to_sparse(B::AbstractMatrix) = sparse(B)

"""
    build_ilu_preconditioner(A, B; τ) → LinearOperator

Build an ILU(τ) right preconditioner for the vectorised Sylvester operator.
Deduplicates identical diagonal entries of B so only one factorisation per
unique value is computed.  Application loops over the m columns of the
solution matrix.
"""
function build_ilu_preconditioner(A::DenseMatrix{T},
                                  B::AbstractMatrix{T};
                                  τ::Float64 = DEFAULT_ILU_TAU) where {T <: AbstractFloat}
    n = size(A, 1)
    B_sp = to_sparse(B)
    m = size(B_sp, 2)
    diag_B = Vector{T}(undef, m)
    @inbounds for j in 1:m
        diag_B[j] = B_sp[j, j]
    end

    A_sp = sparse(A)
    I_n  = sparse(one(T) * ℒ.I, n, n)

    # Factorise one block per unique diagonal value
    cache = Dict{T, ILUFactorization}()
    for d in diag_B
        haskey(cache, d) && continue
        block = I_n - d .* A_sp
        droptol!(block, eps())
        cache[d] = ilu(block; τ = τ)
    end

    # Map each column to its factorisation
    factors = Vector{ILUFactorization}(undef, m)
    @inbounds for j in 1:m
        factors[j] = cache[diag_B[j]]
    end

    nm = n * m
    function precond_ldiv!(y, x)
        X = reshape(x, n, m)
        Y = reshape(y, n, m)
        @inbounds for j in 1:m
            ℒ.ldiv!(view(Y, :, j), factors[j], view(X, :, j))
        end
        y
    end

    LinearOperators.LinearOperator(T, nm, nm, false, false, precond_ldiv!)
end

