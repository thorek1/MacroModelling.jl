# ─── Incomplete LU (ILU) preconditioner for Krylov Sylvester solvers ─────────
#
# Crout ILU(τ) factorisation and Sylvester block-diagonal preconditioner builder.
# Adapted from KrylovPreconditioners.jl (MPL-2.0, © 2023 Alexis Montoison)
# which incorporated IncompleteLU.jl.  Only the subset needed here is retained.

# ─── Sparse-vector accumulator (Gustavson's O(1)-reset technique) ────────────
#
# Dense-length vectors with a generation counter avoid O(n) zeroing each step.
# An entry is "live" when occupied[i] == gen.  No SparseArrays equivalent exists
# for this O(1)-reset pattern; SparseVector would require O(nnz) reset per step.

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
function _flush_column!(A::SparseMatrixCSC, v::SparseAccum, j::Integer,
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
    rv = SparseArrays.rowvals(A);  nz = nonzeros(A);  cp = SparseArrays.getcolptr(A)
    @inbounds for i = 1:total
        row = v.nzind[i]
        push!(rv, row)
        push!(nz, scale * v.nzval[row])
    end
    @inbounds cp[j + 1] = cp[j] + total
    v.gen += 1;  v.nnz = 0
    nothing
end

# ─── ILUFactorization ───────────────────────────────────────────────────────

struct ILUFactorization{Tv,Ti} <: ℒ.Factorization{Tv}
    L::SparseMatrixCSC{Tv,Ti}
    U::SparseMatrixCSC{Tv,Ti}
end

function ℒ.ldiv!(F::ILUFactorization, y::AbstractVecOrMat)
    L = F.L;  Lrows = SparseArrays.rowvals(L);  Lvals = nonzeros(L)
    @inbounds for col in 1:size(L, 2) - 1          # forward (unit lower)
        for idx in SparseArrays.nzrange(L, col)
            y[Lrows[idx]] -= Lvals[idx] * y[col]
        end
    end
    U = F.U;  Urows = SparseArrays.rowvals(U);  Uvals = nonzeros(U)
    @inbounds for col in size(U, 2):-1:1            # backward (upper)
        rng = SparseArrays.nzrange(U, col)
        for idx in last(rng):-1:first(rng) + 1
            y[col] -= Uvals[idx] * y[Urows[idx]]
        end
        y[col] /= Uvals[first(rng)]
    end
    y
end

function ℒ.ldiv!(y::AbstractVector, F::ILUFactorization, x::AbstractVector)
    y .= x
    ℒ.ldiv!(F, y)
end

# ─── Crout ILU(τ) factorisation ─────────────────────────────────────────────
#
# Row access to A uses its precomputed transpose (At) with SparseArrays.nzrange.
# Row access to L and U (built incrementally) uses linked-list indices:
#   nxt[c]  – cursor: next nz-index to visit in column c
#   head[r] – first column with a pending nonzero in row r
#   rnxt[c] – next column after c in the same row chain

function ilu(A::SparseMatrixCSC{ATv,Ti}; τ = 1e-3) where {ATv,Ti}
    n  = size(A, 1)
    Tv = typeof(oneunit(ATv) / (oneunit(ATv) + zero(ATv)))

    # Transpose gives row access to A via standard SparseArrays.nzrange on At
    At = sparse(A')
    At_rows = SparseArrays.rowvals(At);  At_vals = nonzeros(At)
    Arows   = SparseArrays.rowvals(A);   Avals   = nonzeros(A)

    L  = spzeros(Tv, Ti, n, n)
    U  = spzeros(Tv, Ti, n, n)
    Ur = SparseAccum{Tv,Ti}(n)              # row accumulator  (builds U)
    Lc = SparseAccum{Tv,Ti}(n)              # column accumulator (builds L)

    # Linked-list row index for L and U (built incrementally)
    L_nxt = zeros(Ti, n);  L_head = zeros(Ti, n);  L_rnxt = zeros(Ti, n)
    Lrows = SparseArrays.rowvals(L);    Lvals  = nonzeros(L);   Lcp    = SparseArrays.getcolptr(L)

    U_nxt = zeros(Ti, n);  U_head = zeros(Ti, n);  U_rnxt = zeros(Ti, n)
    Urows = SparseArrays.rowvals(U);    Uvals  = nonzeros(U);   Ucp    = SparseArrays.getcolptr(U)

    @inbounds for k = Ti(1):Ti(n)

        # --- Scatter row k of A (upper triangle, c ≥ k) into Ur ---
        for idx in SparseArrays.nzrange(At, k)
            c = At_rows[idx]
            c >= k || continue
            _scatter!(Ur, At_vals[idx], c)
        end

        # --- Scatter column k of A (lower triangle, row > k) into Lc ---
        for idx in SparseArrays.nzrange(A, k)
            row = Arows[idx]
            row > k || continue
            _scatter!(Lc, Avals[idx], row)
        end

        # --- Ur[k:n] -= L[k,i] * U[i, k:n]  for i < k ---
        c = L_head[k]
        while c != 0
            a = -Lvals[L_nxt[c]]
            for idx = U_nxt[c]:Ucp[c + 1] - 1
                _scatter!(Ur, a * Uvals[idx], Urows[idx])
            end
            nc = L_rnxt[c]
            L_nxt[c] += 1
            if L_nxt[c] < Lcp[c + 1]
                row = Lrows[L_nxt[c]]
                L_head[row], L_rnxt[c] = c, L_head[row]
            end
            c = nc
        end

        # --- Lc[k+1:n] -= U[i,k] * L[i, k+1:n]  for i < k ---
        if k < n
            c = U_head[k]
            while c != 0
                a = -Uvals[U_nxt[c]]
                for idx = L_nxt[c]:Lcp[c + 1] - 1
                    _scatter!(Lc, a * Lvals[idx], Lrows[idx])
                end
                nc = U_rnxt[c]
                U_nxt[c] += 1
                if U_nxt[c] < Ucp[c + 1]
                    row = Urows[U_nxt[c]]
                    U_head[row], U_rnxt[c] = c, U_head[row]
                end
                c = nc
            end
        end

        # --- Drop small entries, store columns of U and L ---
        _flush_column!(U, Ur, k, τ)
        _flush_column!(L, Lc, k, τ, inv(Ur.nzval[k]))

        # Register new entries in row-traversal index
        U_nxt[k] = Ucp[k] + 1
        if Ucp[k] < Ucp[k + 1] - 1
            row = Urows[U_nxt[k]]; U_head[row], U_rnxt[k] = k, U_head[row]
        end

        L_nxt[k] = Lcp[k]
        if Lcp[k] < Lcp[k + 1]
            row = Lrows[L_nxt[k]]; L_head[row], L_rnxt[k] = k, L_head[row]
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

_to_sparse(B::SparseMatrixCSC) = B
_to_sparse(B::ThreadedSparseArrays.ThreadedSparseMatrixCSC) = B.A
_to_sparse(B::AbstractMatrix) = sparse(B)

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
    B_sp = _to_sparse(B)
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
