using SparseArrays
using BenchmarkTools
using Base.Threads
using OhMyThreads: tforeach
using LoopVectorization

"""
    spmm_buffer!(C, A, B)

Multiply sparse matrices `A` and `B` storing the result in the
preallocated sparse matrix `C`.

The sparsity pattern of `C` should match the expected result.
This avoids allocations when repeatedly multiplying the same sized
matrices.
"""
function spmm_buffer!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    fill!(C.nzval, zero(eltype(C)))
    SparseArrays._spmatmul!(C, A, B, one(eltype(C)), zero(eltype(C)))
    return C
end


"""
    _col_map!(colmap, C, j)

Fill `colmap` with a lookup for column `j` of `C`. All entries are reset to
zero so the same vector can be reused across columns without additional
allocations.
"""
function _col_map!(colmap::Vector{Int}, C::SparseMatrixCSC, j::Integer)
    fill!(colmap, 0)
    @inbounds for p in C.colptr[j]:(C.colptr[j+1]-1)
        colmap[C.rowval[p]] = p
    end
    return colmap
end


"""
    spmm_buffer_fast!(C, A, B)

Same as [`spmm_buffer_fast!`](@ref) but builds the lookup table internally.
This version is convenient when the map is not available beforehand.
"""
function spmm_buffer_fast!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    colmap = zeros(Int, n)
    fill!(C.nzval, zero(eltype(C)))
    @inbounds for j in 1:size(B,2)
        _col_map!(colmap, C, j)
        for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] += a * b
                end
            end
        end
    end
    return C
end

"""
    spmm_buffer_col!(C, A, B)

Column oriented buffered multiplication that builds the lookup table on each
call.  Useful for quick tests but allocates the map.
"""
function spmm_buffer_col!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    colmap = zeros(Int, n)
    fill!(C.nzval, zero(eltype(C)))
    @inbounds for j in 1:size(B,2)
        _col_map!(colmap, C, j)
        for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] += a * b
                end
            end
        end
    end
    return C
end

"""
    spmm_buffer_threaded!(C, A, B)

Threaded buffered multiplication with internal map creation.
"""
function spmm_buffer_threaded!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC, α, β)
    n = size(C, 1)
    maps = [zeros(Int, n) for _ in 1:nthreads()]
    if β == zero(β)
        fill!(C.nzval, zero(eltype(C)))
    else
        @. C.nzval *= β
    end
    @threads for j in 1:size(B,2)
        colmap = maps[threadid()]
        _col_map!(colmap, C, j)
        @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] += α * a * b
                end
            end
        end
    end
    return C
end

spmm_buffer_threaded!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC) =
    spmm_buffer_threaded!(C, A, B, one(eltype(C)), zero(eltype(C)))

function spmm_buffer_simd!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    colmap = zeros(Int, n)
    fill!(C.nzval, zero(eltype(C)))
    @inbounds for j in 1:size(B,2)
        _col_map!(colmap, C, j)
        for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            @simd for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] = muladd(a, b, C.nzval[idx])
                end
            end
        end
    end
    return C
end

function spmm_buffer_omt!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    maps = [zeros(Int, n) for _ in 1:Threads.nthreads()]
    fill!(C.nzval, zero(eltype(C)))
    tforeach(1:size(B,2)) do j
        colmap = maps[Threads.threadid()]
        _col_map!(colmap, C, j)
        @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] += a * b
                end
            end
        end
    end
    return C
end

function spmm_buffer_simd_threaded!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    maps = [zeros(Int, n) for _ in 1:nthreads()]
    fill!(C.nzval, zero(eltype(C)))
    @threads for j in 1:size(B,2)
        colmap = maps[threadid()]
        _col_map!(colmap, C, j)
        @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            @simd for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] = muladd(a, b, C.nzval[idx])
                end
            end
        end
    end
    return C
end

function spmm_buffer_simd_omt!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    maps = [zeros(Int, n) for _ in 1:Threads.nthreads()]
    fill!(C.nzval, zero(eltype(C)))
    tforeach(1:size(B,2)) do j
        colmap = maps[Threads.threadid()]
        _col_map!(colmap, C, j)
        @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
            k = B.rowval[pb]; b = B.nzval[pb]
            @simd for pa in A.colptr[k]:(A.colptr[k+1]-1)
                i = A.rowval[pa]; a = A.nzval[pa]
                idx = colmap[i]
                if idx != 0
                    C.nzval[idx] = muladd(a, b, C.nzval[idx])
                end
            end
        end
    end
    return C
end


spmm_no_buffer(A::SparseMatrixCSC, B::SparseMatrixCSC) = A * B

function random_sparse(n, m, nnz; T=Float64)
    I = rand(1:n, nnz)
    J = rand(1:m, nnz)
    V = rand(T, nnz)
    sparse(I, J, V, n, m)
end

function expand_pattern(pattern::SparseMatrixCSC, extra::Integer)
    C = copy(pattern)
    n, m = size(pattern)
    added = 0
    while added < extra
        i = rand(1:n)
        j = rand(1:m)
        if C[i, j] == 0
            C[i, j] = one(eltype(C))
            added += 1
        end
    end
    fill!(C.nzval, zero(eltype(C)))
    return C
end

function bench_case(n, p, m, nnzA, nnzB; extra_ratio=0.0)
    A = random_sparse(n, p, nnzA)
    B = random_sparse(p, m, nnzB)

    pattern = A * B
    if extra_ratio > 0
        extra = round(Int, nnz(pattern) * extra_ratio)
        pattern = expand_pattern(pattern, extra)
    end

    C1 = copy(pattern); fill!(C1.nzval, zero(eltype(C1)))
    C2 = copy(pattern); fill!(C2.nzval, zero(eltype(C2)))
    C3 = copy(pattern); fill!(C3.nzval, zero(eltype(C3)))
    C4 = copy(pattern); fill!(C4.nzval, zero(eltype(C4)))
    C5 = copy(pattern); fill!(C5.nzval, zero(eltype(C5)))
    C6 = copy(pattern); fill!(C6.nzval, zero(eltype(C6)))
    C7 = copy(pattern); fill!(C7.nzval, zero(eltype(C7)))

    println("non zeros in A: $(nnz(A))  B: $(nnz(B))  pattern: $(nnz(pattern))")
    println("buffer basic")
    @btime spmm_buffer_fast!($C1, $A, $B)
    println("buffer threaded")
    @btime spmm_buffer_threaded!($C2, $A, $B)
    println("buffer simd")
    @btime spmm_buffer_simd!($C3, $A, $B)
    println("buffer omt")
    @btime spmm_buffer_omt!($C4, $A, $B)
    println("buffer simd threaded")
    @btime spmm_buffer_simd_threaded!($C5, $A, $B)
    println("buffer simd omt")
    @btime spmm_buffer_simd_omt!($C6, $A, $B)
    println("buffer threaded five args")
    @btime spmm_buffer_threaded!($C7, $A, $B, 1.0, 0.0)
    println("no buffer")
    @btime spmm_no_buffer($A, $B)

    C_ref = spmm_no_buffer(A, B)
    @assert isapprox(C1, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C2, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C3, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C4, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C5, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C6, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C7, C_ref; atol=1e-12, rtol=1e-8)
end

function run()
    n = 1000
    p = 1000
    m = 1000
    nnz_values = round.(Int, 10 .^ range(3, log10(2e5); length=7))
    for nnzA in nnz_values
        nnzB = nnzA
        println("\n---- nnz = $nnzA ----")
        bench_case(n, p, m, nnzA, nnzB)
    end
end

run()
