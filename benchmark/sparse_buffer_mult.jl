using SparseArrays
using BenchmarkTools
using Base.Threads
using OhMyThreads: tforeach, TaskLocalValue
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
function _col_map!(colmap::Vector{I}, C::SparseMatrixCSC, j::Int) where I <: Integer
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
    colmap = zeros(UInt, n)
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
    colmap = zeros(UInt, n)
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
    maps = [zeros(UInt, n) for _ in 1:nthreads()]
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


function spmm_buffer_tasks!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC, α, β)
    n    = size(C, 1)
    nB   = size(B, 2)
    nth  = nthreads()
    sz   = cld(nB, nth)                # columns per chunk

    # scale or zero C.nzval
    if β == zero(β)
        fill!(C.nzval, zero(eltype(C)))
    else
        @. C.nzval *= β
    end

    tasks = Vector{Task}(undef, nth)
    for t in 1:nth
        j0 = (t-1)*sz + 1
        j1 = min(t*sz, nB)
        tasks[t] = @spawn begin
            colmap = zeros(UInt, n)
            for j in j0:j1
                _col_map!(colmap, C, j)
                @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
                    k = B.rowval[pb]; b = B.nzval[pb]
                    @inbounds @simd for pa in A.colptr[k]:(A.colptr[k+1]-1)
                        i   = A.rowval[pa]
                        a   = A.nzval[pa]
                        idx = colmap[i]
                        if idx != 0
                            C.nzval[idx] += α * a * b
                        end
                    end
                end
            end
        end
    end

    # wait for all tasks to finish
    foreach(fetch, tasks)

    return C
end

spmm_buffer_tasks!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC) =
    spmm_buffer_tasks!(C, A, B, one(eltype(C)), zero(eltype(C)))


# function spmm_buffer_omt!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
#     n = size(C, 1)
#     maps = [zeros(Int, n) for _ in 1:Threads.nthreads()]
#     fill!(C.nzval, zero(eltype(C)))
#     tforeach(1:size(B,2)) do j
#         colmap = maps[Threads.threadid()]
#         _col_map!(colmap, C, j)
#         @inbounds for pb in B.colptr[j]:(B.colptr[j+1]-1)
#             k = B.rowval[pb]; b = B.nzval[pb]
#             for pa in A.colptr[k]:(A.colptr[k+1]-1)
#                 i = A.rowval[pa]; a = A.nzval[pa]
#                 idx = colmap[i]
#                 if idx != 0
#                     C.nzval[idx] += a * b
#                 end
#             end
#         end
#     end
#     return C
# end


function spmm_buffer_omt_fast!(C::SparseMatrixCSC{Tv,Int},
                               A::SparseMatrixCSC{Tv,Int},
                               B::SparseMatrixCSC{Tv,Int}) where {Tv<:Number}
    n = size(C,1)
    # zero result
    fill!(C.nzval, zero(Tv))

    # task-local map buffer
    tlv = TaskLocalValue{Vector{Int}}(() -> zeros(UInt, n))

    # hoist field accesses
    Acolptr, Arowval, Anzval = A.colptr, A.rowval, A.nzval
    Bcolptr, Browval, Bnzval = B.colptr, B.rowval, B.nzval
    Ccolptr, Cnzval       = C.colptr, C.nzval

    # parallel over columns of B
    tforeach(1:size(B,2)) do j
        colmap = tlv[]                     # zeroed Int[n]
        _col_map!(colmap, C, j)

        # precompute B’s column slice
        pb_start = Bcolptr[j]
        pb_end   = Bcolptr[j+1] - 1

        @inbounds for pb in pb_start:pb_end
            k = Browval[pb]; b = Bnzval[pb]
            # precompute A’s column slice
            pa_start = Acolptr[k]
            pa_end   = Acolptr[k+1] - 1

            @inbounds @simd for pa in pa_start:pa_end
                i   = Arowval[pa]
                a   = Anzval[pa]
                idx = colmap[i]
                if idx != 0
                    Cnzval[idx] += a * b
                end
            end
        end
    end

    return C
end

function spmm_buffer_omt!(C::SparseMatrixCSC, A::SparseMatrixCSC, B::SparseMatrixCSC)
    n = size(C, 1)
    # zero out result
    fill!(C.nzval, zero(eltype(C)))

    # create a task-local map vector, initialised once per task
    tlv = TaskLocalValue{Vector{UInt}}(() -> zeros(UInt, n))

    # loop over columns of B in parallel
    tforeach(1:size(B, 2)) do j
        colmap = tlv[]                  # per-task zeroed Int[n]
        _col_map!(colmap, C, j)

        @inbounds for pb in B.colptr[j]:(B.colptr[j+1] - 1)
            k = B.rowval[pb]; b = B.nzval[pb]
            for pa in A.colptr[k]:(A.colptr[k+1] - 1)
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
    # println("buffer threaded")
    # @btime spmm_buffer_threaded!($C2, $A, $B)
    # println("buffer SparseArrays")
    # @btime spmm_buffer!($C3, $A, $B)
    # println("buffer simd")
    # @btime spmm_buffer_simd!($C3, $A, $B)
    # println("buffer omt")
    # @btime spmm_buffer_omt!($C3, $A, $B)
    println("buffer tasks")
    @btime spmm_buffer_tasks!($C4, $A, $B)
    # println("buffer simd threaded")
    # @btime spmm_buffer_simd_threaded!($C5, $A, $B)
    # println("buffer simd omt")
    # @btime spmm_buffer_simd_omt!($C6, $A, $B)
    # println("buffer threaded five args")
    # @btime spmm_buffer_threaded!($C7, $A, $B, 1.0, 0.0)
    println("no buffer")
    @btime spmm_no_buffer($A, $B)

    C_ref = spmm_no_buffer(A, B)
    @assert isapprox(C1, C_ref; atol=1e-12, rtol=1e-8)
    # @assert isapprox(C2, C_ref; atol=1e-12, rtol=1e-8)
    # @assert isapprox(C3, C_ref; atol=1e-12, rtol=1e-8)
    @assert isapprox(C4, C_ref; atol=1e-12, rtol=1e-8)
    # @assert isapprox(C5, C_ref; atol=1e-12, rtol=1e-8)
    # @assert isapprox(C6, C_ref; atol=1e-12, rtol=1e-8)
    # @assert isapprox(C7, C_ref; atol=1e-12, rtol=1e-8)
end

function run()
    n = 1000
    p = 100000
    m = 100000
    nnz_values = round.(Int, 10 .^ range(4, 6; length=3))
    for nnzA in nnz_values
        nnzB = nnzA
        println("\n---- nnz = $nnzA ----")
        bench_case(n, p, m, nnzA, nnzB)
    end
end

run()



A = random_sparse(100, 1000000, Int(1e5))
B = random_sparse(1000000, 1000000, Int(1e5))
C = A * B
C.nzval .= 0

@profview for i in 1:100 spmm_buffer_tasks!(C,A,B) end
@profview for i in 1:1000 spmm_buffer_fast!(C,A,B) end

using ThreadedSparseArrays
thA = ThreadedSparseMatrixCSC(A)
thB = ThreadedSparseMatrixCSC(B)

@benchmark $thA * $thB
@benchmark $A * $B
@benchmark spmm_buffer_tasks!($C,$A,$B)
@benchmark spmm_buffer_fast!($C,$A,$B)