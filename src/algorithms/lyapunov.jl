# Available algorithms: 
# :doubling     - fast and precise
# :bartels_stewart     - fast for small matrices and precise, dense matrices only
# :bicgstab     - less precise
# :gmres        - less precise

# :iterative    - slow and precise
# :speedmapping - slow and very precise

# solves: A * X * A' + C = X
@stable default_mode = "disable" begin

function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::AbstractMatrix{T},
                                workspace::lyapunov_workspace;
                                lyapunov_algorithm::Symbol = :doubling,
                                tol::AbstractFloat = 1e-14,
                                acceptance_tol::AbstractFloat = 1e-12,
                                verbose::Bool = false,
                                symmetric_rhs::Bool = false)::Union{Tuple{Matrix{T}, Bool}, Tuple{ThreadedSparseArrays.ThreadedSparseMatrixCSC{T, Int, SparseMatrixCSC{T, Int}}, Bool}} where T <: Float64
                                # timer::TimerOutput = TimerOutput(),
    # Ownership: low-level methods below are mixed. Bartels-Stewart and sparse
    # doubling paths return owned matrices, while dense doubling and Krylov
    # paths can return workspace-backed buffers such as workspace.𝐂/workspace.𝐗.
    # This dispatcher currently returns X directly, so callers must not retain
    # the result across workspace reuse unless they make their own copy.
    # Update workspace dimension if needed (for cases like Kalman filter where dimension differs from initial setup)
    n = size(A, 1)
    if workspace.n != n
        workspace.n = n
    end
    
    # @timeit_debug timer "Solve lyapunov equation" begin
    # @timeit_debug timer "Choose matrix formats" begin
        
    if lyapunov_algorithm ≠ :bartels_stewart
        A = choose_matrix_format(A)
    else
        # A = choose_matrix_format(A, density_threshold = 0.0)
        A = collect(A)
    end

    # C = choose_matrix_format(C, density_threshold = 0.0)
    C = collect(C) # C is always dense because the output will be dense in all of these cases as we use this function to compute dense covariance matrices
 
    # end # timeit_debug           
    # @timeit_debug timer "Solve" begin

    X, i, reached_tol = solve_lyapunov_equation(A, C, Val(lyapunov_algorithm), workspace; tol = tol, symmetric_rhs = symmetric_rhs) # timer = timer)

    if verbose
        println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: $lyapunov_algorithm")
    end
    
    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :doubling
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:doubling), workspace; tol = tol, symmetric_rhs = symmetric_rhs) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: doubling")
        end
    end

    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :bicgstab
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bicgstab), workspace; tol = tol, symmetric_rhs = symmetric_rhs) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bicgstab")
        end
    end

    if !(reached_tol < acceptance_tol) && lyapunov_algorithm ≠ :bartels_stewart && length(C) < 5e7 # try sylvester if previous one didn't solve it
        A = collect(A)

        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bartels_stewart), workspace; tol = tol, symmetric_rhs = symmetric_rhs) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bartels_stewart")
        end
    end
    # end # timeit_debug
    # end # timeit_debug
    
    # if (reached_tol > tol) println("Lyapunov failed: $reached_tol") end

    return X, reached_tol < acceptance_tol
end



function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:bartels_stewart},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::AbstractFloat = 1e-14,
                                    symmetric_rhs::Bool = false)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns owned dense matrix from MatrixEquations.lyapd.
    # Note: workspace is unused by bartels_stewart but accepted for API consistency
    𝐂 = try 
        MatrixEquations.lyapd(A, C)::Matrix{T}
    catch
        return C, 0, 1.0
    end
    
    # 𝐂¹ = A * 𝐂 * A' + C

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom   
    
    reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

    # if reached_tol > tol
    #     println("Lyapunov: lyapunov $reached_tol")
    # end

    return 𝐂, 0, reached_tol # return info on convergence
end



function solve_lyapunov_equation(   A::AbstractSparseMatrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14,
                                    symmetric_rhs::Bool = false)::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns owned sparse storage created locally in this method.
    # Note: workspace is unused for sparse matrices but accepted for API consistency
    𝐂  = copy(C)
    𝐀  = copy(A)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐀' + 𝐂

        𝐀 = 𝐀^2

        droptol!(𝐀, eps())

        # Enforce symmetry to prevent numerical drift
        if symmetric_rhs
            𝐂¹ = (𝐂¹ + 𝐂¹') / 2
        end

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        𝐂 = 𝐂¹
    end

    # 𝐂¹ = 𝐀 * 𝐂 * 𝐀' + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

    # if reached_tol > tol
    #     println("Lyapunov: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14,
                                    symmetric_rhs::Bool = false)::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns owned sparse storage created locally in this method.
    # Note: workspace is unused for sparse matrices but accepted for API consistency
    𝐂  = copy(C)
    𝐀  = copy(A)

    𝐀² = similar(𝐀)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐀' + 𝐂

        ℒ.mul!(𝐀², 𝐀, 𝐀)
        copyto!(𝐀, 𝐀²)

        # droptol!(𝐀, eps())

        # Enforce symmetry to prevent numerical drift
        if symmetric_rhs
            𝐂¹ = (𝐂¹ + 𝐂¹') / 2
        end

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        𝐂 = 𝐂¹
    end

    # 𝐂¹ = 𝐀 * 𝐂 * 𝐀' + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

    # if reached_tol > tol
    #     println("Lyapunov: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_lyapunov_equation(   A::AbstractSparseMatrix{T},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14,
                                    symmetric_rhs::Bool = false)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns owned dense storage created locally in this method.
    # Note: workspace is unused for sparse matrices but accepted for API consistency
    𝐂  = copy(C)
    𝐀  = copy(A)
    𝐂A = collect(𝐀)    
    𝐂¹ = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # 𝐂¹ .= 𝐀 * 𝐂 * 𝐀' + 𝐂
        # When C is symmetric, use Symmetric wrapper for dsymm dispatch
        if symmetric_rhs
            ℒ.mul!(𝐂A, 𝐀, ℒ.Symmetric(𝐂, :U))
            ℒ.mul!(𝐂¹, 𝐂A, 𝐀', 1, 1)
        else
            ℒ.mul!(𝐂A, 𝐂, 𝐀')
            ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)
        end

        # 𝐀 *= 𝐀
        𝐀 = 𝐀^2 # faster than A *= A
        # copyto!(𝐂A,𝐀)
        # 𝐀 = sparse(𝐀 * 𝐂A)
        # 𝐀 = sparse(𝐂A * 𝐀) # faster than sparse-dense matmul but slower than sparse sparse matmul
        
        droptol!(𝐀, eps())

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        # Enforce symmetry to prevent numerical drift when exploiting symmetric structure
        if symmetric_rhs
            ℒ.copytri!(𝐂¹, 'U')
        end

        copy!(𝐂,𝐂¹)
        # 𝐂 = 𝐂¹
    end

    # ℒ.mul!(𝐂A, 𝐂, A')
    # ℒ.mul!(𝐂¹, A, 𝐂A)
    # ℒ.axpy!(1, C, 𝐂¹)

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

    # if reached_tol > tol
    #     println("Lyapunov: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14,
                                    symmetric_rhs::Bool = false)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns workspace-backed dense buffer workspace.𝐂.
    # Ensure doubling buffers are allocated
    ensure_lyapunov_doubling_buffers!(workspace)
    
    # Use workspaces for dense-dense case
    𝐂  = workspace.𝐂
    𝐂¹ = workspace.𝐂¹
    𝐀  = workspace.𝐀
    𝐂A = workspace.𝐂A
    𝐀² = workspace.𝐀²
    
    copyto!(𝐂, C)
    copyto!(𝐂¹, C)
    copyto!(𝐀, A)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # When C is symmetric, use Symmetric wrapper so mul! dispatches to BLAS dsymm
        # Matmul order: A*C then (A*C)*A' — first mul! benefits from dsymm (reads only upper triangle of C)
        if symmetric_rhs
            ℒ.mul!(𝐂A, 𝐀, ℒ.Symmetric(𝐂, :U))
            ℒ.mul!(𝐂¹, 𝐂A, 𝐀', 1, 1)
        else
            ℒ.mul!(𝐂A, 𝐂, 𝐀')
            ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)
        end

        ℒ.mul!(𝐀², 𝐀, 𝐀)
        copyto!(𝐀, 𝐀²)
        
        # Enforce symmetry to prevent numerical drift when exploiting symmetric structure
        if symmetric_rhs
            ℒ.copytri!(𝐂¹, 'U')
        end

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copyto!(𝐂, 𝐂¹)
    end

    # ℒ.mul!(𝐂A, 𝐂, A')
    # ℒ.mul!(𝐂¹, A, 𝐂A)
    # ℒ.axpy!(1, C, 𝐂¹)

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom
    
    reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

    # if reached_tol > tol
    #     println("Lyapunov: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:bicgstab},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14,
                                symmetric_rhs::Bool = false)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns workspace-backed dense Krylov buffer workspace.𝐗.
    
    if symmetric_rhs
        # vech-space Krylov: solve for n(n+1)/2 unique elements only
        ensure_lyapunov_krylov_vech_solver!(workspace, :bicgstab)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        n = size(A, 1)
        n_vech = n * (n + 1) ÷ 2
        b_vech = workspace.b_vech

        function lyapunov_vech_bicgstab!(sol, 𝐱)
            # Unpack vech → upper triangle of 𝐗, mirror to full symmetric
            k = 1
            @inbounds for j in 1:n, i in 1:j
                𝐗[i, j] = 𝐱[k]
                k += 1
            end
            ℒ.copytri!(𝐗, 'U')
            # X - A*X*A' using dsymm for the first matmul
            ℒ.mul!(tmp̄, ℒ.Symmetric(𝐗, :U), A')  # dsymm: tmp̄ = X * A'
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)               # 𝐗 = X - A * X * A'
            # Pack upper triangle → sol
            k = 1
            @inbounds for j in 1:n, i in 1:j
                sol[k] = 𝐗[i, j]
                k += 1
            end
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, n_vech, n_vech, true, true, lyapunov_vech_bicgstab!)

        # Pack C upper triangle into b_vech
        k = 1
        @inbounds for j in 1:n, i in 1:j
            b_vech[k] = C[i, j]
            k += 1
        end

        Krylov.bicgstab!(workspace.bicgstab_vech, lyapunov_op, b_vech, rtol = tol, atol = tol)

        # Unpack solution vech → full symmetric 𝐗
        k = 1
        @inbounds for j in 1:n, i in 1:j
            𝐗[i, j] = workspace.bicgstab_vech.x[k]
            k += 1
        end
        ℒ.copytri!(𝐗, 'U')

        reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

        return 𝐗, workspace.bicgstab_vech.stats.niter, reached_tol
    else
        # Standard full-space Krylov
        ensure_lyapunov_krylov_solver!(workspace, :bicgstab)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        b = workspace.b

        function lyapunov_bicgstab!(sol,𝐱)
            copyto!(𝐗, 𝐱)
            ℒ.mul!(tmp̄, 𝐗, A')
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)
            copyto!(sol, 𝐗)
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov_bicgstab!)

        copyto!(b, vec(C))
        Krylov.bicgstab!(workspace.bicgstab, lyapunov_op, b, rtol = tol, atol = tol)
        copyto!(𝐗, workspace.bicgstab.x)

        reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

        return 𝐗, workspace.bicgstab.stats.niter, reached_tol
    end
end


function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:gmres},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14,
                                symmetric_rhs::Bool = false)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns workspace-backed dense Krylov buffer workspace.𝐗.
    
    if symmetric_rhs
        # vech-space Krylov: solve for n(n+1)/2 unique elements only
        ensure_lyapunov_krylov_vech_solver!(workspace, :gmres)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        n = size(A, 1)
        n_vech = n * (n + 1) ÷ 2
        b_vech = workspace.b_vech

        function lyapunov_vech_gmres!(sol, 𝐱)
            # Unpack vech → upper triangle of 𝐗, mirror to full symmetric
            k = 1
            @inbounds for j in 1:n, i in 1:j
                𝐗[i, j] = 𝐱[k]
                k += 1
            end
            ℒ.copytri!(𝐗, 'U')
            # X - A*X*A' using dsymm for the first matmul
            ℒ.mul!(tmp̄, ℒ.Symmetric(𝐗, :U), A')  # dsymm: tmp̄ = X * A'
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)               # 𝐗 = X - A * X * A'
            # Pack upper triangle → sol
            k = 1
            @inbounds for j in 1:n, i in 1:j
                sol[k] = 𝐗[i, j]
                k += 1
            end
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, n_vech, n_vech, true, true, lyapunov_vech_gmres!)

        # Pack C upper triangle into b_vech
        k = 1
        @inbounds for j in 1:n, i in 1:j
            b_vech[k] = C[i, j]
            k += 1
        end

        Krylov.gmres!(workspace.gmres_vech, lyapunov_op, b_vech, rtol = tol, atol = tol)

        # Unpack solution vech → full symmetric 𝐗
        k = 1
        @inbounds for j in 1:n, i in 1:j
            𝐗[i, j] = workspace.gmres_vech.x[k]
            k += 1
        end
        ℒ.copytri!(𝐗, 'U')

        reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

        return 𝐗, workspace.gmres_vech.stats.niter, reached_tol
    else
        # Standard full-space Krylov
        ensure_lyapunov_krylov_solver!(workspace, :gmres)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        b = workspace.b

        function lyapunov_gmres!(sol,𝐱)
            copyto!(𝐗, 𝐱)
            ℒ.mul!(tmp̄, 𝐗, A')
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)
            copyto!(sol, 𝐗)
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov_gmres!)

        copyto!(b, vec(C))
        Krylov.gmres!(workspace.gmres, lyapunov_op, b, rtol = tol, atol = tol)
        copyto!(𝐗, workspace.gmres.x)

        reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

        return 𝐗, workspace.gmres.stats.niter, reached_tol
    end
end


# function solve_lyapunov_equation(A::AbstractMatrix{Float64},
#                                 C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
#                                 ::Val{:iterative};
#                                 tol::AbstractFloat = 1e-14,
#                                 timer::TimerOutput = TimerOutput())
#     𝐂  = copy(C)
#     𝐂¹ = copy(C)
#     𝐂A = copy(C)
    
#     max_iter = 10000
    
#     iters = max_iter

#     for i in 1:max_iter
#         ℒ.mul!(𝐂A, 𝐂, A')
#         ℒ.mul!(𝐂¹, A, 𝐂A)
#         ℒ.axpy!(1, C, 𝐂¹)
    
#         if i % 10 == 0
#             normdiff = ℒ.norm(𝐂¹ - 𝐂)
#             if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
#             # if isapprox(𝐂¹, 𝐂, rtol = tol)
#                 iters = i
#                 break
#             end
#         end
    
#         copyto!(𝐂, 𝐂¹)
#     end

#     # ℒ.mul!(𝐂A, 𝐂, A')
#     # ℒ.mul!(𝐂¹, A, 𝐂A)
#     # ℒ.axpy!(1, C, 𝐂¹)

#     # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

#     # ℒ.axpy!(-1, 𝐂, 𝐂¹)

#     # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom
    
#     reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

#     # if reached_tol > tol
#     #     println("Lyapunov: iterative $reached_tol")
#     # end

#     return 𝐂, iters, reached_tol # return info on convergence
# end


# function solve_lyapunov_equation(A::AbstractMatrix{Float64},
#                                     C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
#                                     ::Val{:speedmapping};
#                                     tol::AbstractFloat = 1e-14,
#                                     timer::TimerOutput = TimerOutput())
#     𝐂A = similar(C)

#     soll = speedmapping(C; 
#             m! = (X, x) -> begin
#                 ℒ.mul!(𝐂A, x, A')
#                 ℒ.mul!(X, A, 𝐂A)
#                 ℒ.axpy!(1, C, X)
#             end, stabilize = false, maps_limit = 1000, tol = tol)
    
#     𝐂 = soll.minimizer

#     reached_tol = ℒ.norm(A * 𝐂 * A' + C - 𝐂) / ℒ.norm(𝐂)

#     # if reached_tol > tol
#     #     println("Lyapunov: speedmapping $reached_tol")
#     # end

#     return 𝐂, soll.maps, reached_tol
# end

end # dispatch_doctor
