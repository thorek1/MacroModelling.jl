# Available algorithms: 
# :doubling     - fast and precise
# :bartels_stewart     - fast for small matrices and precise, dense matrices only
# :bicgstab     - less precise
# :gmres        - less precise

# :iterative    - slow and precise
# :speedmapping - slow and very precise

# solves: A * X * A' + C = X
@stable default_mode = "disable" begin

# Pack upper triangle of a symmetric matrix into a vech vector (in-place).
function vech!(vech_vector::AbstractVector, symmetric_matrix::AbstractMatrix)
    matrix_size = size(symmetric_matrix, 1)
    @inbounds for column in 1:matrix_size
        offset = div(column * (column - 1), 2)
        @simd for row in 1:column
            vech_vector[offset + row] = symmetric_matrix[row, column]
        end
    end
    return vech_vector
end

# Unpack a vech vector into a full symmetric matrix (in-place).
function fill_symmetric_from_vech!(symmetric_matrix::AbstractMatrix, vech_vector::AbstractVector)
    matrix_size = size(symmetric_matrix, 1)
    # Fill the upper triangle
    @inbounds for column in 1:matrix_size
        offset = div(column * (column - 1), 2)
        @simd for row in 1:column
            symmetric_matrix[row, column] = vech_vector[offset + row]
        end
    end
    # Copy the upper triangle to the lower triangle
    @inbounds for column in 1:matrix_size
        @simd for row in (column + 1):matrix_size
            symmetric_matrix[row, column] = symmetric_matrix[column, row]
        end
    end
    return symmetric_matrix
end

# Approximate symmetry check (allocation-free).  Returns true when
# max|C[i,j] - C[j,i]| ≤ rtol · max|C[i,j]| over all off-diagonal pairs.
function _is_approx_symmetric(C::AbstractMatrix;
                              rtol::Real = sqrt(eps(real(eltype(C)))))
    m, n = size(C)
    m == n || return false
    max_asym = zero(real(eltype(C)))
    max_abs  = zero(real(eltype(C)))
    @inbounds for j in 1:n, i in 1:(j - 1)
        max_asym = max(max_asym, abs(C[i, j] - C[j, i]))
        max_abs  = max(max_abs,  abs(C[i, j]), abs(C[j, i]))
    end
    return max_abs == 0 ? true : max_asym ≤ rtol * max_abs
end

function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::AbstractMatrix{T},
                                workspace::lyapunov_workspace;
                                initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                lyapunov_algorithm::Symbol = :doubling,
                                tol::SolverTolerances = SolverTolerances(atol = 1e-14,
                                                                          rtol = 1e-14,
                                                                          initial_guess_acceptance_tol = 1e-12,
                                                                          acceptance_tol = 1e-12),
                                verbose::Bool = false,
                                has_unit_roots::Bool = false)::Union{Tuple{Matrix{T}, Bool}, Tuple{ThreadedSparseArrays.ThreadedSparseMatrixCSC{T, Int, SparseMatrixCSC{T, Int}}, Bool}} where T <: Float64
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
        
    if lyapunov_algorithm == :bartels_stewart && !_has_bartels_stewart()
        error("The :bartels_stewart algorithm requires the MatrixEquations package. Run `using MatrixEquations` to enable it.")
    end

    if lyapunov_algorithm ≠ :bartels_stewart
        A = choose_matrix_format(A)
    else
        # A = choose_matrix_format(A, density_threshold = 0.0)
        A = collect(A)
    end

    # C = choose_matrix_format(C, density_threshold = 0.0)
    C = collect(C) # C is always dense because the output will be dense in all of these cases as we use this function to compute dense covariance matrices

    initial_guess_acceptance_tol = tol.initial_guess_acceptance_tol
    acceptance_tol = tol.acceptance_tol

    if length(initial_guess) > 0
        guess = initial_guess
        if size(guess) == size(C)
            ensure_lyapunov_doubling_buffers!(workspace)
            _tmp = workspace.𝐂A
            _res = workspace.𝐂¹
            ℒ.mul!(_tmp, guess, A')
            ℒ.mul!(_res, A, _tmp)
            ℒ.axpy!(1, C, _res)
            ℒ.axpy!(-1, guess, _res)

            denom = max(ℒ.norm(guess), ℒ.norm(C))
            reached_tol = denom == 0 ? 0.0 : ℒ.norm(_res) / denom
            if reached_tol < initial_guess_acceptance_tol
                if verbose println("Lyapunov equation - initial guess achieves relative tol of $reached_tol (initial guess tol: $initial_guess_acceptance_tol)") end
                return choose_matrix_format(guess), true
            end
        end
    end
 
    # end # timeit_debug           
    # @timeit_debug timer "Solve" begin

    # Fast path: when unit roots are known from QME solve, skip directly to Schur deflation
    # instead of wasting O(n³) on solvers guaranteed to fail.
    if has_unit_roots
        A_dense = collect(A)
        C_dense = collect(C)

        X_deflated, deflation_solved = solve_lyapunov_schur_deflation(A_dense, C_dense, workspace;
                                                                        tol = tol,
                                                                        verbose = verbose)
        if deflation_solved
            if verbose
                println("Lyapunov equation - solved via Schur deflation (unit roots pre-detected)")
            end
            return X_deflated, true
        end
        # If deflation failed despite the flag, fall through to standard solvers
    end

    X, i, reached_tol = solve_lyapunov_equation(A, C, Val(lyapunov_algorithm), workspace; tol = tol) # timer = timer)

    if verbose
        println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: $lyapunov_algorithm")
    end
    
    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :doubling
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:doubling), workspace; tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: doubling")
        end
    end

    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :bicgstab
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bicgstab), workspace; tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bicgstab")
        end
    end

    if !(reached_tol < acceptance_tol) && lyapunov_algorithm ≠ :bartels_stewart && length(C) < 5e7 && _has_bartels_stewart() # try bartels_stewart if previous one didn't solve it
        A = collect(A)

        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bartels_stewart), workspace; tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bartels_stewart")
        end
    end

    # Schur deflation fallback: when all standard solvers fail, check for unit-root
    # eigenvalues and solve only the stationary subspace.
    if !(reached_tol < acceptance_tol)
        A_dense = collect(A)
        C_dense = collect(C)

        X_deflated, deflation_solved = solve_lyapunov_schur_deflation(A_dense, C_dense, workspace;
                                                                       tol = tol,
                                                                       verbose = verbose)
        if deflation_solved
            X = X_deflated
            reached_tol = zero(T) # signal success
            if verbose
                println("Lyapunov equation - solved via Schur deflation (unit-root subspace set to NaN)")
            end
        end
    end

    return X, reached_tol < acceptance_tol
end


# Keep the low-level bartels-stewart signature available in core so fallback
# paths remain well-typed when MatrixEquations is not loaded.
function solve_lyapunov_equation(A::AbstractMatrix{T},
                                          C::AbstractMatrix{T},
                                          ::Val{:bartels_stewart},
                                          workspace::lyapunov_workspace;
                                          tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
     return Matrix(C), 0, T(Inf)
end



function solve_lyapunov_equation(   A::AbstractSparseMatrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::SolverTolerances = SolverTolerances())::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol.rtol
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
                                    tol::SolverTolerances = SolverTolerances())::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol.rtol
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
                                    tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns owned dense storage created locally in this method.
    # Note: workspace is unused for sparse matrices but accepted for API consistency
    𝐂  = copy(C)
    𝐀  = copy(A)
    𝐂A = collect(𝐀)    
    𝐂¹ = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # Sparse A: standard matmul is efficient; Symmetric wrapper lacks optimised sparse dispatch
        ℒ.mul!(𝐂A, 𝐂, 𝐀')
        ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

        # 𝐀 *= 𝐀
        𝐀 = 𝐀^2 # faster than A *= A
        # copyto!(𝐂A,𝐀)
        # 𝐀 = sparse(𝐀 * 𝐂A)
        # 𝐀 = sparse(𝐂A * 𝐀) # faster than sparse-dense matmul but slower than sparse sparse matmul
        
        droptol!(𝐀, eps())

        if i % 2 == 0
            copyto!(𝐂A, 𝐂¹)
            ℒ.axpy!(-1, 𝐂, 𝐂A)
            normdiff = ℒ.norm(𝐂A)
            maxnorm = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))
            if !isfinite(normdiff) || normdiff / maxnorm < tol.rtol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
        # 𝐂 = 𝐂¹
    end

    ℒ.mul!(𝐂A, 𝐂, A')
    ℒ.mul!(𝐂¹, A, 𝐂A)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = ℒ.norm(𝐂¹) / ℒ.norm(𝐂)

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:doubling},
                                    workspace::lyapunov_workspace;
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
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
        # Always use dgemm — dsymm is slower at typical DSGE sizes (n ≤ 400)
        ℒ.mul!(𝐂A, 𝐂, 𝐀')
        ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

        ℒ.mul!(𝐀², 𝐀, 𝐀)
        copyto!(𝐀, 𝐀²)

        if i % 2 == 0
            copyto!(𝐂A, 𝐂¹)
            ℒ.axpy!(-1, 𝐂, 𝐂A)
            normdiff = ℒ.norm(𝐂A)
            maxnorm = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))
            if !isfinite(normdiff) || normdiff / maxnorm < tol.rtol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copyto!(𝐂, 𝐂¹)
    end

    ℒ.mul!(𝐂A, 𝐂, A')
    ℒ.mul!(𝐂¹, A, 𝐂A)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐂, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / ℒ.norm(𝐂)

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:bicgstab},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns workspace-backed dense Krylov buffer workspace.𝐗.
    
    if _is_approx_symmetric(C)
        # vech-space Krylov: solve for n(n+1)/2 unique elements only
        ensure_lyapunov_krylov_vech_solver!(workspace, :bicgstab)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        n = size(A, 1)
        n_vech = n * (n + 1) ÷ 2
        b_vech = workspace.b_vech

        function lyapunov_vech_bicgstab!(sol, 𝐱)
            fill_symmetric_from_vech!(𝐗, 𝐱)
            ℒ.mul!(tmp̄, 𝐗, A')
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)
            vech!(sol, 𝐗)
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, n_vech, n_vech, true, true, lyapunov_vech_bicgstab!)

        vech!(b_vech, C)

        Krylov.bicgstab!(workspace.bicgstab_vech, lyapunov_op, b_vech, rtol = tol.rtol, atol = tol.atol)

        fill_symmetric_from_vech!(𝐗, workspace.bicgstab_vech.x)

        # Allocation-free residual: reuse tmp̄ for intermediate, 𝐗 is the solution
        ensure_lyapunov_doubling_buffers!(workspace)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(workspace.𝐂¹, A, tmp̄)
        ℒ.axpy!(1, C, workspace.𝐂¹)
        ℒ.axpy!(-1, 𝐗, workspace.𝐂¹)
        reached_tol = ℒ.norm(workspace.𝐂¹) / ℒ.norm(𝐗)

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
        Krylov.bicgstab!(workspace.bicgstab, lyapunov_op, b, rtol = tol.rtol, atol = tol.atol)
        copyto!(𝐗, workspace.bicgstab.x)

        # Allocation-free residual
        ensure_lyapunov_doubling_buffers!(workspace)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(workspace.𝐂¹, A, tmp̄)
        ℒ.axpy!(1, C, workspace.𝐂¹)
        ℒ.axpy!(-1, 𝐗, workspace.𝐂¹)
        reached_tol = ℒ.norm(workspace.𝐂¹) / ℒ.norm(𝐗)

        return 𝐗, workspace.bicgstab.stats.niter, reached_tol
    end
end


function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:gmres},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ownership: returns workspace-backed dense Krylov buffer workspace.𝐗.
    
    if _is_approx_symmetric(C)
        # vech-space Krylov: solve for n(n+1)/2 unique elements only
        ensure_lyapunov_krylov_vech_solver!(workspace, :gmres)
        tmp̄ = workspace.tmp̄
        𝐗 = workspace.𝐗
        n = size(A, 1)
        n_vech = n * (n + 1) ÷ 2
        b_vech = workspace.b_vech

        function lyapunov_vech_gmres!(sol, 𝐱)
            fill_symmetric_from_vech!(𝐗, 𝐱)
            ℒ.mul!(tmp̄, 𝐗, A')
            ℒ.mul!(𝐗, A, tmp̄, -1, 1)
            vech!(sol, 𝐗)
        end

        lyapunov_op = LinearOperators.LinearOperator(Float64, n_vech, n_vech, true, true, lyapunov_vech_gmres!)

        vech!(b_vech, C)

        Krylov.gmres!(workspace.gmres_vech, lyapunov_op, b_vech, rtol = tol.rtol, atol = tol.atol)

        fill_symmetric_from_vech!(𝐗, workspace.gmres_vech.x)

        # Allocation-free residual
        ensure_lyapunov_doubling_buffers!(workspace)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(workspace.𝐂¹, A, tmp̄)
        ℒ.axpy!(1, C, workspace.𝐂¹)
        ℒ.axpy!(-1, 𝐗, workspace.𝐂¹)
        reached_tol = ℒ.norm(workspace.𝐂¹) / ℒ.norm(𝐗)

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
        Krylov.gmres!(workspace.gmres, lyapunov_op, b, rtol = tol.rtol, atol = tol.atol)
        copyto!(𝐗, workspace.gmres.x)

        # Allocation-free residual
        ensure_lyapunov_doubling_buffers!(workspace)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(workspace.𝐂¹, A, tmp̄)
        ℒ.axpy!(1, C, workspace.𝐂¹)
        ℒ.axpy!(-1, 𝐗, workspace.𝐂¹)
        reached_tol = ℒ.norm(workspace.𝐂¹) / ℒ.norm(𝐗)

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

# Schur deflation for Lyapunov equations with unit-root eigenvalues.
#
# When A has eigenvalues on or outside the unit circle, the standard Lyapunov
# equation A*X*A' + C = X has no finite solution. This function decomposes A via
# real Schur factorization, reorders so that unstable eigenvalues (|λ| ≥ 1 - unit_root_tol)
# come first, then solves the Lyapunov equation only for the stationary (lower-right)
# block. Original-basis entries whose variance is contaminated by unit-root directions
# are set to NaN.
#
# Returns (X, solved::Bool) where X is n×n with NaN for unit-root-affected entries.

# Type-stable wrapper for ordered Schur decomposition via LAPACK gees!.
# gees! returns a union type (eigenvalue vector is Float64 or ComplexF64),
# so this barrier function isolates the type instability and returns only
# the concrete types needed by callers: (T_matrix, Z_vectors, n_selected).
function _ordered_schur!(A_work::Matrix{T}, unit_root_tol::Float64,
                         schur_ws::FastLapackInterface.SchurWs{T}) where T <: AbstractFloat
    ℒ.LAPACK.gees!(schur_ws, 'V', A_work;
                    select = FastLapackInterface.ed,
                    criterium = (1 - unit_root_tol)^2,
                    resize = true)
    vs = schur_ws.vs::Matrix{T}
    n_sel = schur_ws.sdim[]::Int
    return (A_work, vs, n_sel)
end

function solve_lyapunov_schur_deflation(A::DenseMatrix{T},
                                         C::DenseMatrix{T},
                                         workspace::lyapunov_workspace;
                                         tol::SolverTolerances = SolverTolerances(),
                                         verbose::Bool = false,
                                         unit_root_tol::Float64 = 1e-8)::Tuple{Matrix{T}, Bool} where T <: AbstractFloat
    n = size(A, 1)

    # Real Schur decomposition with eigenvalue reordering in one step via LAPACK gees!.
    # FastLapackInterface.ed selects eigenvalues on the exterior of the disk (|λ|² ≥ criterium),
    # placing unstable eigenvalues in the top-left block.
    # After: Tmat = [T_uu T_us; 0 T_ss] where T_ss is the stable block.
    A_work = copy(A)
    Tmat, U, n_unstable = _ordered_schur!(A_work, unit_root_tol, workspace.schur_ws)

    if n_unstable == 0
        # No unit roots found — deflation not applicable, signal failure so caller
        # does not silently accept a potentially incorrect result
        return Matrix{T}(undef, 0, 0), false
    end

    if n_unstable == n
        # All eigenvalues are unit roots — no stationary subspace
        return fill(T(NaN), n, n), true
    end

    n_stable = n - n_unstable
    stable_range = (n_unstable + 1):n

    T_ss = Tmat[stable_range, stable_range]

    # Transform noise covariance to Schur basis
    C_schur = U' * C * U
    C_ss = C_schur[stable_range, stable_range]

    # Symmetrize (numerical noise from rotation can break symmetry)
    C_ss = (C_ss + C_ss') / 2

    # Solve the reduced Lyapunov equation: X_ss = T_ss * X_ss * T_ss' + C_ss
    # This converges because all eigenvalues of T_ss are strictly inside the unit circle.
    # Try multiple algorithms directly (not via dispatch, to avoid recursion into Schur deflation).
    ws_stable = Lyapunov_workspace(n_stable)
    X_ss_result, sub_iters, sub_tol = solve_lyapunov_equation(T_ss, C_ss, Val(:doubling), ws_stable; tol = tol)

    if sub_tol > tol.acceptance_tol
        X_ss_result, sub_iters, sub_tol = solve_lyapunov_equation(T_ss, C_ss, Val(:bicgstab), ws_stable; tol = tol)
    end

    if sub_tol > tol.acceptance_tol && _has_bartels_stewart() && length(C_ss) < 5e7
        X_ss_result, sub_iters, sub_tol = solve_lyapunov_equation(T_ss, C_ss, Val(:bartels_stewart), ws_stable; tol = tol)
    end

    if sub_tol > tol.acceptance_tol
        if verbose
            println("Schur deflation: stable sub-block Lyapunov failed (tol=$sub_tol)")
        end
        return Matrix{T}(undef, 0, 0), false
    end

    X_ss = collect(X_ss_result)

    # Map back to original coordinates.
    # Only the stationary component contributes finite variance:
    #   Σ_stationary = U_s * X_ss * U_s'
    U_s = U[:, stable_range]
    Σ = U_s * X_ss * U_s'

    # Identify which original variables have any loading on unstable Schur vectors.
    # These variables have infinite unconditional variance → set to NaN.
    U_u = @view U[:, 1:n_unstable]
    unstable_loading = vec(sum(abs2, U_u; dims = 2))  # ‖U_u[i,:]‖²
    unit_root_vars = unstable_loading .> unit_root_tol

    # Set rows and columns of unit-root-affected variables to NaN
    for i in 1:n
        if unit_root_vars[i]
            Σ[i, :] .= T(NaN)
            Σ[:, i] .= T(NaN)
        end
    end

    if verbose
        println("Schur deflation: $n_unstable unstable eigenvalue(s), ",
                "$n_stable stable, $(count(unit_root_vars)) variable(s) set to NaN")
    end

    return Σ, true
end


end # dispatch_doctor
