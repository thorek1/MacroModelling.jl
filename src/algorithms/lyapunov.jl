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
                                verbose::Bool = false)::Union{Tuple{Matrix{T}, Bool}, Tuple{ThreadedSparseArrays.ThreadedSparseMatrixCSC{T, Int, SparseMatrixCSC{T, Int}}, Bool}} where T <: Float64
                                # timer::TimerOutput = TimerOutput(),
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

    if !(reached_tol < acceptance_tol) && lyapunov_algorithm ≠ :bartels_stewart && length(C) < 5e7 # try sylvester if previous one didn't solve it
        A = collect(A)

        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bartels_stewart), workspace; tol = tol) # timer = timer)

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
                                    tol::AbstractFloat = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
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
                                    tol::Float64 = 1e-14)::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
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
                                    tol::Float64 = 1e-14)::Tuple{<:AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
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
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Note: workspace is unused for sparse matrices but accepted for API consistency
    𝐂  = copy(C)
    𝐀  = copy(A)
    𝐂A = collect(𝐀)    
    𝐂¹ = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # 𝐂¹ .= 𝐀 * 𝐂 * 𝐀' + 𝐂
        ℒ.mul!(𝐂A, 𝐂, 𝐀')
        ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

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
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
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
        ℒ.mul!(𝐂A, 𝐂, 𝐀')
        ℒ.mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

        ℒ.mul!(𝐀², 𝐀, 𝐀)
        copyto!(𝐀, 𝐀²)
        
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

    return copy(𝐂), iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:bicgstab},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ensure Krylov buffers and bicgstab solver are allocated
    ensure_lyapunov_bicgstab_solver!(workspace)
    
    # Use workspaces
    tmp̄ = workspace.tmp̄
    𝐗 = workspace.𝐗
    b = workspace.b

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, -1, 1)
        copyto!(sol, 𝐗)
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    # Use vectorized C in workspace
    copyto!(b, vec(C))
    
    # Use pre-allocated solver
    Krylov.bicgstab!(workspace.bicgstab_workspace, lyapunov, b, rtol = tol, atol = tol)

    copyto!(𝐗, workspace.bicgstab_workspace.x)

    # ℒ.mul!(tmp̄, A, 𝐗 * A')
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

    # if reached_tol > tol
    #     println("Lyapunov: bicgstab $reached_tol")
    # end

    return copy(𝐗), workspace.bicgstab_workspace.stats.niter, reached_tol
end


function solve_lyapunov_equation(A::AbstractMatrix{T},
                                C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                ::Val{:gmres},
                                workspace::lyapunov_workspace;
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # Ensure Krylov buffers and gmres solver are allocated
    ensure_lyapunov_gmres_solver!(workspace)
    
    # Use workspaces
    tmp̄ = workspace.tmp̄
    𝐗 = workspace.𝐗
    b = workspace.b

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, -1, 1)
        copyto!(sol, 𝐗)
        # sol = @view reshape(𝐗, size(sol))
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    # Use vectorized C in workspace
    copyto!(b, vec(C))
    
    # Use pre-allocated solver
    Krylov.gmres!(workspace.gmres_workspace, lyapunov, b, rtol = tol, atol = tol)

    copyto!(𝐗, workspace.gmres_workspace.x)

    # ℒ.mul!(tmp̄, A, 𝐗 * A')
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

    # if reached_tol > tol
    #     println("Lyapunov: gmres $reached_tol")
    # end

    return copy(𝐗), workspace.gmres_workspace.stats.niter, reached_tol
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
