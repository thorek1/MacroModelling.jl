# Available algorithms: 
# :doubling     - fast and precise
# :bartels_stewart     - fast for small matrices and precise, dense matrices only
# :bicgstab     - less precise
# :gmres        - less precise

# :iterative    - slow and precise
# :speedmapping - slow and very precise

# solves: A * X * A' + C = X

function solve_lyapunov_equation(A::AbstractMatrix{Float64},
                                C::AbstractMatrix{Float64};
                                lyapunov_algorithm::Symbol = :doubling,
                                tol::AbstractFloat = 1e-14,
                                acceptance_tol::AbstractFloat = 1e-12,
                                verbose::Bool = false)::Tuple{AbstractMatrix{Float64}, Bool}
                                # timer::TimerOutput = TimerOutput(),
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

    X, i, reached_tol = solve_lyapunov_equation(A, C, Val(lyapunov_algorithm), tol = tol) # timer = timer)

    if verbose
        println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: $lyapunov_algorithm")
    end
    
    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :doubling
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:doubling), tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: doubling")
        end
    end

    if reached_tol > acceptance_tol && lyapunov_algorithm ≠ :bicgstab
        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bicgstab), tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bicgstab")
        end
    end

    if !(reached_tol < acceptance_tol) && lyapunov_algorithm ≠ :bartels_stewart && length(C) < 5e7 # try sylvester if previous one didn't solve it
        A = collect(A)

        C = collect(C)

        X, i, reached_tol = solve_lyapunov_equation(A, C, Val(:bartels_stewart), tol = tol) # timer = timer)

        if verbose
            println("Lyapunov equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bartels_stewart")
        end
    end
    # end # timeit_debug
    # end # timeit_debug
    
    # if (reached_tol > tol) println("Lyapunov failed: $reached_tol") end

    return X, reached_tol < acceptance_tol
end

function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64};
                lyapunov_algorithm::Symbol = :doubling,
                tol::AbstractFloat = 1e-14,
                acceptance_tol::AbstractFloat = 1e-12,
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = false)

    P, solved = solve_lyapunov_equation(A, C, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)

    # pullback 
    # https://arxiv.org/abs/2011.11430  
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, slvd = solve_lyapunov_equation(A', ∂P[1], lyapunov_algorithm = lyapunov_algorithm,  tol = tol, verbose = verbose)
    
        solved = solved && slvd

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), ∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
end



function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-14,
                                    acceptance_tol::AbstractFloat = 1e-12,
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * Â' + Â * P̂ * Ã' + C̃

        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_lyapunov_equation(Â, X, lyapunov_algorithm = lyapunov_algorithm, tol = tol, verbose = verbose)
        
        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_lyapunov_equation(A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                ::Val{:bartels_stewart};
                                # timer::TimerOutput = TimerOutput(),
                                tol::AbstractFloat = 1e-14)
    𝐂 = try 
        MatrixEquations.lyapd(A, C)
    catch
        return C, false, 0, 1.0
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



function solve_lyapunov_equation(   A::AbstractSparseMatrix{Float64},
                                    C::AbstractSparseMatrix{Float64},
                                    ::Val{:doubling};
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14)
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


function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::AbstractSparseMatrix{Float64},
                                    ::Val{:doubling};
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14)
    𝐂  = copy(C)
    𝐀  = copy(A)

    𝐀² = similar(𝐀)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐀' + 𝐂

        mul!(𝐀², 𝐀, 𝐀)
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


function solve_lyapunov_equation(   A::AbstractSparseMatrix{Float64},
                                    C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    ::Val{:doubling};
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14)
    𝐂  = copy(C)
    𝐀  = copy(A)
    𝐂A = collect(𝐀)    
    𝐂¹ = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # 𝐂¹ .= 𝐀 * 𝐂 * 𝐀' + 𝐂
        mul!(𝐂A, 𝐂, 𝐀')
        mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

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




function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    ::Val{:doubling};
                                    # timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-14)
    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐀  = copy(A)

    𝐂A = similar(𝐀)
    𝐀² = similar(𝐀)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        mul!(𝐂A, 𝐂, 𝐀')
        mul!(𝐂¹, 𝐀, 𝐂A, 1, 1)

        mul!(𝐀², 𝐀, 𝐀)
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

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_lyapunov_equation(A::AbstractMatrix{Float64},
                                C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                ::Val{:bicgstab};
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14)
    tmp̄ = similar(C)
    𝐗 = similar(C)

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, -1, 1)
        copyto!(sol, 𝐗)
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    𝐂, info = Krylov.bicgstab(lyapunov, [vec(C);], rtol = tol, atol = tol)

    copyto!(𝐗, 𝐂)

    # ℒ.mul!(tmp̄, A, 𝐗 * A')
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

    # if reached_tol > tol
    #     println("Lyapunov: bicgstab $reached_tol")
    # end

    return 𝐗, info.niter, reached_tol
end


function solve_lyapunov_equation(A::AbstractMatrix{Float64},
                                C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                ::Val{:gmres};
                                # timer::TimerOutput = TimerOutput(),
                                tol::Float64 = 1e-14)
    tmp̄ = similar(C)
    𝐗 = similar(C)

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, -1, 1)
        copyto!(sol, 𝐗)
        # sol = @view reshape(𝐗, size(sol))
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    𝐂, info = Krylov.gmres(lyapunov, [vec(C);], rtol = tol, atol = tol)

    copyto!(𝐗, 𝐂)

    # ℒ.mul!(tmp̄, A, 𝐗 * A')
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    reached_tol = ℒ.norm(A * 𝐗 * A' + C - 𝐗) / ℒ.norm(𝐗)

    # if reached_tol > tol
    #     println("Lyapunov: gmres $reached_tol")
    # end

    return 𝐗, info.niter, reached_tol
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
