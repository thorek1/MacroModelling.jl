function solve_lyapunov_equation(A::AbstractMatrix{Float64},
                                C::AbstractMatrix{Float64};
                                lyapunov_algorithm::Symbol = :doubling,
                                tol::AbstractFloat = 1e-12)
    if A isa AbstractSparseMatrix
        if length(A.nzval) / length(A) > .1 || lyapunov_algorithm == :lyapunov
            A = collect(A)
        elseif VERSION >= v"1.9"
            A = ThreadedSparseArrays.ThreadedSparseMatrixCSC(A)
        end
    end

    if C isa AbstractSparseMatrix
        if length(C.nzval) / length(C) > .1 || lyapunov_algorithm == :lyapunov
            C = collect(C)
        elseif VERSION >= v"1.9"
            C = ThreadedSparseArrays.ThreadedSparseMatrixCSC(C)
        end
    end
    
    solve_lyapunov_equation(A, C, Val(lyapunov_algorithm), tol = tol)
end

function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64};
                lyapunov_algorithm::Symbol = :doubling,
                tol::AbstractFloat = 1e-12)

    P, solved = solve_lyapunov_equation(A, C, lyapunov_algorithm = lyapunov_algorithm, tol = tol)

    # pullback 
    # https://arxiv.org/abs/2011.11430  
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], lyapunov_algorithm = lyapunov_algorithm, tol = tol)
    
        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), ∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
end



function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-12) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, lyapunov_algorithm = lyapunov_algorithm, tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * Â' + Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, lyapunov_algorithm = lyapunov_algorithm, tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_lyapunov_equation(A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                C::DenseMatrix{Float64},
                                ::Val{:lyapunov};
                                tol::AbstractFloat = 1e-12)
    𝐂 = MatrixEquations.lyapd(A, C)
    
    solved = isapprox(𝐂, A * 𝐂 * A' + C, rtol = tol)

    return 𝐂, solved # return info on convergence
end



function solve_lyapunov_equation(   A::AbstractSparseMatrix{Float64},
                                    C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-14)
    𝐂  = copy(C)
    𝐀  = copy(A)
    CA = collect(𝐀)    
    𝐂¹ = copy(C)

    max_iter = 500
    
    for i in 1:max_iter
        # 𝐂¹ .= 𝐀 * 𝐂 * 𝐀' + 𝐂
        mul!(CA, 𝐂, 𝐀')
        mul!(𝐂¹, 𝐀, CA, 1, 1)

        𝐀 *= 𝐀
        
        droptol!(𝐀, eps())

        if i > 10 && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break 
            end
        end

        copy!(𝐂,𝐂¹)
        # 𝐂 = 𝐂¹
    end

    solved = isapprox(𝐂, A * 𝐂 * A' + C, rtol = tol)

    return 𝐂, solved # return info on convergence
end




function solve_lyapunov_equation(   A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::DenseMatrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-14)
    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐀  = copy(A)

    CA = similar(𝐀)
    𝐀² = similar(𝐀)

    max_iter = 500
    
    for i in 1:max_iter
        mul!(CA, 𝐂, 𝐀')
        mul!(𝐂¹, 𝐀, CA, 1, 1)

        mul!(𝐀², 𝐀, 𝐀)
        copyto!(𝐀, 𝐀²)
        
        if i > 10 && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break 
            end
        end

        copyto!(𝐂, 𝐂¹)
    end

    solved = isapprox(𝐂, 𝐀 * 𝐂 * 𝐀' + C, rtol = tol)

    return 𝐂, solved # return info on convergence
end




function solve_lyapunov_equation(A::AbstractMatrix{Float64},
    C::DenseMatrix{Float64},
    ::Val{:bicgstab};
    tol::Float64 = 1e-12)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, -1, 1)
        copyto!(sol, 𝐗)
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    𝐂, info = Krylov.bicgstab(lyapunov, [vec(C);], rtol = tol)

    copyto!(𝐗, 𝐂)

    solved = info.solved

    return 𝐗, solved, solved # return info on convergence
end


function solve_lyapunov_equation(A::AbstractMatrix{Float64},
    C::DenseMatrix{Float64},
    ::Val{:gmres};
    tol::Float64 = 1e-12)

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

    𝐂, info = Krylov.gmres(lyapunov, [vec(C);],rtol = tol)

    copyto!(𝐗, 𝐂)

    solved = info.solved

    return 𝐗, solved # return info on convergence
end


function solve_lyapunov_equation(A::AbstractMatrix{Float64},
    C::DenseMatrix{Float64},
    ::Val{:iterative};
    tol::AbstractFloat = 1e-14)

    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐂A = copy(C)
    
    max_iter = 10000
    
    for i in 1:max_iter
        ℒ.mul!(𝐂A, 𝐂, A')
        ℒ.mul!(𝐂¹, A, 𝐂A)
        ℒ.axpy!(1, C, 𝐂¹)
    
        if i % 10 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break
            end
        end
    
        copyto!(𝐂, 𝐂¹)
    end

    ℒ.mul!(𝐂A, 𝐂, A')
    ℒ.mul!(𝐂¹, A, 𝐂A)
    ℒ.axpy!(1, C, 𝐂¹)

    solved = isapprox(𝐂¹, 𝐂, rtol = tol)

    return 𝐂, solved # return info on convergence
end


function solve_lyapunov_equation(A::AbstractMatrix{Float64},
    C::DenseMatrix{Float64},
    ::Val{:speedmapping};
    tol::AbstractFloat = 1e-12)

    if !(C isa DenseMatrix)
        C = collect(C)
    end

    CA = similar(C)

    soll = speedmapping(C; 
            m! = (X, x) -> begin
                ℒ.mul!(CA, x, A')
                ℒ.mul!(X, A, CA)
                ℒ.axpy!(1, C, X)
            end, stabilize = false, maps_limit = 10000, tol = tol)
    
    𝐂 = soll.minimizer

    solved = soll.converged

    return 𝐂, solved
end
