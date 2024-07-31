# Available algorithms: 
# :sylvester    - fast and precise
# :bicgstab     - less precise
# :gmres        - less precise
# :iterative    - slow and precise
# :speedmapping - slow and very precise

function solve_sylvester_equation(A::AbstractMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::AbstractMatrix{Float64};
                                    sylvester_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-12)
    if A isa AbstractSparseMatrix
        if length(A.nzval) / length(A) > .1 || sylvester_algorithm == :sylvester
            A = collect(A)
        elseif VERSION >= v"1.9"
            A = ThreadedSparseArrays.ThreadedSparseMatrixCSC(A)
        end
    end

    if B isa AbstractSparseMatrix
        if length(B.nzval) / length(B) > .1 || sylvester_algorithm == :sylvester
            B = collect(B)
        elseif VERSION >= v"1.9"
            B = ThreadedSparseArrays.ThreadedSparseMatrixCSC(B)
        end
    end

    if C isa AbstractSparseMatrix
        if A isa DenseMatrix || length(C.nzval) / length(C) > .1 || sylvester_algorithm == :sylvester
            C = collect(C)
        elseif VERSION >= v"1.9"
            C = ThreadedSparseArrays.ThreadedSparseMatrixCSC(C)
        end
    end
    
    solve_sylvester_equation(A, B, C, Val(sylvester_algorithm), tol = tol)
end


function rrule(::typeof(solve_sylvester_equation),
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64};
    sylvester_algorithm::Symbol = :gmres,
    tol::AbstractFloat = 1e-12)

    P, solved = solve_sylvester_equation(A, B, C, sylvester_algorithm = sylvester_algorithm, tol = tol)

    # pullback
    function solve_sylvester_equation_pullback(∂P)
        ∂C, solved = solve_sylvester_equation(A', B', ∂P[1], sylvester_algorithm = sylvester_algorithm, tol = tol)

        ∂A = ∂C * B' * P'

        ∂B = P' * A' * ∂C

        return NoTangent(), -∂A, -∂B, ∂C, NoTangent()
    end

    return (P, solved), solve_sylvester_equation_pullback
end



function solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    sylvester_algorithm::Symbol = :gmres,
                                    tol::AbstractFloat = 1e-12) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, sylvester_algorithm = sylvester_algorithm, tol = tol)

    Ã = copy(Â)
    B̃ = copy(B̂)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * B̂ - Â * P̂ * B̃ + C̃

        P, solved = solve_sylvester_equation(Â, B̂, X, sylvester_algorithm = sylvester_algorithm, tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_sylvester_equation(  A::AbstractMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::AbstractMatrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)
    𝐁  = copy(B)
    𝐂  = copy(-C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        𝐀 = 𝐀^2
        𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                println(i)
                iters = i
                break 
            end
        end

        𝐂 = 𝐂¹
    end

    𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = ℒ.norm(𝐂¹ - 𝐂) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
                                    B::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::DenseMatrix{Float64},
                                    ::Val{:sylvester};
                                    tol::AbstractFloat = 1e-12)
    𝐂 = MatrixEquations.sylvd(-A, B, -C)
    
    solved = isapprox(𝐂, A * 𝐂 * B - C, rtol = tol)

    return 𝐂, solved # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:bicgstab};
    tol::Float64 = 1e-12)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function sylvester!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, B)
        ℒ.mul!(𝐗, A, tmp̄, 1, -1)
        copyto!(sol, 𝐗)
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

    𝐂, info = Krylov.bicgstab(sylvester, [vec(C);], rtol = tol)

    copyto!(𝐗, 𝐂)

    solved = info.solved

    return 𝐗, solved, solved # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:gmres};
    tol::Float64 = 1e-12)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function sylvester!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        ℒ.mul!(tmp̄, 𝐗, B)
        ℒ.mul!(𝐗, A, tmp̄, 1, -1)
        copyto!(sol, 𝐗)
        # sol = @view reshape(𝐗, size(sol))
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

    𝐂, info = Krylov.gmres(sylvester, [vec(C);],rtol = tol)

    copyto!(𝐗, 𝐂)

    solved = info.solved

    return 𝐗, solved # return info on convergence
end


function solve_sylvester_equation(A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:iterative};
    tol::AbstractFloat = 1e-14)

    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐂B = copy(C)
    
    max_iter = 10000
    
    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, B)
        ℒ.mul!(𝐂¹, A, 𝐂B)
        ℒ.axpy!(-1, C, 𝐂¹)
    
        if i % 10 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break
            end
        end
    
        copyto!(𝐂, 𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, B)
    ℒ.mul!(𝐂¹, A, 𝐂B)
    ℒ.axpy!(-1, C, 𝐂¹)

    solved = isapprox(𝐂¹, 𝐂, rtol = tol)

    return 𝐂, solved # return info on convergence
end


function solve_sylvester_equation(A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:speedmapping};
    tol::AbstractFloat = 1e-12)

    if !(C isa DenseMatrix)
        C = collect(C)
    end

    CB = similar(C)

    soll = speedmapping(-C; 
            m! = (X, x) -> begin
                ℒ.mul!(CB, x, B)
                ℒ.mul!(X, A, CB)
                ℒ.axpy!(1, C, X)
            end, stabilize = false, maps_limit = 10000, tol = tol)
    
    𝐂 = soll.minimizer

    solved = soll.converged

    return -𝐂, solved
end
