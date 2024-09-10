# Available algorithms: 
# :doubling     - fast and precise
# :sylvester    - fast and precise, dense matrices only
# :bicgstab     - less precise
# :gmres        - less precise
# :iterative    - slow and precise
# :speedmapping - slow and very precise

# solves: A * X * B + C = X for X

function solve_sylvester_equation(A::AbstractMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::AbstractMatrix{Float64};
                                    sylvester_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-12,
                                    density_threshold::Float64 = .15,
                                    verbose::Bool = false)
    A = choose_matrix_format(A)

    B = choose_matrix_format(B)

    C = choose_matrix_format(C)
    
    X, solved, i, reached_tol = solve_sylvester_equation(A, B, C, Val(sylvester_algorithm), tol = tol)

    if verbose
        println("Sylvester equation - converged to tol $tol: $solved; iterations: $i; reached tol: $reached_tol; algorithm: $sylvester_algorithm")
    end

    return X, solved
end


function rrule(::typeof(solve_sylvester_equation),
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64};
    sylvester_algorithm::Symbol = :doubling,
    tol::AbstractFloat = 1e-12,
    verbose::Bool = false)

    P, solved = solve_sylvester_equation(A, B, C, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

    # pullback
    function solve_sylvester_equation_pullback(∂P)
        ∂C, solved = solve_sylvester_equation(A', B', ∂P[1], sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

        ∂A = ∂C * B' * P'

        ∂B = P' * A' * ∂C

        return NoTangent(), ∂A, ∂B, ∂C, NoTangent()
    end

    return (P, solved), solve_sylvester_equation_pullback
end



function solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    sylvester_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-12,
                                    verbose::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

    Ã = copy(Â)
    B̃ = copy(B̂)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * B̂ + Â * P̂ * B̃ + C̃

        P, solved = solve_sylvester_equation(Â, B̂, X, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_sylvester_equation(  A::AbstractSparseMatrix{Float64},
                                    B::AbstractSparseMatrix{Float64},
                                    C::AbstractSparseMatrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)
    𝐁  = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        𝐀 = 𝐀^2
        𝐁 = 𝐁^2

        droptol!(𝐀, eps())
        droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        𝐂 = 𝐂¹
    end

    𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::AbstractSparseMatrix{Float64},
                                    B::AbstractSparseMatrix{Float64},
                                    C::Matrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # 𝐀 = 𝐀^2
        # 𝐁 = 𝐁^2

        droptol!(𝐀, eps())
        droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end




function solve_sylvester_equation(  A::Matrix{Float64},
                                    B::AbstractSparseMatrix{Float64},
                                    C::Matrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        # 𝐀 = 𝐀^2
        𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::AbstractSparseMatrix{Float64},
                                    B::Matrix{Float64},
                                    C::Matrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    # 𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        𝐀 = 𝐀^2
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # 𝐁 = 𝐁^2

        droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::Matrix{Float64},
                                    B::Matrix{Float64},
                                    C::AbstractSparseMatrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # ℒ.mul!(𝐂B, 𝐂, 𝐁)
        # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        # ℒ.axpy!(1, 𝐂, 𝐂¹)
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        # 𝐀 = 𝐀^2
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # 𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        # copy!(𝐂,𝐂¹)
        𝐂 = 𝐂¹
    end

    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::Matrix{Float64},
                                    B::AbstractSparseMatrix{Float64},
                                    C::AbstractSparseMatrix{Float64},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # ℒ.mul!(𝐂B, 𝐂, 𝐁)
        # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        # ℒ.axpy!(1, 𝐂, 𝐂¹)
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        # 𝐀 = 𝐀^2
        # ℒ.mul!(𝐁¹,𝐁,𝐁)
        # copy!(𝐁,𝐁¹)
        𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        # copy!(𝐂,𝐂¹)
        𝐂 = 𝐂¹
    end

    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    B::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(C)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # 𝐀 = 𝐀^2
        # 𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
                                    B::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::DenseMatrix{Float64},
                                    ::Val{:sylvester};
                                    tol::AbstractFloat = 1e-12)
    𝐂 = MatrixEquations.sylvd(-A, B, C)
    
    𝐂¹ = A * 𝐂 * B + C

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom
    
    return 𝐂, reached_tol < tol, 0, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:bicgstab};
    tol::Float64 = 1e-8)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function sylvester!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, B)
        ℒ.mul!(𝐗, A, tmp̄, 1, 1)
        copyto!(sol, 𝐗)
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

    𝐂, info = Krylov.bicgstab(sylvester, [vec(C);], rtol = tol)

    copyto!(𝐗, 𝐂)

    ℒ.mul!(tmp̄, A, 𝐗 * B)
    ℒ.axpy!(1, C, tmp̄)

    denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, 𝐗, tmp̄)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    return 𝐗, reached_tol < tol, info.niter, reached_tol
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:gmres};
    tol::Float64 = 1e-8)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function sylvester!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        ℒ.mul!(tmp̄, 𝐗, B)
        ℒ.mul!(𝐗, A, tmp̄, 1, 1)
        copyto!(sol, 𝐗)
        # sol = @view reshape(𝐗, size(sol))
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

    𝐂, info = Krylov.gmres(sylvester, [vec(C);],rtol = tol)

    copyto!(𝐗, 𝐂)

    ℒ.mul!(tmp̄, A, 𝐗 * B)
    ℒ.axpy!(1, C, tmp̄)

    denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, 𝐗, tmp̄)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    return 𝐗, reached_tol < tol, info.niter, reached_tol
end


function solve_sylvester_equation(A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:iterative};
    tol::AbstractFloat = 1e-12)

    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐂B = copy(C)
    
    max_iter = 10000
    
    iters = max_iter

    for i in 1:max_iter
        ℒ.mul!(𝐂B, 𝐂, B)
        ℒ.mul!(𝐂¹, A, 𝐂B)
        ℒ.axpy!(1, C, 𝐂¹)
    
        if i % 10 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break
            end
        end
    
        copyto!(𝐂, 𝐂¹)
    end

    ℒ.mul!(𝐂B, 𝐂, B)
    ℒ.mul!(𝐂¹, A, 𝐂B)
    ℒ.axpy!(1, C, 𝐂¹)

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom
    
    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
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
    
    return soll.minimizer, soll.converged, soll.maps, soll.norm_∇
end
