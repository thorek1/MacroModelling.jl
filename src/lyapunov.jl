function solve_lyapunov_equation(A::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
    C::DenseMatrix{Float64},
    ::Val{:lyapunov};
    tol::AbstractFloat = 1e-12)
    𝐂 = MatrixEquations.lyapd(A, -C)
    
    solved = isapprox(𝐂, A * 𝐂 * A' - C, rtol = tol)

    return 𝐂, solved # return info on convergence
end



function solve_lyapunov_equation(   A::S,
                                    C::S,
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-14) where S <: AbstractSparseMatrix{Float64}
    𝐂  = -C

    max_iter = 500
    
    for i in 1:max_iter
        𝐂¹ = A * 𝐂 * A' + 𝐂

        A *= A
        
        droptol!(A, eps())

        if i > 10 && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break 
            end
        end

        𝐂 = 𝐂¹
    end

    solved = isapprox(𝐂, A * 𝐂 * A' - C, rtol = tol)

    return 𝐂, solved # return info on convergence
end




function solve_lyapunov_equation(   A::M,
                                    C::M,
                                    ::Val{:doubling};
                                    tol::Float64 = 1e-14) where M <: DenseMatrix{Float64}
    𝐂  = copy(-C)
    𝐂¹ = copy(-C)

    CA = similar(A)
    A² = similar(A)

    max_iter = 500
    
    for i in 1:max_iter
        mul!(CA, 𝐂, A')
        mul!(𝐂¹, A, CA, 1, 1)

        mul!(A², A, A)
        copyto!(A, A²)
        
        if i > 10 && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break 
            end
        end

        copyto!(𝐂, 𝐂¹)
    end

    solved = isapprox(𝐂, A * 𝐂 * A' - C, rtol = tol)

    return 𝐂, solved # return info on convergence
end




function solve_lyapunov_equation(A::DenseMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:bicgstab};
    tol::Float64 = 1e-12)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, 1, 1)
        copyto!(sol, 𝐗)
    end

    lyapunov = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, lyapunov!)

    𝐂, info = Krylov.bicgstab(lyapunov, [vec(C);], rtol = tol)

    copyto!(𝐗, 𝐂)

    solved = info.solved

    return 𝐗, solved, solved # return info on convergence
end


function solve_lyapunov_equation(A::DenseMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:gmres};
    tol::Float64 = 1e-12)

    tmp̄ = similar(C)
    𝐗 = similar(C)

    function lyapunov!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        ℒ.mul!(tmp̄, 𝐗, A')
        ℒ.mul!(𝐗, A, tmp̄, 1, 1)
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
    C::AbstractMatrix{Float64},
    ::Val{:iterative};
    tol::AbstractFloat = 1e-14)

    𝐂  = copy(C)
    𝐂¹ = copy(C)
    𝐂A = copy(C)
    
    max_iter = 10000
    
    for i in 1:max_iter
        ℒ.mul!(𝐂A, 𝐂, A')
        ℒ.mul!(𝐂¹, A, 𝐂A)
        ℒ.axpy!(-1, C, 𝐂¹)
    
        if i % 10 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                break
            end
        end
    
        copyto!(𝐂, 𝐂¹)
    end

    ℒ.mul!(𝐂A, 𝐂, A')
    ℒ.mul!(𝐂¹, A, 𝐂A)
    ℒ.axpy!(-1, C, 𝐂¹)

    solved = isapprox(𝐂¹, 𝐂, rtol = tol)

    return 𝐂, solved # return info on convergence
end


function solve_lyapunov_equation(A::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:speedmapping};
    tol::AbstractFloat = 1e-12)

    if !(C isa DenseMatrix)
        C = collect(C)
    end

    CA = similar(C)

    soll = speedmapping(-C; 
            m! = (X, x) -> begin
                ℒ.mul!(CA, x, A')
                ℒ.mul!(X, A, CA)
                ℒ.axpy!(1, C, X)
            end, stabilize = false, maps_limit = 10000, tol = tol)
    
    𝐂 = soll.minimizer

    solved = soll.converged

    return -𝐂, solved
end


function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64},
                ::Val{:speedmapping};
                tol::AbstractFloat = 1e-12)

    P, solved = solve_lyapunov_equation(A, C, Val(:speedmapping), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:speedmapping), tol = tol)
    
        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
end



function rrule(::typeof(solve_lyapunov_equation),
                A::AbstractMatrix{Float64},
                C::AbstractMatrix{Float64},
                ::Val{:doubling};
                tol::AbstractFloat = 1e-14)

    P, solved = solve_lyapunov_equation(A, C, Val(:speedmapping), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:doubling), tol = tol)
    
        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end
    
    return (P, solved), solve_lyapunov_equation_pullback
end




function rrule(::typeof(solve_lyapunov_equation),
    A::DenseMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:gmres};
    tol::AbstractFloat = 1e-12)

    P, solved = solve_lyapunov_equation(A, C, Val(:gmres), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:gmres), tol = tol)

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end

    return (P, solved), solve_lyapunov_equation_pullback
end

function rrule(::typeof(solve_lyapunov_equation),
    A::DenseMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:bicgstab};
    tol::AbstractFloat = 1e-12)

    P, solved = solve_lyapunov_equation(A, C, Val(:bicgstab), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:bicgstab), tol = tol)

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end

    return (P, solved), solve_lyapunov_equation_pullback
end



function rrule(::typeof(solve_lyapunov_equation),
    A::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:lyapunov};
    tol::AbstractFloat = 1e-12)

    P, solved = solve_lyapunov_equation(A, C, Val(:lyapunov), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:lyapunov), tol = tol)

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end

    return (P, solved), solve_lyapunov_equation_pullback
end



function rrule(::typeof(solve_lyapunov_equation),
    A::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    ::Val{:iterative};
    tol::AbstractFloat = 1e-14)

    P, solved = solve_lyapunov_equation(A, C, Val(:iterative), tol = tol)

    # pullback
    function solve_lyapunov_equation_pullback(∂P)
        ∂C, solved = solve_lyapunov_equation(A', ∂P[1], Val(:iterative), tol = tol)

        ∂A = ∂C * A * P' + ∂C' * A * P

        return NoTangent(), -∂A, ∂C, NoTangent()
    end

    return (P, solved), solve_lyapunov_equation_pullback
end


function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    ::Val{:speedmapping};
                                    tol::AbstractFloat = 1e-12) where {Z,S,N}

    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:speedmapping), tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, Val(:speedmapping), tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end




function solve_lyapunov_equation(  A::DenseMatrix{ℱ.Dual{Z,S,N}},
    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
    ::Val{:gmres};
    tol::AbstractFloat = 1e-12) where {Z,S,N}

# unpack: AoS -> SoA
Â = ℱ.value.(A)
Ĉ = ℱ.value.(C)

P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:gmres), tol = tol)

Ã = copy(Â)
C̃ = copy(Ĉ)

P̃ = zeros(length(P̂), N)

for i in 1:N
Ã .= ℱ.partials.(A, i)
C̃ .= ℱ.partials.(C, i)

X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

P, solved = solve_lyapunov_equation(Â, X, Val(:gmres), tol = tol)

P̃[:,i] = vec(P)
end

return reshape(map(P̂, eachrow(P̃)) do v, p
ℱ.Dual{Z}(v, p...) # Z is the tag
end, size(P̂)), solved
end


function solve_lyapunov_equation(   A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    ::Val{:doubling};
                                    tol::AbstractFloat = 1e-14) where {Z,S,N}

    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:doubling), tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, Val(:doubling), tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_lyapunov_equation(  A::DenseMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    ::Val{:bicgstab};
                                    tol::AbstractFloat = 1e-12) where {Z,S,N}

    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:bicgstab), tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, Val(:bicgstab), tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_lyapunov_equation(  A::DenseMatrix{ℱ.Dual{Z,S,N}},
                                    C::DenseMatrix{ℱ.Dual{Z,S,N}},
                                    ::Val{:lyapunov};
                                    tol::AbstractFloat = 1e-12) where {Z,S,N}

    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:lyapunov), tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, Val(:speedmapping), tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    ::Val{:iterative};
                                    tol::AbstractFloat = 1e-14) where {Z,S,N}

    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, Val(:iterative), tol = tol)

    Ã = copy(Â)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = - Ã * P̂ * Â' - Â * P̂ * Ã' + C̃

        P, solved = solve_lyapunov_equation(Â, X, Val(:speedmapping), tol = tol)

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end

