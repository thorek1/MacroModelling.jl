# Available algorithms: 
# :doubling     - fast and precise, expensive part: B^2
# :sylvester    - fast and precise, dense matrices only
# :bicgstab     - less precise, fastest for large problems
# :gmres        - less precise, fastest for large problems
# :iterative    - slow and precise
# :speedmapping - slow and very precise

# solves: A * X * B + C = X for X

function solve_sylvester_equation(A::M,
                                    B::N,
                                    C::O;
                                    # init::AbstractMatrix{Float64} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-14,
                                    timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}
    @timeit_debug timer "Choose matrix formats" begin

    if sylvester_algorithm == :sylvester
        B = collect(B)
    else
        B = choose_matrix_format(B)# |> collect
    end

    if sylvester_algorithm ∈ [:bicgstab, :gmres, :sylvester]
        A = collect(A)

        C = collect(C)
    else
        A = choose_matrix_format(A)# |> sparse

        C = choose_matrix_format(C)# |> sparse
    end
    
    end # timeit_debug
                                    
    @timeit_debug timer "Solve sylvester equation" begin

    X, solved, i, reached_tol = solve_sylvester_equation(A, B, C, Val(sylvester_algorithm), 
                                                        # init = init, 
                                                        tol = tol, 
                                                        timer = timer)

    end # timeit_debug

    if verbose
        println("Sylvester equation - converged to tol $tol: $solved; iterations: $i; reached tol: $reached_tol; algorithm: $sylvester_algorithm")
    end

    X = choose_matrix_format(X)# |> sparse

    return X, solved
end


function rrule(::typeof(solve_sylvester_equation),
    A::M,
    B::N,
    C::O;
    # init::AbstractMatrix{Float64},
    sylvester_algorithm::Symbol = :doubling,
    tol::AbstractFloat = 1e-12,
    timer::TimerOutput = TimerOutput(),
    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}

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
                                    # init::AbstractMatrix{Float64},
                                    sylvester_algorithm::Symbol = :doubling,
                                    tol::AbstractFloat = 1e-12,
                                    timer::TimerOutput = TimerOutput(),
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
                                    # init::AbstractMatrix{Float64},
                                    ::Val{:doubling};
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)
    𝐁  = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(𝐂)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    @timeit_debug timer "Doubling solve" begin
    @timeit_debug timer "Setup buffers" begin

    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = similar(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(𝐂)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    end # timeit_debug

    for i in 1:max_iter
        @timeit_debug timer "Update C" begin
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂
        end # timeit_debug

        @timeit_debug timer "Square A" begin
        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        end # timeit_debug

        # 𝐀 = 𝐀^2
        @timeit_debug timer "Square B" begin
        𝐁 = 𝐁^2
        # ℒ.mul!(𝐁¹,𝐁,𝐁)
        # copy!(𝐁,𝐁¹)
        end # timeit_debug


        # droptol!(𝐀, eps())
        @timeit_debug timer "droptol B" begin
        droptol!(𝐁, eps())
        end # timeit_debug

        if i > 10# && i % 2 == 0
            if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        @timeit_debug timer "Copy C" begin
        copy!(𝐂,𝐂¹)
        end # timeit_debug
    end

    @timeit_debug timer "Finalise" begin
    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom
    end # timeit_debug
    end # timeit_debug

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::AbstractSparseMatrix{Float64},
                                    B::Matrix{Float64},
                                    C::Matrix{Float64},
                                    ::Val{:doubling};
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    # 𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-12)
                                    # see doi:10.1016/j.aml.2009.01.012
    @timeit_debug timer "Setup buffers" begin
                                        
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)
    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(𝐂)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    end # timeit_debug

    for i in 1:max_iter
        @timeit_debug timer "Update C" begin
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂
        end # timeit_debug

        @timeit_debug timer "Square A" begin
        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        end # timeit_debug
        @timeit_debug timer "Square B" begin
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        end # timeit_debug
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

        @timeit_debug timer "Copy C" begin
        copy!(𝐂,𝐂¹)
        end # timeit_debug
    end

    @timeit_debug timer "Finalise" begin
        
    ℒ.mul!(𝐂B, 𝐂, 𝐁)
    ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    ℒ.axpy!(-1, 𝐂, 𝐂¹)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    end # timeit_debug

    return 𝐂, reached_tol < tol, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
                                    B::Union{ℒ.Adjoint{Float64,Matrix{Float64}},DenseMatrix{Float64}},
                                    C::DenseMatrix{Float64},
                                    ::Val{:sylvester};
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::AbstractFloat = 1e-12)
    𝐂 = MatrixEquations.sylvd(-A, B, C)
    
    𝐂¹ = A * 𝐂 * B + C

    denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom
    
    return 𝐂, reached_tol < tol, 0, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::DenseMatrix{Float64},
                                    ::Val{:bicgstab};
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-8)
    @timeit_debug timer "Preallocate matrices" begin
    tmp̄ = zero(C)
    𝐗 = zero(C)
    end # timeit_debug  

    # idxs = findall(abs.(C) .> eps()) # this does not work since the problem is incomplete

    function sylvester!(sol,𝐱)
        @timeit_debug timer "Copy1" begin
        # @inbounds 𝐗[idxs] = 𝐱
        copyto!(𝐗, 𝐱)
        end # timeit_debug
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        @timeit_debug timer "Mul1" begin
        # tmp̄ = A * 𝐗 * B 
        ℒ.mul!(tmp̄, A, 𝐗)
        end # timeit_debug
        @timeit_debug timer "Mul2" begin
        ℒ.mul!(𝐗, tmp̄, B, -1, 1)
        # ℒ.axpby!(-1, tmp̄, 1, 𝐗)
        end # timeit_debug
        @timeit_debug timer "Copy2" begin
        # @inbounds @views copyto!(sol, 𝐗[idxs])
        copyto!(sol, 𝐗)
        end # timeit_debug
        # sol = @view reshape(𝐗, size(sol))
    end

    # sylvester = LinearOperators.LinearOperator(Float64, length(idxs), length(idxs), false, false, sylvester!)
    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), false, false, sylvester!)
    
    # unclear if this helps in most cases
    # # ==== Preconditioner Setup ====
    # # Approximate the diagonal of the Sylvester operator J = A ⊗ B - I
    # # For AXB - X, the diagonal can be approximated as diag(A) * diag(B) - 1
    # diag_J_matrix = ℒ.diag(B) * ℒ.diag(A)' .+ 1
    # diag_J_matrix .+= ℒ.norm(C)
    # # println(ℒ.norm(C))
    # diag_J = vec(diag_J_matrix)

    # # # To avoid division by zero, set a minimum threshold
    # # diag_J = max.(abs.(diag_J), 1e-6)

    # # Compute the inverse of the diagonal preconditioner
    # inv_diag_J = 1.0 ./ diag_J

    # # println(length(inv_diag_J))
    # # Define the preconditioner as a LinearOperator
    # function preconditioner!(y, x)
    #     # ℒ.ldiv!(y, 1e4, x)
    #     # ℒ.mul!(y, inv_diag_J, x)
    #     @. y = inv_diag_J * x
    # end

    # precond = LinearOperators.LinearOperator(Float64, length(C), length(C), false, false, preconditioner!)

    @timeit_debug timer "BICGSTAB solve" begin
    # if length(init) == 0
        # 𝐂, info = Krylov.bicgstab(sylvester, C[idxs], rtol = tol / 10, atol = tol / 10)#, M = precond)
        𝐂, info = Krylov.bicgstab(sylvester, [vec(C);], rtol = tol / 10, atol = tol / 10)#, M = precond)
    # else
    #     𝐂, info = Krylov.bicgstab(sylvester, [vec(C);], [vec(init);], rtol = tol / 10)
    # end
    end # timeit_debug
    
    @timeit_debug timer "Postprocess" begin

    # @inbounds 𝐗[idxs] = 𝐂
    copyto!(𝐗, 𝐂)

    ℒ.mul!(tmp̄, A, 𝐗 * B)
    ℒ.axpy!(1, C, tmp̄)

    denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, 𝐗, tmp̄)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom
    
    end # timeit_debug
    
    if reached_tol > tol || !isfinite(reached_tol)
        @timeit_debug timer "GMRES refinement" begin

        𝐂, info = Krylov.gmres(sylvester, [vec(C);], 
                                [vec(𝐂);], # start value helps
                                rtol = tol / 10, atol = tol / 10)#, M = precond)

        # @inbounds 𝐗[idxs] = 𝐂
        copyto!(𝐗, 𝐂)
    
        ℒ.mul!(tmp̄, A, 𝐗 * B)
        ℒ.axpy!(1, C, tmp̄)
    
        denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))
    
        ℒ.axpy!(-1, 𝐗, tmp̄)
    
        reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom
    
        end # timeit_debug
    end

    if !(typeof(C) <: DenseMatrix)
        𝐗 = choose_matrix_format(𝐗, density_threshold = 1.0)
    end

    return 𝐗, reached_tol < tol, info.niter, reached_tol
end


function solve_sylvester_equation(A::DenseMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::DenseMatrix{Float64},
                                    ::Val{:gmres};
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::Float64 = 1e-8)
    @timeit_debug timer "Preallocate matrices" begin
    tmp̄ = similar(C)
    𝐗 = similar(C)
    end # timeit_debug   

    function sylvester!(sol,𝐱)
        @timeit_debug timer "Copy1" begin
        copyto!(𝐗, 𝐱)
        end # timeit_debug
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        @timeit_debug timer "Mul1" begin
        # tmp̄ = A * 𝐗 * B 
        ℒ.mul!(tmp̄, A, 𝐗)
        end # timeit_debug
        @timeit_debug timer "Mul2" begin
        ℒ.mul!(𝐗, tmp̄, B, -1, 1)
        # ℒ.axpby!(-1, tmp̄, 1, 𝐗)
        end # timeit_debug
        @timeit_debug timer "Copy2" begin
        copyto!(sol, 𝐗)
        end # timeit_debug
        # sol = @view reshape(𝐗, size(sol))
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), false, false, sylvester!)
    
    # # ==== Preconditioner Setup ====
    # # Approximate the diagonal of the Sylvester operator J = A ⊗ B + I
    # # For AXB - X, the diagonal can be approximated as diag(A) * diag(B) + 1
    # diag_J_matrix = ℒ.diag(B) * ℒ.diag(A)' .+ 1.0
    # diag_J = vec(diag_J_matrix)  # Vector of length 39,270

    # # # To avoid division by zero, set a minimum threshold
    # # diag_J = max.(abs.(diag_J), 1e-12)

    # # Compute the inverse of the diagonal preconditioner
    # inv_diag_J = 1.0 ./ diag_J

    # # println(length(inv_diag_J))
    # # Define the preconditioner as a LinearOperator
    # function preconditioner!(y, x)
    #     # ℒ.mul!(y, inv_diag_J, x)
    #     @. y = inv_diag_J .* x
    # end

    # precond = LinearOperators.LinearOperator(Float64, length(C), length(C), false, false, preconditioner!)

    @timeit_debug timer "GMRES solve" begin
    # if length(init) == 0
        𝐂, info = Krylov.gmres(sylvester, [vec(C);], rtol = tol / 10, atol = tol / 10)#, M = precond)
    # else
    #     𝐂, info = Krylov.gmres(sylvester, [vec(C);], [vec(init);], rtol = tol / 10)#, restart = true, M = precond)
    # end
    end # timeit_debug

    @timeit_debug timer "Postprocess" begin

    copyto!(𝐗, 𝐂)

    ℒ.mul!(tmp̄, A, 𝐗 * B)
    ℒ.axpy!(1, C, tmp̄)

    denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, 𝐗, tmp̄)

    reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom

    end # timeit_debug

    return 𝐗, reached_tol < tol, info.niter, reached_tol
end


function solve_sylvester_equation(A::AbstractMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::AbstractMatrix{Float64},
                                    ::Val{:iterative};
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::AbstractFloat = 1e-12)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
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
                                    # init::AbstractMatrix{Float64},
                                    timer::TimerOutput = TimerOutput(),
                                    tol::AbstractFloat = 1e-12)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = copy(C)

    if !(C isa DenseMatrix)
        C = collect(C)
    end

    CB = similar(C)

    soll = speedmapping(-𝐂; 
            m! = (X, x) -> begin
                ℒ.mul!(CB, x, B)
                ℒ.mul!(X, A, CB)
                ℒ.axpy!(1, C, X)
            end, stabilize = false, maps_limit = 10000, tol = tol)

    return soll.minimizer, soll.converged, soll.maps, soll.norm_∇
end
