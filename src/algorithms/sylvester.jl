# Available algorithms: 
# :doubling     - fast, expensive part: B^2
# :bartels_stewart    - fast, dense matrices only
# :bicgstab     - fastest for large problems, might not reach desired precision, warm start not always helpful
# :dqgmres      - fastest for large problems, might not reach desired precision, stable path with warm start
# :gmres      - fastest for large problems, might not reach desired precision, can be effective, not efficient

# :iterative    - slow
# :speedmapping - slow

# solves: A * X * B + C = X for X
@stable default_mode = "disable" begin
function solve_sylvester_equation(A::M,
                                    B::N,
                                    C::O;
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    acceptance_tol::AbstractFloat = 1e-10,
                                    tol::AbstractFloat = 1e-14,
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    verbose::Bool = false)::Union{Tuple{Matrix{Float64}, Bool}, Tuple{SparseMatrixCSC{Float64, Int}, Bool}, Tuple{ThreadedSparseArrays.ThreadedSparseMatrixCSC{Float64, Int, SparseMatrixCSC{Float64, Int}}, Bool}} where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}
                                    # timer::TimerOutput = TimerOutput(),
    # @timeit_debug timer "Choose matrix formats" begin

    if sylvester_algorithm == :bartels_stewart
        b = collect(B)
    else
        b = choose_matrix_format(B)# |> collect
    end

    if sylvester_algorithm ∈ [:bicgstab, :gmres, :dqgmres, :bartels_stewart]
        a = collect(A)

        c = collect(C)
    else
        a = choose_matrix_format(A)# |> sparse

        c = choose_matrix_format(C)# |> sparse
    end
    
    # end # timeit_debug
    # @timeit_debug timer "Check if guess solves it already" begin

    if length(initial_guess) > 0
        𝐂  = a * initial_guess * b + c - initial_guess
        
        reached_tol = ℒ.norm(𝐂) / ℒ.norm(initial_guess)

        if reached_tol < acceptance_tol
            if verbose println("Sylvester equation - previous solution achieves relative tol of $reached_tol") end

            # X = choose_matrix_format(initial_guess)

            return initial_guess, true
        end
    end
    
    # end # timeit_debug
    # @timeit_debug timer "Solve sylvester equation" begin

    x, i, reached_tol = solve_sylvester_equation(a, b, c, Val(sylvester_algorithm), 
                                                        initial_guess = initial_guess, 
                                                        tol = tol,
                                                        𝕊ℂ = 𝕊ℂ,
                                                        # timer = timer, 
                                                        verbose = verbose)

    if verbose && i != 0
        println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: $sylvester_algorithm")
    end

    if (!isfinite(reached_tol) || !(reached_tol < acceptance_tol)) && (sylvester_algorithm ≠ :bartels_stewart) && (length(B) < 5e7) # try sylvester if previous one didn't solve it
        aa = collect(A)

        bb = collect(B)

        cc = collect(C)

        x, i, reached_tol = solve_sylvester_equation(aa, bb, cc, 
                                                            Val(:bartels_stewart), 
                                                            initial_guess = zeros(0,0), 
                                                            tol = tol, 
                                                            𝕊ℂ = 𝕊ℂ,
                                                            # timer = timer, 
                                                            verbose = verbose)

        if verbose && i != 0
            println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: bartels_stewart")
        end
    end

    if (!isfinite(reached_tol) || !(reached_tol < acceptance_tol)) && reached_tol < sqrt(acceptance_tol)
        aa = collect(A)

        cc = collect(C)

        X, i, Reached_tol = solve_sylvester_equation(aa, b, cc, 
                                                            Val(:dqgmres), 
                                                            initial_guess = x, 
                                                            tol = tol, 
                                                            𝕊ℂ = 𝕊ℂ,
                                                            # timer = timer, 
                                                            verbose = verbose)
        if Reached_tol < reached_tol
            x = X
            reached_tol = Reached_tol
        end

        if verbose# && i != 0
            println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $Reached_tol; algorithm: dqgmres (refinement of previous solution)")
        end
    end

    if (!isfinite(reached_tol) || !(reached_tol < acceptance_tol)) && sylvester_algorithm ≠ :gmres
        aa = collect(A)

        cc = collect(C)

        x, i, reached_tol = solve_sylvester_equation(aa, b, cc, 
                                                            Val(:gmres), 
                                                            initial_guess = zeros(0,0), 
                                                            tol = tol, 
                                                            𝕊ℂ = 𝕊ℂ,
                                                            # timer = timer, 
                                                            verbose = verbose)

        if verbose# && i != 0
            println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: gmres")
        end
    end

    if (!isfinite(reached_tol) || !(reached_tol < acceptance_tol)) && reached_tol < sqrt(acceptance_tol)
        aa = collect(A)

        cc = collect(C)

        X, i, Reached_tol = solve_sylvester_equation(aa, b, cc, 
                                                            Val(:dqgmres), 
                                                            initial_guess = x, 
                                                            tol = tol, 
                                                            𝕊ℂ = 𝕊ℂ,
                                                            # timer = timer, 
                                                            verbose = verbose)
        if Reached_tol < reached_tol
            x = X
            reached_tol = Reached_tol
        end

        if verbose# && i != 0
            println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $Reached_tol; algorithm: dqgmres (refinement of previous solution)")
        end
    end

    if (!isfinite(reached_tol) || !(reached_tol < acceptance_tol)) && sylvester_algorithm ≠ :doubling
        aa = collect(A)

        cc = collect(C)

        x, i, reached_tol = solve_sylvester_equation(aa, b, cc, 
                                                            Val(:doubling), 
                                                            initial_guess = zeros(0,0), 
                                                            tol = tol, 
                                                            𝕊ℂ = 𝕊ℂ,
                                                            # timer = timer, 
                                                            verbose = verbose)

        if verbose# && i != 0
            println("Sylvester equation - converged to tol $acceptance_tol: $(reached_tol < acceptance_tol); iterations: $i; reached tol: $reached_tol; algorithm: doubling")
        end
    end

    # if !solved && sylvester_algorithm ≠ :bicgstab
    #     aa = collect(A)

    #     cc = collect(C)

    #     x, solved, i, reached_tol = solve_sylvester_equation(aa, b, cc, 
    #                                                         Val(:bicgstab), 
    #                                                         initial_guess = zeros(0,0), 
    #                                                         tol = tol, 
    #                                                         verbose = verbose,
    #                                                         timer = timer)

    #     if verbose# && i != 0
    #         println("Sylvester equation - converged to tol $tol: $solved; iterations: $i; reached tol: $reached_tol; algorithm: bicgstab")
    #     end
    # end

    # if !solved && reached_tol < sqrt(tol)
    #     aa = collect(A)

    #     cc = collect(C)

    #     X, solved, i, Reached_tol = solve_sylvester_equation(aa, b, cc, 
    #                                                         Val(:dqgmres), 
    #                                                         initial_guess = x, 
    #                                                         tol = tol, 
    #                                                         verbose = verbose,
    #                                                         timer = timer)
    #     if Reached_tol < reached_tol
    #         x = X
    #         reached_tol = Reached_tol
    #     end

    #     if verbose# && i != 0
    #         println("Sylvester equation - converged to tol $tol: $solved; iterations: $i; reached tol: $Reached_tol; algorithm: dqgmres (refinement of previous solution)")
    #     end
    # end

    # if !solved && sylvester_algorithm ≠ :doubling
    #     aa = collect(A)

    #     cc = collect(C)

    #     x, solved, i, reached_tol = solve_sylvester_equation(aa, b, cc, 
    #                                                         Val(:doubling), 
    #                                                         initial_guess = zeros(0,0), 
    #                                                         tol = tol, 
    #                                                         verbose = verbose,
    #                                                         timer = timer)

    #     if verbose# && i != 0
    #         println("Sylvester equation - converged to tol $tol: $solved; iterations: $i; reached tol: $reached_tol; algorithm: doubling")
    #     end
    # end

    # end # timeit_debug

    X = choose_matrix_format(x)# |> sparse

    # if (reached_tol > tol) println("Sylvester failed: $reached_tol") end

    return X, reached_tol < acceptance_tol
end

end # dispatch_doctor


function rrule(::typeof(solve_sylvester_equation),
    A::M,
    B::N,
    C::O;
    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
    sylvester_algorithm::Symbol = :doubling,
    acceptance_tol::AbstractFloat = 1e-10,
    tol::AbstractFloat = 1e-14,
    𝕊ℂ::sylvester_caches = Sylvester_caches(),
    # timer::TimerOutput = TimerOutput(),
    verbose::Bool = false) where {M <: AbstractMatrix{Float64}, N <: AbstractMatrix{Float64}, O <: AbstractMatrix{Float64}}

    P, solved = solve_sylvester_equation(A, B, C, 
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose, 
                                        initial_guess = initial_guess)

                                        println("C norm: $(ℒ.norm(C))")
    # pullback
    function solve_sylvester_equation_pullback(∂P)
        if ℒ.norm(∂P[1]) < tol return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() end

        ∂C, slvd = solve_sylvester_equation(A', B', ∂P[1], 
                                            sylvester_algorithm = sylvester_algorithm, 
                                            tol = tol, 
                                            𝕊ℂ = 𝕊ℂ,
                                            verbose = verbose)

        solved = solved && slvd

        ∂A = ∂C * B' * P'

        ∂B = P' * A' * ∂C

        return NoTangent(), ∂A, ∂B, ∂C, NoTangent()
    end

    return (P, solved), solve_sylvester_equation_pullback
end

@stable default_mode = "disable" begin

function solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    acceptance_tol::AbstractFloat = 1e-10,
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    tol::AbstractFloat = 1e-14,
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, 
                                        sylvester_algorithm = sylvester_algorithm, 
                                        tol = tol, 
                                        𝕊ℂ = 𝕊ℂ,
                                        verbose = verbose, 
                                        initial_guess = initial_guess)

    Ã = copy(Â)
    B̃ = copy(B̂)
    C̃ = copy(Ĉ)
    
    P̃ = zeros(S, length(P̂), N)
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂ * B̂ + Â * P̂ * B̃ + C̃
        
        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_sylvester_equation(Â, B̂, X, 
                                            sylvester_algorithm = sylvester_algorithm, 
                                            𝕊ℂ = 𝕊ℂ,
                                            tol = tol, 
                                            verbose = verbose)

        solved = solved && slvd

        P̃[:,i] = vec(P)
    end
    
    return reshape(map(P̂, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂)), solved
end



function solve_sylvester_equation(  A::AbstractSparseMatrix{T},
                                    B::AbstractSparseMatrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{AbstractSparseMatrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # guess_provided = true
    
    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end
    
    𝐀  = copy(A)
    𝐁  = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        𝐀 = 𝐀^2
        𝐁 = 𝐁^2

        droptol!(𝐀, eps())
        droptol!(𝐁, eps())

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

    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::AbstractSparseMatrix{T},
                                    B::AbstractSparseMatrix{T},
                                    C::Matrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012    
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    # # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)
    
    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    ℒ.axpy!(1, initial_guess, 𝐂)

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end




function solve_sylvester_equation(  A::Matrix{T},
                                    B::AbstractSparseMatrix{T},
                                    C::Matrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # @timeit_debug timer "Doubling solve" begin
    # @timeit_debug timer "Setup buffers" begin

    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = similar(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(𝐂)
    𝐂B = similar(C)

    max_iter = 500

    iters = max_iter

    # end # timeit_debug

    for i in 1:max_iter
        # @timeit_debug timer "Update C" begin
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂
        # end # timeit_debug

        # @timeit_debug timer "Square A" begin
        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        # end # timeit_debug

        # 𝐀 = 𝐀^2
        # @timeit_debug timer "Square B" begin
        𝐁 = 𝐁^2
        # ℒ.mul!(𝐁¹,𝐁,𝐁)
        # copy!(𝐁,𝐁¹)
        # end # timeit_debug


        # droptol!(𝐀, eps())
        # @timeit_debug timer "droptol B" begin
        droptol!(𝐁, eps())
        # end # timeit_debug

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        # @timeit_debug timer "Copy C" begin
        copy!(𝐂,𝐂¹)
        # end # timeit_debug
    end

    # @timeit_debug timer "Finalise" begin
    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    # # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # end # timeit_debug
    # end # timeit_debug

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::AbstractSparseMatrix{T},
                                    B::Matrix{T},
                                    C::Matrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{T} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012  On Smith-type iterative algorithms for the Stein matrix equation
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    # 𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        copy!(𝐂,𝐂¹)
    end

    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::Matrix{T},
                                    B::Matrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
    # 𝐂B = copy(C)

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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
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
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end



function solve_sylvester_equation(  A::AbstractSparseMatrix{T},
                                    B::Matrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    # 𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
    # 𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    for i in 1:max_iter
        # ℒ.mul!(𝐂B, 𝐂, 𝐁)
        # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        # ℒ.axpy!(1, 𝐂, 𝐂¹)
        𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

        # ℒ.mul!(𝐀¹,𝐀,𝐀)
        # copy!(𝐀,𝐀¹)
        𝐀 = 𝐀^2
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # 𝐁 = 𝐁^2

        droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
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
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::Matrix{T},
                                    B::AbstractSparseMatrix{T},
                                    C::AbstractSparseMatrix{T},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    # 𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)
    𝐂¹ = similar(𝐂)
    # 𝐂B = copy(C)

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

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
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
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    𝐂 += initial_guess

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(  A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    B::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:doubling};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
                                    # see doi:10.1016/j.aml.2009.01.012
    # @timeit_debug timer "Setup buffers" begin
    
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end
                                  
    𝐀  = copy(A)    
    𝐀¹ = copy(A)
    𝐁  = copy(B)
    𝐁¹ = copy(B)
    # 𝐂  = length(init) == 0 ? copy(C) : copy(init)
    𝐂  = A * initial_guess * B + C - initial_guess #copy(C)

    # ℒ.rmul!(𝐂, -1)
    𝐂¹  = similar(𝐂)
    𝐂B = copy(C)

    max_iter = 500

    iters = max_iter

    # end # timeit_debug

    for i in 1:max_iter
        # @timeit_debug timer "Update C" begin
        ℒ.mul!(𝐂B, 𝐂, 𝐁)
        ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
        ℒ.axpy!(1, 𝐂, 𝐂¹)
        # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂
        # end # timeit_debug

        # @timeit_debug timer "Square A" begin
        ℒ.mul!(𝐀¹,𝐀,𝐀)
        copy!(𝐀,𝐀¹)
        # end # timeit_debug
        # @timeit_debug timer "Square B" begin
        ℒ.mul!(𝐁¹,𝐁,𝐁)
        copy!(𝐁,𝐁¹)
        # end # timeit_debug
        # 𝐀 = 𝐀^2
        # 𝐁 = 𝐁^2

        # droptol!(𝐀, eps())
        # droptol!(𝐁, eps())

        if i % 2 == 0
            normdiff = ℒ.norm(𝐂¹ - 𝐂)
            if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
            # if isapprox(𝐂¹, 𝐂, rtol = tol)
                iters = i
                break 
            end
        end

        # @timeit_debug timer "Copy C" begin
        copy!(𝐂,𝐂¹)
        # end # timeit_debug
    end

    # @timeit_debug timer "Finalise" begin
        
    # ℒ.mul!(𝐂B, 𝐂, 𝐁)
    # ℒ.mul!(𝐂¹, 𝐀, 𝐂B)
    # ℒ.axpy!(1, 𝐂, 𝐂¹)
    # 𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # ℒ.axpy!(-1, 𝐂, 𝐂¹)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

    ℒ.axpy!(1, initial_guess, 𝐂)

    reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # end # timeit_debug

    # if reached_tol > tol
    #     println("Sylvester: doubling $reached_tol")
    # end

    return 𝐂, iters, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{T},
                                    B::Union{ℒ.Adjoint{Float64, Matrix{T}}, DenseMatrix{T}},
                                    C::DenseMatrix{T},
                                    ::Val{:bartels_stewart};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::AbstractFloat = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end
      
    # 𝐂¹  = A * initial_guess * B + C - initial_guess #copy(C)
    𝐂¹ = zero(C)
    tmp̄ = zero(C)
    
    ℒ.mul!(tmp̄, initial_guess, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)

    ℒ.axpy!(-1, initial_guess, 𝐂¹)

    𝐂 = try 
        MatrixEquations.sylvd(-A, B, 𝐂¹)
    catch
        return C, 0, 1.0
    end

    # 𝐂¹ = A * 𝐂 * B + C

    # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹ - 𝐂) / denom

    𝐂 += initial_guess

    ℒ.mul!(tmp̄, 𝐂, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐂, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / max(ℒ.norm(𝐂), ℒ.norm(C))
    # reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

    # if reached_tol > tol
    #     println("Sylvester: sylvester $reached_tol")
    # end

    return 𝐂, -1, reached_tol # return info on convergence
end


function solve_sylvester_equation(A::DenseMatrix{T},
                                    B::AbstractMatrix{T},
                                    C::DenseMatrix{T},
                                    ::Val{:bicgstab};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # @timeit_debug timer "Preallocate matrices" begin

    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    # 𝐂¹  = A * initial_guess * B + C - initial_guess #copy(C)
    # println(ℒ.norm(𝐂¹))

    if length(𝕊ℂ.tmp) == 0
        𝕊ℂ.tmp = zero(C)
    end

    if length(𝕊ℂ.𝐗) == 0
        𝕊ℂ.𝐗 = zero(C)
    end
    
    if length(𝕊ℂ.𝐂) == 0
        𝕊ℂ.𝐂 = zero(C)
    end
    
    𝐂¹ = 𝕊ℂ.𝐂
    tmp̄ = 𝕊ℂ.tmp
    𝐗 = 𝕊ℂ.𝐗 

    # end # timeit_debug  

    ℒ.mul!(𝐗, initial_guess, B)
    ℒ.mul!(tmp̄, A, 𝐗)
    ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, initial_guess, tmp̄)

    copyto!(𝐂¹, tmp̄)

    # println(ℒ.norm(𝐂¹))

    # idxs = findall(abs.(C) .> eps()) # this does not work since the problem is incomplete

    function sylvester!(sol,𝐱)
        # @timeit_debug timer "Copy1" begin
        # @inbounds 𝐗[idxs] = 𝐱
        copyto!(𝐗, 𝐱)
        # end # timeit_debug
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        # @timeit_debug timer "Mul1" begin
        # tmp̄ = A * 𝐗 * B 
        ℒ.mul!(tmp̄, A, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Mul2" begin
        ℒ.mul!(𝐗, tmp̄, B, -1, 1)
        # ℒ.axpby!(-1, tmp̄, 1, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Copy2" begin
        # @inbounds @views copyto!(sol, 𝐗[idxs])
        copyto!(sol, 𝐗)
        # end # timeit_debug
        # sol = @view reshape(𝐗, size(sol))
    end

    # sylvester = LinearOperators.LinearOperator(Float64, length(idxs), length(idxs), true, true, sylvester!)
    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)
    
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

    # precond = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, preconditioner!)

    if 𝕊ℂ.krylov_caches.bicgstab.m == 0
        𝕊ℂ.krylov_caches.bicgstab =  BicgstabWorkspace(length(C), length(C), Vector{T})
    end
    # @timeit_debug timer "BICGSTAB solve" begin
    # if length(init) == 0
        # 𝐂, info = Krylov.bicgstab(sylvester, C[idxs], rtol = tol / 10, atol = tol / 10)#, M = precond)
        # 𝐂, info = Krylov.bicgstab(sylvester, [vec(𝕊ℂ.𝐂);], 
        Krylov.bicgstab!(   𝕊ℂ.krylov_caches.bicgstab,
                            sylvester, [vec(𝐂¹);], 
                            # [vec(initial_guess);], 
                            itmax = min(5000,max(500,Int(round(sqrt(length(𝐂¹)*10))))),
                            timemax = 10.0,
                            rtol = tol, 
                            atol = tol)#, M = precond)
    # else
    #     𝐂, info = Krylov.bicgstab(sylvester, [vec(C);], [vec(init);], rtol = tol / 10)
    # end
    # end # timeit_debug

    # @timeit_debug timer "Postprocess" begin

    # # @inbounds 𝕊ℂ.𝐗[idxs] = 𝐂
    copyto!(𝐗, 𝕊ℂ.krylov_caches.bicgstab.x)

    # ℒ.mul!(tmp̄, A, 𝐗 * B)
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom
    𝐗 += initial_guess # problematic with copyto! above. cant be axpy!

    ℒ.mul!(tmp̄, 𝐗, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐗, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / max(ℒ.norm(𝐗), ℒ.norm(C))
    # reached_tol = ℒ.norm(A * 𝐗 * B + C - 𝐗) / max(ℒ.norm(𝐗), ℒ.norm(C))

    # end # timeit_debug
    
    if !(typeof(C) <: DenseMatrix)
        𝐗 = choose_matrix_format(𝐗, density_threshold = 1.0)
    end

    # if reached_tol > tol
    #     println("Sylvester: bicgstab $reached_tol")
    # end
    
    # iter = info.niter
    iter = 𝕊ℂ.krylov_caches.bicgstab.stats.niter

    # return 𝕊ℂ.𝐗, iter, reached_tol
    return 𝐗, iter, reached_tol
end


function solve_sylvester_equation(A::DenseMatrix{T},
                                    B::AbstractMatrix{T},
                                    C::DenseMatrix{T},
                                    ::Val{:dqgmres};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # @timeit_debug timer "Preallocate matrices" begin

    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    # 𝐂¹  = A * initial_guess * B + C - initial_guess #copy(C)
    # println(ℒ.norm(𝐂¹))

    if length(𝕊ℂ.tmp) == 0
        𝕊ℂ.tmp = zero(C)
    end

    if length(𝕊ℂ.𝐗) == 0
        𝕊ℂ.𝐗 = zero(C)
    end
    
    if length(𝕊ℂ.𝐂) == 0
        𝕊ℂ.𝐂 = zero(C)
    end
    
    𝐂¹ = 𝕊ℂ.𝐂
    tmp̄ = 𝕊ℂ.tmp
    𝐗 = 𝕊ℂ.𝐗 

    # end # timeit_debug  

    ℒ.mul!(𝐗, initial_guess, B)
    ℒ.mul!(tmp̄, A, 𝐗)
    ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, initial_guess, tmp̄)

    copyto!(𝐂¹, tmp̄)

    # println(ℒ.norm(𝐂¹))

    # idxs = findall(abs.(C) .> eps()) # this does not work since the problem is incomplete

    function sylvester!(sol,𝐱)
        # @timeit_debug timer "Copy1" begin
        # @inbounds 𝐗[idxs] = 𝐱
        copyto!(𝐗, 𝐱)
        # end # timeit_debug
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        # @timeit_debug timer "Mul1" begin
        # tmp̄ = A * 𝐗 * B 
        ℒ.mul!(tmp̄, A, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Mul2" begin
        ℒ.mul!(𝐗, tmp̄, B, -1, 1)
        # ℒ.axpby!(-1, tmp̄, 1, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Copy2" begin
        # @inbounds @views copyto!(sol, 𝐗[idxs])
        copyto!(sol, 𝐗)
        # end # timeit_debug
        # sol = @view reshape(𝐗, size(sol))
    end

    # sylvester = LinearOperators.LinearOperator(Float64, length(idxs), length(idxs), true, true, sylvester!)
    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)
    
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

    # precond = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, preconditioner!)

    if 𝕊ℂ.krylov_caches.dqgmres.m == 0
        𝕊ℂ.krylov_caches.dqgmres =  DqgmresWorkspace(length(C), length(C), Vector{T})
    end
    # @timeit_debug timer "DQGMRES solve" begin
    # if length(init) == 0
        # 𝐂, info = Krylov.dqgmres(sylvester, C[idxs], rtol = tol / 10, atol = tol / 10)#, M = precond)
        # 𝐂, info = Krylov.dqgmres(sylvester, [vec(𝕊ℂ.𝐂);], 
        Krylov.dqgmres!(𝕊ℂ.krylov_caches.dqgmres,
                        sylvester, [vec(𝐂¹);], 
                        # [vec(initial_guess);], 
                        itmax = min(5000,max(500,Int(round(sqrt(length(𝐂¹)*10))))),
                        timemax = 10.0,
                        rtol = tol, 
                        atol = tol)#, M = precond)
    # else
    #     𝐂, info = Krylov.dqgmres(sylvester, [vec(C);], [vec(init);], rtol = tol / 10)
    # end
    # end # timeit_debug

    # @timeit_debug timer "Postprocess" begin

    # # @inbounds 𝕊ℂ.𝐗[idxs] = 𝐂
    copyto!(𝐗, 𝕊ℂ.krylov_caches.dqgmres.x)

    # ℒ.mul!(tmp̄, A, 𝐗 * B)
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom
    𝐗 += initial_guess # problematic with copyto! above. cant be axpy!

    ℒ.mul!(tmp̄, 𝐗, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐗, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / max(ℒ.norm(𝐗), ℒ.norm(C))
    # reached_tol = ℒ.norm(A * 𝐗 * B + C - 𝐗) / max(ℒ.norm(𝐗), ℒ.norm(C))

    # end # timeit_debug
    
    if !(typeof(C) <: DenseMatrix)
        𝐗 = choose_matrix_format(𝐗, density_threshold = 1.0)
    end

    # if reached_tol > tol
    #     println("Sylvester: dqgmres $reached_tol")
    # end
    
    # iter = info.niter
    iter = 𝕊ℂ.krylov_caches.dqgmres.stats.niter

    # return 𝕊ℂ.𝐗, iter, reached_tol
    return 𝐗, iter, reached_tol
end


function solve_sylvester_equation(A::DenseMatrix{T},
                                    B::AbstractMatrix{T},
                                    C::DenseMatrix{T},
                                    ::Val{:gmres};
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    𝕊ℂ::sylvester_caches = Sylvester_caches(),
                                    # timer::TimerOutput = TimerOutput(),
                                    verbose::Bool = false,
                                    tol::Float64 = 1e-14)::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat
    # @timeit_debug timer "Preallocate matrices" begin

    # guess_provided = true

    if length(initial_guess) == 0
        # guess_provided = false
        initial_guess = zero(C)
    end

    # 𝐂¹  = A * initial_guess * B + C - initial_guess #copy(C)
    # println(ℒ.norm(𝐂¹))

    if length(𝕊ℂ.tmp) == 0
        𝕊ℂ.tmp = zero(C)
    end

    if length(𝕊ℂ.𝐗) == 0
        𝕊ℂ.𝐗 = zero(C)
    end
    
    if length(𝕊ℂ.𝐂) == 0
        𝕊ℂ.𝐂 = zero(C)
    end
    
    𝐂¹ = 𝕊ℂ.𝐂
    tmp̄ = 𝕊ℂ.tmp
    𝐗 = 𝕊ℂ.𝐗 

    # end # timeit_debug  

    ℒ.mul!(𝐗, initial_guess, B)
    ℒ.mul!(tmp̄, A, 𝐗)
    ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    ℒ.axpy!(-1, initial_guess, tmp̄)

    copyto!(𝐂¹, tmp̄)

    # println(ℒ.norm(𝐂¹))

    # idxs = findall(abs.(C) .> eps()) # this does not work since the problem is incomplete

    function sylvester!(sol,𝐱)
        # @timeit_debug timer "Copy1" begin
        # @inbounds 𝐗[idxs] = 𝐱
        copyto!(𝐗, 𝐱)
        # end # timeit_debug
        # 𝐗 = @view reshape(𝐱, size(𝐗))
        # @timeit_debug timer "Mul1" begin
        # tmp̄ = A * 𝐗 * B 
        ℒ.mul!(tmp̄, A, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Mul2" begin
        ℒ.mul!(𝐗, tmp̄, B, -1, 1)
        # ℒ.axpby!(-1, tmp̄, 1, 𝐗)
        # end # timeit_debug
        # @timeit_debug timer "Copy2" begin
        # @inbounds @views copyto!(sol, 𝐗[idxs])
        copyto!(sol, 𝐗)
        # end # timeit_debug
        # sol = @view reshape(𝐗, size(sol))
    end

    # sylvester = LinearOperators.LinearOperator(Float64, length(idxs), length(idxs), true, true, sylvester!)
    sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)
    
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

    # precond = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, preconditioner!)

    if 𝕊ℂ.krylov_caches.gmres.m == 0
        𝕊ℂ.krylov_caches.gmres =  GmresWorkspace(length(C), length(C), Vector{T})
    end
    # @timeit_debug timer "GMRES solve" begin
    # if length(init) == 0
        # 𝐂, info = Krylov.gmres(sylvester, C[idxs], rtol = tol / 10, atol = tol / 10)#, M = precond)
        # 𝐂, info = Krylov.gmres(sylvester, [vec(𝕊ℂ.𝐂);], 
        Krylov.gmres!(𝕊ℂ.krylov_caches.gmres,
                        sylvester, [vec(𝐂¹);], 
                        # [vec(initial_guess);], 
                        itmax = min(5000,max(500,Int(round(sqrt(length(𝐂¹)*10))))),
                        timemax = 10.0,
                        rtol = tol, 
                        atol = tol)#, M = precond)
    # else
    #     𝐂, info = Krylov.gmres(sylvester, [vec(C);], [vec(init);], rtol = tol / 10)
    # end
    # end # timeit_debug

    # @timeit_debug timer "Postprocess" begin

    # # @inbounds 𝕊ℂ.𝐗[idxs] = 𝐂
    copyto!(𝐗, 𝕊ℂ.krylov_caches.gmres.x)

    # ℒ.mul!(tmp̄, A, 𝐗 * B)
    # ℒ.axpy!(1, C, tmp̄)

    # denom = max(ℒ.norm(𝐗), ℒ.norm(tmp̄))

    # ℒ.axpy!(-1, 𝐗, tmp̄)

    # reached_tol = denom == 0 ? 0.0 : ℒ.norm(tmp̄) / denom
    𝐗 += initial_guess # problematic with copyto! above. cant be axpy!

    ℒ.mul!(tmp̄, 𝐗, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐗, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / max(ℒ.norm(𝐗), ℒ.norm(C))
    # reached_tol = ℒ.norm(A * 𝐗 * B + C - 𝐗) / max(ℒ.norm(𝐗), ℒ.norm(C))

    # end # timeit_debug
    
    if !(typeof(C) <: DenseMatrix)
        𝐗 = choose_matrix_format(𝐗, density_threshold = 1.0)
    end

    # if reached_tol > tol
    #     println("Sylvester: gmres $reached_tol")
    # end
    
    # iter = info.niter
    iter = 𝕊ℂ.krylov_caches.gmres.stats.niter

    # return 𝕊ℂ.𝐗, iter, reached_tol
    return 𝐗, iter, reached_tol
end


# function solve_sylvester_equation(A::AbstractMatrix{Float64},
#                                     B::AbstractMatrix{Float64},
#                                     C::AbstractMatrix{Float64},
#                                     ::Val{:iterative};
#                                     initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
#                                     timer::TimerOutput = TimerOutput(),
#                                     verbose::Bool = false,
#                                     tol::AbstractFloat = 1e-14)
#     # guess_provided = true

#     if length(initial_guess) == 0
#         # guess_provided = false
#         initial_guess = zero(C)
#     end

#     𝐂  = A * initial_guess * B + C - initial_guess
#     𝐂⁰  = copy(𝐂)
#     # 𝐂  = copy(C)
 
#     𝐂¹ = similar(C)
#     𝐂B = similar(C)
    
#     max_iter = 10000
    
#     iters = max_iter

#     for i in 1:max_iter
#         # @timeit_debug timer "Update" begin
#         ℒ.mul!(𝐂B, 𝐂, B)
#         ℒ.mul!(𝐂¹, A, 𝐂B)
#         ℒ.axpy!(1, 𝐂⁰, 𝐂¹)
    
#         if i % 10 == 0
#             normdiff = ℒ.norm(𝐂¹ - 𝐂)
#             if !isfinite(normdiff) || normdiff / max(ℒ.norm(𝐂), ℒ.norm(𝐂¹)) < tol
#             # if isapprox(𝐂¹, 𝐂, rtol = tol)
#                 iters = i
#                 break
#             end
#         end
    
#         copyto!(𝐂, 𝐂¹)
#         # end # timeit_debug
#     end

#     # ℒ.mul!(𝐂B, 𝐂, B)
#     # ℒ.mul!(𝐂¹, A, 𝐂B)
#     # ℒ.axpy!(1, C, 𝐂¹)

#     # denom = max(ℒ.norm(𝐂), ℒ.norm(𝐂¹))

#     # ℒ.axpy!(-1, 𝐂, 𝐂¹)

#     # reached_tol = denom == 0 ? 0.0 : ℒ.norm(𝐂¹) / denom

#     𝐂 += initial_guess

#     reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

#     # if reached_tol > tol
#     #     println("Sylvester: iterative $reached_tol")
#     # end

#     return 𝐂, iters, reached_tol # return info on convergence
# end


# function solve_sylvester_equation(A::AbstractMatrix{Float64},
#                                     B::AbstractMatrix{Float64},
#                                     C::AbstractMatrix{Float64},
#                                     ::Val{:speedmapping};
#                                     initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
#                                     timer::TimerOutput = TimerOutput(),
#                                     verbose::Bool = false,
#                                     tol::AbstractFloat = 1e-14)
#     # guess_provided = true

#     if length(initial_guess) == 0
#         # guess_provided = false
#         initial_guess = zero(C)
#     end

#     𝐂  = A * initial_guess * B + C - initial_guess 
#     # 𝐂 = copy(C)

#     if !(C isa DenseMatrix)
#         C = collect(C)
#     end

#     CB = similar(C)

#     soll = speedmapping(-𝐂; 
#             m! = (X, x) -> begin
#                 ℒ.mul!(CB, x, B)
#                 ℒ.mul!(X, A, CB)
#                 ℒ.axpy!(1, 𝐂, X)
#             end, stabilize = false, maps_limit = 10000, tol = tol)

#     𝐂 = soll.minimizer

#     𝐂 += initial_guess

#     reached_tol = ℒ.norm(A * 𝐂 * B + C - 𝐂) / max(ℒ.norm(𝐂), ℒ.norm(C))

#     # if reached_tol > tol
#     #     println("Sylvester: speedmapping $reached_tol")
#     # end

#     return 𝐂, soll.maps, reached_tol
# end

end # dispatch_doctor