# Solves A * X ^ 2 + B * X + C = 0

# Algorithms:
# Schur decomposition (:schur) - fastest, most reliable
# Doubling algorithm (:doubling) - fast, reliable, most precise
# Linear time iteration algorithm (:linear_time_iteration) [ -(A * X + B) \ C = X̂ ] - slow
# Quadratic iteration algorithm (:quadratic_iteration) [ B \ A * X ^ 2 + B \ C = X̂ ] - very slow

function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        quadratic_matrix_equation_solver::Symbol = :doubling, 
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where R <: Real
    solve_quadratic_matrix_equation(A, B, C, 
                                    Val(quadratic_matrix_equation_solver), 
                                    T; 
                                    initial_guess = initial_guess,
                                    timer = timer,
                                    verbose = verbose)
end

function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:schur}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where R <: Real
    @timeit_debug timer "Prepare indice" begin
   
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)
    indices_past_not_future_in_comb = indexin(T.past_not_future_idx, comb)

    end # timeit_debug
    @timeit_debug timer "Assemble matrices" begin

    Ã₊ =  A[:,future_not_past_and_mixed_in_comb]
    
    Ã₋ =  C[:,past_not_future_and_mixed_in_comb]
    
    Ã₀₊ =  B[:,future_not_past_and_mixed_in_comb]

    Ã₀₋ =  B[:,indices_past_not_future_in_comb] * ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:]

    Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
    I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]
    
    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = ℒ.I(T.nPast_not_future_and_mixed)[T.mixed_in_past_idx,:]
    
    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    
    ℒ.rmul!(Ã₋,-1)
    ℒ.rmul!(Ã₀₊,-1)
    E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))
    
    end # timeit_debug
    @timeit_debug timer "Schur decomposition" begin

    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = try
        ℒ.schur!(D, E)
    catch
        return A, false
    end

    eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

    end # timeit_debug
    @timeit_debug timer "Reorder Schur decomposition" begin

    try
        ℒ.ordschur!(schdcmp, eigenselect)
    catch
        return A, false
    end

    end # timeit_debug
    @timeit_debug timer "Postprocess" begin

    Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
    Z₁₁ = schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    S₁₁    = schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    @timeit_debug timer "Matrix inversions" begin

    Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)
    
    if !ℒ.issuccess(Ẑ₁₁)
        return A, false
    end

    Ŝ₁₁ = ℒ.lu!(S₁₁, check = false)
    
    if !ℒ.issuccess(Ŝ₁₁)
        return A, false
    end

    end # timeit_debug
    @timeit_debug timer "Matrix divisions" begin

    # D      = Z₂₁ / Ẑ₁₁
    ℒ.rdiv!(Z₂₁, Ẑ₁₁)
    D = Z₂₁
    
    # L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    ℒ.ldiv!(Ŝ₁₁, T₁₁)
    ℒ.mul!(S₁₁, Z₁₁, T₁₁)
    ℒ.rdiv!(S₁₁, Ẑ₁₁)
    L = S₁₁

    sol = vcat(L[T.not_mixed_in_past_idx,:], D)

    end # timeit_debug
    end # timeit_debug

    return sol[T.dynamic_order,:] * ℒ.I(length(comb))[past_not_future_and_mixed_in_comb,:], true
end


function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:doubling}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        tol::AbstractFloat = eps(),
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 100) where R <: Real
    # Johannes Huber, Alexander Meyer-Gohde, Johanna Saecker (2024). Solving Linear DSGE Models with Structure Preserving Doubling Methods.
    # https://www.imfs-frankfurt.de/forschung/imfs-working-papers/details.html?tx_mmpublications_publicationsdetail%5Bcontroller%5D=Publication&tx_mmpublications_publicationsdetail%5Bpublication%5D=461&cHash=f53244e0345a27419a9d40a3af98c02f
    # https://arxiv.org/abs/2212.09491
    @timeit_debug timer "Invert B" begin

    guess_provided = true

    if length(initial_guess) == 0
        guess_provided = false
        initial_guess = zero(A)
    end

    B̄ = B + A * initial_guess

    B̂ = ℒ.lu(B̄, check = false)

    if !ℒ.issuccess(B̂)
        return A, false
    end

    # Compute initial values X, Y, E, F
    ℒ.ldiv!(B̂, A)
    ℒ.ldiv!(B̂, C)

    E = C
    F = A
    X = -E - initial_guess
    Y = -F
    end # timeit_debug
    @timeit_debug timer "Prellocate" begin

    X_new = similar(X)
    Y_new = similar(Y)
    E_new = similar(E) 
    F_new = similar(F)
    
    temp1 = similar(Y)  # Temporary for intermediate operations
    temp2 = similar(Y)  # Temporary for intermediate operations
    temp3 = similar(Y)  # Temporary for intermediate operations

    n = size(X, 1)
    II = ℒ.I(n)  # Temporary for identity matrix

    Xtol = 1.0
    Ytol = 1.0

    solved = false

    iter = max_iter
    end # timeit_debug
    @timeit_debug timer "Loop" begin

    for i in 1:max_iter
        @timeit_debug timer "Compute EI" begin

        # Compute EI = I - Y * X
        ℒ.mul!(temp1, Y, X)
        ℒ.axpby!(1, II, -1, temp1)
        # temp1 = II - Y * X

        end # timeit_debug
        @timeit_debug timer "Invert EI" begin

        fEI = ℒ.lu!(temp1, check = false)

        if !ℒ.issuccess(fEI)
            return A, false
        end

        end # timeit_debug
        @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        ℒ.ldiv!(temp3, fEI, E)
        ℒ.mul!(E_new, E, temp3)
        # E_new = E / fEI * E

        end # timeit_debug
        @timeit_debug timer "Compute FI" begin
            
        # Compute FI = I - X * Y
        ℒ.mul!(temp2, X, Y)
        ℒ.axpby!(1, II, -1, temp2)
        # copy!(E_new, II)
        # ℒ.mul!(E_new, X, Y, -1, 1)
        # temp1 .= II - X * Y

        end # timeit_debug
        @timeit_debug timer "Invert FI" begin

        fFI = ℒ.lu!(temp2, check = false)
        
        if !ℒ.issuccess(fFI)
            return A, false
        end

        end # timeit_debug
        @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        ℒ.ldiv!(temp3, fFI, F)
        ℒ.mul!(F_new, F, temp3)
        # F_new = F / fFI * F

        end # timeit_debug
        @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(temp3, X, E)
        ℒ.ldiv!(fFI, temp3)
        ℒ.mul!(X_new, F, temp3)
        # X_new = F / fFI * X * E
        if i > 5 || guess_provided Xtol = ℒ.norm(X_new) end
        ℒ.axpy!(1, X, X_new)
        # X_new += X

        end # timeit_debug
        @timeit_debug timer "Compute Y_new" begin

        # Compute Y_new = Y + E * EI * Y * F
        ℒ.mul!(X, Y, F) # use X as temporary storage
        ℒ.ldiv!(fEI, X)
        ℒ.mul!(Y_new, E, X)
        # Y_new = E / fEI * Y * F
        if i > 5 || guess_provided Ytol = ℒ.norm(Y_new) end
        ℒ.axpy!(1, Y, Y_new)
        # Y_new += Y
        
        # println("Iter: $i; xtol: $Xtol; ytol: $Ytol")

        # Check for convergence
        if Xtol < tol# && Yreltol < tol # i % 2 == 0 && 
            solved = true
            iter = i
            break
        end

        end # timeit_debug
        @timeit_debug timer "Copy" begin

        # Update values for the next iteration
        copy!(X, X_new)
        copy!(Y, Y_new)
        copy!(E, E_new)
        copy!(F, F_new)
        end # timeit_debug
    end
    end # timeit_debug

    ℒ.axpy!(1, initial_guess, X_new)

    if verbose println("Quadratic matrix equation solver: doubling - converged: $solved in $iter iterations to tolerance: $Xtol") end

    return X_new, solved
end



function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:linear_time_iteration}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        tol::AbstractFloat = 1e-8, # lower tol not possible for NAWM (and probably other models this size)
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 5000) where R <: Real
    # Iterate over: A * X * X + C + B * X = (A * X + B) * X + C = X + (A * X + B) \ C

    @timeit_debug timer "Preallocate" begin

    F = similar(C)
    t = similar(C)

    initial_guess = length(initial_guess) > 0 ? initial_guess : zero(A)

    end # timeit_debug
    @timeit_debug timer "Loop" begin

    sol = @suppress begin 
        speedmapping(initial_guess; m! = (F̄, F) -> begin 
            ℒ.mul!(t, A, F)
            ℒ.axpby!(-1, B, -1, t)
            t̂ = ℒ.lu!(t, check = false)
            ℒ.ldiv!(F̄, t̂, C)
        end,
        tol = tol, maps_limit = max_iter)#, σ_min = 0.0, stabilize = false, orders = [3,3,2])
    end

    X = sol.minimizer

    # reached_tol = ℒ.norm(A * X * X + B * X + C)
    
    # converged = reached_tol < tol

    # if verbose println("Converged: $converged in $(sol.maps) iterations to tolerance: $reached_tol") end

    if verbose println("Quadratic matrix equation solver: linear_time_iteration - converged: $(sol.converged) in $(sol.maps) iterations to tolerance: $(sol.norm_∇)") end
    
    end # timeit_debug

    return X, sol.converged
end



function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:quadratic_iteration}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        tol::AbstractFloat = 1e-8, # lower tol not possible for NAWM (and probably other models this size)
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 50000) where R <: Real
    # Iterate over: Ā * X * X + C̄ + X

    B̂ =  ℒ.lu(B)

    C̄ = B̂ \ C
    Ā = B̂ \ A
    
    X = similar(Ā)
    X̄ = similar(Ā)
    
    X² = similar(X)

    initial_guess = length(initial_guess) > 0 ? initial_guess : zero(A)

    sol = @suppress begin
        speedmapping(initial_guess; m! = (X̄, X) -> begin 
                                                ℒ.mul!(X², X, X)
                                                ℒ.mul!(X̄, Ā, X²)
                                                ℒ.axpy!(1, C̄, X̄)
                                            end,
        tol = tol, maps_limit = max_iter, σ_min = 0.0, stabilize = false, orders = [3,3,2])
    end

    X = -sol.minimizer

    if verbose println("Quadratic matrix equation solver: linear_time_iteration - converged: $(sol.converged) in $(sol.maps) iterations to tolerance: $(sol.norm_∇)") end
    
    return X, sol.converged
end



function solve_quadratic_matrix_equation(A::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        B::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        C::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        T::timings; 
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                        quadratic_matrix_equation_solver::Symbol = :doubling, 
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where {Z,S,N}
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    X, solved = solve_quadratic_matrix_equation(Â, B̂, Ĉ, 
                                                Val(quadratic_matrix_equation_solver), 
                                                T; 
                                                initial_guess = initial_guess,
                                                timer = timer,
                                                verbose = verbose)

    AXB = Â * X + B̂
    
    AXBfact = ℒ.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * Â

    X² = X * X

    X̃ = zeros(length(X), N)

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        dA = ℱ.partials.(A, i)
        dB = ℱ.partials.(B, i)
        dC = ℱ.partials.(C, i)
    
        CC = invAXB * (dA * X² + dB * X + dC)
    
        dX, solved = solve_sylvester_equation(AA, -X, -CC, sylvester_algorithm = :sylvester)

        X̃[:,i] = vec(dX)
    end
    
    return reshape(map(X, eachrow(X̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(X)), solved
end
