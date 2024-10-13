# Solves A * X ^ 2 + B * X + C = 0
# Algorithms:
# Schur decomposition - fastest, most reliable
# Doubling algorithm - fast, reliable

function solve_quadratic_matrix_equation(A::Matrix{S}, 
                                        B::Matrix{S}, 
                                        C::Matrix{S}, 
                                        ::Val{:Schur}, 
                                        T::timings; 
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where S
    n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
    
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)
    indices_past_not_future_in_comb = indexin(T.past_not_future_idx, comb)

    Ã₊ = @view A[:,future_not_past_and_mixed_in_comb]
    
    Ã₋ = @view C[:,past_not_future_and_mixed_in_comb]
    
    Ã₀₊ = @view B[:,future_not_past_and_mixed_in_comb]

    Ã₀₋ = @views B[:,indices_past_not_future_in_comb] * ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:]

    Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
    I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]
    
    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = ℒ.I(T.nPast_not_future_and_mixed)[T.mixed_in_past_idx,:]
    
    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    
    ℒ.rmul!(Ã₋,-1)
    ℒ.rmul!(Ã₀₊,-1)
    E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))
    
    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = ℒ.schur!(D, E)
    
    eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

    ℒ.ordschur!(schdcmp, eigenselect)

    Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
    Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)
    
    Ŝ₁₁ = ℒ.lu!(S₁₁, check = false)
    
    # D      = Z₂₁ / Ẑ₁₁
    ℒ.rdiv!(Z₂₁, Ẑ₁₁)
    D = Z₂₁
    
    # L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    ℒ.ldiv!(Ŝ₁₁, T₁₁)
    ℒ.mul!(n₋₋, Z₁₁, T₁₁)
    ℒ.rdiv!(n₋₋, Ẑ₁₁)
    L = n₋₋
    
    sol = vcat(L[T.not_mixed_in_past_idx,:], D)

    return sol[T.dynamic_order,:] * ℒ.I(length(comb))[past_not_future_and_mixed_in_comb,:]
end


function solve_quadratic_matrix_equation(A::Matrix{S}, 
                                        B::Matrix{S}, 
                                        C::Matrix{S}, 
                                        ::Val{:Doubling}, 
                                        T::timings;
                                        tol::AbstractFloat = eps(),
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 100) where S
    @timeit_debug timer "Prepare" begin
    B̂ = ℒ.factorize(B)

    # Compute initial values X, Y, E, F
    E = B̂ \ C
    F = B̂ \ A
    X = -E
    Y = -F

    X_new = similar(X)
    Y_new = similar(Y)
    E_new = similar(E) 
    F_new = similar(F)
    
    temp1 = similar(Y)  # Temporary for intermediate operations

    n = size(X, 1)
    II = ℒ.I(n)  # Temporary for identity matrix

    Xtol = 1.0
    Ytol = 1.0

    end # timeit_debug
    @timeit_debug timer "Loop" begin

    for i in 1:max_iter
        @timeit_debug timer "Compute EI" begin

        # Compute EI = I - Y * X
        ℒ.mul!(temp1, Y, X)
        ℒ.axpby!(1, II, -1, temp1)

        end # timeit_debug
        @timeit_debug timer "Invert EI" begin

        fEI = ℒ.lu(temp1, check = false)

        end # timeit_debug
        @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        ℒ.ldiv!(temp1, fEI, E)
        ℒ.mul!(E_new, E, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute FI" begin
            
        # Compute FI = I - X * Y
        copy!(temp1, II)
        ℒ.mul!(temp1, X, Y, -1, 1)

        end # timeit_debug
        @timeit_debug timer "Invert FI" begin

        fFI = ℒ.lu(temp1, check = false)
        
        end # timeit_debug
        @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        ℒ.ldiv!(temp1, fFI, F)
        ℒ.mul!(F_new, F, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(X_new, X, E)
        ℒ.ldiv!(temp1, fFI, X_new)
        ℒ.mul!(X_new, F, temp1)
        if i > 5 Xtol = ℒ.norm(X_new) end
        ℒ.axpy!(1, X, X_new)

        end # timeit_debug
        @timeit_debug timer "Compute Y_new" begin

        # Compute Y_new = Y + E * EI * Y * F
        ℒ.mul!(Y_new, Y, F)
        ℒ.ldiv!(temp1, fEI, Y_new)
        ℒ.mul!(Y_new, E, temp1)
        if i > 5 Ytol = ℒ.norm(Y_new) end
        ℒ.axpy!(1, Y, Y_new)
        
        # println("Iter: $i; xtol: $(ℒ.norm(X_new - X)); ytol: $(ℒ.norm(Y_new - Y))")

        # Check for convergence
        if Xtol < tol && Ytol < tol
            if verbose println("Converged in $i iterations.") end
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

    return X_new  # Converged to solution 
end

