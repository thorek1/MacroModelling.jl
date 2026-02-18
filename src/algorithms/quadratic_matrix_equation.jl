# Solves A * X ^ 2 + B * X + C = 0

# Algorithms:
# Schur decomposition (:schur) - fastest, most reliable
# Doubling algorithm (:doubling) - fast, reliable, most precise
# Linear time iteration algorithm (:linear_time_iteration) [ -(A * X + B) \ C = X̂ ] - slow
# Quadratic iteration algorithm (:quadratic_iteration) [ B \ A * X ^ 2 + B \ C = X̂ ] - very slow

@stable default_mode = "disable" begin

function factorize_generalized_schur!(D::AbstractMatrix{R},
                                      E::AbstractMatrix{R},
                                      schur_ws::schur_workspace{R},
                                      eigenselect::AbstractVector{Bool};
                                      use_fastlapack_schur::Bool = true) where {R <: AbstractFloat}
    if use_fastlapack_schur && R <: Union{Float32, Float64}
        qz_dims = (size(D, 1), size(D, 2))
        if schur_ws.fast_qz_dims != qz_dims
            schur_ws.fast_qz_ws = FastLapackInterface.GeneralizedSchurWs(D)
            schur_ws.fast_qz_dims = qz_dims
        end

        try
            S, T, _, _, _, Z = ℒ.LAPACK.gges!(schur_ws.fast_qz_ws, 'V', 'V', D, E;
                                              select = FastLapackInterface.id,
                                              criterium = 1.0,
                                              resize = true)
            return (S = S, T = T, Z = Z), true
        catch
            return nothing, false
        end
    else
        schdcmp = try
            ℒ.schur!(D, E)
        catch
            return nothing, false
        end

        @. eigenselect = abs(schdcmp.β / schdcmp.α) < 1

        try
            ℒ.ordschur!(schdcmp, eigenselect)
        catch
            return nothing, false
        end

        return schdcmp, true
    end
end

function solve_quadratic_matrix_equation(A::AbstractMatrix{R},
                                        B::AbstractMatrix{R},
                                        C::AbstractMatrix{R},
                                        constants::constants,
                                        workspace::qme_workspace{R,S};
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        quadratic_matrix_equation_algorithm::Symbol = :schur,
                                        use_fastlapack_schur::Bool = true,
                                        tol::AbstractFloat = 1e-14,
                                        acceptance_tol::AbstractFloat = 1e-8,
                                        verbose::Bool = false) where {R <: Real, S <: Real}
    T = constants.post_model_macro
    

    if length(initial_guess) > 0
        X = initial_guess
        X² = workspace.temp3

        # Compute residual: A*X² + B*X + C
        # X² into temporary buffer
        ℒ.mul!(X², X, X)
        # A*X² into AXX buffer
        ℒ.mul!(workspace.AXX, A, X²)
        
        AXXnorm = max(ℒ.norm(workspace.AXX), ℒ.norm(C))
        
        # AXX += B*X
        ℒ.mul!(workspace.AXX, B, X, 1, 1)
        # AXX += C
        ℒ.axpy!(1, C, workspace.AXX)
    
        reached_tol = ℒ.norm(workspace.AXX) / AXXnorm

        if reached_tol < (acceptance_tol * length(initial_guess) / 1e6)# 1e-12 is too large eps is too small; if the low tol is used it can be that a small change in the parameters still yields an acceptable solution but as a better tol can be reached it is actually not accurate
            if verbose println("Quadratic matrix equation solver previous solution has tolerance: $reached_tol") end

            return initial_guess, true
        end
    end

    sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                        Val(quadratic_matrix_equation_algorithm), 
                                                        constants,
                                                        workspace; 
                                                        initial_guess = initial_guess,
                                                        use_fastlapack_schur = use_fastlapack_schur,
                                                        tol = tol,
                                                        # timer = timer,
                                                        verbose = verbose)

    if verbose println("Quadratic matrix equation solver: $quadratic_matrix_equation_algorithm - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end

    if reached_tol > acceptance_tol
        if quadratic_matrix_equation_algorithm ≠ :schur # try schur if previous one didn't solve it
            sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                                Val(:schur), 
                                                                constants,
                                                                workspace; 
                                                                initial_guess = initial_guess,
                                                                use_fastlapack_schur = use_fastlapack_schur,
                                                                tol = tol,
                                                                # timer = timer,
                                                                verbose = verbose)

            if verbose println("Quadratic matrix equation solver: schur - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end
        else quadratic_matrix_equation_algorithm ≠ :doubling
            sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                                Val(:doubling), 
                                                                constants,
                                                                workspace; 
                                                                initial_guess = initial_guess,
                                                                use_fastlapack_schur = use_fastlapack_schur,
                                                                tol = tol,
                                                                # timer = timer,
                                                                verbose = verbose)

            if verbose println("Quadratic matrix equation solver: doubling - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end
        end
    end

    # if (reached_tol > tol) println("QME failed: $reached_tol") end

    return sol, reached_tol < acceptance_tol
end

function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:schur}, 
                                        constants::constants,
                                        workspace::qme_workspace; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        use_fastlapack_schur::Bool = true,
                                        tol::AbstractFloat = 1e-14,
                                        # timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false)::Tuple{Matrix{R}, Int64, R} where R <: AbstractFloat
    
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    
    # Ensure schur workspace is properly sized
    n = T.nVars - T.nPresent_only
    nMixed = T.nMixed
    nPfm = T.nPast_not_future_and_mixed
    nFnpm = T.nFuture_not_past_and_mixed
    
    # Get schur workspace from constants.workspaces (need to pass through from caller)
    # For now, create locally but in future this should come from workspaces
    schur_ws = Schur_workspace(n, nMixed, nPfm, nFnpm, T = R)
    
    # Use cached indices from constants instead of recomputing
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    indices_past_not_future_in_comb = idx_constants.indices_past_not_future_in_comb
    
    # Use views for read-only slices
    Ã₊_view = @view A[:, future_not_past_and_mixed_in_comb]
    
    # Copy C and B slices that need negation into workspace buffers
    copyto!(schur_ws.Ã₋, @view C[:, past_not_future_and_mixed_in_comb])
    copyto!(schur_ws.Ã₀₊, @view B[:, future_not_past_and_mixed_in_comb])
    
    # Compute Ã₀₋ = B[:,indices_past_not_future_in_comb] * I_nPast[not_mixed_in_past_idx,:]
    # Use cached constant matrix for I_nPast_not_mixed
    ℒ.mul!(schur_ws.Ã₀₋, @view(B[:, indices_past_not_future_in_comb]), idx_constants.I_nPast_not_mixed)
    
    # Use cached constant matrices for zeros and identity blocks
    Z₊ = idx_constants.schur_Z₊
    I₊ = idx_constants.schur_I₊
    Z₋ = idx_constants.schur_Z₋
    I₋ = idx_constants.schur_I₋
    
    # Assemble D matrix in-place: D = [[Ã₀₋ Ã₊], [I₋ Z₊]]
    D = schur_ws.D
    # Top-left block: Ã₀₋
    copyto!(view(D, 1:n, 1:nPfm), schur_ws.Ã₀₋)
    # Top-right block: Ã₊
    copyto!(view(D, 1:n, nPfm+1:nPfm+nFnpm), Ã₊_view)
    # Bottom-left block: I₋
    copyto!(view(D, n+1:n+nMixed, 1:nPfm), I₋)
    # Bottom-right block: Z₊
    copyto!(view(D, n+1:n+nMixed, nPfm+1:nPfm+nFnpm), Z₊)
    
    # Negate Ã₋ and Ã₀₊ for E matrix
    ℒ.rmul!(schur_ws.Ã₋, -1)
    ℒ.rmul!(schur_ws.Ã₀₊, -1)
    
    # Assemble E matrix in-place: E = [[Ã₋ Ã₀₊], [Z₋ I₊]]
    E = schur_ws.E
    # Top-left block: Ã₋ (already negated)
    copyto!(view(E, 1:n, 1:nPfm), schur_ws.Ã₋)
    # Top-right block: Ã₀₊ (already negated)
    copyto!(view(E, 1:n, nPfm+1:nPfm+nFnpm), schur_ws.Ã₀₊)
    # Bottom-left block: Z₋
    copyto!(view(E, n+1:n+nMixed, 1:nPfm), Z₋)
    # Bottom-right block: I₊
    copyto!(view(E, n+1:n+nMixed, nPfm+1:nPfm+nFnpm), I₊)
    
    schdcmp, schur_ok = factorize_generalized_schur!(D,
                                                     E,
                                                     schur_ws,
                                                     schur_ws.eigenselect;
                                                     use_fastlapack_schur = use_fastlapack_schur)

    if !schur_ok
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    # Extract blocks from reordered Schur form (need owned copies for lu!)
    copyto!(schur_ws.Z₂₁, @view schdcmp.Z[nPfm+1:end, 1:nPfm])
    # Z₁₁ can be a view since it's only used as RHS in mul!
    Z₁₁ = @view schdcmp.Z[1:nPfm, 1:nPfm]
    
    copyto!(schur_ws.S₁₁, @view schdcmp.S[1:nPfm, 1:nPfm])
    copyto!(schur_ws.T₁₁, @view schdcmp.T[1:nPfm, 1:nPfm])

    # LU factorization of Z₁₁ (non-mutating since Z₁₁ is a view)
    Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)
    
    if !ℒ.issuccess(Ẑ₁₁)
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    # LU factorization of S₁₁ (mutating - overwrites workspace buffer)
    Ŝ₁₁ = ℒ.lu!(schur_ws.S₁₁, check = false)
    
    if !ℒ.issuccess(Ŝ₁₁)
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    # Compute D = Z₂₁ / Ẑ₁₁ (overwrites Z₂₁ buffer)
    ℒ.rdiv!(schur_ws.Z₂₁, Ẑ₁₁)
    
    # Compute L = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    # First: T₁₁ ← Ŝ₁₁ \ T₁₁ (overwrites T₁₁ buffer)
    ℒ.ldiv!(Ŝ₁₁, schur_ws.T₁₁)
    # Then: S₁₁ ← Z₁₁ * T₁₁ (reuse S₁₁ buffer)
    ℒ.mul!(schur_ws.S₁₁, Z₁₁, schur_ws.T₁₁)
    # Finally: S₁₁ ← S₁₁ / Ẑ₁₁ (overwrites S₁₁ buffer)
    ℒ.rdiv!(schur_ws.S₁₁, Ẑ₁₁)
    
    # Assemble sol = vcat(L[not_mixed_in_past_idx,:], D) in-place
    sol = schur_ws.sol
    copyto!(view(sol, 1:length(T.not_mixed_in_past_idx), :), 
            @view schur_ws.S₁₁[T.not_mixed_in_past_idx, :])
    copyto!(view(sol, length(T.not_mixed_in_past_idx)+1:size(sol,1), :), 
            schur_ws.Z₂₁)
    
    # Final reordering: X = sol[dynamic_order,:] * Ir[past_not_future_and_mixed_in_comb,:]
    # Use cached Ir_past_selector and mul! into workspace X buffer
    X = schur_ws.X
    ℒ.mul!(X, @view(sol[T.dynamic_order, :]), idx_constants.Ir_past_selector)
    
    # Compute residual: A*X² + B*X + C
    # X² into temp_X2 buffer
    ℒ.mul!(schur_ws.temp_X2, X, X)
    # A*X² into AXX buffer
    ℒ.mul!(schur_ws.AXX, A, schur_ws.temp_X2)
    
    AXXnorm = max(ℒ.norm(schur_ws.AXX), ℒ.norm(C))
    
    # AXX += B*X
    ℒ.mul!(schur_ws.AXX, B, X, 1, 1)
    # AXX += C
    ℒ.axpy!(1, C, schur_ws.AXX)
    
    reached_tol = ℒ.norm(schur_ws.AXX) / AXXnorm
    
    # Return a copy of X (to avoid returning a reference to mutable workspace)
    return copy(X), 0, reached_tol
end


function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:doubling}, 
                                        constants::constants,
                                        workspace::qme_workspace{R,S}; 
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        use_fastlapack_schur::Bool = true,
                                        tol::AbstractFloat = 1e-14,
                                        # timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 100)::Tuple{Matrix{R}, Int64, R} where {R <: AbstractFloat, S <: Real}
    T = constants.post_model_macro
    # Johannes Huber, Alexander Meyer-Gohde, Johanna Saecker (2024). Solving Linear DSGE Models with Structure Preserving Doubling Methods.
    # https://www.imfs-frankfurt.de/forschung/imfs-working-papers/details.html?tx_mmpublications_publicationsdetail%5Bcontroller%5D=Publication&tx_mmpublications_publicationsdetail%5Bpublication%5D=461&cHash=f53244e0345a27419a9d40a3af98c02f
    # https://arxiv.org/abs/2212.09491
    # @timeit_debug timer "Invert B" begin

    guess_provided = true
    n = size(A, 1)

    if length(initial_guess) == 0
        guess_provided = false
        initial_guess = zero(A)
    end

    # Extract workspaces
    E = workspace.E
    F = workspace.F
    X = workspace.X
    Y = workspace.Y
    X_new = workspace.X_new
    Y_new = workspace.Y_new
    E_new = workspace.E_new
    F_new = workspace.F_new
    temp1 = workspace.temp1
    temp2 = workspace.temp2
    temp3 = workspace.temp3
    B̄ = workspace.B̄
    AXX = workspace.AXX
    
    # Initialize from inputs
    copy!(E, C)
    copy!(F, A)
    copy!(B̄, B)

    ℒ.mul!(B̄, A, initial_guess, 1, 1)
    
    B̂ = ℒ.lu!(B̄, check = false)

    if !ℒ.issuccess(B̂)
        return A, 0, 1.0
    end

    # Compute initial values X, Y, E, F
    ℒ.ldiv!(E, B̂, C)
    ℒ.ldiv!(F, B̂, A)

    # X = -E - initial_guess (in-place)
    copy!(X, E)
    ℒ.rmul!(X, -1)
    ℒ.axpy!(-1, initial_guess, X)
    # Y = -F (in-place)
    copy!(Y, F)
    ℒ.rmul!(Y, -1)
    # end # timeit_debug
    # @timeit_debug timer "Prellocate" begin

    II = workspace.I_n  # Pre-computed identity matrix reference

    Xtol = 1.0
    Ytol = 1.0

    solved = false

    iter = max_iter
    # end # timeit_debug
    # @timeit_debug timer "Loop" begin

    for i in 1:max_iter
        # @timeit_debug timer "Compute EI" begin

        # Compute EI = I - Y * X
        ℒ.mul!(temp1, Y, X)
        ℒ.axpby!(1, II, -1, temp1)
        # temp1 = II - Y * X

        # end # timeit_debug
        # @timeit_debug timer "Invert EI" begin

        fEI = ℒ.lu!(temp1, check = false)

        if !ℒ.issuccess(fEI)
            return A, iter, 1.0
        end

        # end # timeit_debug
        # @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        ℒ.ldiv!(temp3, fEI, E)
        ℒ.mul!(E_new, E, temp3)
        # E_new = E / fEI * E

        # end # timeit_debug
        # @timeit_debug timer "Compute FI" begin
            
        # Compute FI = I - X * Y
        ℒ.mul!(temp2, X, Y)
        ℒ.axpby!(1, II, -1, temp2)
        # copy!(E_new, II)
        # ℒ.mul!(E_new, X, Y, -1, 1)
        # temp1 .= II - X * Y

        # end # timeit_debug
        # @timeit_debug timer "Invert FI" begin

        fFI = ℒ.lu!(temp2, check = false)
        
        if !ℒ.issuccess(fFI)
            return A, iter, 1.0
        end

        # end # timeit_debug
        # @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        ℒ.ldiv!(temp3, fFI, F)
        ℒ.mul!(F_new, F, temp3)
        # F_new = F / fFI * F

        # end # timeit_debug
        # @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(temp3, X, E)
        ℒ.ldiv!(fFI, temp3)
        ℒ.mul!(X_new, F, temp3)
        # X_new = F / fFI * X * E
        if i > 5 || guess_provided 
            Xtol = ℒ.norm(X_new) 
            # relXtol = Xtol / ℒ.norm(X) 
        end
        ℒ.axpy!(1, X, X_new)
        # X_new += X

        # end # timeit_debug
        # @timeit_debug timer "Compute Y_new" begin

        # Compute Y_new = Y + E * EI * Y * F
        ℒ.mul!(X, Y, F) # use X as temporary storage
        ℒ.ldiv!(fEI, X)
        ℒ.mul!(Y_new, E, X)
        # Y_new = E / fEI * Y * F
        if i > 5 || guess_provided 
            Ytol = ℒ.norm(Y_new) 
            # relYtol = Ytol / ℒ.norm(Y) 
        end
        ℒ.axpy!(1, Y, Y_new)
        # Y_new += Y
        
        # println("Iter: $i; xtol: $Xtol; ytol: $Ytol; rel ytol: $relYtol; rel xtol: $relXtol")

        # Check for convergence
        if Xtol < tol # && Yreltol < tol # i % 2 == 0 && 
            solved = true
            iter = i
            break
        end

        # end # timeit_debug
        # @timeit_debug timer "Copy" begin

        # Update values for the next iteration
        copy!(X, X_new)
        copy!(Y, Y_new)
        copy!(E, E_new)
        copy!(F, F_new)
        # end # timeit_debug
    end
    # end # timeit_debug

    ℒ.axpy!(1, initial_guess, X_new)

    # Compute residual to verify solution quality
    # AXX = A * X_new^2 (use temp1 for X^2)
    ℒ.mul!(temp1, X_new, X_new)
    ℒ.mul!(AXX, A, temp1)
    
    AXXnorm = max(ℒ.norm(AXX), ℒ.norm(C))
    
    ℒ.mul!(AXX, B, X_new, 1, 1)

    ℒ.axpy!(1, C, AXX)
    
    reached_tol = ℒ.norm(AXX) / AXXnorm
    
    # if reached_tol > tol
    #     println("QME: doubling $reached_tol")
    # end

    # Return a copy of X_new (to avoid returning a reference to mutable workspace)
    return copy(X_new), iter, reached_tol
end



# function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
#                                         B::AbstractMatrix{R}, 
#                                         C::AbstractMatrix{R}, 
#                                         ::Val{:linear_time_iteration}, 
#                                         T::timings; 
#                                         initial_guess::AbstractMatrix{R} = zeros(0,0),
#                                         tol::AbstractFloat = 1e-14, # lower tol not possible for NAWM (and probably other models this size)
#                                         timer::TimerOutput = TimerOutput(),
#                                         verbose::Bool = false,
#                                         max_iter::Int = 5000) where R <: Real
#     # Iterate over: A * X * X + C + B * X = (A * X + B) * X + C = X + (A * X + B) \ C

#     # @timeit_debug timer "Preallocate" begin

#     F = similar(C)
#     t = similar(C)

#     initial_guess = length(initial_guess) > 0 ? initial_guess : zero(A)

#     # end # timeit_debug
#     # @timeit_debug timer "Loop" begin

#     sol = @suppress begin 
#         speedmapping(initial_guess; m! = (F̄, F) -> begin 
#             ℒ.mul!(t, A, F)
#             ℒ.axpby!(-1, B, -1, t)
#             t̂ = ℒ.lu!(t, check = false)
#             ℒ.ldiv!(F̄, t̂, C)
#         end,
#         tol = tol, maps_limit = max_iter)#, σ_min = 0.0, stabilize = false, orders = [3,3,2])
#     end

#     X = sol.minimizer

#     AXX = A * X^2
    
#     AXXnorm = ℒ.norm(AXX)
    
#     ℒ.mul!(AXX, B, X, 1, 1)

#     ℒ.axpy!(1, C, AXX)
    
#     reached_tol = ℒ.norm(AXX) / AXXnorm
    
#     # end # timeit_debug

#     # if reached_tol > tol
#     #     println("QME: linear time iteration $reached_tol")
#     # end

#     return X, sol.maps, reached_tol
# end



# function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
#                                         B::AbstractMatrix{R}, 
#                                         C::AbstractMatrix{R}, 
#                                         ::Val{:quadratic_iteration}, 
#                                         T::timings; 
#                                         initial_guess::AbstractMatrix{R} = zeros(0,0),
#                                         tol::AbstractFloat = 1e-14, # lower tol not possible for NAWM (and probably other models this size)
#                                         timer::TimerOutput = TimerOutput(),
#                                         verbose::Bool = false,
#                                         max_iter::Int = 50000) where R <: Real
#     # Iterate over: Ā * X * X + C̄ + X

#     B̂ =  ℒ.lu(B)

#     C̄ = B̂ \ C
#     Ā = B̂ \ A
    
#     X = similar(Ā)
#     X̄ = similar(Ā)
    
#     X² = similar(X)

#     initial_guess = length(initial_guess) > 0 ? initial_guess : zero(A)

#     sol = @suppress begin
#         speedmapping(initial_guess; m! = (X̄, X) -> begin 
#                                                 ℒ.mul!(X², X, X)
#                                                 ℒ.mul!(X̄, Ā, X²)
#                                                 ℒ.axpy!(1, C̄, X̄)
#                                             end,
#         tol = tol, maps_limit = max_iter)#, σ_min = 0.0, stabilize = false, orders = [3,3,2])
#     end

#     X = -sol.minimizer

#     AXX = A * X^2
    
#     AXXnorm = ℒ.norm(AXX)
    
#     ℒ.mul!(AXX, B, X, 1, 1)

#     ℒ.axpy!(1, C, AXX)
    
#     reached_tol = ℒ.norm(AXX) / AXXnorm

#     # if reached_tol > tol
#     #     println("QME: quadratic iteration $reached_tol")
#     # end

#     return X, sol.maps, reached_tol
# end




end # dispatch_doctor
