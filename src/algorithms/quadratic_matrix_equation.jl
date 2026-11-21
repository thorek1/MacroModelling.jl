@stable default_mode = "disable" begin

# Solves A * X ^ 2 + B * X + C = 0

# Algorithms:
# Schur decomposition (:schur) - fastest, most reliable
# Doubling algorithm (:doubling) - fast, reliable, most precise
# Linear time iteration algorithm (:linear_time_iteration) [ -(A * X + B) \ C = X̂ ] - slow
# Quadratic iteration algorithm (:quadratic_iteration) [ B \ A * X ^ 2 + B \ C = X̂ ] - very slow


function solve_quadratic_matrix_equation(A::AbstractMatrix{R},
                                        B::AbstractMatrix{R},
                                        C::AbstractMatrix{R},
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                        use_fastlapack_schur::Bool = true,
                                        use_fastlapack_lu::Bool = true,
                                        tol::SolverTolerances = SolverTolerances(atol = 1e-14,
                                                                                rtol = 1e-14,
                                                                                initial_guess_acceptance_tol = 1e-10,
                                                                                acceptance_tol = 1e-6),
                                        verbose::Bool = false,
                                        caching::Bool = true)::Tuple{Matrix{R}, Bool} where {R <: AbstractFloat}
    T = constants.post_model_macro
    n = T.nVars - T.nPresent_only
    nPfm = T.nPast_not_future_and_mixed

    qme_ws = ensure_qme_doubling_workspace!(workspaces, n)
    ensure_schur_workspace!(workspaces,
                            n,
                            T.nMixed,
                            nPfm,
                            T.nFuture_not_past_and_mixed)

    initial_guess_acceptance_tol = tol.initial_guess_acceptance_tol
    acceptance_tol = tol.acceptance_tol
    

    if caching && length(initial_guess) > 0
        X = initial_guess
        X² = qme_ws.temp3

        # Compute residual: A*X² + B*X + C
        # X² into temporary buffer
        ℒ.mul!(X², X, X)
        # A*X² into AXX buffer
        ℒ.mul!(qme_ws.AXX, A, X²)
        norm_AXX = ℒ.norm(qme_ws.AXX)
        
        # B*X into X² buffer (no longer needed for X²)
        ℒ.mul!(X², B, X)
        norm_BX = ℒ.norm(X²)
        
        # Accumulate residual: AXX += B*X + C
        ℒ.axpy!(1, X², qme_ws.AXX)
        ℒ.axpy!(1, C, qme_ws.AXX)
    
        # Standard relative residual: norm(r) / (norm(A*X²) + norm(B*X) + norm(C))
        reached_tol = ℒ.norm(qme_ws.AXX) / (norm_AXX + norm_BX + ℒ.norm(C))

        if reached_tol < (initial_guess_acceptance_tol * length(initial_guess) / 1e6)# 1e-12 is too large eps is too small; if the low tol is used it can be that a small change in the parameters still yields an acceptable solution but as a better tol can be reached it is actually not accurate
            if verbose println("Quadratic matrix equation solver previous solution has tolerance: $reached_tol") end

            existing_sol = cache.qme_solution
            if existing_sol isa Matrix{R} && size(existing_sol) == size(initial_guess)
                copyto!(existing_sol, initial_guess)
                return existing_sol, true
            else
                new_sol = Matrix{R}(initial_guess)
                cache.qme_solution = new_sol
                return new_sol, true
            end
        end
    end

    sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                        Val(quadratic_matrix_equation_algorithm), 
                                                        constants,
                                                        workspaces,
                                                        cache;
                                                        initial_guess = initial_guess,
                                                        use_fastlapack_schur = use_fastlapack_schur,
                                                        use_fastlapack_lu = use_fastlapack_lu,
                                                        tol = tol,
                                                        # timer = timer,
                                                        verbose = verbose,
                                                        caching = caching)

    if verbose println("Quadratic matrix equation solver: $quadratic_matrix_equation_algorithm - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end

    if reached_tol > acceptance_tol
        if quadratic_matrix_equation_algorithm ≠ :schur # try schur if previous one didn't solve it
            sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                                Val(:schur), 
                                                                constants,
                                                                workspaces,
                                                                cache;
                                                                initial_guess = initial_guess,
                                                                use_fastlapack_schur = use_fastlapack_schur,
                                                                use_fastlapack_lu = use_fastlapack_lu,
                                                                tol = tol,
                                                                # timer = timer,
                                                                verbose = verbose,
                                                                caching = caching)

            if verbose println("Quadratic matrix equation solver: schur - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end
        else quadratic_matrix_equation_algorithm ≠ :doubling
            sol, iterations, reached_tol = solve_quadratic_matrix_equation(A, B, C, 
                                                                Val(:doubling), 
                                                                constants,
                                                                workspaces,
                                                                cache;
                                                                initial_guess = initial_guess,
                                                                use_fastlapack_schur = use_fastlapack_schur,
                                                                use_fastlapack_lu = use_fastlapack_lu,
                                                                tol = tol,
                                                                # timer = timer,
                                                                verbose = verbose,
                                                                caching = caching)

            if verbose println("Quadratic matrix equation solver: doubling - converged: $(reached_tol < acceptance_tol) in $iterations iterations to tolerance: $reached_tol") end
        end
    end

    # if (reached_tol > tol) println("QME failed: $reached_tol") end

    return sol, reached_tol < acceptance_tol
end


# Internal switch for the standard-Schur QME path (tests compare it against QZ).
const QME_STANDARD_SCHUR = Ref(true)

# QME solve through the standard Schur form of the companion pencil (D, E).
# When E (or D) is well conditioned, the right deflating subspaces of the pencil are the
# invariant subspaces of M = E⁻¹D (or D⁻¹E). Real Schur (dgees) with ordering is cheaper
# than the generalized QZ (dgges). The policy is then recovered from the omitted
# (trailing) Schur vectors W = [W_p; W_f] with one small solve G = -(W_fᵀ)⁻¹W_pᵀ, and
# one n-square solve against the original coefficients:
#   (A X + B) X = -C  with  A X = A[:, future] G  ⇒  X[:, past] = -(B + A[:, future] G)⁻¹ C[:, past].
# Returns (X, relative residual, success, has unit roots). D and E are not modified.
function qme_standard_schur_solve(A::AbstractMatrix{Float64},
                                    B::AbstractMatrix{Float64},
                                    C::AbstractMatrix{Float64},
                                    D::AbstractMatrix{Float64},
                                    E::AbstractMatrix{Float64},
                                    T,
                                    idx_constants,
                                    nPfm::Int,
                                    nFnpm::Int;
                                    unit_root_tol::Float64 = 1e-6,
                                    rcond_tol::Float64 = 1e-10)
    n = size(A, 1)
    N = size(D, 1)
    fail = (zeros(0, 0), Inf, false, false)
    (N == nPfm + nFnpm && nPfm > 0 && nFnpm > 0) || return fail

    r = 1.0 - unit_root_tol

    # Standard form. Pencil eigenvalues λ (D x = λ E x) with |λ| ≥ r go to the leading block.
    # σ = Inf: M = E⁻¹D, μ = λ.   Finite σ: M = (D - σE)⁻¹E, μ = 1/(λ - σ), so λ = σ + 1/μ
    # (an infinite λ is μ = 0). The first well-conditioned factorization is used.
    F = similar(D)
    M = similar(D)
    σ = NaN
    for shift in (Inf, 0.0, -1.0, 0.5, 2.0)
        if isinf(shift)
            copyto!(F, E)
        else
            @. F = D - shift * E
        end
        anorm = ℒ.opnorm(F, 1)
        Ff = ℒ.lu!(F, check = false)
        ℒ.issuccess(Ff) || continue
        ℒ.LAPACK.gecon!('1', Ff.factors, anorm) > rcond_tol || continue
        copyto!(M, isinf(shift) ? D : E)
        ℒ.ldiv!(Ff, M)
        σ = shift
        break
    end
    isnan(σ) && return fail
    all(isfinite, M) || return fail

    # Diagonal similarity balancing (no permutation, so the past/future split is kept).
    _, _, scale = ℒ.LAPACK.gebal!('S', M)

    Fs = try
        ℒ.schur!(M)
    catch
        return fail
    end

    pencil_λ(μ) = isinf(σ) ? abs(μ) : (μ == 0 ? Inf : abs(σ + 1 / μ))
    select = [pencil_λ(μ) ≥ r for μ in Fs.values]
    count(select) == nPfm || return fail
    has_ur = any(μ -> abs(pencil_λ(μ) - 1) ≤ unit_root_tol, Fs.values)

    try
        ℒ.ordschur!(Fs, select)
    catch
        return fail
    end
    Q = Fs.Z

    # Omitted subspace in balanced coordinates; G maps past to future coordinates.
    W_p = Q[1:nPfm, nPfm+1:N]
    W_f = Q[nPfm+1:N, nPfm+1:N]
    W_ff = ℒ.lu!(copy(W_f'), check = false)
    ℒ.issuccess(W_ff) || return fail
    G = W_ff \ Matrix(W_p')
    ℒ.rmul!(G, -1)
    # undo balancing: x = Dₛ x̃  ⇒  G = Dₛ,f G̃ Dₛ,p⁻¹
    for j in 1:nPfm, i in 1:nFnpm
        G[i, j] *= scale[nPfm + i] / scale[j]
    end
    all(isfinite, G) || return fail

    nL = length(T.not_mixed_in_past_idx)
    sol = zeros(nL + nFnpm, nPfm)
    sol[nL+1:end, :] .= G
    Xg = sol[T.dynamic_order, :] * idx_constants.Ir_past_selector

    fut = idx_constants.future_not_past_and_mixed_in_comb
    past = idx_constants.past_not_future_and_mixed_in_comb
    K = Matrix{Float64}(B)
    ℒ.mul!(K, @view(A[:, fut]), @view(Xg[fut, :]), 1, 1)
    Kf = ℒ.lu!(K, check = false)
    ℒ.issuccess(Kf) || return fail
    Xp = Kf \ Matrix(@view C[:, past])
    ℒ.rmul!(Xp, -1)
    all(isfinite, Xp) || return fail

    X = zeros(n, n)
    X[:, past] .= Xp

    # Relative residual  ‖A X² + B X + C‖ / (‖A X²‖ + ‖B X‖ + ‖C‖), using the column structure of X.
    AX = @view(A[:, fut]) * @view(X[fut, past])          # n × nPfm  (= (A X)[:, past])
    AXX = AX * @view(X[past, past])                       # (A X²)[:, past]
    BX = B * Xp
    res = AXX .+ BX .+ @view(C[:, past])
    nC = ℒ.norm(C)
    # C, A X², B X only have nonzero columns at past.
    reached_tol = ℒ.norm(res) / (ℒ.norm(AXX) + ℒ.norm(BX) + nC)

    return X, reached_tol, true, has_ur
end

function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:schur}, 
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        use_fastlapack_schur::Bool = true,
                                        use_fastlapack_lu::Bool = true,
                                        tol::SolverTolerances = SolverTolerances(),
                                        # timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        caching::Bool = true)::Tuple{Matrix{R}, Int64, R} where R <: AbstractFloat
    
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    
    # Ensure schur workspace is properly sized
    n = T.nVars - T.nPresent_only
    nMixed = T.nMixed
    nPfm = T.nPast_not_future_and_mixed
    nFnpm = T.nFuture_not_past_and_mixed
    
    schur_ws_local = ensure_schur_workspace!(workspaces, n, nMixed, nPfm, nFnpm)
    
    # Use cached indices from constants instead of recomputing
    future_not_past_and_mixed_in_comb = idx_constants.future_not_past_and_mixed_in_comb
    past_not_future_and_mixed_in_comb = idx_constants.past_not_future_and_mixed_in_comb
    indices_past_not_future_in_comb = idx_constants.indices_past_not_future_in_comb
    
    # Use views for read-only slices
    Ã₊_view = @view A[:, future_not_past_and_mixed_in_comb]
    
    # Copy C and B slices that need negation into workspace buffers
    copyto!(schur_ws_local.Ã₋, @view C[:, past_not_future_and_mixed_in_comb])
    copyto!(schur_ws_local.Ã₀₊, @view B[:, future_not_past_and_mixed_in_comb])
    
    # Compute Ã₀₋ = B[:,indices_past_not_future_in_comb] * I_nPast[not_mixed_in_past_idx,:]
    # Use cached constant matrix for I_nPast_not_mixed
    ℒ.mul!(schur_ws_local.Ã₀₋, @view(B[:, indices_past_not_future_in_comb]), idx_constants.I_nPast_not_mixed)
    
    # Use cached constant matrices for zeros and identity blocks
    Z₊ = idx_constants.schur_Z₊
    I₊ = idx_constants.schur_I₊
    Z₋ = idx_constants.schur_Z₋
    I₋ = idx_constants.schur_I₋
    
    # Assemble D matrix in-place: D = [[Ã₀₋ Ã₊], [I₋ Z₊]]
    D = schur_ws_local.D
    # Top-left block: Ã₀₋
    copyto!(view(D, 1:n, 1:nPfm), schur_ws_local.Ã₀₋)
    # Top-right block: Ã₊
    copyto!(view(D, 1:n, nPfm+1:nPfm+nFnpm), Ã₊_view)
    # Bottom-left block: I₋
    copyto!(view(D, n+1:n+nMixed, 1:nPfm), I₋)
    # Bottom-right block: Z₊
    copyto!(view(D, n+1:n+nMixed, nPfm+1:nPfm+nFnpm), Z₊)
    
    # Negate Ã₋ and Ã₀₊ for E matrix
    ℒ.rmul!(schur_ws_local.Ã₋, -1)
    ℒ.rmul!(schur_ws_local.Ã₀₊, -1)
    
    # Assemble E matrix in-place: E = [[Ã₋ Ã₀₊], [Z₋ I₊]]
    E = schur_ws_local.E
    # Top-left block: Ã₋ (already negated)
    copyto!(view(E, 1:n, 1:nPfm), schur_ws_local.Ã₋)
    # Top-right block: Ã₀₊ (already negated)
    copyto!(view(E, 1:n, nPfm+1:nPfm+nFnpm), schur_ws_local.Ã₀₊)
    # Bottom-left block: Z₋
    copyto!(view(E, n+1:n+nMixed, 1:nPfm), Z₋)
    # Bottom-right block: I₊
    copyto!(view(E, n+1:n+nMixed, nPfm+1:nPfm+nFnpm), I₊)
    
    if QME_STANDARD_SCHUR[] && R === Float64
        X_std, reached_tol_std, ok_std, has_ur_std = qme_standard_schur_solve(A, B, C, D, E, T, idx_constants, nPfm, nFnpm)
        if ok_std && reached_tol_std < min(tol.acceptance_tol, sqrt(eps(Float64)))
            X = if caching
                existing_sol = cache.qme_solution
                if existing_sol isa Matrix{R} && size(existing_sol) == (n, n)
                    copyto!(existing_sol, X_std)
                else
                    cache.qme_solution = X_std
                end
            else
                X_std
            end
            if caching
                cache.has_unit_roots = has_ur_std
            end
            if verbose println("Quadratic matrix equation solver: standard schur - converged with tolerance $reached_tol_std") end
            return X, 0, reached_tol_std
        end
    end

    schur_ws_local.fast_qz_ws,
    schur_ws_local.fast_qz_dims,
    schdcmp,
    schur_ok,
    has_ur = factorize_generalized_schur!((use_fastlapack_schur ? Val(:FastLapack) : Val(:Julia)), D,
                                            E,
                                            schur_ws_local.fast_qz_ws,
                                            schur_ws_local.fast_qz_dims,
                                            schur_ws_local.eigenselect)

    if !schur_ok
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    if caching
        cache.has_unit_roots = has_ur
    end

    # Extract blocks from reordered Schur form (need owned copies for lu!)
    copyto!(schur_ws_local.Z₁₁, @view schdcmp.Z[1:nPfm, 1:nPfm])
    copyto!(schur_ws_local.Z₂₁, @view schdcmp.Z[nPfm+1:end, 1:nPfm])
    # Z₁₁ can be a view for matrix multiplication, but LU factorization needs an owned copy.
    Z₁₁ = @view schdcmp.Z[1:nPfm, 1:nPfm]
    
    copyto!(schur_ws_local.S₁₁, @view schdcmp.S[1:nPfm, 1:nPfm])
    copyto!(schur_ws_local.T₁₁, @view schdcmp.T[1:nPfm, 1:nPfm])

    schur_ws_local.fast_lu_ws_z11,
    schur_ws_local.fast_lu_dims_z11,
    solved_Z₁₁,
    Ẑ₁₁ = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), schur_ws_local.Z₁₁,
                        schur_ws_local.fast_lu_ws_z11,
                        schur_ws_local.fast_lu_dims_z11)
    
    if !solved_Z₁₁
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    # LU factorization of S₁₁ (mutating - overwrites workspace buffer)
    schur_ws_local.fast_lu_ws_s11,
    schur_ws_local.fast_lu_dims_s11,
    solved_S₁₁,
    Ŝ₁₁ = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), schur_ws_local.S₁₁,
                        schur_ws_local.fast_lu_ws_s11,
                        schur_ws_local.fast_lu_dims_s11)
    
    if !solved_S₁₁
        if verbose println("Quadratic matrix equation solver: schur - converged: false") end
        return A, 0, 1.0
    end

    # Compute D = Z₂₁ / Ẑ₁₁ (overwrites Z₂₁ buffer)
    solve_lu_right!(schur_ws_local.Z₁₁,
                    schur_ws_local.Z₂₁,
                    schur_ws_local.fast_lu_ws_z11,
                    Ẑ₁₁,
                    schur_ws_local.fast_lu_rhs_t_z21;
                    use_fastlapack_lu = use_fastlapack_lu)
    
    # Compute L = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    # First: T₁₁ ← Ŝ₁₁ \ T₁₁ (overwrites T₁₁ buffer)
    solve_lu_left!(schur_ws_local.S₁₁,
                   schur_ws_local.T₁₁,
                   schur_ws_local.fast_lu_ws_s11,
                   Ŝ₁₁;
                   use_fastlapack_lu = use_fastlapack_lu)
    # Then: S₁₁ ← Z₁₁ * T₁₁ (reuse S₁₁ buffer)
    ℒ.mul!(schur_ws_local.S₁₁, Z₁₁, schur_ws_local.T₁₁)
    # Finally: S₁₁ ← S₁₁ / Ẑ₁₁ (overwrites S₁₁ buffer)
    solve_lu_right!(schur_ws_local.Z₁₁,
                    schur_ws_local.S₁₁,
                    schur_ws_local.fast_lu_ws_z11,
                    Ẑ₁₁,
                    schur_ws_local.fast_lu_rhs_t_s11;
                    use_fastlapack_lu = use_fastlapack_lu)
    
    # Assemble sol = vcat(L[not_mixed_in_past_idx,:], D) in-place
        sol = schur_ws_local.sol
    copyto!(view(sol, 1:length(T.not_mixed_in_past_idx), :), 
            @view schur_ws_local.S₁₁[T.not_mixed_in_past_idx, :])
    copyto!(view(sol, length(T.not_mixed_in_past_idx)+1:size(sol,1), :), 
            schur_ws_local.Z₂₁)
    
    # Final reordering: X = sol[dynamic_order,:] * Ir[past_not_future_and_mixed_in_comb,:]
    # n == n_comb (= nFnpm + nPfm - nMixed) so the result is (n, n), same as doubling.
    # Prefer cache-backed storage to avoid extra allocations.
    X = if caching
        existing_sol = cache.qme_solution
        if existing_sol isa Matrix{R} && size(existing_sol) == (n, n)
            existing_sol
        else
            cache.qme_solution = zeros(R, n, n)
        end
    else
        zeros(R, n, n)
    end

    ℒ.mul!(X, @view(sol[T.dynamic_order, :]), idx_constants.Ir_past_selector)
    
    # Compute residual: A*X² + B*X + C
    # X² into temp_X2 buffer
    ℒ.mul!(schur_ws_local.temp_X2, X, X)
    # A*X² into AXX buffer
    ℒ.mul!(schur_ws_local.AXX, A, schur_ws_local.temp_X2)
    
    norm_AXX = ℒ.norm(schur_ws_local.AXX)
    
    # B*X into temp_X2 buffer (no longer needed for X²)
    ℒ.mul!(schur_ws_local.temp_X2, B, X)
    norm_BX = ℒ.norm(schur_ws_local.temp_X2)
    
    # Accumulate residual: AXX += B*X + C
    ℒ.axpy!(1, schur_ws_local.temp_X2, schur_ws_local.AXX)
    ℒ.axpy!(1, C, schur_ws_local.AXX)
    
    # Standard relative residual: norm(r) / (norm(A*X²) + norm(B*X) + norm(C))
    reached_tol = ℒ.norm(schur_ws_local.AXX) / (norm_AXX + norm_BX + ℒ.norm(C))
    
    return X, 0, reached_tol
end


function solve_quadratic_matrix_equation(A::AbstractMatrix{R}, 
                                        B::AbstractMatrix{R}, 
                                        C::AbstractMatrix{R}, 
                                        ::Val{:doubling}, 
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        initial_guess::AbstractMatrix{R} = zeros(0,0),
                                        use_fastlapack_schur::Bool = true,
                                        use_fastlapack_lu::Bool = true,
                                        tol::SolverTolerances = SolverTolerances(),
                                        # timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 50,
                                        caching::Bool = true)::Tuple{Matrix{R}, Int64, R} where {R <: AbstractFloat}
    T = constants.post_model_macro
    ensure_first_order_constants!(constants)
    idx_constants = constants.post_complete_parameters
    workspace = ensure_qme_doubling_workspace!(workspaces, size(A, 1))
    # Johannes Huber, Alexander Meyer-Gohde, Johanna Saecker (2024). Solving Linear DSGE Models with Structure Preserving Doubling Methods.
    # https://www.imfs-frankfurt.de/forschung/imfs-working-papers/details.html?tx_mmpublications_publicationsdetail%5Bcontroller%5D=Publication&tx_mmpublications_publicationsdetail%5Bpublication%5D=461&cHash=f53244e0345a27419a9d40a3af98c02f
    # https://arxiv.org/abs/2212.09491
    # @timeit_debug timer "Invert B" begin

    guess_provided = true
    n = size(A, 1)

    # if length(initial_guess) == 0
    #     guess_provided = false
    #     initial_guess = zero(A)
    # end

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

    # ℒ.mul!(B̄, A, initial_guess, 1, 1)
    
    workspace.fast_lu_ws_qme_a,
    workspace.fast_lu_dims_qme_a,
    solved_B,
    B̂ = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), B̄,
                       workspace.fast_lu_ws_qme_a,
                       workspace.fast_lu_dims_qme_a)

    if !solved_B
        return A, 0, 1.0
    end

    # Compute initial values X, Y, E, F
    solve_lu_left!(B̄, E, workspace.fast_lu_ws_qme_a, B̂;
                   use_fastlapack_lu = use_fastlapack_lu)
    solve_lu_left!(B̄, F, workspace.fast_lu_ws_qme_a, B̂;
                   use_fastlapack_lu = use_fastlapack_lu)

    # X = -E - initial_guess (in-place)
    copy!(X, E)
    ℒ.rmul!(X, -1)
    # ℒ.axpy!(-1, initial_guess, X)
    # Y = -F (in-place)
    copy!(Y, F)
    ℒ.rmul!(Y, -1)
    # end # timeit_debug
    # @timeit_debug timer "Prellocate" begin

    II = idx_constants.I_n  # Pre-computed identity matrix reference

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

        workspace.fast_lu_ws_qme_a,
        workspace.fast_lu_dims_qme_a,
        solved_EI,
        fEI = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), temp1,
                            workspace.fast_lu_ws_qme_a,
                            workspace.fast_lu_dims_qme_a)

        if !solved_EI
            return A, iter, 1.0
        end

        # end # timeit_debug
        # @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        copyto!(temp3, E)
        solve_lu_left!(temp1, temp3, workspace.fast_lu_ws_qme_a, fEI;
                   use_fastlapack_lu = use_fastlapack_lu)
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

        workspace.fast_lu_ws_qme_b,
        workspace.fast_lu_dims_qme_b,
        solved_FI,
        fFI = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), temp2,
                            workspace.fast_lu_ws_qme_b,
                            workspace.fast_lu_dims_qme_b)
        
        if !solved_FI
            return A, iter, 1.0
        end

        # end # timeit_debug
        # @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        copyto!(temp3, F)
        solve_lu_left!(temp2, temp3, workspace.fast_lu_ws_qme_b, fFI;
                   use_fastlapack_lu = use_fastlapack_lu)
        ℒ.mul!(F_new, F, temp3)
        # F_new = F / fFI * F

        # end # timeit_debug
        # @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(temp3, X, E)
        solve_lu_left!(temp2, temp3, workspace.fast_lu_ws_qme_b, fFI;
                   use_fastlapack_lu = use_fastlapack_lu)
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
        solve_lu_left!(temp1, X, workspace.fast_lu_ws_qme_a, fEI;
                   use_fastlapack_lu = use_fastlapack_lu)
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
        if Xtol < tol.atol # && Yreltol < tol # i % 2 == 0 && 
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

    # ℒ.axpy!(1, initial_guess, X_new)

    # Compute residual to verify solution quality
    # AXX = A * X_new^2 (use temp1 for X^2)
    ℒ.mul!(temp1, X_new, X_new)
    ℒ.mul!(AXX, A, temp1)
    
    norm_AXX = ℒ.norm(AXX)

    # B*X into temp1 buffer (no longer needed for X²)
    ℒ.mul!(temp1, B, X_new)
    norm_BX = ℒ.norm(temp1)

    # Accumulate residual: AXX += B*X + C
    ℒ.axpy!(1, temp1, AXX)
    ℒ.axpy!(1, C, AXX)
    
    # Standard relative residual: norm(r) / (norm(A*X²) + norm(B*X) + norm(C))
    reached_tol = ℒ.norm(AXX) / (norm_AXX + norm_BX + ℒ.norm(C))

    # if reached_tol > tol
    #     println("QME: doubling $reached_tol")
    # end

    X_cache = if caching
        existing_sol = cache.qme_solution
        if existing_sol isa Matrix{R} && size(existing_sol) == size(X_new)
            existing_sol
        else
            cache.qme_solution = zeros(R, size(X_new, 1), size(X_new, 2))
        end
    else
        zeros(R, size(X_new, 1), size(X_new, 2))
    end
    copyto!(X_cache, X_new)

    return X_cache, iter, reached_tol
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





end # @stable
