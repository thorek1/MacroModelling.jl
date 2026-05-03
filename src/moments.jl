@stable default_mode = "disable" begin


"""
    sparse_ABAt(A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T};
                tol::Real = eps(T)) -> SparseMatrixCSC{T}

Compute `C = A * B * A'` returning a sparse symmetric matrix, where `B` is
symmetric.  Uses a column-by-column sparse-accumulator (SPA) kernel that
computes only the lower triangle with advancing-pointer row pruning, then
emits both `(i,j)` and `(j,i)` entries into COO vectors for direct CSC
assembly.

`tol` controls the drop tolerance during assembly (entries with `|v| < tol`
are discarded).
"""
function sparse_ABAt(A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T};
                     tol::Real = eps(T)) where T <: Real
    m, n = size(A)
    @assert size(B) == (n, n) "B must be n×n where A is m×n"

    # Linked-list row index for A: avoids allocating sparse(A').
    # row_head[row] -> first nz index in that row; row_next[nz] -> next nz in same row;
    # row_col[nz] -> column of that nonzero entry.
    A_rv   = SparseArrays.rowvals(A)
    A_nz   = nonzeros(A)
    A_cp   = SparseArrays.getcolptr(A)
    nnzA   = nnz(A)
    row_head = zeros(Int, m)
    row_next = zeros(Int, nnzA)
    row_col  = Vector{Int}(undef, nnzA)
    @inbounds for col in n:-1:1
        for idx in A_cp[col]:(A_cp[col + 1] - 1)
            row = A_rv[idx]
            row_next[idx] = row_head[row]
            row_head[row] = idx
            row_col[idx]  = col
        end
    end

    B_rows  = SparseArrays.rowvals(B)
    B_vals  = nonzeros(B)

    A_rows  = A_rv
    A_vals  = A_nz
    A_colptr = A_cp

    # SPA workspace: generation-marker pattern avoids zeroing w each column
    w      = Vector{T}(undef, n)
    mark_w = zeros(Int, n)
    w_nz   = Vector{Int}(undef, n)

    # Dense accumulator for output column (lower triangle only)
    c = zeros(T, m)

    # Advancing pointers: A_start[k] tracks first unconsumed position in A[:,k].
    # Monotonic j means pointers only advance; total work bounded by nnz(A).
    A_start = A_colptr[1:n]

    # COO output: emit both (i,j) and (j,i) for off-diagonal entries.
    # Upper bound: m*(m+1)/2 lower-triangle entries + m*(m-1)/2 mirrors = m²
    max_entries = m * m
    coo_I = Vector{Int}(undef, max_entries)
    coo_J = Vector{Int}(undef, max_entries)
    coo_V = Vector{T}(undef, max_entries)
    cnt   = 0

    @inbounds for j in 1:m
        row_head[j] == 0 && continue

        # Phase A: w = B * Aᵀ[:,j]  via SPA (iterate row j of A via linked-list)
        w_cnt = 0
        p = row_head[j]
        while p != 0
            l    = row_col[p]
            a_jl = A_nz[p]
            for q in SparseArrays.nzrange(B, l)
                k    = B_rows[q]
                b_kl = B_vals[q]
                if mark_w[k] != j
                    mark_w[k] = j
                    w_cnt += 1
                    w_nz[w_cnt] = k
                    w[k] = b_kl * a_jl
                else
                    w[k] += b_kl * a_jl
                end
            end
            p = row_next[p]
        end

        # Sort for sequential A-column access (cache-friendly advancing pointers)
        sort!(view(w_nz, 1:w_cnt))

        # Phase B: c[i] += (A * w)[i] for i ≥ j  (lower triangle only)
        for idx in 1:w_cnt
            k    = w_nz[idx]
            wk   = w[k]
            hi_k = A_colptr[k + 1] - 1
            p    = A_start[k]
            while p ≤ hi_k && A_rows[p] < j
                p += 1
            end
            A_start[k] = p
            for q in p:hi_k
                c[A_rows[q]] += A_vals[q] * wk
            end
        end

        # Phase C: gather nonzeros from c[j:m], emit both (i,j) and (j,i)
        for i in j:m
            val  = c[i]
            c[i] = zero(T)
            if abs(val) >= tol
                cnt += 1
                coo_I[cnt] = i
                coo_J[cnt] = j
                coo_V[cnt] = val
                if i != j
                    cnt += 1
                    coo_I[cnt] = j
                    coo_J[cnt] = i
                    coo_V[cnt] = val
                end
            end
        end
    end

    resize!(coo_I, cnt)
    resize!(coo_J, cnt)
    resize!(coo_V, cnt)

    return sparse(coo_I, coo_J, coo_V, m, m)
end


function calculate_covariance(parameters::Vector{R}, 
                                𝓂::ℳ; 
                                opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{R}, Matrix{R}, Matrix{R}, Vector{R}, Bool} where R <: Real
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    idx_constants = constants.post_complete_parameters
    T = constants.post_model_macro
    
    _nsss_result = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    SS_and_pars = _nsss_result[1]::Vector{R}
    solution_error = _nsss_result[2][1]
    
    if solution_error > opts.tol.nsss.acceptance_tol
        return zeros(0,0), zeros(0,0), zeros(0,0), SS_and_pars, solution_error < opts.tol.nsss.acceptance_tol
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces) 

    sol, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                            constants,
                                                            𝓂.workspaces,
                                                            𝓂.caches;
                                                            initial_guess = 𝓂.caches.qme_solution,
                                                            opts = opts,
                                                            parameter_values = parameters)

    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    # Direct constants access instead of model access
    A = @views sol[:, 1:T.nPast_not_future_and_mixed] * idx_constants.diag_nVars[T.past_not_future_and_mixed_idx,:]

    C = @views sol[:, T.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    if !solved
        return CC, sol, ∇₁, SS_and_pars, solved
    end

    # Check Lyapunov cache: if valid for current parameters, skip the solve
    cached_covar = 𝓂.caches.covariance_first_order
    if R === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.covariance_first_order, parameters) &&
       !isempty(cached_covar) && size(cached_covar) == (T.nVars, T.nVars)
        return cached_covar, sol, ∇₁, SS_and_pars, true
    end

    # Ensure lyapunov workspace is properly sized and get it
    lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, T.nVars, :first_order)

    covar_raw, solved = solve_lyapunov_equation(A, CC, lyap_ws,
                            initial_guess = cached_covar,
                            lyapunov_algorithm = opts.lyapunov_algorithm, 
                            tol = opts.tol.first_order.lyapunov,
                            verbose = opts.verbose,
                            has_unit_roots = 𝓂.caches.has_unit_roots)

    # Safety net: if Lyapunov result contains NaN (deflation was used),
    # ensure the flag is set for subsequent calls even if QME didn't detect it
    if solved && any(isnan, covar_raw)
        𝓂.caches.has_unit_roots = true
    end

    # Cache the result for reuse
    if R === Float64 && solved
        if size(𝓂.caches.covariance_first_order) != size(covar_raw)
            𝓂.caches.covariance_first_order = Matrix{Float64}(undef, size(covar_raw)...)
        end
        copyto!(𝓂.caches.covariance_first_order, covar_raw)
        𝓂.caches.valid_for.covariance_first_order = Float64.(parameters)
    end

    covar_stable = covar_raw

    return covar_stable, sol , ∇₁, SS_and_pars, solved
end


function calculate_mean(parameters::Vector{R}, 
                        𝓂::ℳ; 
                        algorithm = :pruned_second_order, 
                        opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{R}, Bool} where R <: Real
                        # Matrix{R}, Matrix{R}, AbstractSparseMatrix{R}, AbstractSparseMatrix{R}, 
                        
    # Theoretical mean identical for 2nd and 3rd order pruned solution.
    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Theoretical mean available only for first order, pruned second and pruned third order perturbation solutions."

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    
    _nsss_result = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    SS_and_pars = _nsss_result[1]::Vector{R}
    solution_error = _nsss_result[2][1]
    
    if algorithm == :first_order
        mean_of_variables = SS_and_pars[1:T.nVars]

        solved = solution_error < opts.tol.nsss.acceptance_tol
    else
        ensure_moments_constants!(constants)
        so = constants.second_order
        ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix
        
        𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                            constants,
                                                            𝓂.workspaces,
                                                            𝓂.caches;
                                                            initial_guess = 𝓂.caches.qme_solution,
                                                            opts = opts,
                                                            parameter_values = parameters)
        
        update_perturbation_counter!(𝓂.counters, solved, order = 1)

        if !solved 
            mean_of_variables = fill(R(NaN), T.nVars)
        else
            ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)# * 𝓂.constants.second_order.𝐔∇₂
            
            𝐒₂, solved = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                        opts = opts, parameter_values = parameters)

            update_perturbation_counter!(𝓂.counters, solved, order = 2)

            if !solved 
                mean_of_variables = fill(R(NaN), T.nVars)
            else
                𝐒₂ *= 𝓂.constants.second_order.𝐔₂

                𝐒₂ = sparse(𝐒₂) # ensure stable sparse type

                nᵉ = T.nExo
                nˢ = T.nPast_not_future_and_mixed

                kron_states = so.kron_states
                kron_shocks = so.kron_e_e
                kron_volatility = so.kron_v_v

                # first order
                states_to_variables¹ = sparse(𝐒₁[:,1:T.nPast_not_future_and_mixed])

                states_to_states¹ = 𝐒₁[T.past_not_future_and_mixed_idx, 1:T.nPast_not_future_and_mixed]
                shocks_to_states¹ = 𝐒₁[T.past_not_future_and_mixed_idx, (T.nPast_not_future_and_mixed + 1):end]

                # second order
                states_to_variables²        = 𝐒₂[:, kron_states]
                shocks_to_variables²        = 𝐒₂[:, kron_shocks]
                volatility_to_variables²    = 𝐒₂[:, kron_volatility]

                states_to_states²       = 𝐒₂[T.past_not_future_and_mixed_idx, kron_states] |> collect
                shocks_to_states²       = 𝐒₂[T.past_not_future_and_mixed_idx, kron_shocks]
                volatility_to_states²   = 𝐒₂[T.past_not_future_and_mixed_idx, kron_volatility]

                kron_states_to_states¹ = ℒ.kron(states_to_states¹, states_to_states¹) |> collect
                kron_shocks_to_states¹ = ℒ.kron(shocks_to_states¹, shocks_to_states¹)

                n_sts = T.nPast_not_future_and_mixed

                # Set up in pruned state transition matrices
                pruned_states_to_pruned_states = [  states_to_states¹       zeros(R,n_sts, n_sts)   zeros(R,n_sts, n_sts^2)
                                                    zeros(R,n_sts, n_sts)   states_to_states¹       states_to_states² / 2
                                                    zeros(R,n_sts^2, 2 * n_sts)                     kron_states_to_states¹   ]

                pruned_states_to_variables = [states_to_variables¹  states_to_variables¹  states_to_variables² / 2]

                vec_Iₑ = so.vec_Iₑ
                pruned_states_vol_and_shock_effect = [  zeros(R,n_sts) 
                                                        vec(volatility_to_states²) / 2 + shocks_to_states² / 2 * vec_Iₑ
                                                        kron_shocks_to_states¹ * vec_Iₑ]

                variables_vol_and_shock_effect = (vec(volatility_to_variables²) + shocks_to_variables² * vec_Iₑ) / 2

                ## First-order moments, ie mean of variables
                mean_of_pruned_states   = (ℒ.I(size(pruned_states_to_pruned_states, 1)) - pruned_states_to_pruned_states) \ pruned_states_vol_and_shock_effect
                mean_of_variables   = SS_and_pars[1:T.nVars] + pruned_states_to_variables * mean_of_pruned_states + variables_vol_and_shock_effect
            end
        end
    end

    return mean_of_variables, solved
    # return mean_of_variables, 𝐒₁, ∇₁, 𝐒₂, ∇₂, true
end


function calculate_second_order_moments(parameters::Vector{R}, 
                                        𝓂::ℳ;
                                        opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{R}, Vector{R}, Matrix{R}, Matrix{R}, Vector{R}, Matrix{R}, Matrix{R}, AbstractSparseMatrix{R,Int}, AbstractSparseMatrix{R,Int}, Bool} where R <: Real

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars, solved = calculate_covariance(parameters, 𝓂, opts = opts)

    if solved
        # Initialize constants at entry point
        constants = initialise_constants!(𝓂)
        ensure_moments_constants!(constants)
        so = constants.second_order
        T = constants.post_model_macro
        nᵉ = T.nExo

        nˢ = T.nPast_not_future_and_mixed

        iˢ = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx

        Σᶻ₁ = Σʸ₁[iˢ, iˢ]

        # precalc second order
        ## mean
        I_plus_s_s = so.I_plus_s_s

        ## covariance
        e⁴ = so.e4

        # second order
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)# * 𝓂.constants.second_order.𝐔∇₂

        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                opts = opts, parameter_values = parameters)
        
        update_perturbation_counter!(𝓂.counters, solved2, order = 2)

        if solved2
            𝐒₂ *= 𝓂.constants.second_order.𝐔₂

            𝐒₂ = sparse(𝐒₂) # ensure stable sparse type

            kron_s_s = so.kron_states
            kron_e_e = so.kron_e_e
            kron_v_v = so.kron_v_v
            kron_s_e = so.kron_s_e

            # first order
            s_to_y₁ = 𝐒₁[:, 1:nˢ]
            e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]
            
            s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
            e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


            # second order
            s_s_to_y₂ = 𝐒₂[:, kron_s_s]
            e_e_to_y₂ = 𝐒₂[:, kron_e_e]
            v_v_to_y₂ = 𝐒₂[:, kron_v_v]
            s_e_to_y₂ = 𝐒₂[:, kron_s_e]

            s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
            e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
            v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
            s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

            # Compression matrices
            sub_idx = ensure_moments_substate_indices!(𝓂, nˢ)
            D₂ˢ = sub_idx.D₂ˢ
            L₂ˢ = sub_idx.L₂ˢ
            n₂ˢ = size(D₂ˢ, 2)  # nˢ(nˢ+1)/2

            s_to_s₁_by_s_to_s₁ = L₂ˢ * ℒ.kron(s_to_s₁, s_to_s₁) * D₂ˢ
            e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
            s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

            # # Set up in pruned state transition matrices (block 3 compressed: nˢ² → n₂ˢ)
            ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + n₂ˢ)
                        zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2 * D₂ˢ
                        zeros(n₂ˢ, 2*nˢ)    s_to_s₁_by_s_to_s₁                  ]

            ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                        zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                        zeros(n₂ˢ,nᵉ)   L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁]

            ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2 * D₂ˢ]

            ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

            vec_Iₑ = so.vec_Iₑ
            ŝv₂ = [ zeros(nˢ) 
                    vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec_Iₑ
                    L₂ˢ * e_to_s₁_by_e_to_s₁ * vec_Iₑ]

            yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec_Iₑ) / 2

            ## Mean
            μˢ⁺₂ = (ℒ.I(size(ŝ_to_ŝ₂, 1)) - ŝ_to_ŝ₂) \ ŝv₂
            Δμˢ₂ = vec((ℒ.I(size(s_to_s₁, 1)) - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec_Iₑ) / 2))
            μʸ₂  = SS_and_pars[1:𝓂.constants.post_model_macro.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

            slvd = solved && solved2
        else
            μʸ₂ = zeros(R,0)
            Δμˢ₂ = zeros(R,0)
            # Σʸ₁ = zeros(R,0,0)
            # Σᶻ₁ = zeros(R,0,0)
            # SS_and_pars = zeros(R,0)
            # 𝐒₁ = zeros(R,0,0)
            # ∇₁ = zeros(R,0,0)
            # 𝐒₂ = spzeros(R,0,0)
            # ∇₂ = spzeros(R,0,0)
            slvd = solved2
        end
    else
        μʸ₂ = zeros(R,0)
        Δμˢ₂ = zeros(R,0)
        # Σʸ₁ = zeros(R,0,0)
        Σᶻ₁ = zeros(R,0,0)
        # SS_and_pars = zeros(R,0)
        # 𝐒₁ = zeros(R,0,0)
        # ∇₁ = zeros(R,0,0)
        𝐒₂ = spzeros(R,0,0)
        ∇₂ = spzeros(R,0,0)
        slvd = solved
    end

    return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, slvd
end



function calculate_second_order_moments_with_covariance(parameters::Vector{R}, 𝓂::ℳ;
                                                        opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{R}, Matrix{R}, Vector{R}, Vector{R}, Matrix{R}, Matrix{R}, Matrix{R}, Matrix{R}, Matrix{R}, Vector{R}, Matrix{R}, Matrix{R}, AbstractMatrix{R}, AbstractSparseMatrix{R,Int}, Bool} where R <: Real

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars, solved = calculate_covariance(parameters, 𝓂, opts = opts)

    if solved
        ensure_moments_constants!(𝓂.constants)
        so = 𝓂.constants.second_order
        nᵉ = 𝓂.constants.post_model_macro.nExo

        nˢ = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed

        iˢ = 𝓂.constants.post_model_macro.past_not_future_and_mixed_idx

        Σᶻ₁ = Σʸ₁[iˢ, iˢ]

        # precalc second order
        ## mean
        I_plus_s_s = so.I_plus_s_s

        ## covariance
        e⁴ = so.e4

        # second order
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian, 𝓂.workspaces)# * 𝓂.constants.second_order.𝐔∇₂

        𝐒₂_raw, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                opts = opts, parameter_values = parameters)

        update_perturbation_counter!(𝓂.counters, solved2, order = 2)
        
        if solved2
            𝐒₂ = (sparse(𝐒₂_raw) * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{R, Int}

            kron_s_s = so.kron_states
            kron_e_e = so.kron_e_e
            kron_v_v = so.kron_v_v
            kron_s_e = so.kron_s_e

            # Substate duplication/elimination matrices for symmetric Kronecker compression
            sub_idx = ensure_moments_substate_indices!(𝓂, nˢ)
            D₂ˢ = sub_idx.D₂ˢ
            L₂ˢ = sub_idx.L₂ˢ
            n₂ˢ = size(D₂ˢ, 2)  # nˢ(nˢ+1)/2

            # first order
            s_to_y₁ = 𝐒₁[:, 1:nˢ]
            e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]
            
            s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
            e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


            # second order
            s_s_to_y₂ = 𝐒₂[:, kron_s_s]
            e_e_to_y₂ = 𝐒₂[:, kron_e_e]
            v_v_to_y₂ = 𝐒₂[:, kron_v_v]
            s_e_to_y₂ = 𝐒₂[:, kron_s_e]

            s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
            e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
            v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
            s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

            s_to_s₁_by_s_to_s₁ = L₂ˢ * ℒ.kron(s_to_s₁, s_to_s₁) * D₂ˢ
            e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
            s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

            # # Set up in pruned state transition matrices (block 3 compressed: nˢ² → n₂ˢ)
            ŝ_to_ŝ₂ = [ s_to_s₁             spzeros(nˢ, nˢ + n₂ˢ)
                        spzeros(nˢ, nˢ)     s_to_s₁             s_s_to_s₂ / 2 * D₂ˢ
                        spzeros(n₂ˢ, 2*nˢ)  s_to_s₁_by_s_to_s₁                  ]

            ê_to_ŝ₂ = [ e_to_s₁         spzeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                        spzeros(nˢ,nᵉ)  e_e_to_s₂ / 2       s_e_to_s₂
                        spzeros(n₂ˢ,nᵉ) L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁]

            ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2 * D₂ˢ]

            ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

            vec_Iₑ = so.vec_Iₑ
            ŝv₂ = [ zeros(nˢ) 
                    vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec_Iₑ
                    L₂ˢ * e_to_s₁_by_e_to_s₁ * vec_Iₑ]

            yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec_Iₑ) / 2

            ## Mean
            μˢ⁺₂ = collect(ℒ.I(size(ŝ_to_ŝ₂, 1)) - ŝ_to_ŝ₂) \ ŝv₂
            Δμˢ₂ = vec((ℒ.I(size(s_to_s₁, 1)) - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec_Iₑ) / 2))
            μʸ₂  = SS_and_pars[1:𝓂.constants.post_model_macro.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

            # Covariance
            Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
                    zeros(nᵉ^2, nᵉ)    so.e4_minus_vecIₑ_outer     zeros(nᵉ^2, nᵉ * nˢ)
                    zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

            C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

            # Check 2nd-order Lyapunov cache
            cached_covar_2nd = 𝓂.caches.covariance_second_order
            n_ŝ₂ = size(ŝ_to_ŝ₂, 1)
            if R === Float64 && !isempty(parameters) &&
               cache_valid_for_parameters(𝓂.caches.valid_for.covariance_second_order, parameters) &&
               !isempty(cached_covar_2nd) && size(cached_covar_2nd) == (n_ŝ₂, n_ŝ₂)
                Σᶻ₂ = cached_covar_2nd
                info = true
            else
                # Ensure second-order lyapunov workspace and solve
                lyap_ws_2nd = ensure_lyapunov_workspace!(𝓂.workspaces, n_ŝ₂, :second_order)

                Σᶻ₂, info = solve_lyapunov_equation(ŝ_to_ŝ₂, C, lyap_ws_2nd,
                                        initial_guess = cached_covar_2nd,
                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                        tol = opts.tol.second_order.lyapunov,
                                        verbose = opts.verbose,
                                        has_unit_roots = 𝓂.caches.has_unit_roots)

                # Cache the result for reuse
                if R === Float64 && info
                    if size(𝓂.caches.covariance_second_order) != size(Σᶻ₂)
                        𝓂.caches.covariance_second_order = Matrix{Float64}(undef, size(Σᶻ₂)...)
                    end
                    copyto!(𝓂.caches.covariance_second_order, Σᶻ₂)
                    𝓂.caches.valid_for.covariance_second_order = Float64.(parameters)
                end
            end

            if info
                Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

                autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'

                slvd = solved && solved2 && info
            else
                nVars = 𝓂.constants.post_model_macro.nVars
                Σʸ₂ = fill(R(NaN), nVars, nVars)
                Σᶻ₂ = zeros(R,0,0)
                μʸ₂ = fill(R(NaN), nVars)
                Δμˢ₂ = zeros(R,0)
                autocorr_tmp = zeros(R,0,0)
                ŝ_to_ŝ₂ = zeros(R,0,0)
                ŝ_to_y₂ = zeros(R,0,0)
                slvd = info
            end
        else
            nVars = 𝓂.constants.post_model_macro.nVars
            Σʸ₂ = fill(R(NaN), nVars, nVars)
            Σᶻ₂ = zeros(R,0,0)
            μʸ₂ = fill(R(NaN), nVars)
            Δμˢ₂ = zeros(R,0)
            autocorr_tmp = zeros(R,0,0)
            ŝ_to_ŝ₂ = zeros(R,0,0)
            ŝ_to_y₂ = zeros(R,0,0)
            slvd = solved2
        end
    else
        nVars = 𝓂.constants.post_model_macro.nVars
        Σʸ₂ = fill(R(NaN), nVars, nVars)
        Σᶻ₂ = zeros(R,0,0)
        μʸ₂ = fill(R(NaN), nVars)
        Δμˢ₂ = zeros(R,0)
        autocorr_tmp = zeros(R,0,0)
        ŝ_to_ŝ₂ = zeros(R,0,0)
        ŝ_to_y₂ = zeros(R,0,0)
        # Σʸ₁ = zeros(R,0,0)
        Σᶻ₁ = zeros(R,0,0)
        # SS_and_pars = zeros(R,0)
        # 𝐒₁ = zeros(R,0,0)
        # ∇₁ = zeros(R,0,0)
        𝐒₂_raw = zeros(R,0,0)
        ∇₂ = spzeros(R,0,0)
        slvd = solved
    end

    return Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, slvd
end


# Block-triangular Lyapunov solver for third-order pruned state covariance.
# Solves the block-triangular Lyapunov equation for the third-order pruned state covariance.
# Accepts pre-sliced sub-blocks of the transition matrix [A_UU 0; A_LU A_LL]
# and RHS matrix C (only C_LU and C_LL blocks needed).
# Reuses Σ̂ᶻ₂ for the upper block; Sylvester for the cross-block; Lyapunov for the lower block.
function solve_block_triangular_lyapunov(A_UU::AbstractMatrix{T},
                                          A_LU::AbstractMatrix{T},
                                          A_LL::AbstractMatrix{T},
                                          C_LU::AbstractMatrix{T},
                                          C_LL::AbstractMatrix{T},
                                          Σᶻ₂_upper::AbstractMatrix{T},
                                          𝓂_workspaces::workspaces,
                                          opts::CalculationOptions;
                                          n₃ˢ::Int = 0) where T <: Real
    N_upper = size(A_UU, 1)
    N_lower = size(A_LL, 1)

    # Step 1: X_UU = Σ̂ᶻ₂ (already solved)
    X_UU = Σᶻ₂_upper

    # Step 2: X_LU via discrete Sylvester  (A_LL X_LU A_UU' + RHS = X_LU)
    RHS_LU = A_LU * X_UU * A_UU' + C_LU
    
    sylv_ws = 𝓂_workspaces.sylvester_block
    X_LU, sylv_solved = solve_sylvester_equation(A_LL, A_UU', RHS_LU, sylv_ws,
                                                  tol = opts.tol.third_order.sylvester,
                                                  verbose = opts.verbose)

    # Step 3: X_LL via Lyapunov with modified RHS
    C_LL_mod = C_LL +
               A_LU * X_UU  * A_LU' +
               A_LL * X_LU  * A_LU' +
               A_LU * X_LU' * A_LL'

    if n₃ˢ > 0 && N_lower > n₃ˢ
        # A_LL has sub-block structure: decompose into A₆₆ (lower-right n₃ˢ×n₃ˢ) and upper blocks
        n_upper_LL = N_lower - n₃ˢ
        ru_ll = 1:n_upper_LL
        rl_ll = (n_upper_LL+1):N_lower

        A_LL_UU = A_LL[ru_ll, ru_ll]
        A_LL_UL = A_LL[ru_ll, rl_ll]
        A_LL_LL = A_LL[rl_ll, rl_ll]

        C_mod_UU = C_LL_mod[ru_ll, ru_ll]
        C_mod_UL = C_LL_mod[ru_ll, rl_ll]
        C_mod_LL = C_LL_mod[rl_ll, rl_ll]

        # Step 3a: X₆₆ via standard Lyapunov
        lyap_ws_66 = ensure_lyapunov_workspace!(𝓂_workspaces, n₃ˢ, :block)
        X_66, _ = solve_lyapunov_equation(A_LL_LL, C_mod_LL, lyap_ws_66,
                              tol = opts.tol.third_order.lyapunov,
                              verbose = opts.verbose)

        # Step 3b: X_{upper,6} via Sylvester
        RHS_UL6 = A_LL_UL * X_66 * A_LL_LL' + C_mod_UL
        X_UL6, _ = solve_sylvester_equation(A_LL_UU, A_LL_LL', RHS_UL6, sylv_ws,
                             tol = opts.tol.third_order.sylvester,
                             verbose = opts.verbose)

        # Step 3c: X_{upper,upper} via Lyapunov
        C_UU_mod2 = C_mod_UU +
                     A_LL_UL * X_66 * A_LL_UL' +
                     A_LL_UU * X_UL6 * A_LL_UL' +
                     A_LL_UL * X_UL6' * A_LL_UU'

        lyap_ws_inner = ensure_lyapunov_workspace!(𝓂_workspaces, n_upper_LL, :block)
        X_UU_LL, _ = solve_lyapunov_equation(A_LL_UU, C_UU_mod2, lyap_ws_inner,
                              tol = opts.tol.third_order.lyapunov,
                              verbose = opts.verbose)

        X_LL = zeros(T, N_lower, N_lower)
        X_LL[ru_ll, ru_ll] = X_UU_LL
        X_LL[ru_ll, rl_ll] = X_UL6
        X_LL[rl_ll, ru_ll] = X_UL6'
        X_LL[rl_ll, rl_ll] = X_66
    else
        # Standard Lyapunov on full lower block
        lyap_ws = ensure_lyapunov_workspace!(𝓂_workspaces, N_lower, :block)
        X_LL_result, _ = solve_lyapunov_equation(A_LL, C_LL_mod, lyap_ws,
                              tol = opts.tol.third_order.lyapunov,
                              verbose = opts.verbose)
        X_LL = X_LL_result
    end

    # Reassemble full solution
    N = N_upper + N_lower
    ru = 1:N_upper
    rl = (N_upper+1):N
    Σᶻ₃ = Matrix{T}(undef, N, N)
    Σᶻ₃[ru, ru] = X_UU
    Σᶻ₃[ru, rl] = X_LU'
    Σᶻ₃[rl, ru] = X_LU
    Σᶻ₃[rl, rl] = X_LL

    return Σᶻ₃, sylv_solved
end


function calculate_third_order_moments_with_autocorrelation(parameters::Vector{T}, 
                                            observables::Union{Symbol_input,String_input},
                                            𝓂::ℳ; 
                                            autocorrelation_periods::U = 1:5,
                                            third_order_block_lyapunov_method::Bool = false,
                                            covariance::Union{Symbol_input,String_input} = Symbol[],
                                            opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{T}, Vector{T}, Matrix{T}, Vector{T}, Bool} where {U, T <: Real}

    second_order_moments = calculate_second_order_moments_with_covariance(parameters, 𝓂; opts = opts)

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = second_order_moments

    if !solved
        return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
    end

    # Check 3rd-order autocorrelation cache
    nVars_ac = 𝓂.constants.post_model_macro.nVars
    obs_key_ac = if observables == :full_covar
        collect(1:nVars_ac)
    else
        obs_idx_ac = parse_variables_input_to_index(observables, 𝓂.constants) |> sort
        if covariance == Symbol[]
            collect(obs_idx_ac)
        else
            covar_idx_ac = parse_variables_input_to_index(covariance, 𝓂.constants) |> sort
            sort(union(obs_idx_ac, covar_idx_ac))
        end
    end
    ac_periods = collect(Int, autocorrelation_periods)
    cached_covar_ac = 𝓂.caches.covariance_third_order
    cached_autocorr = 𝓂.caches.covariance_third_order_autocorr
    if T === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.covariance_third_order_autocorr, parameters) &&
       !isempty(cached_covar_ac) && size(cached_covar_ac) == (nVars_ac, nVars_ac) &&
       !isempty(cached_autocorr) && size(cached_autocorr, 1) == nVars_ac &&
       𝓂.caches.valid_for.covariance_third_order_autocorr_obs_key == obs_key_ac &&
       𝓂.caches.valid_for.covariance_third_order_autocorr_periods == ac_periods
        return cached_covar_ac, μʸ₂, cached_autocorr, SS_and_pars, true
    end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐒₂ = (sparse(𝐒₂_raw) * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{T, Int}

    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)# * 𝓂.constants.third_order.𝐔∇₃

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw, 
                                                𝓂.constants,
                                                𝓂.workspaces,
                                                𝓂.caches;
                                                initial_guess = 𝓂.caches.third_order_solution,
                                                opts = opts, parameter_values = parameters)

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3
        return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
    end

    𝐒₃ *= 𝓂.constants.third_order.𝐔₃

    𝐒₃ = sparse(𝐒₃) # ensure stable sparse type
    
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃, 𝓂.constants, observables, covariance = covariance, tol = opts.tol.third_order.dependencies_tol)

    nᵉ = 𝓂.constants.post_model_macro.nExo

    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v
    kron_e_v = to.kron_e_v
    e_in_s⁺ = so.e_in_s⁺
    v_in_s⁺ = so.v_in_s⁺

    # precalc second order
    ## covariance
    e⁴ = so.e4

    # precalc third order
    e⁶ = to.e6

    # cached reshaped matrices and vec(I)
    vec_Iₑ = so.vec_Iₑ
    e4_nᵉ²_nᵉ² = so.e4_nᵉ²_nᵉ²
    e4_nᵉ_nᵉ³ = so.e4_nᵉ_nᵉ³
    e4_minus_vecIₑ_outer = so.e4_minus_vecIₑ_outer
    e6_nᵉ³_nᵉ³ = to.e6_nᵉ³_nᵉ³

    # Expand compressed Σᶻ₂ (block 3 is vech-compressed) back to full form for third-order indexing
    nˢ_full = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    sub_idx_full = ensure_moments_substate_indices!(𝓂, nˢ_full)
    D₂ˢ_full = sub_idx_full.D₂ˢ
    n₂ˢ_full = size(D₂ˢ_full, 2)
    E₂_exp = [sparse(ℒ.I, 2*nˢ_full, 2*nˢ_full)  spzeros(2*nˢ_full, n₂ˢ_full)
              spzeros(nˢ_full^2, 2*nˢ_full)        D₂ˢ_full]
    Σᶻ₂ = E₂_exp * Σᶻ₂ * E₂_exp'

    Σʸ₃ = zeros(T, size(Σʸ₂))

    autocorr = zeros(T, size(Σʸ₂,1), length(autocorrelation_periods))

    solved_lyapunov = true

    # Threads.@threads for ords in orders 
    for ords in orders 
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)

        sort!(dependencies_all_vars)

        dependencies = intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, 𝓂.constants.post_model_macro.var)

        dependencies_in_states_idx = indexin(dependencies, 𝓂.constants.post_model_macro.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.constants.post_model_macro.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx, 
                dependencies_in_states_idx .+ 𝓂.constants.post_model_macro.nPast_not_future_and_mixed, 
                findall(ℒ.kron(𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed,dependencies),), 𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed,dependencies),))) .+ 2*𝓂.constants.post_model_macro.nPast_not_future_and_mixed)
        
        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]
        
        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))

        substate_indices = ensure_moments_substate_indices!(𝓂, nˢ)
        I_plus_s_s = substate_indices.I_plus_s_s
        e_es = substate_indices.e_es
        e_ss = substate_indices.e_ss
        ss_s = substate_indices.ss_s
        s_s = substate_indices.s_s
        D₂ˢ = substate_indices.D₂ˢ
        L₂ˢ = substate_indices.L₂ˢ
        D₃ˢ = substate_indices.D₃ˢ
        L₃ˢ = substate_indices.L₃ˢ
        n₂ˢ = size(D₂ˢ, 2)
        n₃ˢ = size(D₃ˢ, 2)

        # first order
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1):end]
        
        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1):end]

        # second order
        dep_kron = ensure_moments_dependency_kron_indices!(𝓂, dependencies, s_in_s⁺)
        kron_s_s = dep_kron.kron_s_s
        kron_s_e = dep_kron.kron_s_e

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)
        s_to_s₁_by_s_to_s₁_c = L₂ˢ * s_to_s₁_by_s_to_s₁ * D₂ˢ

        # third order
        kron_s_v = dep_kron.kron_s_v

        s_s_s_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

        s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]

        # Set up pruned state transition sub-blocks
        N_upper = 2 * nˢ + n₂ˢ
        N_lower = nˢ + nˢ^2 + n₃ˢ

        A_UU = [s_to_s₁                spzeros(nˢ, nˢ + n₂ˢ)
                spzeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2 * D₂ˢ
                spzeros(n₂ˢ, 2 * nˢ)               s_to_s₁_by_s_to_s₁_c]

        A_LU = [s_v_v_to_s₃ / 2                    spzeros(nˢ, nˢ + n₂ˢ)
                ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    spzeros(nˢ^2, nˢ + n₂ˢ)
                spzeros(n₃ˢ, 2 * nˢ + n₂ˢ)]

        A_LL = [s_to_s₁           s_s_to_s₂             s_s_s_to_s₃ / 6 * D₃ˢ
                spzeros(nˢ^2, nˢ) s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2) * D₃ˢ
                spzeros(n₃ˢ, nˢ + nˢ^2)               L₃ˢ * ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁) * D₃ˢ]

        ê_to_ŝ₃ = [ e_to_s₁   spzeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            spzeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   spzeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            spzeros(n₂ˢ,nᵉ)  L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁  spzeros(n₂ˢ, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            e_v_v_to_s₃ / 2    spzeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                            ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    spzeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                            spzeros(n₃ˢ, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) L₃ˢ * (ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss)   L₃ˢ * (ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es)  L₃ˢ * ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2 * D₂ˢ   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6 * D₃ˢ]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I(size(s_to_s₁_by_s_to_s₁, 1)) - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec_Iₑ) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * e4_nᵉ_nᵉ³' / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    e4_nᵉ_nᵉ³
                spzeros(nᵉ^2, nᵉ)    e4_minus_vecIₑ_outer     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), e4_nᵉ_nᵉ³)
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)    spzeros(nˢ*nᵉ^2,nᵉ^3)
                e4_nᵉ_nᵉ³'  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', e4_nᵉ_nᵉ³')     ℒ.kron(vec(Σ̂ᶻ₁)', e4_nᵉ_nᵉ³')  spzeros(nᵉ^3, nˢ*nᵉ^2)     e6_nᵉ³_nᵉ³]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   spzeros(nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]
        
        droptol!(A_UU, eps())
        droptol!(A_LU, eps())
        droptol!(A_LL, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        # Third-order Lyapunov solve
        if third_order_block_lyapunov_method
            # Block-triangular: reuse second-order covariance
            E₂_comp = [sparse(ℒ.I, 2*nˢ, 2*nˢ)  spzeros(2*nˢ, nˢ^2)
                        spzeros(n₂ˢ, 2*nˢ)       L₂ˢ]
            Σ̂ᶻ₂_compressed = E₂_comp * Σ̂ᶻ₂ * E₂_comp'

            # Compute C sub-blocks directly (avoid building full N×N matrix)
            ê_U = ê_to_ŝ₃[1:N_upper, :]
            ê_L = ê_to_ŝ₃[(N_upper+1):end, :]
            E_cU = Eᴸᶻ[:, 1:N_upper]
            E_cL = Eᴸᶻ[:, (N_upper+1):end]

            Q = E_cU * A_LU' + E_cL * A_LL'
            R = E_cU * A_UU'
            C_LU = ê_L * (Γ₃ * ê_U' + R) + Q' * ê_U'
            eQ = ê_L * Q
            C_LL = sparse_ABAt(ê_L, Γ₃) + eQ + eQ'
            droptol!(C_LU, eps())
            droptol!(C_LL, eps())

            Σᶻ₃, info = solve_block_triangular_lyapunov(A_UU, A_LU, A_LL, C_LU, C_LL,
                                                          Σ̂ᶻ₂_compressed,
                                                          𝓂.workspaces, opts,
                                                          n₃ˢ = n₃ˢ)

            # Assemble full ŝ_to_ŝ₃ (needed for autocorrelation)
            ŝ_to_ŝ₃ = [A_UU spzeros(N_upper, N_lower); A_LU A_LL]
        else
            ŝ_to_ŝ₃ = [A_UU spzeros(N_upper, N_lower); A_LU A_LL]

            A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
            droptol!(A, eps())

            C = sparse_ABAt(ê_to_ŝ₃, Γ₃) + A + A'
            droptol!(C, eps())

            lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)
            Σᶻ₃, info = solve_lyapunov_equation(ŝ_to_ŝ₃, C, lyap_ws_3rd,
                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                        tol = opts.tol.third_order.lyapunov,
                                        verbose = opts.verbose,
                                        has_unit_roots = 𝓂.caches.has_unit_roots)
        end

        if !info
            return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + sparse_ABAt(ê_to_y₃, Γ₃) + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.constants.post_model_macro.var), indexin(variance_observable, 𝓂.constants.post_model_macro.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end

        autocorr_tmp = ŝ_to_ŝ₃ * Eᴸᶻ' * ê_to_y₃' + ê_to_ŝ₃ * Γ₃ * ê_to_y₃'

        s_to_s₁ⁱ = zero(s_to_s₁)
        s_to_s₁ⁱ += ℒ.diagm(ones(nˢ))

        ŝ_to_ŝ₃ⁱ = zero(ŝ_to_ŝ₃)
        ŝ_to_ŝ₃ⁱ += ℒ.diagm(ones(size(Σᶻ₃,1)))

        Σᶻ₃ⁱ = Σᶻ₃

        for i in autocorrelation_periods
            Σᶻ₃ⁱ .= ŝ_to_ŝ₃ * Σᶻ₃ⁱ + ê_to_ŝ₃ * Eᴸᶻ
            s_to_s₁ⁱ *= s_to_s₁

            Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
            ℒ.kron(s_to_s₁ⁱ * Σ̂ᶻ₁,vec_Iₑ)   spzeros(nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(s_to_s₁ⁱ * μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(s_to_s₁ⁱ * reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(s_to_s₁ⁱ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
            spzeros(nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]

            for obs in variance_observable
                autocorr[indexin([obs], 𝓂.constants.post_model_macro.var), i] .= ℒ.diag(ŝ_to_y₃ * Σᶻ₃ⁱ * ŝ_to_y₃' + ŝ_to_y₃ * ŝ_to_ŝ₃ⁱ * autocorr_tmp + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃')[indexin([obs], variance_observable)] ./ max.(ℒ.diag(Σʸ₃tmp), eps(Float64))[indexin([obs], variance_observable)]

                autocorr[indexin([obs], 𝓂.constants.post_model_macro.var), i][ℒ.diag(Σʸ₃tmp)[indexin([obs], variance_observable)] .< opts.tol.third_order.lyapunov.acceptance_tol] .= 0
            end

            ŝ_to_ŝ₃ⁱ *= ŝ_to_ŝ₃
        end

    end

    # Compute obs_key for cache storage
    nVars_autocorr = 𝓂.constants.post_model_macro.nVars
    obs_key_autocorr = if observables == :full_covar
        collect(1:nVars_autocorr)
    else
        obs_idx = parse_variables_input_to_index(observables, 𝓂.constants) |> sort
        if covariance == Symbol[]
            collect(obs_idx)
        else
            covar_idx = parse_variables_input_to_index(covariance, 𝓂.constants) |> sort
            sort(union(obs_idx, covar_idx))
        end
    end

    # Cache the 3rd-order covariance for reuse (also benefits calculate_third_order_moments)
    all_solved = solved && solved3 && solved_lyapunov
    if T === Float64 && all_solved
        if size(𝓂.caches.covariance_third_order) != size(Σʸ₃)
            𝓂.caches.covariance_third_order = Matrix{Float64}(undef, size(Σʸ₃)...)
        end
        copyto!(𝓂.caches.covariance_third_order, Σʸ₃)
        𝓂.caches.valid_for.covariance_third_order = Float64.(parameters)
        𝓂.caches.valid_for.covariance_third_order_obs_key = obs_key_autocorr

        # Cache autocorrelation
        if size(𝓂.caches.covariance_third_order_autocorr) != size(autocorr)
            𝓂.caches.covariance_third_order_autocorr = Matrix{Float64}(undef, size(autocorr)...)
        end
        copyto!(𝓂.caches.covariance_third_order_autocorr, autocorr)
        𝓂.caches.valid_for.covariance_third_order_autocorr = Float64.(parameters)
        𝓂.caches.valid_for.covariance_third_order_autocorr_obs_key = obs_key_autocorr
        𝓂.caches.valid_for.covariance_third_order_autocorr_periods = collect(Int, autocorrelation_periods)
    end

    return Σʸ₃, μʸ₂, autocorr, SS_and_pars, all_solved
end

function calculate_third_order_moments(parameters::Vector{T}, 
                                            observables::Union{Symbol_input,String_input},
                                            𝓂::ℳ;
                                            covariance::Union{Symbol_input,String_input} = Symbol[],
                                            third_order_block_lyapunov_method::Bool = false,
                                            opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{T}, Vector{T}, Vector{T}, Bool} where T <: Real
    second_order_moments = calculate_second_order_moments_with_covariance(parameters, 𝓂; opts = opts)

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = second_order_moments

    if !solved
        nVars = 𝓂.constants.post_model_macro.nVars
        return fill(T(NaN), nVars, nVars), fill(T(NaN), nVars), fill(T(NaN), nVars), false
    end

    # Check 3rd-order covariance cache: if valid for current parameters AND same observables/covariance, skip
    nVars_check = 𝓂.constants.post_model_macro.nVars
    obs_key = if observables == :full_covar
        collect(1:nVars_check)
    else
        obs_idx = parse_variables_input_to_index(observables, 𝓂.constants) |> sort
        if covariance == Symbol[]
            collect(obs_idx)
        else
            covar_idx = parse_variables_input_to_index(covariance, 𝓂.constants) |> sort
            sort(union(obs_idx, covar_idx))
        end
    end
    cached_covar_3rd = 𝓂.caches.covariance_third_order
    if T === Float64 && !isempty(parameters) &&
       cache_valid_for_parameters(𝓂.caches.valid_for.covariance_third_order, parameters) &&
       !isempty(cached_covar_3rd) && size(cached_covar_3rd) == (nVars_check, nVars_check) &&
       𝓂.caches.valid_for.covariance_third_order_obs_key == obs_key
        return cached_covar_3rd, μʸ₂, SS_and_pars, true
    end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐒₂ = (sparse(𝐒₂_raw) * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{T, Int}

    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives, 𝓂.workspaces)# * 𝓂.constants.third_order.𝐔∇₃

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw, 
                                                𝓂.constants,
                                                𝓂.workspaces,
                                                𝓂.caches;
                                                initial_guess = 𝓂.caches.third_order_solution,
                                                opts = opts, parameter_values = parameters)

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)
    
    if !solved3
        nVars = 𝓂.constants.post_model_macro.nVars
        return fill(T(NaN), nVars, nVars), fill(T(NaN), nVars), fill(T(NaN), nVars), false
    end

    𝐒₃ *= 𝓂.constants.third_order.𝐔₃

    𝐒₃ = sparse(𝐒₃) # ensure stable sparse type
    
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃, 𝓂.constants, observables, covariance = covariance, tol = opts.tol.third_order.dependencies_tol)

    nᵉ = 𝓂.constants.post_model_macro.nExo

    kron_e_e = so.kron_e_e
    kron_v_v = so.kron_v_v
    kron_e_v = to.kron_e_v
    e_in_s⁺ = so.e_in_s⁺
    v_in_s⁺ = so.v_in_s⁺

    # precalc second order
    ## covariance
    e⁴ = so.e4

    # precalc third order
    e⁶ = to.e6

    # cached reshaped matrices and vec(I)
    vec_Iₑ = so.vec_Iₑ
    e4_nᵉ²_nᵉ² = so.e4_nᵉ²_nᵉ²
    e4_nᵉ_nᵉ³ = so.e4_nᵉ_nᵉ³
    e4_minus_vecIₑ_outer = so.e4_minus_vecIₑ_outer
    e6_nᵉ³_nᵉ³ = to.e6_nᵉ³_nᵉ³

    # Expand compressed Σᶻ₂ (block 3 is vech-compressed) back to full form for third-order indexing
    nˢ_full = 𝓂.constants.post_model_macro.nPast_not_future_and_mixed
    sub_idx_full = ensure_moments_substate_indices!(𝓂, nˢ_full)
    D₂ˢ_full = sub_idx_full.D₂ˢ
    n₂ˢ_full = size(D₂ˢ_full, 2)
    E₂_exp = [sparse(ℒ.I, 2*nˢ_full, 2*nˢ_full)  spzeros(2*nˢ_full, n₂ˢ_full)
              spzeros(nˢ_full^2, 2*nˢ_full)        D₂ˢ_full]
    Σᶻ₂ = E₂_exp * Σᶻ₂ * E₂_exp'

    Σʸ₃ = zeros(T, size(Σʸ₂))

    solved_lyapunov = true

    # Threads.@threads for ords in orders 
    for ords in orders 
        variance_observable, dependencies_all_vars = ords

        sort!(variance_observable)

        sort!(dependencies_all_vars)

        dependencies = intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed, dependencies_all_vars)

        obs_in_y = indexin(variance_observable, 𝓂.constants.post_model_macro.var)

        dependencies_in_states_idx = indexin(dependencies, 𝓂.constants.post_model_macro.past_not_future_and_mixed)

        dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.constants.post_model_macro.var))

        nˢ = length(dependencies)

        iˢ = dependencies_in_var_idx

        Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

        dependencies_extended_idx = vcat(dependencies_in_states_idx, 
                dependencies_in_states_idx .+ 𝓂.constants.post_model_macro.nPast_not_future_and_mixed, 
                findall(ℒ.kron(𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed,dependencies),), 𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (intersect(𝓂.constants.post_model_macro.past_not_future_and_mixed,dependencies),))) .+ 2*𝓂.constants.post_model_macro.nPast_not_future_and_mixed)
        
        Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]
        
        Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

        s_in_s⁺ = BitVector(vcat(𝓂.constants.post_model_macro.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))

        substate_indices = ensure_moments_substate_indices!(𝓂, nˢ)
        I_plus_s_s = substate_indices.I_plus_s_s
        e_es = substate_indices.e_es
        e_ss = substate_indices.e_ss
        ss_s = substate_indices.ss_s
        s_s = substate_indices.s_s
        D₂ˢ = substate_indices.D₂ˢ
        L₂ˢ = substate_indices.L₂ˢ
        D₃ˢ = substate_indices.D₃ˢ
        L₃ˢ = substate_indices.L₃ˢ
        n₂ˢ = size(D₂ˢ, 2)
        n₃ˢ = size(D₃ˢ, 2)

        # first order
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1):end]
        
        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1):end]

        # second order
        dep_kron = ensure_moments_dependency_kron_indices!(𝓂, dependencies, s_in_s⁺)
        kron_s_s = dep_kron.kron_s_s
        kron_s_e = dep_kron.kron_s_e

        s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
        e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
        s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

        s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
        e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
        v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
        s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

        s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
        e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
        s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)
        s_to_s₁_by_s_to_s₁_c = L₂ˢ * s_to_s₁_by_s_to_s₁ * D₂ˢ

        # third order
        kron_s_v = dep_kron.kron_s_v

        s_s_s_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

        s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
        s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
        s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
        e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
        s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
        e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]

        # Set up pruned state transition sub-blocks
        N_upper = 2 * nˢ + n₂ˢ
        N_lower = nˢ + nˢ^2 + n₃ˢ

        A_UU = [s_to_s₁                spzeros(nˢ, nˢ + n₂ˢ)
                spzeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2 * D₂ˢ
                spzeros(n₂ˢ, 2 * nˢ)               s_to_s₁_by_s_to_s₁_c]

        A_LU = [s_v_v_to_s₃ / 2                    spzeros(nˢ, nˢ + n₂ˢ)
                ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    spzeros(nˢ^2, nˢ + n₂ˢ)
                spzeros(n₃ˢ, 2 * nˢ + n₂ˢ)]

        A_LL = [s_to_s₁           s_s_to_s₂             s_s_s_to_s₃ / 6 * D₃ˢ
                spzeros(nˢ^2, nˢ) s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2) * D₃ˢ
                spzeros(n₃ˢ, nˢ + nˢ^2)               L₃ˢ * ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁) * D₃ˢ]

        ê_to_ŝ₃ = [ e_to_s₁   spzeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            spzeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   spzeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            spzeros(n₂ˢ,nᵉ)  L₂ˢ * e_to_s₁_by_e_to_s₁  L₂ˢ * I_plus_s_s * s_to_s₁_by_e_to_s₁  spzeros(n₂ˢ, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                            e_v_v_to_s₃ / 2    spzeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                            ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    spzeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                            spzeros(n₃ˢ, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) L₃ˢ * (ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss)   L₃ˢ * (ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es)  L₃ˢ * ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2 * D₂ˢ   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6 * D₃ˢ]

        ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

        μˢ₃δμˢ₁ = reshape((ℒ.I(size(s_to_s₁_by_s_to_s₁, 1)) - s_to_s₁_by_s_to_s₁) \ vec( 
                                    (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                    s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                    s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec_Iₑ) / 2 +
                                    s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                    (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                    e_e_e_to_s₃ * e4_nᵉ_nᵉ³' / 6 +
                                    s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                    e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                    ), nˢ, nˢ)

        Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    e4_nᵉ_nᵉ³
                spzeros(nᵉ^2, nᵉ)    e4_minus_vecIₑ_outer     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
                ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, e4_nᵉ_nᵉ³)
                ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), e4_nᵉ_nᵉ³)
                spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, e4_nᵉ²_nᵉ²)    spzeros(nˢ*nᵉ^2,nᵉ^3)
                e4_nᵉ_nᵉ³'  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', e4_nᵉ_nᵉ³')     ℒ.kron(vec(Σ̂ᶻ₁)', e4_nᵉ_nᵉ³')  spzeros(nᵉ^3, nˢ*nᵉ^2)     e6_nᵉ³_nᵉ³]


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   spzeros(nˢ*nᵉ^2, nˢ + n₂ˢ)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3) * L₃ˢ', vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + n₂ˢ + nˢ^2 + n₃ˢ)]
        
        droptol!(A_UU, eps())
        droptol!(A_LU, eps())
        droptol!(A_LL, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())

        # Third-order Lyapunov solve
        if third_order_block_lyapunov_method
            # Block-triangular: reuse second-order covariance
            E₂_comp = [sparse(ℒ.I, 2*nˢ, 2*nˢ)  spzeros(2*nˢ, nˢ^2)
                        spzeros(n₂ˢ, 2*nˢ)       L₂ˢ]
            Σ̂ᶻ₂_compressed = E₂_comp * Σ̂ᶻ₂ * E₂_comp'

            # Compute C sub-blocks directly (avoid building full N×N matrix)
            ê_U = ê_to_ŝ₃[1:N_upper, :]
            ê_L = ê_to_ŝ₃[(N_upper+1):end, :]
            E_cU = Eᴸᶻ[:, 1:N_upper]
            E_cL = Eᴸᶻ[:, (N_upper+1):end]

            Q = E_cU * A_LU' + E_cL * A_LL'
            R = E_cU * A_UU'
            C_LU = ê_L * (Γ₃ * ê_U' + R) + Q' * ê_U'
            eQ = ê_L * Q
            C_LL = sparse_ABAt(ê_L, Γ₃) + eQ + eQ'
            droptol!(C_LU, eps())
            droptol!(C_LL, eps())

            Σᶻ₃, info = solve_block_triangular_lyapunov(A_UU, A_LU, A_LL, C_LU, C_LL,
                                                          Σ̂ᶻ₂_compressed,
                                                          𝓂.workspaces, opts,
                                                          n₃ˢ = n₃ˢ)
        else
            ŝ_to_ŝ₃ = [A_UU spzeros(N_upper, N_lower); A_LU A_LL]

            A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
            droptol!(A, eps())

            C = sparse_ABAt(ê_to_ŝ₃, Γ₃) + A + A'
            droptol!(C, eps())

            lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)
            Σᶻ₃, info = solve_lyapunov_equation(ŝ_to_ŝ₃, C, lyap_ws_3rd,
                                        lyapunov_algorithm = opts.lyapunov_algorithm,
                                        tol = opts.tol.third_order.lyapunov,
                                        verbose = opts.verbose,
                                        has_unit_roots = 𝓂.caches.has_unit_roots)
        end

        if !info
            nVars = 𝓂.constants.post_model_macro.nVars
            return fill(T(NaN), nVars, nVars), fill(T(NaN), nVars), fill(T(NaN), nVars), false
        end
    
        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + sparse_ABAt(ê_to_y₃, Γ₃) + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'
        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.constants.post_model_macro.var), indexin(variance_observable, 𝓂.constants.post_model_macro.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end
    end

    # Cache the 3rd-order result for reuse
    all_solved = solved && solved3 && solved_lyapunov
    if T === Float64 && all_solved
        if size(𝓂.caches.covariance_third_order) != size(Σʸ₃)
            𝓂.caches.covariance_third_order = Matrix{Float64}(undef, size(Σʸ₃)...)
        end
        copyto!(𝓂.caches.covariance_third_order, Σʸ₃)
        𝓂.caches.valid_for.covariance_third_order = Float64.(parameters)
        𝓂.caches.valid_for.covariance_third_order_obs_key = obs_key
    end

    return Σʸ₃, μʸ₂, SS_and_pars, all_solved
end


function determine_efficient_order(𝐒₁::Matrix{<: Real}, 
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Precompute state indices to avoid repeated indexin calls
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

        # Iterative propagation without redundant allocations
        while true
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        dependencies = T.past_not_future_and_mixed[dependencies_in_states]

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

                # Iterative propagation without redundant allocations
                while true
                    new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
                    if new_deps == dependencies_in_states
                        break
                    end
                    dependencies_in_states = new_deps
                end

                dependencies = T.past_not_future_and_mixed[dependencies_in_states]
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Build lookup dictionary for faster searches
        var_to_idx = Dict{Symbol, Int}()
        for (idx, order) in enumerate(orders)
            for var in order.first
                var_to_idx[var] = idx
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables using lookup dictionary
                idx_i = var_to_idx[covariance_vars[i]]
                idx_j = var_to_idx[covariance_vars[j]]
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end


function determine_efficient_order(𝐒₁::Matrix{<: Real},
                                    𝐒₂::AbstractMatrix{<: Real},
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    nᵉ = T.nExo
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Build selector for state variables in the augmented state vector [states; 1; shocks]
    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    
    # Kronecker product indices for state-state interactions
    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    # Precompute state indices and matrix slices to avoid repeated operations
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]
    𝐒₂_states = nnz(𝐒₂) > 0 ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        
        # First order dependencies
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0
        
        # Second order dependencies from quadratic terms (s ⊗ s)
        if nnz(𝐒₂) > 0
            s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
            
            # Check which state variable pairs have influence
            # Vectorized approach: reshape to nˢ×nˢ and check column/row sums
            s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
        end

        # Propagate dependencies through the system (iterative closure)
        # considering both first and second order propagation
        while true
            prev_dependencies = dependencies_in_states
            
            # First order propagation
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)
            
            # Second order propagation: if state i and state j are dependencies,
            # their product can affect states
            if !isnothing(𝐒₂_states)
                # Generate selector vector for columns where both states are dependencies
                selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                if any(selector)
                    # Check which states are affected by the selected products
                    affected = vec(sum(abs, 𝐒₂_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end
            
            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        dependencies = T.past_not_future_and_mixed[dependencies_in_states]

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                
                # First order dependencies
                dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0
                
                # Second order dependencies from quadratic terms (s ⊗ s)
                if nnz(𝐒₂) > 0
                    s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
                    # Vectorized approach: reshape to nˢ×nˢ and check column/row sums
                    s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
                    dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
                end

                # Propagate dependencies through the system
                # Precompute matrix slices
                𝐒₁_states_local = 𝐒₁[state_idx_in_var, 1:nˢ]
                𝐒₂_states_local = nnz(𝐒₂) > 0 ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing
                
                while true
                    prev_dependencies = dependencies_in_states
                    
                    # First order propagation
                    new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states_local) .> tol)
                    
                    # Second order propagation
                    if !isnothing(𝐒₂_states_local)
                        # Generate selector vector for columns where both states are dependencies
                        selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                        if any(selector)
                            affected = vec(sum(abs, 𝐒₂_states_local[:, selector], dims=2) .> tol)
                            new_deps = new_deps .| affected
                        end
                    end
                    
                    if new_deps == dependencies_in_states
                        break
                    end
                    dependencies_in_states = new_deps
                end

                dependencies = T.past_not_future_and_mixed[dependencies_in_states]
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables (they should exist now)
                idx_i = findfirst(x -> covariance_vars[i] in x.first, orders)
                idx_j = findfirst(x -> covariance_vars[j] in x.first, orders)
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end


function determine_efficient_order(𝐒₁::Matrix{<: Real},
                                    𝐒₂::AbstractSparseMatrix{<: Real},
                                    𝐒₃::AbstractSparseMatrix{<: Real},
                                    constants::constants,
                                    variables::Union{Symbol_input,String_input};
                                    covariance::Union{Symbol_input,String_input} = Symbol[],
                                    tol::AbstractFloat = eps())

    T = constants.post_model_macro
    

    orders = Pair{Vector{Symbol}, Vector{Symbol}}[]

    nˢ = T.nPast_not_future_and_mixed
    nᵉ = T.nExo
    
    if variables == :full_covar
        return [T.var => T.past_not_future_and_mixed]
    else
        var_idx = MacroModelling.parse_variables_input_to_index(variables, constants) |> sort
        observables = T.var[var_idx]
    end

    # Build selectors for state variables in the augmented state vector [states; 1; shocks]
    s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
    
    # Kronecker product indices for interactions
    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_s_s_s = ℒ.kron(kron_s_s, s_in_s⁺)
    
    # Precompute state indices and matrix slices
    state_idx_in_var = indexin(T.past_not_future_and_mixed, T.var) .|> Int
    𝐒₁_states = 𝐒₁[state_idx_in_var, 1:nˢ]
    has_S₂ = nnz(𝐒₂) > 0
    has_S₃ = nnz(𝐒₃) > 0
    𝐒₂_states = has_S₂ ? 𝐒₂[state_idx_in_var, kron_s_s] : nothing
    𝐒₃_states = has_S₃ ? 𝐒₃[state_idx_in_var, kron_s_s_s] : nothing

    function compute_dependencies(obs_in_var_idx::Vector{Int})
        # First order dependencies
        dependencies_in_states = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:nˢ], dims=1) .> tol) .> 0

        # Second order dependencies from quadratic terms (s ⊗ s)
        if has_S₂
            s_s_to_y₂ = 𝐒₂[obs_in_var_idx, kron_s_s]
            s_s_matrix = reshape(vec(sum(abs, s_s_to_y₂, dims=1) .> tol), nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_matrix, dims=2) .> 0) .| vec(sum(s_s_matrix, dims=1) .> 0)
        end

        # Third order dependencies from cubic terms (s ⊗ s ⊗ s)
        if has_S₃
            s_s_s_to_y₃ = 𝐒₃[obs_in_var_idx, kron_s_s_s]
            s_s_s_tensor = reshape(vec(sum(abs, s_s_s_to_y₃, dims=1) .> tol), nˢ, nˢ, nˢ)
            dependencies_in_states = dependencies_in_states .| vec(sum(s_s_s_tensor, dims=(2,3)) .> 0) .|
                                                             vec(sum(s_s_s_tensor, dims=(1,3)) .> 0) .|
                                                             vec(sum(s_s_s_tensor, dims=(1,2)) .> 0)
        end

        # Propagate dependencies through the system (iterative closure)
        while true
            prev_dependencies = dependencies_in_states

            # First order propagation
            new_deps = dependencies_in_states .| vec(abs.(dependencies_in_states' * 𝐒₁_states) .> tol)

            # Second order propagation
            if !isnothing(𝐒₂_states)
                selector = vec(ℒ.kron(prev_dependencies, prev_dependencies))
                if any(selector)
                    affected = vec(sum(abs, 𝐒₂_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end

            # Third order propagation
            if !isnothing(𝐒₃_states)
                selector = vec(ℒ.kron(ℒ.kron(prev_dependencies, prev_dependencies), prev_dependencies))
                if any(selector)
                    affected = vec(sum(abs, 𝐒₃_states[:, selector], dims=2) .> tol)
                    new_deps = new_deps .| affected
                end
            end

            if new_deps == dependencies_in_states
                break
            end
            dependencies_in_states = new_deps
        end

        return T.past_not_future_and_mixed[dependencies_in_states]
    end

    for obs in observables
        obs_in_var_idx = indexin([obs],T.var) .|> Int
        dependencies = compute_dependencies(obs_in_var_idx)

        push!(orders,[obs] => sort(dependencies))
    end
    
    # If covariance variables are specified, compute dependencies and add entries for those pairs
    if !(covariance == Symbol[])
        covar_var_idx = MacroModelling.parse_variables_input_to_index(covariance, constants) |> sort
        covariance_vars = T.var[covar_var_idx]
        
        # Compute dependencies for covariance variables (if not already computed)
        for covar_var in covariance_vars
            # Check if this variable's dependencies are already computed
            if isnothing(findfirst(x -> covar_var in x.first, orders))
                obs_in_var_idx = indexin([covar_var], T.var) .|> Int
                dependencies = compute_dependencies(obs_in_var_idx)
                push!(orders,[covar_var] => sort(dependencies))
            end
        end
        
        # Add entries for all pairs of covariance variables
        for i in 1:length(covariance_vars)
            for j in (i+1):length(covariance_vars)
                # Find dependencies for both variables (they should exist now)
                idx_i = findfirst(x -> covariance_vars[i] in x.first, orders)
                idx_j = findfirst(x -> covariance_vars[j] in x.first, orders)
                
                deps_i = orders[idx_i].second
                deps_j = orders[idx_j].second
                # Union of dependencies for covariance computation
                combined_deps = sort(union(deps_i, deps_j))
                push!(orders, [covariance_vars[i], covariance_vars[j]] => combined_deps)
            end
        end
    end

    sort!(orders, by = x -> length(x[2]), rev = true)

    return combine_pairs(orders)
end



function combine_pairs(v::Vector{Pair{Vector{Symbol}, Vector{Symbol}}})
    i = 1
    while i <= length(v)
        subset_found = false
        for j in i+1:length(v)
            # Check if v[i].second and v[j].second are equal or if one is subset of the other
            if v[i].second == v[j].second
                # Exact match: combine first elements and remove duplicate
                v[i] = v[i].first ∪ v[j].first => v[i].second
                deleteat!(v, j)
                subset_found = true
                break
            elseif all(elem -> elem in v[j].second, v[i].second) || all(elem -> elem in v[i].second, v[j].second)
                # One is subset of the other: combine the first elements and assign to the one with the larger second element
                if length(v[i].second) > length(v[j].second)
                    v[i] = v[i].first ∪ v[j].first => v[i].second
                    deleteat!(v, j)
                else
                    v[j] = v[i].first ∪ v[j].first => v[j].second
                    deleteat!(v, i)
                end
                subset_found = true
                break
            end
        end
        # If no subset was found for v[i], move to the next element
        if !subset_found
            i += 1
        end
    end
    return v
end


end # @stable
