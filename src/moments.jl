@stable default_mode = "disable" begin

function calculate_covariance(parameters::Vector{R}, 
                                𝓂::ℳ; 
                                opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{R}, Matrix{R}, Matrix{R}, Vector{R}, Bool} where R <: Real
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    idx_constants = constants.post_complete_parameters
    T = constants.post_model_macro
    
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    
    if solution_error > opts.tol.NSSS_acceptance_tol
        return zeros(0,0), zeros(0,0), zeros(0,0), SS_and_pars, solution_error < opts.tol.NSSS_acceptance_tol
    end

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian) 

    sol, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                            constants,
                                                            𝓂.workspaces,
                                                            𝓂.caches;
                                                            initial_guess = 𝓂.caches.qme_solution,
                                                            opts = opts)

    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    # Direct constants access instead of model access
    A = @views sol[:, 1:T.nPast_not_future_and_mixed] * idx_constants.diag_nVars[T.past_not_future_and_mixed_idx,:]

    C = @views sol[:, T.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    if !solved
        return CC, sol, ∇₁, SS_and_pars, solved
    end

    # Ensure lyapunov workspace is properly sized and get it
    lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, T.nVars, :first_order)

    covar_raw, solved = solve_lyapunov_equation(A, CC, lyap_ws,
                            lyapunov_algorithm = opts.lyapunov_algorithm, 
                            tol = opts.tol.lyapunov_tol,
                            acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                            verbose = opts.verbose)

    return covar_raw, sol , ∇₁, SS_and_pars, solved
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
    
    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    
    if algorithm == :first_order
        mean_of_variables = SS_and_pars[1:T.nVars]

        solved = solution_error < opts.tol.NSSS_acceptance_tol
    else
        ensure_moments_constants!(constants)
        so = constants.second_order
        ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix
        
        𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                            constants,
                                                            𝓂.workspaces,
                                                            𝓂.caches;
                                                            initial_guess = 𝓂.caches.qme_solution,
                                                            opts = opts)
        
        update_perturbation_counter!(𝓂.counters, solved, order = 1)

        if !solved 
            mean_of_variables = SS_and_pars[1:T.nVars]
        else
            ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)# * 𝓂.constants.second_order.𝐔∇₂
            
            𝐒₂, solved = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                        opts = opts)

            update_perturbation_counter!(𝓂.counters, solved, order = 2)

            if !solved 
                mean_of_variables = SS_and_pars[1:T.nVars]
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
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)# * 𝓂.constants.second_order.𝐔∇₂

        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                opts = opts)
        
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

            s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
            e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
            s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

            # # Set up in pruned state transition matrices
            ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
                        zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
                        zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

            ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                        zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

            ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

            ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

            vec_Iₑ = so.vec_Iₑ
            ŝv₂ = [ zeros(nˢ) 
                    vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec_Iₑ
                    e_to_s₁_by_e_to_s₁ * vec_Iₑ]

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
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)# * 𝓂.constants.second_order.𝐔∇₂

        𝐒₂_raw, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                opts = opts)

        update_perturbation_counter!(𝓂.counters, solved2, order = 2)
        
        if solved2
            𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{R, Int}

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

            s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
            e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
            s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

            # # Set up in pruned state transition matrices
            ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
                        zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
                        zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

            ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
                        zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

            ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

            ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

            vec_Iₑ = so.vec_Iₑ
            ŝv₂ = [ zeros(nˢ) 
                    vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec_Iₑ
                    e_to_s₁_by_e_to_s₁ * vec_Iₑ]

            yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec_Iₑ) / 2

            ## Mean
            μˢ⁺₂ = (ℒ.I(size(ŝ_to_ŝ₂, 1)) - ŝ_to_ŝ₂) \ ŝv₂
            Δμˢ₂ = vec((ℒ.I(size(s_to_s₁, 1)) - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec_Iₑ) / 2))
            μʸ₂  = SS_and_pars[1:𝓂.constants.post_model_macro.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

            # Covariance
            Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
                    zeros(nᵉ^2, nᵉ)    so.e4_minus_vecIₑ_outer     zeros(nᵉ^2, nᵉ * nˢ)
                    zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

            C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

            # Ensure second-order lyapunov workspace and solve
            lyap_ws_2nd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₂, 1), :second_order)

            Σᶻ₂, info = solve_lyapunov_equation(ŝ_to_ŝ₂, C, lyap_ws_2nd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose)

            if info
                Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

                autocorr_tmp = ŝ_to_ŝ₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_ŝ₂ * Γ₂ * ê_to_y₂'

                slvd = solved && solved2 && info
            else
                Σʸ₂ = zeros(R,0,0)
                Σᶻ₂ = zeros(R,0,0)
                μʸ₂ = zeros(R,0)
                Δμˢ₂ = zeros(R,0)
                autocorr_tmp = zeros(R,0,0)
                ŝ_to_ŝ₂ = zeros(R,0,0)
                ŝ_to_y₂ = zeros(R,0,0)
                slvd = info
            end
        else
            Σʸ₂ = zeros(R,0,0)
            Σᶻ₂ = zeros(R,0,0)
            μʸ₂ = zeros(R,0)
            Δμˢ₂ = zeros(R,0)
            autocorr_tmp = zeros(R,0,0)
            ŝ_to_ŝ₂ = zeros(R,0,0)
            ŝ_to_y₂ = zeros(R,0,0)
            slvd = solved2
        end
    else
        Σʸ₂ = zeros(R,0,0)
        Σᶻ₂ = zeros(R,0,0)
        μʸ₂ = zeros(R,0)
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




function calculate_third_order_moments_with_autocorrelation(parameters::Vector{T}, 
                                            observables::Union{Symbol_input,String_input},
                                            𝓂::ℳ; 
                                            autocorrelation_periods::U = 1:5,
                                            covariance::Union{Symbol_input,String_input} = Symbol[],
                                            opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{T}, Vector{T}, Matrix{T}, Vector{T}, Bool} where {U, T <: Real}

    second_order_moments = calculate_second_order_moments_with_covariance(parameters, 𝓂; opts = opts)

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = second_order_moments

    if !solved
        return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
    end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{T, Int}

    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)# * 𝓂.constants.third_order.𝐔∇₃

	    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw, 
	                                                𝓂.constants,
                                                    𝓂.workspaces,
                                                    𝓂.caches;
	                                                initial_guess = 𝓂.caches.third_order_solution,
                                                    opts = opts)

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)

    if !solved3
        return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
    end

    𝐒₃ *= 𝓂.constants.third_order.𝐔₃

    𝐒₃ = sparse(𝐒₃) # ensure stable sparse type
    
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃, 𝓂.constants, observables, covariance = covariance, tol = opts.tol.dependencies_tol)

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

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

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


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
        
        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())
        
        A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A, eps())

        C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'
        droptol!(C, eps())

        # Ensure third-order lyapunov workspace and solve
        lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)

        Σᶻ₃, info = solve_lyapunov_equation(ŝ_to_ŝ₃, C, lyap_ws_3rd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose)

        if !info
            return zeros(T,0,0), zeros(T,0), zeros(T,0,0), zeros(T,0), false
        end

        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

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

            Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
            ℒ.kron(s_to_s₁ⁱ * Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(s_to_s₁ⁱ * μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(s_to_s₁ⁱ * reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(s_to_s₁ⁱ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
            spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]

            for obs in variance_observable
                autocorr[indexin([obs], 𝓂.constants.post_model_macro.var), i] .= ℒ.diag(ŝ_to_y₃ * Σᶻ₃ⁱ * ŝ_to_y₃' + ŝ_to_y₃ * ŝ_to_ŝ₃ⁱ * autocorr_tmp + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃')[indexin([obs], variance_observable)] ./ max.(ℒ.diag(Σʸ₃tmp), eps(Float64))[indexin([obs], variance_observable)]

                autocorr[indexin([obs], 𝓂.constants.post_model_macro.var), i][ℒ.diag(Σʸ₃tmp)[indexin([obs], variance_observable)] .< opts.tol.lyapunov_acceptance_tol] .= 0
            end

            ŝ_to_ŝ₃ⁱ *= ŝ_to_ŝ₃
        end
    end

    return Σʸ₃, μʸ₂, autocorr, SS_and_pars, solved && solved3 && solved_lyapunov
end

function calculate_third_order_moments(parameters::Vector{T}, 
                                            observables::Union{Symbol_input,String_input},
                                            𝓂::ℳ;
                                            covariance::Union{Symbol_input,String_input} = Symbol[],
                                            opts::CalculationOptions = merge_calculation_options())::Tuple{Matrix{T}, Vector{T}, Vector{T}, Bool} where T <: Real
    second_order_moments = calculate_second_order_moments_with_covariance(parameters, 𝓂; opts = opts)

    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂_raw, ∇₂, solved = second_order_moments

    if !solved
        return zeros(T,0,0), zeros(T,0), zeros(T,0), false
    end

    # Expand compressed 𝐒₂_raw to full for moments computation
    𝐒₂ = sparse(𝐒₂_raw * 𝓂.constants.second_order.𝐔₂)::SparseMatrixCSC{T, Int}

    ensure_moments_constants!(𝓂.constants)
    so = 𝓂.constants.second_order
    to = 𝓂.constants.third_order

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)# * 𝓂.constants.third_order.𝐔∇₃

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂_raw, 
                                                𝓂.constants,
                                                𝓂.workspaces,
                                                𝓂.caches;
                                                initial_guess = 𝓂.caches.third_order_solution,
                                                opts = opts)

    update_perturbation_counter!(𝓂.counters, solved3, order = 3)
    
    if !solved3
        return zeros(T,0,0), zeros(T,0), zeros(T,0), false
    end

    𝐒₃ *= 𝓂.constants.third_order.𝐔₃

    𝐒₃ = sparse(𝐒₃) # ensure stable sparse type
    
    orders = determine_efficient_order(𝐒₁, 𝐒₂, 𝐒₃, 𝓂.constants, observables, covariance = covariance, tol = opts.tol.dependencies_tol)

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

        # Set up pruned state transition matrices
        ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                            zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                            zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                            s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                            ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                            zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

        ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                        e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                        ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                        zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

        ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

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


        Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
                ℒ.kron(Σ̂ᶻ₁,vec_Iₑ)   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec_Iₑ)    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec_Iₑ)  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec_Iₑ)
                spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
        
        droptol!(ŝ_to_ŝ₃, eps())
        droptol!(ê_to_ŝ₃, eps())
        droptol!(Eᴸᶻ, eps())
        droptol!(Γ₃, eps())
        
        A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
        droptol!(A, eps())

        C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'
        droptol!(C, eps())

        # Ensure third-order lyapunov workspace and solve
        lyap_ws_3rd = ensure_lyapunov_workspace!(𝓂.workspaces, size(ŝ_to_ŝ₃, 1), :third_order)

        Σᶻ₃, info = solve_lyapunov_equation(ŝ_to_ŝ₃, C, lyap_ws_3rd,
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose)

        if !info
            return zeros(T,0,0), zeros(T,0), zeros(T,0), false
        end
    
        solved_lyapunov = solved_lyapunov && info

        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃' + ê_to_y₃ * Eᴸᶻ * ŝ_to_y₃' + ŝ_to_y₃ * Eᴸᶻ' * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.constants.post_model_macro.var), indexin(variance_observable, 𝓂.constants.post_model_macro.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end
    end

    return Σʸ₃, μʸ₂, SS_and_pars, solved && solved3 && solved_lyapunov
end

end
