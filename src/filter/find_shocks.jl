@stable default_mode = "disable" begin

# Algorithms
# - LagrangeNewton: fast, but no guarantee of convergence to global minimum
# - COBYLA: best known chances of convergence to global minimum; ok speed for third order; lower tol on optimality conditions (1e-7)
# - SLSQP: relatively slow and not guaranteed to converge to global minimum
# For higher-order solutions the global minimum-norm shocks problem is NP-hard, since the
# number of feasible roots that satisfy the constraints grows exponentially and would
# require exhaustive checking. LagrangeNewton will return the minimum-norm solution whose
# basin of attraction contains the origin.

# Generalized find_shocks for conditional forecasts
# This function finds shocks that minimize their squared magnitude while satisfying 
# conditional forecast constraints (only some variables match target values)
# Uses analytical derivatives from perturbation solution matrices (like find_shocks)

function find_shocks_conditional_forecast(::Val{:LagrangeNewton},
                                         initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
                                         all_shocks::Vector{Float64},
                                         conditions::Vector{Float64},
                                         cond_var_idx::Vector{Int},
                                         free_shock_idx::Vector{Int},
                                         state_update::Function,
                                        #  pruning_arg::Bool,
                                         𝐒₁::AbstractMatrix{Float64},
                                         𝐒₂::Union{AbstractMatrix{Float64}, Nothing},
                                         𝐒₃::Union{AbstractMatrix{Float64}, Nothing},
                                         constants::constants,
                                         ws::find_shocks_workspace{Float64};
                                         max_iter::Int = 1000,
                                         tol::Float64 = 1e-13,
                                         verbose::Bool = false,
                                         rescue_shocks::Bool = false)

    # Note: state_update and pruning_arg are accepted for interface consistency
    # but not used. We infer pruning from initial_state type and compute states
    # directly using perturbation matrices.
    pruning = initial_state isa Vector{Vector{Float64}}

    T = constants.post_model_macro
    so = constants.second_order
    to = constants.third_order
    n_exo = T.nExo
    third_order = !isnothing(𝐒₃)

    ensure_conditional_forecast_constants!(constants; third_order = third_order)

    shockvar_no_vol_cols = so.shockvar_no_vol_cols
    shockvar²_cols = so.shockvar²_cols
    shock²_cols = so.shock²_cols
    var_vol²_cols = so.var_vol²_cols
    var²_cols = so.var²_cols

    var_vol³_cols = to.var_vol³_cols
    shockvar³2_cols = to.shockvar³2_cols
    shockvar³_cols = to.shockvar³_cols
    shock³_cols = to.shock³_cols

    fixed_shock_idx = setdiff(1:n_exo, free_shock_idx)

    if isempty(cond_var_idx) && isempty(fixed_shock_idx)
        return zeros(length(free_shock_idx)), true
    end

    J = ℒ.I(n_exo)

    nPast = T.nPast_not_future_and_mixed
    n_global = nPast + 1 + n_exo
    third_order_pruning = third_order && pruning
    ensure_find_shocks_state_buffers!(ws, n_exo, nPast;
                                      third_order = third_order,
                                      third_order_pruning = third_order_pruning)
    kron_state_vol = ws.kron_state_vol
    kron_I_state = ws.kron_I_state

    n_exo² = n_exo * (n_exo + 1) ÷ 2

    if isnothing(𝐒₃)
        # Second order (pruned or non-pruned)
        if pruning
            state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
            state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
            state_vol = vcat(state₁, 1)

            𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
            ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

            if isnothing(𝐒₂)
                𝐒ⁱ = copy(𝐒¹ᵉ)
                𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo²)
            else
                𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_cols]
                𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_cols]
                𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_cols]

                compressed_kron²_power!(kron_state_vol, state_vol)
                ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

                ℒ.kron!(kron_I_state, J, state_vol)
                𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
                𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
            end
        else
            state = initial_state[T.past_not_future_and_mixed_idx]
            state_vol = vcat(state, 1)

            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

            if isnothing(𝐒₂)
                𝐒ⁱ = copy(𝐒¹ᵉ)
                𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo²)
            else
                𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_cols]
                𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_cols]
                𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_cols]

                compressed_kron²_power!(kron_state_vol, state_vol)
                ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

                ℒ.kron!(kron_I_state, J, state_vol)
                𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
                𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
            end
        end

        𝐒ⁱ³ᵉ = nothing
    else # third_order
        # Third order (pruned or non-pruned)
        if pruning
            state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
            state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
            state₃ = initial_state[3][T.past_not_future_and_mixed_idx]
            state_vol = vcat(state₁, 1)

            𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_cols]
            𝐒²⁻ = @views 𝐒₂[cond_var_idx, var²_cols]
            𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_cols]
            𝐒²⁻ᵛᵉ = @views 𝐒₂[cond_var_idx, shockvar_no_vol_cols]
            𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_cols]

            𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_cols]
            𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_cols]
            𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_cols]
                𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_cols]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
            ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
            ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

            compressed_kron²_power!(kron_state_vol, state_vol)
            ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

            kron_state₁₂ = ws.kron_state₁₂
            compressed_kron²!(kron_state₁₂, state₁, state₂)
            ℒ.mul!(shock_independent, 𝐒²⁻, kron_state₁₂, -1, 1)

            kron_state_vol3 = ws.kron_state_vol3
            compressed_kron³_power!(kron_state_vol3, state_vol)
            ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

            ℒ.kron!(kron_I_state, J, state_vol)
            kron_I_state₂ = ws.kron_I_state₂
            ℒ.kron!(kron_I_state₂, J, state₂)
            kron_I_state_state = ws.kron_I_state_state
            ℒ.kron!(kron_I_state_state, J, kron_state_vol)

            𝐒ⁱ = 𝐒¹ᵉ +
            𝐒²⁻ᵉ * kron_I_state +
            𝐒²⁻ᵛᵉ * kron_I_state₂ +
            𝐒³⁻ᵉ² * kron_I_state_state / 2

            𝐒³⁻ᵉ_state = compressed_triple_state_to_pair(state_vol,
                                                          n_global,
                                                          nPast + 1,
                                                          n_exo,
                                                          shockvar³_cols)
            𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * 𝐒³⁻ᵉ_state
            𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
        else
            state = initial_state[T.past_not_future_and_mixed_idx]
            state_vol = vcat(state, 1)

            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_cols]
            𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_cols]
            𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_cols]

            𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_cols]
            𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_cols]
            𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_cols]
            𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_cols]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

            compressed_kron²_power!(kron_state_vol, state_vol)
            ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

            kron_state_vol3 = ws.kron_state_vol3
            compressed_kron³_power!(kron_state_vol3, state_vol)
            ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

            ℒ.kron!(kron_I_state, J, state_vol)
            kron_I_state_state = ws.kron_I_state_state
            ℒ.kron!(kron_I_state_state, J, kron_state_vol)

            𝐒ⁱ = 𝐒¹ᵉ +
            𝐒²⁻ᵉ * kron_I_state +
            𝐒³⁻ᵉ² * kron_I_state_state / 2

            𝐒³⁻ᵉ_state = compressed_triple_state_to_pair(state_vol,
                                                          n_global,
                                                          nPast + 1,
                                                          n_exo,
                                                          shockvar³_cols)
            𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * 𝐒³⁻ᵉ_state
            𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
        end
    end

    if !isempty(fixed_shock_idx)
        n_cond = length(cond_var_idx)
        n_fixed = length(fixed_shock_idx)

        𝐒ⁱ_aug = zeros(n_cond + n_fixed, n_exo)
        𝐒ⁱ_aug[1:n_cond, :] = 𝐒ⁱ
        for (row, idx) in enumerate(fixed_shock_idx)
            𝐒ⁱ_aug[n_cond + row, idx] = 1.0
        end

        𝐒ⁱ²ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ²ᵉ, 2))
        𝐒ⁱ²ᵉ_aug[1:n_cond, :] = 𝐒ⁱ²ᵉ

        if isnothing(𝐒ⁱ³ᵉ)
            𝐒ⁱ³ᵉ_aug = nothing
        else
            𝐒ⁱ³ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ³ᵉ, 2))
            𝐒ⁱ³ᵉ_aug[1:n_cond, :] = 𝐒ⁱ³ᵉ
        end

        shock_independent = vcat(shock_independent, all_shocks[fixed_shock_idx])
        𝐒ⁱ = 𝐒ⁱ_aug
        𝐒ⁱ²ᵉ = 𝐒ⁱ²ᵉ_aug
        𝐒ⁱ³ᵉ = 𝐒ⁱ³ᵉ_aug
    end

    initial_guess = copy(all_shocks)
    
    # Ensure workspaces are allocated
    third_order = !isnothing(𝐒ⁱ³ᵉ)
    ensure_find_shocks_buffers!(ws, n_exo; third_order = third_order)

    if isnothing(𝐒ⁱ³ᵉ)
        x, matched = find_shocks(Val(:LagrangeNewton),
                                 initial_guess,
                                 ws.kron_buffer,
                                 ws.kron_buffer2,
                                 J,
                                 𝐒ⁱ,
                                 𝐒ⁱ²ᵉ,
                                 shock_independent;
                                 max_iter = max_iter,
                                 tol = tol,
                                 verbose = verbose)
        if !matched && rescue_shocks
            # Limited-budget restarts on the augmented conditional-forecast system.
            x, matched = find_shocks_with_restarts(Val(:LagrangeNewton),
                                 ws.kron_buffer,
                                 ws.kron_buffer2,
                                 J,
                                 𝐒ⁱ,
                                 𝐒ⁱ²ᵉ,
                                 shock_independent;
                                 max_iter = max_iter,
                                 tol = tol,
                                 verbose = verbose)
        end
    else
        x, matched = find_shocks(Val(:LagrangeNewton),
                                 initial_guess,
                                 ws.kron_buffer,
                                 ws.kron_buffer²,
                                 ws.kron_buffer2,
                                 ws.kron_buffer3,
                                 ws.kron_buffer4,
                                 J,
                                 𝐒ⁱ,
                                 𝐒ⁱ²ᵉ,
                                 𝐒ⁱ³ᵉ,
                                 shock_independent;
                                 max_iter = max_iter,
                                 tol = tol,
                                 verbose = verbose)
        if !matched && rescue_shocks
            # Limited-budget restarts on the augmented conditional-forecast system.
            x, matched = find_shocks_with_restarts(Val(:LagrangeNewton),
                                 ws.kron_buffer,
                                 ws.kron_buffer²,
                                 ws.kron_buffer2,
                                 ws.kron_buffer3,
                                 ws.kron_buffer4,
                                 J,
                                 𝐒ⁱ,
                                 𝐒ⁱ²ᵉ,
                                 𝐒ⁱ³ᵉ,
                                 shock_independent;
                                 max_iter = max_iter,
                                 tol = tol,
                                 verbose = verbose)
        end
    end

    return x[free_shock_idx], matched
end



# function find_shocks_conditional_forecast(::Val{:SLSQP},
#                                          initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
#                                          all_shocks::Vector{Float64},
#                                          conditions::Vector{Float64},
#                                          cond_var_idx::Vector{Int},
#                                          free_shock_idx::Vector{Int},
#                                          state_update::Function,
#                                         #  pruning_arg::Bool,
#                                          𝐒₁::AbstractMatrix{Float64},
#                                          𝐒₂::Union{AbstractMatrix{Float64}, Nothing},
#                                          𝐒₃::Union{AbstractMatrix{Float64}, Nothing},
#                                          T::timings;
#                                          max_iter::Int = 1000,
#                                          tol::Float64 = 1e-13,
#                                          verbose::Bool = false)

#     # Note: state_update and pruning_arg are accepted for interface consistency
#     # but not used. We infer pruning from initial_state type and compute states
#     # directly using perturbation matrices.
#     pruning = initial_state isa Vector{Vector{Float64}}

#     n_past = T.nPast_not_future_and_mixed
#     n_exo = T.nExo
#     third_order = !isnothing(𝐒₃)

#     s_in_s⁺ = BitVector(vcat(ones(Bool, n_past), zeros(Bool, n_exo + 1)))
#     sv_in_s⁺ = BitVector(vcat(ones(Bool, n_past + 1), zeros(Bool, n_exo)))
#     e_in_s⁺ = BitVector(vcat(zeros(Bool, n_past + 1), ones(Bool, n_exo)))

#     tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
#     shock_idxs = tmp.nzind

#     tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
#     shock²_idxs = tmp.nzind

#     shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

#     tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
#     var_vol²_idxs = tmp.nzind

#     tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
#     var²_idxs = tmp.nzind

#     tmp = ℒ.kron(e_in_s⁺, s_in_s⁺) |> sparse
#     shockvar_idxs = tmp.nzind

#     if third_order
#         tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
#         var_vol³_idxs = tmp.nzind

#         tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
#         shock_idxs2 = tmp.nzind

#         tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
#         shock_idxs3 = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
#         shock³_idxs = tmp.nzind

#         tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
#         shockvar1_idxs = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
#         shockvar2_idxs = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
#         shockvar3_idxs = tmp.nzind

#         shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
#         shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)
#     end

#     fixed_shock_idx = setdiff(1:n_exo, free_shock_idx)

#     if isempty(cond_var_idx) && isempty(fixed_shock_idx)
#         return zeros(length(free_shock_idx)), true
#     end

#     J = ℒ.I(n_exo)

#     if isnothing(𝐒₃)
#         # Second order (pruned or non-pruned)
#         if pruning
#             state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
#             state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state₁, 1)

#             𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

#             if isnothing(𝐒₂)
#                 𝐒ⁱ = copy(𝐒¹ᵉ)
#                 𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
#             else
#                 𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#                 𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#                 𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#                 kron_state_vol = ℒ.kron(state_vol, state_vol)
#                 ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#                 kron_I_state = ℒ.kron(J, state_vol)
#                 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
#                 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
#             end
#         else
#             state = initial_state[T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state, 1)

#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

#             if isnothing(𝐒₂)
#                 𝐒ⁱ = copy(𝐒¹ᵉ)
#                 𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
#             else
#                 𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#                 𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#                 𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#                 kron_state_vol = ℒ.kron(state_vol, state_vol)
#                 ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#                 kron_I_state = ℒ.kron(J, state_vol)
#                 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
#                 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
#             end
#         end

#         𝐒ⁱ³ᵉ = nothing
#     else
#         # Third order (pruned or non-pruned)
#         II = sparse(ℒ.I(n_exo^2))

#         if pruning
#             state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
#             state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
#             state₃ = initial_state[3][T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state₁, 1)

#             𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#             𝐒²⁻ = @views 𝐒₂[cond_var_idx, var²_idxs]
#             𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#             𝐒²⁻ᵛᵉ = @views 𝐒₂[cond_var_idx, shockvar_idxs]
#             𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#             𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
#             𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
#             𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
#             𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

#             kron_state_vol = ℒ.kron(state_vol, state_vol)
#             ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#             kron_state₁₂ = ℒ.kron(state₁, state₂)
#             ℒ.mul!(shock_independent, 𝐒²⁻, kron_state₁₂, -1, 1)

#             kron_state_vol3 = ℒ.kron(state_vol, kron_state_vol)
#             ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

#             kron_I_state = ℒ.kron(J, state_vol)
#             kron_I_state₂ = ℒ.kron(J, state₂)
#             kron_I_state_state = ℒ.kron(J, kron_state_vol)

#             𝐒ⁱ = 𝐒¹ᵉ +
#                  𝐒²⁻ᵉ * kron_I_state +
#                  𝐒²⁻ᵛᵉ * kron_I_state₂ +
#                  𝐒³⁻ᵉ² * kron_I_state_state / 2

#             𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
#             𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
#         else
#             state = initial_state[T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state, 1)

#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#             𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#             𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#             𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
#             𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
#             𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
#             𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

#             kron_state_vol = ℒ.kron(state_vol, state_vol)
#             ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#             kron_state_vol3 = ℒ.kron(state_vol, kron_state_vol)
#             ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

#             kron_I_state = ℒ.kron(J, state_vol)
#             kron_I_state_state = ℒ.kron(J, kron_state_vol)

#             𝐒ⁱ = 𝐒¹ᵉ +
#                  𝐒²⁻ᵉ * kron_I_state +
#                  𝐒³⁻ᵉ² * kron_I_state_state / 2

#             𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
#             𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
#         end
#     end

#     if !isempty(fixed_shock_idx)
#         n_cond = length(cond_var_idx)
#         n_fixed = length(fixed_shock_idx)

#         𝐒ⁱ_aug = zeros(n_cond + n_fixed, n_exo)
#         𝐒ⁱ_aug[1:n_cond, :] = 𝐒ⁱ
#         for (row, idx) in enumerate(fixed_shock_idx)
#             𝐒ⁱ_aug[n_cond + row, idx] = 1.0
#         end

#         𝐒ⁱ²ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ²ᵉ, 2))
#         𝐒ⁱ²ᵉ_aug[1:n_cond, :] = 𝐒ⁱ²ᵉ

#         if isnothing(𝐒ⁱ³ᵉ)
#             𝐒ⁱ³ᵉ_aug = nothing
#         else
#             𝐒ⁱ³ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ³ᵉ, 2))
#             𝐒ⁱ³ᵉ_aug[1:n_cond, :] = 𝐒ⁱ³ᵉ
#         end

#         shock_independent = vcat(shock_independent, all_shocks[fixed_shock_idx])
#         𝐒ⁱ = 𝐒ⁱ_aug
#         𝐒ⁱ²ᵉ = 𝐒ⁱ²ᵉ_aug
#         𝐒ⁱ³ᵉ = 𝐒ⁱ³ᵉ_aug
#     end

#     initial_guess = copy(all_shocks)

#     if isnothing(𝐒ⁱ³ᵉ)
#         kron_buffer = zeros(n_exo^2)
#         kron_buffer2 = ℒ.kron(J, zeros(n_exo))

#         x, matched = find_shocks(Val(:SLSQP),
#                                  initial_guess,
#                                  kron_buffer,
#                                  kron_buffer2,
#                                  J,
#                                  𝐒ⁱ,
#                                  𝐒ⁱ²ᵉ,
#                                  shock_independent;
#                                  max_iter = max_iter,
#                                  tol = tol,
#                                  verbose = verbose)
#     else
#         kron_buffer = zeros(n_exo^2)
#         kron_buffer² = zeros(n_exo^3)
#         kron_buffer2 = ℒ.kron(J, zeros(n_exo))
#         kron_buffer3 = ℒ.kron(J, kron_buffer)
#         kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(n_exo))

#         x, matched = find_shocks(Val(:SLSQP),
#                                  initial_guess,
#                                  kron_buffer,
#                                  kron_buffer²,
#                                  kron_buffer2,
#                                  kron_buffer3,
#                                  kron_buffer4,
#                                  J,
#                                  𝐒ⁱ,
#                                  𝐒ⁱ²ᵉ,
#                                  𝐒ⁱ³ᵉ,
#                                  shock_independent;
#                                  max_iter = max_iter,
#                                  tol = tol,
#                                  verbose = verbose)
#     end

#     return x[free_shock_idx], matched
# end



# function find_shocks_conditional_forecast(::Val{:COBYLA},
#                                          initial_state::Union{Vector{Float64}, Vector{Vector{Float64}}},
#                                          all_shocks::Vector{Float64},
#                                          conditions::Vector{Float64},
#                                          cond_var_idx::Vector{Int},
#                                          free_shock_idx::Vector{Int},
#                                          state_update::Function,
#                                         #  pruning_arg::Bool,
#                                          𝐒₁::AbstractMatrix{Float64},
#                                          𝐒₂::Union{AbstractMatrix{Float64}, Nothing},
#                                          𝐒₃::Union{AbstractMatrix{Float64}, Nothing},
#                                          T::timings;
#                                          max_iter::Int = 1000,
#                                          tol::Float64 = 1e-13,
#                                          verbose::Bool = false)

#     # Note: state_update and pruning_arg are accepted for interface consistency
#     # but not used. We infer pruning from initial_state type and compute states
#     # directly using perturbation matrices.
#     pruning = initial_state isa Vector{Vector{Float64}}

#     n_past = T.nPast_not_future_and_mixed
#     n_exo = T.nExo
#     third_order = !isnothing(𝐒₃)

#     s_in_s⁺ = BitVector(vcat(ones(Bool, n_past), zeros(Bool, n_exo + 1)))
#     sv_in_s⁺ = BitVector(vcat(ones(Bool, n_past + 1), zeros(Bool, n_exo)))
#     e_in_s⁺ = BitVector(vcat(zeros(Bool, n_past + 1), ones(Bool, n_exo)))

#     tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
#     shock_idxs = tmp.nzind

#     tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
#     shock²_idxs = tmp.nzind

#     shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

#     tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
#     var_vol²_idxs = tmp.nzind

#     tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
#     var²_idxs = tmp.nzind

#     tmp = ℒ.kron(e_in_s⁺, s_in_s⁺) |> sparse
#     shockvar_idxs = tmp.nzind

#     if third_order
#         tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
#         var_vol³_idxs = tmp.nzind

#         tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
#         shock_idxs2 = tmp.nzind

#         tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
#         shock_idxs3 = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
#         shock³_idxs = tmp.nzind

#         tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
#         shockvar1_idxs = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
#         shockvar2_idxs = tmp.nzind

#         tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
#         shockvar3_idxs = tmp.nzind

#         shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
#         shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)
#     end

#     fixed_shock_idx = setdiff(1:n_exo, free_shock_idx)

#     if isempty(cond_var_idx) && isempty(fixed_shock_idx)
#         return zeros(length(free_shock_idx)), true
#     end

#     J = ℒ.I(n_exo)

#     if isnothing(𝐒₃)
#         # Second order (pruned or non-pruned)
#         if pruning
#             state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
#             state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state₁, 1)

#             𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

#             if isnothing(𝐒₂)
#                 𝐒ⁱ = copy(𝐒¹ᵉ)
#                 𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
#             else
#                 𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#                 𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#                 𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#                 kron_state_vol = ℒ.kron(state_vol, state_vol)
#                 ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#                 kron_I_state = ℒ.kron(J, state_vol)
#                 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
#                 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
#             end
#         else
#             state = initial_state[T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state, 1)

#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

#             if isnothing(𝐒₂)
#                 𝐒ⁱ = copy(𝐒¹ᵉ)
#                 𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
#             else
#                 𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#                 𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#                 𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#                 kron_state_vol = ℒ.kron(state_vol, state_vol)
#                 ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#                 kron_I_state = ℒ.kron(J, state_vol)
#                 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
#                 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
#             end
#         end

#         𝐒ⁱ³ᵉ = nothing
#     else
#         # Third order (pruned or non-pruned)
#         II = sparse(ℒ.I(n_exo^2))

#         if pruning
#             state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
#             state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
#             state₃ = initial_state[3][T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state₁, 1)

#             𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#             𝐒²⁻ = @views 𝐒₂[cond_var_idx, var²_idxs]
#             𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#             𝐒²⁻ᵛᵉ = @views 𝐒₂[cond_var_idx, shockvar_idxs]
#             𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#             𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
#             𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
#             𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
#             𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
#             ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

#             kron_state_vol = ℒ.kron(state_vol, state_vol)
#             ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#             kron_state₁₂ = ℒ.kron(state₁, state₂)
#             ℒ.mul!(shock_independent, 𝐒²⁻, kron_state₁₂, -1, 1)

#             kron_state_vol3 = ℒ.kron(state_vol, kron_state_vol)
#             ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

#             kron_I_state = ℒ.kron(J, state_vol)
#             kron_I_state₂ = ℒ.kron(J, state₂)
#             kron_I_state_state = ℒ.kron(J, kron_state_vol)

#             𝐒ⁱ = 𝐒¹ᵉ +
#                  𝐒²⁻ᵉ * kron_I_state +
#                  𝐒²⁻ᵛᵉ * kron_I_state₂ +
#                  𝐒³⁻ᵉ² * kron_I_state_state / 2

#             𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
#             𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
#         else
#             state = initial_state[T.past_not_future_and_mixed_idx]
#             state_vol = vcat(state, 1)

#             𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
#             𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

#             𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
#             𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
#             𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

#             𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
#             𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
#             𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
#             𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

#             shock_independent = copy(conditions)
#             ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

#             kron_state_vol = ℒ.kron(state_vol, state_vol)
#             ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

#             kron_state_vol3 = ℒ.kron(state_vol, kron_state_vol)
#             ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

#             kron_I_state = ℒ.kron(J, state_vol)
#             kron_I_state_state = ℒ.kron(J, kron_state_vol)

#             𝐒ⁱ = 𝐒¹ᵉ +
#                  𝐒²⁻ᵉ * kron_I_state +
#                  𝐒³⁻ᵉ² * kron_I_state_state / 2

#             𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
#             𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
#         end
#     end

#     if !isempty(fixed_shock_idx)
#         n_cond = length(cond_var_idx)
#         n_fixed = length(fixed_shock_idx)

#         𝐒ⁱ_aug = zeros(n_cond + n_fixed, n_exo)
#         𝐒ⁱ_aug[1:n_cond, :] = 𝐒ⁱ
#         for (row, idx) in enumerate(fixed_shock_idx)
#             𝐒ⁱ_aug[n_cond + row, idx] = 1.0
#         end

#         𝐒ⁱ²ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ²ᵉ, 2))
#         𝐒ⁱ²ᵉ_aug[1:n_cond, :] = 𝐒ⁱ²ᵉ

#         if isnothing(𝐒ⁱ³ᵉ)
#             𝐒ⁱ³ᵉ_aug = nothing
#         else
#             𝐒ⁱ³ᵉ_aug = zeros(n_cond + n_fixed, size(𝐒ⁱ³ᵉ, 2))
#             𝐒ⁱ³ᵉ_aug[1:n_cond, :] = 𝐒ⁱ³ᵉ
#         end

#         shock_independent = vcat(shock_independent, all_shocks[fixed_shock_idx])
#         𝐒ⁱ = 𝐒ⁱ_aug
#         𝐒ⁱ²ᵉ = 𝐒ⁱ²ᵉ_aug
#         𝐒ⁱ³ᵉ = 𝐒ⁱ³ᵉ_aug
#     end

#     initial_guess = copy(all_shocks)

#     if isnothing(𝐒ⁱ³ᵉ)
#         kron_buffer = zeros(n_exo^2)
#         kron_buffer2 = ℒ.kron(J, zeros(n_exo))

#         x, matched = find_shocks(Val(:COBYLA),
#                                  initial_guess,
#                                  kron_buffer,
#                                  kron_buffer2,
#                                  J,
#                                  𝐒ⁱ,
#                                  𝐒ⁱ²ᵉ,
#                                  shock_independent;
#                                  max_iter = max_iter,
#                                  tol = tol,
#                                  verbose = verbose)
#     else
#         kron_buffer = zeros(n_exo^2)
#         kron_buffer² = zeros(n_exo^3)
#         kron_buffer2 = ℒ.kron(J, zeros(n_exo))
#         kron_buffer3 = ℒ.kron(J, kron_buffer)
#         kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(n_exo))

#         x, matched = find_shocks(Val(:COBYLA),
#                                  initial_guess,
#                                  kron_buffer,
#                                  kron_buffer²,
#                                  kron_buffer2,
#                                  kron_buffer3,
#                                  kron_buffer4,
#                                  J,
#                                  𝐒ⁱ,
#                                  𝐒ⁱ²ᵉ,
#                                  𝐒ⁱ³ᵉ,
#                                  shock_independent;
#                                  max_iter = max_iter,
#                                  tol = tol,
#                                  verbose = verbose)
#     end

#     return x[free_shock_idx], matched
# end

function shock_polynomial_residual!(residual, part, pairs, triples, x,
                                     linear, quadratic, cubic, target)
    compressed_kron²_power!(pairs, x)
    # residual = linear*x + quadratic*k₂(x) + cubic*k₃(x) - target.
    ℒ.mul!(residual, linear, x)
    scale = max(ℒ.norm(residual), ℒ.norm(target))
    ℒ.mul!(part, quadratic, pairs)
    scale = max(scale, ℒ.norm(part))
    ℒ.axpy!(1, part, residual)
    if cubic !== nothing
        compressed_kron³_power!(triples, x)
        ℒ.mul!(part, cubic, triples)
        scale = max(scale, ℒ.norm(part))
        ℒ.axpy!(1, part, residual)
    end
    ℒ.axpy!(-1, target, residual)
    return scale
end

# A square observation map needs only Newton's n×n Jacobian, rather than the
# 2n×2n shock/multiplier KKT system. Backtracking keeps the observation residual
# decreasing; the caller retains LagrangeNewton as a fallback when this stalls.
function find_square_shocks(initial_guess, pairs, triples, pair_jacobian,
                             triple_jacobian, identity, linear, quadratic, cubic,
                             target; max_iter, tol)
    x = copy(initial_guess)
    residual = similar(target)
    trial_residual = similar(target)
    part = similar(target)
    trial = similar(x)
    step = similar(x)
    jacobian = similar(linear)
    scale = shock_polynomial_residual!(residual, part, pairs, triples, x,
                                        linear, quadratic, cubic, target)
    for _ in 1:min(max_iter, 60)
        compressed_kron²!(pair_jacobian, x, identity)
        copyto!(jacobian, linear)
        # J(x) = linear + 2quadratic*D₂(x) + 3cubic*D₃(x).
        ℒ.mul!(jacobian, quadratic, pair_jacobian, 2, 1)
        if cubic !== nothing
            compressed_kron³!(triple_jacobian, x, x, identity)
            ℒ.mul!(jacobian, cubic, triple_jacobian, 3, 1)
        end
        factor = ℒ.lu!(jacobian, check = false)
        ℒ.issuccess(factor) || return x, false
        # step = J(x)⁻¹ f(x).
        ℒ.ldiv!(step, factor, residual)
        all(isfinite, step) || return x, false
        current = ℒ.norm(residual)
        feasible = tol * scale
        if current <= feasible
            # Apply the final correction even at an exact primal root so Dual
            # inputs retain the implicit root derivative through J⁻¹f.
            @. trial = x - step
            trial_scale = shock_polynomial_residual!(trial_residual, part, pairs,
                triples, trial, linear, quadratic, cubic, target)
            if ℒ.norm(trial_residual) <= tol * trial_scale
                copyto!(x, trial)
            end
            return x, true
        end
        alpha = one(eltype(x))
        accepted = false
        for _ in 1:30
            @. trial = x - alpha * step
            trial_scale = shock_polynomial_residual!(trial_residual, part, pairs,
                triples, trial, linear, quadratic, cubic, target)
            trial_norm = ℒ.norm(trial_residual)
            if isfinite(trial_norm) && trial_norm <= (1 - 1e-4 * alpha) * current
                copyto!(x, trial)
                copyto!(residual, trial_residual)
                scale = trial_scale
                accepted = true
                break
            end
            alpha /= 2
        end
        accepted || return x, false
    end
    return x, false
end

"""
Solve for minimum-norm shocks that satisfy linear and quadratic constraints using a
Lagrange-Newton stepper. For higher-order solutions the global minimum-norm problem is
NP-hard because the number of feasible roots grows exponentially; any gradient-based
solver started at the origin (not just LagrangeNewton) will converge to the root whose
basin contains the origin rather than guaranteeing the global optimum.

Square observation maps first use damped Newton steps on the shock Jacobian;
Lagrange-Newton is retained for underidentified maps and as a fallback.
"""
function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{R},
                    kron_buffer::Vector{R},
                    kron_buffer2::AbstractMatrix{R},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{R},
                    𝐒ⁱ²ᵉ::AbstractMatrix{R},
                    shock_independent::Vector{R};
                    max_iter::Int = 1000,
                    tol::AbstractFloat = 1e-13,
                    verbose::Bool = false) where R <: Real
    if size(𝐒ⁱ, 1) == size(𝐒ⁱ, 2) && max_iter > 0
        square_x, matched = find_square_shocks(initial_guess, kron_buffer, nothing,
            kron_buffer2, nothing, J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, nothing, shock_independent;
            max_iter = max_iter, tol = tol)
        matched && return square_x, true
    end
    x = copy(initial_guess)
    
    λ = zeros(R, size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm_part = one(R)
    
    Δnorm = R(1e12)

    x̂ = copy(shock_independent)

    part_buf = zeros(R, size(𝐒ⁱ, 1))
    if !all(iszero, x)
        shock_polynomial_residual!(x̂, part_buf, kron_buffer, nothing, x,
                                   𝐒ⁱ, 𝐒ⁱ²ᵉ, nothing, shock_independent)
        ℒ.rmul!(x̂, -1)
    end

    x̄ = zeros(R, size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)
    
    fxλ = zeros(R, length(xλ))
    
    fxλp = zeros(R, length(xλ), length(xλ))

    n_shock = size(𝐒ⁱ, 2)
    tmp = zeros(R, n_shock, n_shock)
    tmp_coeff = zeros(R, length(kron_buffer))

    lI = R(-2) * Matrix(ℒ.I(n_shock))

    iter = 0
    @inbounds for i in 1:max_iter
        iter = i
        compressed_kron²!(kron_buffer2, x, J)

        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpby!(1, 𝐒ⁱ, 2, ∂x)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x))]

        ℒ.mul!(tmp_coeff, 𝐒ⁱ²ᵉ', λ)
        compressed_pair_hessian!(tmp, tmp_coeff)
        ℒ.axpby!(1, lI, 2, tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x

        # fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        # f̂xλp = ℒ.lu(fxλp, check = false)

        # if !ℒ.issuccess(f̂xλp)
        #     return x, false
        # end

        f̂xλp = ℒ.lu(fxλp, check = false)

        if !ℒ.issuccess(f̂xλp)
            return x, false
        end

        ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        
        if !all(isfinite,Δxλ) break end

        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ

        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        compressed_kron²_power!(kron_buffer, x)

        # Part-relative residual: ‖F‖ / max(‖A·x‖, ‖Q·k‖, ‖b‖)
        ℒ.mul!(part_buf, 𝐒ⁱ²ᵉ, kron_buffer)
        norm_q = ℒ.norm(part_buf)
        ℒ.mul!(x̂, 𝐒ⁱ, x)
        norm_a = ℒ.norm(x̂)
        ℒ.axpy!(1, part_buf, x̂)
        norm_part = max(norm_a, norm_q, norm1)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        residual = norm_part > 0 ? ℒ.norm(x̂) / norm_part : ℒ.norm(x̂)

        if residual < tol && ℒ.norm(Δxλ) / max(one(R), ℒ.norm(xλ)) < sqrt(tol)
            break
        end
    end

    # Final part-relative residual at the returned x (covers early breaks
    # and max_iter exhaustion; also defines residual if the loop never
    # assigned it).
    compressed_kron²_power!(kron_buffer, x)
    ℒ.mul!(part_buf, 𝐒ⁱ²ᵉ, kron_buffer)
    norm_q = ℒ.norm(part_buf)
    ℒ.mul!(x̂, 𝐒ⁱ, x)
    norm_a = ℒ.norm(x̂)
    ℒ.axpy!(1, part_buf, x̂)
    norm_part = max(norm_a, norm_q, norm1)
    ℒ.axpby!(1, shock_independent, -1, x̂)
    residual = norm_part > 0 ? ℒ.norm(x̂) / norm_part : ℒ.norm(x̂)

    step_norm = ℒ.norm(Δxλ) / max(one(R), ℒ.norm(xλ))
    matched = residual < tol && step_norm < sqrt(tol)
    verbose && @info "LagrangeNewton solve (2nd order)" iterations = iter residual = residual step_norm = step_norm matched = matched
    return x, matched
end

"""
Limited-budget random-restart fallback around the LagrangeNewton shock solver
(second-order systems only).

The primary zero-start solve preserves the filter's min-norm-leaning behavior
wherever it works. Only when it fails, up to `n_restarts` additional starts
are tried — the linear solution plus coordinate-axis probes at a scale ladder —
each capped at `restart_max_iter` iterations, and the minimum-norm match is
returned. If those also fail, a short homotopy in `b`
(F_λ(x) = Sⁱx + Qk(x) − λb, λ: 0 → 1, warm-started stage by stage from the
exact x = 0 root at λ = 0) tracks the origin-connected branch, then a
homotopy in `Q` (G_μ(x) = Sⁱx + μQk(x) − b from the linear solution) tracks
the branch family disconnected from the origin path, then a
null-direction escape (SVD of the Jacobian at the best failure point, a few
fixed steps along the null space, fresh Newton tries) hops off shelf floors,
and finally — when roots live at radius ≫ 1, where every axis probe and both
homotopies miss — a large-radius multistart of plain damped Newton from
`randn` directions (see stage 5). That last stage is only reached when all
previous stages fail, so the happy path pays nothing for it. Acceptance uses
the part-relative residual ‖F‖ / max(‖A·x‖, ‖Q·k‖, ‖b‖) with the same `tol`
as the primary solver. Set `rescue_newton = false` to disable stage 5.

Stage-5 directions use `Random.randn` (global RNG). Rescue is therefore not
bit-reproducible across calls with the same inputs; stages 1–4 remain
deterministic. Reverse-mode AD (Zygote/Mooncake) uses the rrule pullback and
does not differentiate through the direction draw; ForwardDiff Dual paths
draw Float64 directions and convert.
"""

function find_shocks_with_restarts(::Val{:LagrangeNewton},
                    kron_buffer::Vector{R},
                    kron_buffer2::AbstractMatrix{R},
                    J,
                    𝐒ⁱ::AbstractMatrix{R},
                    𝐒ⁱ²ᵉ::AbstractMatrix{R},
                    shock_independent::Vector{R};
                    tol::AbstractFloat = 1e-13,
                    verbose::Bool = false,
                    n_restarts::Int = 16,
                    restart_max_iter::Int = 200,
                    rescue_newton::Bool = true) where R <: Real
    n = size(𝐒ⁱ, 2)
    buf = similar(kron_buffer)
    part_buf = similar(shock_independent)
    res_buf = similar(shock_independent)
    # Part-relative residual: ‖F‖ / max(‖A·x‖, ‖Q·k‖, ‖b‖) for
    # A·x + Q·ck2p(x) − b = 0 (same measure as the primary find_shocks).
    function res_norm(x, Qm, bv)
        compressed_kron²_power!(buf, x)
        ℒ.mul!(res_buf, 𝐒ⁱ, x)
        norm_a = ℒ.norm(res_buf)
        ℒ.mul!(part_buf, Qm, buf)
        norm_q = ℒ.norm(part_buf)
        ℒ.axpy!(1, part_buf, res_buf)
        norm_b = ℒ.norm(bv)
        ℒ.axpby!(1, bv, -1, res_buf)
        nr = ℒ.norm(res_buf)
        part = max(norm_a, norm_q, norm_b)
        part > 0 ? nr / part : nr
    end
    # Starts: the linear solution (when S^i is nonsingular) plus
    # coordinate-axis probes at a scale ladder, alternating signs.
    starts = Vector{Vector{R}}()
    luS = ℒ.lu(𝐒ⁱ, check = false)
    if ℒ.issuccess(luS)
        xl = luS \ shock_independent
        if all(isfinite, xl)
            push!(starts, xl)
        end
    end
    s_lin = isempty(starts) ? one(R) : max(ℒ.norm(starts[end]) / sqrt(n), one(R))
    k = 0
    for s in (s_lin, one(R), 10 * one(R), 100 * one(R))
        length(starts) > n_restarts && break
        for d in 1:n
            length(starts) > n_restarts && break
            v = zeros(R, n)
            v[d] = isodd(k) ? s : -s
            k += 1
            push!(starts, v)
        end
    end
    best_x = nothing
    best_res = R(Inf)
    best_n = R(Inf)
    fail_x = zeros(R, n)
    fail_res = R(Inf)
    for st in starts
        xk, _ = find_shocks(Val(:LagrangeNewton), st, kron_buffer, kron_buffer2,
                            J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, shock_independent;
                            max_iter = restart_max_iter, tol = tol, verbose = false)
        rk = res_norm(xk, 𝐒ⁱ²ᵉ, shock_independent)
        nk = ℒ.norm(xk)
        # Minimum-norm pick among matches: below tol all residuals are
        # numerical noise, while norm differences move the likelihood's
        # shocks² penalty one-for-one.
        if rk < tol && nk < best_n
            best_x, best_res, best_n = xk, rk, nk
        end
        if rk < fail_res
            fail_x, fail_res = xk, rk
        end
    end
    if best_x === nothing
        # Homotopy in b: F_λ(x) = Sⁱx + Qk(x) − λb for λ: 0 → 1 in Kλ stages.
        # x = 0 solves λ = 0 exactly, so tracking starts inside a basin by
        # construction; each stage warm-starts Newton from the previous root.
        # Finds the origin-connected branch even when no probe start converges
        # (narrow/disconnected basins). Deterministic plain arithmetic.
        Kλ = 12
        x_track = zeros(R, n)
        h_ok = true
        for kλ in 1:Kλ
            λ = R(kλ) / R(Kλ)
            bl = λ * shock_independent
            xk, _ = find_shocks(Val(:LagrangeNewton), x_track, kron_buffer, kron_buffer2,
                                J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, bl;
                                max_iter = restart_max_iter, tol = tol, verbose = false)
            if res_norm(xk, 𝐒ⁱ²ᵉ, bl) < tol
                x_track = xk
            else
                h_ok = false
                break
            end
        end
        if h_ok
            best_x = x_track
            best_res = res_norm(x_track, 𝐒ⁱ²ᵉ, shock_independent)
            best_n = ℒ.norm(x_track)
        end
    end
    if best_x === nothing
        # Q-homotopy from the linear solution: G_μ(x) = Sⁱx + μQk(x) − b.
        # μ = 0 is exactly the linear system (unique root x_lin when Sⁱ is
        # nonsingular); tracking to μ = 1 follows a DIFFERENT branch family
        # than b-homotopy from x = 0, reaching roots disconnected from the
        # origin path. Deterministic plain arithmetic.
        luS2 = ℒ.lu(𝐒ⁱ, check = false)
        if ℒ.issuccess(luS2)
            xl2 = luS2 \ shock_independent
            if all(isfinite, xl2)
                Kμ = 12
                x_track = copy(xl2)
                q_ok = true
                for kμ in 1:Kμ
                    μ = R(kμ) / R(Kμ)
                    Qμ = μ * 𝐒ⁱ²ᵉ
                    xk, _ = find_shocks(Val(:LagrangeNewton), x_track, kron_buffer, kron_buffer2,
                                        J, 𝐒ⁱ, Qμ, shock_independent;
                                        max_iter = restart_max_iter, tol = tol, verbose = false)
                    if res_norm(xk, Qμ, shock_independent) < tol
                        x_track = xk
                    else
                        q_ok = false
                        break
                    end
                end
                if q_ok
                    best_x = x_track
                    best_res = res_norm(x_track, 𝐒ⁱ²ᵉ, shock_independent)
                    best_n = ℒ.norm(x_track)
                end
            end
        end
    end
    if best_x === nothing
        # Null-direction escape at the best failure point: where Newton
        # stalls, the Jacobian is near-singular; stepping along its null
        # space moves x a lot while F barely changes (to first order),
        # hopping off the shelf floor for a fresh Newton try. One SVD plus a
        # few damped solves; deterministic.
        K2b = zeros(R, size(𝐒ⁱ²ᵉ, 2), n)
        JmI = Matrix{R}(ℒ.I(n))
        compressed_kron²!(K2b, fail_x, JmI)
        Jf = 𝐒ⁱ + 2 * (𝐒ⁱ²ᵉ * K2b)
        Fsvd = ℒ.svd(Jf)
        v = Fsvd.V[:, end]
        escale = max(ℒ.norm(fail_x), one(R))
        for δ in (R(0.5), R(2), R(8)), sgn in (one(R), -one(R))
            xs = fail_x + sgn * δ * escale * v
            all(isfinite, xs) || continue
            xk, _ = find_shocks(Val(:LagrangeNewton), xs, kron_buffer, kron_buffer2,
                                J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, shock_independent;
                                max_iter = restart_max_iter, tol = tol, verbose = false)
            rk = res_norm(xk, 𝐒ⁱ²ᵉ, shock_independent)
            nk = ℒ.norm(xk)
            if rk < tol && nk < best_n
                best_x, best_res, best_n = xk, rk, nk
                break
            end
        end
    end
    if best_x === nothing && rescue_newton
        # Stage 5: large-radius multistart plain Newton from randn starts.
        # Roots can sit at radius >> 1 (observed ||x|| ~ 35-423 in the
        # pruned-2nd EA pocket) — far outside the axis probes' reach and
        # past the folds where the b/Q homotopies die, so stages 1-4 all
        # miss while plain Newton from an off-axis direction at large
        # radius hits the basin. Directions are Random.randn (global RNG).
        # Only reached when every earlier stage failed — zero cost on the
        # happy path. Two-tier acceptance on the part-relative residual
        # ‖F‖ / max(‖A·x‖, ‖Q·k‖, ‖b‖) (same measure as the primary solver):
        #   strict: ratio ≤ 1e-12   → keep min-‖x‖
        #   loose:  ratio ≤ 1e-10   → keep min-ratio, only when the whole
        #           search finds no strict hit
        # The part-norm anchor certifies ill-conditioned huge-radius roots
        # (evaluation floor ≈1e-6..1e-5 ≫ tol·‖b‖ on some pocket steps;
        # part-norm ratios ≈4e-13…6e-11 at exact roots vs ≳1e-3 on shelves;
        # some roots need the 1e-10 loose headroom). Loose points are scored
        # by ratio, not norm: a small-‖x‖ non-root can sit right at the
        # loose threshold and would win a min-norm contest against the true
        # root — strict tier wins whenever any strict root exists anywhere.
        # Starts: randn directions × a log scale ladder from 10 out to
        # 1e120 — censuses show roots marching up with each rescued failure
        # (step1 53, step15 4.2e5, step27 8.8e11, step29 2.3e15, step45
        # 2.6e21, step65 2.3e34; ‖b‖ grows 5 → 1.5e59; Case A NUTS-means
        # point: step39 ‖x‖=7.4e97, step40 root at start scale 1e101 with
        # ratio 8.4e-13). Each accepted root is then certified by a
        # LagrangeNewton warm start when LN's own tolerance allows
        # (huge-radius roots: LN may never pass its relative test — keep
        # the plain-Newton candidate).
        JmI5 = Matrix{R}(ℒ.I(n))
        K25 = zeros(R, size( 𝐒ⁱ²ᵉ, 2), n)
        Jf5 = zeros(R, n, n)
        r5 = zeros(R, n)
        t5 = zeros(R, n)
        xc5 = zeros(R, n)
        xt5 = zeros(R, n)
        dx5 = zeros(R, n)
        kb5 = zeros(R, size( 𝐒ⁱ²ᵉ, 2))
        u5 = zeros(R, n)
        nb5 = ℒ.norm(shock_independent)
        rtol5s = R(1e-12)
        rtol5 = R(1e-10)
        nbrs5 = max(128, 16 * n)
        scales5 = (R(10), R(50), R(200), R(400), R(800), R(3e3), R(1e4),
                   R(3e4), R(1e5), R(3e5), R(1e6), R(3e6), R(1e7), R(3e7),
                   R(1e8), R(3e8), R(1e9), R(3e9), R(1e10), R(3e10), R(1e11),
                   R(3e11), R(1e12), R(3e12), R(1e13), R(3e13), R(1e14),
                   R(3e14), R(1e15), R(3e15), R(1e16), R(3e16), R(1e17),
                   R(3e17), R(1e18), R(1e19), R(1e20), R(1e21), R(3e21),
                   R(1e22), R(3e22), R(1e23), R(3e23), R(1e24), R(3e24),
                   R(1e25), R(3e25), R(1e26), R(3e26), R(1e27), R(3e27),
                   R(1e28), R(3e28), R(1e29), R(3e29), R(1e30), R(3e30),
                   R(1e31), R(3e31), R(1e32), R(3e32), R(1e33), R(3e33),
                   R(1e34), R(3e34), R(1e35), R(3e35), R(1e36), R(3e36),
                   R(1e37), R(3e37), R(1e38), R(3e38), R(1e39), R(3e39),
                   R(1e40), R(3e40), R(1e41), R(3e41), R(1e42), R(3e42),
                   R(1e43), R(3e43), R(1e44), R(3e44), R(1e45), R(3e45),
                   R(1e46), R(3e46), R(1e47), R(3e47), R(1e48), R(3e48),
                   R(1e49), R(3e49), R(1e50), R(3e50), R(1e51), R(3e51),
                   R(1e52), R(3e52), R(1e53), R(3e53), R(1e54), R(3e54),
                   R(1e55), R(3e55), R(1e56), R(3e56), R(1e57), R(3e57),
                   R(1e58), R(3e58), R(1e59), R(3e59), R(1e60), R(3e60),
                   R(1e61), R(3e61), R(1e62), R(3e62), R(1e63), R(3e63),
                   R(1e64), R(3e64), R(1e65), R(3e65), R(1e66), R(3e66),
                   R(1e67), R(3e67), R(1e68), R(3e68), R(1e69), R(3e69),
                   R(1e70), R(3e70), R(1e71), R(3e71), R(1e72), R(3e72),
                   R(1e73), R(3e73), R(1e74), R(3e74), R(1e75), R(3e75),
                   R(1e76), R(3e76), R(1e77), R(3e77), R(1e78), R(3e78),
                   R(1e79), R(3e79), R(1e80), R(3e80), R(1e81), R(3e81),
                   R(1e82), R(3e82), R(1e83), R(3e83), R(1e84), R(3e84),
                   R(1e85), R(3e85), R(1e86), R(3e86), R(1e87), R(3e87),
                   R(1e88), R(3e88), R(1e89), R(3e89), R(1e90), R(3e90),
                   R(1e91), R(3e91), R(1e92), R(3e92), R(1e93), R(3e93),
                   R(1e94), R(3e94), R(1e95), R(3e95), R(1e96), R(3e96),
                   R(1e97), R(3e97), R(1e98), R(3e98), R(1e99), R(3e99),
                   R(1e100), R(3e100), R(1e101), R(3e101), R(1e102),
                   R(3e102), R(1e103), R(3e103), R(1e104), R(3e104),
                   R(1e105), R(3e105), R(1e106), R(3e106), R(1e107),
                   R(3e107), R(1e108), R(3e108), R(1e109), R(3e109),
                   R(1e110), R(3e110), R(1e111), R(3e111), R(1e112),
                   R(3e112), R(1e113), R(3e113), R(1e114), R(3e114),
                   R(1e115), R(3e115), R(1e116), R(3e116), R(1e117),
                   R(3e117), R(1e118), R(3e118), R(1e119), R(3e119),
                   R(1e120))
        best_cand5 = zeros(R, n)
        best_cand_n5 = R(Inf)
        best_found5 = false
        best_loose5 = zeros(R, n)
        best_loose_ratio5 = R(Inf)
        best_loose_found5 = false
        for s5 in scales5, k5 in 1:nbrs5
            dn5 = Random.randn(n)
            @inbounds for i in 1:n
                xc5[i] = s5 * R(dn5[i])
            end
            hit_strict5 = false
            for _ in 1:100
                # r5 = 𝐒ⁱ*x + 𝐒ⁱ²ᵉ*ck2p(x) - shock_independent
                compressed_kron²_power!(kb5, xc5)
                ℒ.mul!(r5, 𝐒ⁱ, xc5)
                nu5 = ℒ.norm(r5)
                ℒ.mul!(u5, 𝐒ⁱ²ᵉ, kb5)
                nv5 = ℒ.norm(u5)
                ℒ.axpy!(1, u5, r5)
                ℒ.axpy!(-1, shock_independent, r5)
                nr5 = ℒ.norm(r5)
                # Two-tier acceptance on the part-relative residual (see
                # block comment above): strict tier keeps min-‖x‖; loose
                # tier keeps min-ratio and is only eligible when no strict
                # candidate exists in the whole search. On a strict hit do
                # NOT stop: acceptance fires as soon as the ratio crosses
                # the threshold, which can leave the last Newton step on
                # the table — keep iterating until the line search stalls
                # at the evaluation floor, then record the final point.
                part5 = max(nu5, nv5, nb5)
                ratio5 = part5 > 0 ? nr5 / part5 : nr5
                if ratio5 <= rtol5s
                    hit_strict5 = true
                elseif ratio5 <= rtol5 && ratio5 < best_loose_ratio5
                    copyto!(best_loose5, xc5)
                    best_loose_ratio5 = ratio5
                    best_loose_found5 = true
                end
                # Jf5 = 𝐒ⁱ + 2 𝐒ⁱ²ᵉ * ck2p(x, I)
                compressed_kron²!(K25, xc5, JmI5)
                copyto!(Jf5, 𝐒ⁱ)
                ℒ.mul!(Jf5, 𝐒ⁱ²ᵉ, K25, 2, 1)
                lu5 = ℒ.lu(Jf5, check = false)
                ℒ.issuccess(lu5) || break
                ℒ.ldiv!(dx5, lu5, r5)
                ℒ.rmul!(dx5, -1)
                all(isfinite, dx5) || break
                α5 = one(R)
                accepted5 = false
                for _ls in 1:30
                    @inbounds for i in 1:n
                        xt5[i] = xc5[i] + α5 * dx5[i]
                    end
                    compressed_kron²_power!(kb5, xt5)
                    ℒ.mul!(t5, 𝐒ⁱ, xt5)
                    ℒ.mul!(t5, 𝐒ⁱ²ᵉ, kb5, 1, 1)
                    ℒ.axpy!(-1, shock_independent, t5)
                    if ℒ.norm(t5) < nr5
                        copyto!(xc5, xt5)
                        accepted5 = true
                        break
                    end
                    α5 /= 2
                end
                accepted5 || break
            end
            if hit_strict5
                # Re-evaluate the final (polished) point and record it in
                # the strict tier if it still passes — line search only
                # decreases ‖F‖, so a strict mid-trajectory hit stays
                # strict unless the part-norm ratio moved for other
                # reasons (take the final point only when it qualifies).
                compressed_kron²_power!(kb5, xc5)
                ℒ.mul!(r5, 𝐒ⁱ, xc5)
                nu5 = ℒ.norm(r5)
                ℒ.mul!(u5, 𝐒ⁱ²ᵉ, kb5)
                nv5 = ℒ.norm(u5)
                ℒ.axpy!(1, u5, r5)
                ℒ.axpy!(-1, shock_independent, r5)
                nr5 = ℒ.norm(r5)
                part5 = max(nu5, nv5, nb5)
                ratio5 = part5 > 0 ? nr5 / part5 : nr5
                if ratio5 <= rtol5s
                    nk5 = ℒ.norm(xc5)
                    if nk5 < best_cand_n5
                        copyto!(best_cand5, xc5)
                        best_cand_n5 = nk5
                        best_found5 = true
                    end
                end
            end
        end
        if best_found5 || best_loose_found5
            strict5 = best_found5
            cand5 = strict5 ? best_cand5 : best_loose5
            xk5, mk5 = find_shocks(Val(:LagrangeNewton), copy(cand5), kron_buffer,
                                   kron_buffer2, J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, shock_independent;
                                   max_iter = restart_max_iter, tol = tol, verbose = false)
            # LagrangeNewton seeds its initial merit at ‖b‖ (not ‖F(x₀)‖) and
            # starts from λ = 0, so the first full KKT step can walk an exact
            # root to a different one (observed: 423 -> 738 on the step-5
            # pocket system). From the strict tier keep the min-norm point
            # unless LN returns an equally-normed-or-smaller certified root.
            # From the loose tier any LN-certified root is an upgrade (it
            # passes the ‖b‖-anchored test the loose candidate cannot).
            if mk5 && (ℒ.norm(xk5) < ℒ.norm(cand5) || !strict5)
                best_x = xk5
            else
                best_x = cand5
            end
            best_res = res_norm(best_x, 𝐒ⁱ²ᵉ, shock_independent)
            best_n = ℒ.norm(best_x)
            verbose && @info "LagrangeNewton restarts: large-radius Newton rescue" newton_certified = mk5 strict_tier = strict5 residual = best_res norm = best_n
        end
    end
    matched = best_x !== nothing
    verbose && @info "LagrangeNewton restarts" starts = length(starts) matched = matched best_residual = best_res
    matched && return best_x, true
    return zeros(R, n), false
end

"""
Limited-budget random-restart fallback around the LagrangeNewton shock solver
(third-order systems only): A·x + Q₂·k₂(x) + Q₃·k₃(x) = b.

Same stage structure as the second-order variant (linear/axis starts,
b-homotopy, μ-homotopy scaling both Q₂ and Q₃, null-direction escape with the
cubic Jacobian, large-radius multistart plain Newton), with the part-relative
residual ‖F‖ / max(‖A·x‖, ‖Q₂·k₂‖, ‖Q₃·k₃‖, ‖b‖) as acceptance measure.
Stage-5 directions use `Random.randn` (global RNG). The scale ladder is capped
at 1e99 (cubic terms overflow in Float64 beyond) with isfinite guards.
Set `rescue_newton = false` to disable stage 5.
"""
function find_shocks_with_restarts(::Val{:LagrangeNewton},
                    kron_buffer::Vector{R},
                    kron_buffer²::Vector{R},
                    kron_buffer2::AbstractMatrix{R},
                    kron_buffer3::AbstractMatrix{R},
                    kron_buffer4::AbstractMatrix{R},
                    J,
                    𝐒ⁱ::AbstractMatrix{R},
                    𝐒ⁱ²ᵉ::AbstractMatrix{R},
                    𝐒ⁱ³ᵉ::AbstractMatrix{R},
                    shock_independent::Vector{R};
                    tol::AbstractFloat = 1e-13,
                    verbose::Bool = false,
                    n_restarts::Int = 16,
                    restart_max_iter::Int = 200,
                    rescue_newton::Bool = true) where R <: Real
    n = size(𝐒ⁱ, 2)
    buf2 = similar(kron_buffer)
    buf3 = similar(kron_buffer²)
    part_buf = similar(shock_independent)
    res_buf = similar(shock_independent)
    # Part-relative residual for A·x + Q₂·k₂(x) + Q₃·k₃(x) − b = 0.
    function res_norm3(x, Q2m, Q3m, bv)
        compressed_kron²_power!(buf2, x)
        compressed_kron³_power!(buf3, x)
        ℒ.mul!(res_buf, 𝐒ⁱ, x)
        norm_a = ℒ.norm(res_buf)
        ℒ.mul!(part_buf, Q2m, buf2)
        norm_q2 = ℒ.norm(part_buf)
        ℒ.axpy!(1, part_buf, res_buf)
        ℒ.mul!(part_buf, Q3m, buf3)
        norm_q3 = ℒ.norm(part_buf)
        ℒ.axpy!(1, part_buf, res_buf)
        norm_b = ℒ.norm(bv)
        ℒ.axpby!(1, bv, -1, res_buf)
        nr = ℒ.norm(res_buf)
        part = max(norm_a, norm_q2, norm_q3, norm_b)
        part > 0 ? nr / part : nr
    end
    # Starts: the linear solution (when S^i is nonsingular) plus
    # coordinate-axis probes at a scale ladder, alternating signs.
    starts = Vector{Vector{R}}()
    luS = ℒ.lu(𝐒ⁱ, check = false)
    if ℒ.issuccess(luS)
        xl = luS \ shock_independent
        if all(isfinite, xl)
            push!(starts, xl)
        end
    end
    s_lin = isempty(starts) ? one(R) : max(ℒ.norm(starts[end]) / sqrt(n), one(R))
    k = 0
    for s in (s_lin, one(R), 10 * one(R), 100 * one(R))
        length(starts) > n_restarts && break
        for d in 1:n
            length(starts) > n_restarts && break
            v = zeros(R, n)
            v[d] = isodd(k) ? s : -s
            k += 1
            push!(starts, v)
        end
    end
    best_x = nothing
    best_res = R(Inf)
    best_n = R(Inf)
    fail_x = zeros(R, n)
    fail_res = R(Inf)
    for st in starts
        xk, _ = find_shocks(Val(:LagrangeNewton), st, kron_buffer, kron_buffer²,
                            kron_buffer2, kron_buffer3, kron_buffer4,
                            J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent;
                            max_iter = restart_max_iter, tol = tol, verbose = false)
        rk = res_norm3(xk, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
        nk = ℒ.norm(xk)
        # Minimum-norm pick among matches: below tol all residuals are
        # numerical noise, while norm differences move the likelihood's
        # shocks² penalty one-for-one.
        if rk < tol && nk < best_n
            best_x, best_res, best_n = xk, rk, nk
        end
        if rk < fail_res
            fail_x, fail_res = xk, rk
        end
    end
    if best_x === nothing
        # Homotopy in b: F_λ(x) = Sⁱx + Q₂k₂(x) + Q₃k₃(x) − λb, λ: 0 → 1.
        Kλ = 12
        x_track = zeros(R, n)
        h_ok = true
        for kλ in 1:Kλ
            λ = R(kλ) / R(Kλ)
            bl = λ * shock_independent
            xk, _ = find_shocks(Val(:LagrangeNewton), x_track, kron_buffer, kron_buffer²,
                                kron_buffer2, kron_buffer3, kron_buffer4,
                                J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, bl;
                                max_iter = restart_max_iter, tol = tol, verbose = false)
            if res_norm3(xk, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, bl) < tol
                x_track = xk
            else
                h_ok = false
                break
            end
        end
        if h_ok
            best_x = x_track
            best_res = res_norm3(x_track, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
            best_n = ℒ.norm(x_track)
        end
    end
    if best_x === nothing
        # μ-homotopy scaling both Q₂ and Q₃ from the linear solution:
        # G_μ(x) = Sⁱx + μQ₂k₂(x) + μQ₃k₃(x) − b.
        luS2 = ℒ.lu(𝐒ⁱ, check = false)
        if ℒ.issuccess(luS2)
            xl2 = luS2 \ shock_independent
            if all(isfinite, xl2)
                Kμ = 12
                x_track = copy(xl2)
                q_ok = true
                for kμ in 1:Kμ
                    μ = R(kμ) / R(Kμ)
                    Q2μ = μ * 𝐒ⁱ²ᵉ
                    Q3μ = μ * 𝐒ⁱ³ᵉ
                    xk, _ = find_shocks(Val(:LagrangeNewton), x_track, kron_buffer, kron_buffer²,
                                        kron_buffer2, kron_buffer3, kron_buffer4,
                                        J, 𝐒ⁱ, Q2μ, Q3μ, shock_independent;
                                        max_iter = restart_max_iter, tol = tol, verbose = false)
                    if res_norm3(xk, Q2μ, Q3μ, shock_independent) < tol
                        x_track = xk
                    else
                        q_ok = false
                        break
                    end
                end
                if q_ok
                    best_x = x_track
                    best_res = res_norm3(x_track, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
                    best_n = ℒ.norm(x_track)
                end
            end
        end
    end
    if best_x === nothing
        # Null-direction escape at the best failure point with the cubic
        # Jacobian Jf = Sⁱ + 2·Sⁱ²ᵉ·K₂(x) + 3·Sⁱ³ᵉ·K₃(x).
        K2b = zeros(R, size(𝐒ⁱ²ᵉ, 2), n)
        K3b = zeros(R, size(𝐒ⁱ³ᵉ, 2), n)
        JmI = Matrix{R}(ℒ.I(n))
        compressed_kron²!(K2b, fail_x, JmI)
        compressed_kron³!(K3b, fail_x, fail_x, JmI)
        Jf = 𝐒ⁱ + 2 * (𝐒ⁱ²ᵉ * K2b) + 3 * (𝐒ⁱ³ᵉ * K3b)
        Fsvd = ℒ.svd(Jf)
        v = Fsvd.V[:, end]
        escale = max(ℒ.norm(fail_x), one(R))
        for δ in (R(0.5), R(2), R(8)), sgn in (one(R), -one(R))
            xs = fail_x + sgn * δ * escale * v
            all(isfinite, xs) || continue
            xk, _ = find_shocks(Val(:LagrangeNewton), xs, kron_buffer, kron_buffer²,
                                kron_buffer2, kron_buffer3, kron_buffer4,
                                J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent;
                                max_iter = restart_max_iter, tol = tol, verbose = false)
            rk = res_norm3(xk, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
            nk = ℒ.norm(xk)
            if rk < tol && nk < best_n
                best_x, best_res, best_n = xk, rk, nk
                break
            end
        end
    end
    if best_x === nothing && rescue_newton
        # Stage 5: large-radius multistart plain Newton from randn starts on
        # the cubic system. Two-tier acceptance on the part-relative
        # residual ‖F‖ / max(‖A·x‖, ‖Q₂·k₂‖, ‖Q₃·k₃‖, ‖b‖): strict ratio ≤
        # 1e-12 keeps min-‖x‖, loose ratio ≤ 1e-10 keeps min-ratio only when
        # no strict hit exists. Ladder capped at 1e99 (cubic terms overflow
        # Float64 beyond) with isfinite guards throughout.
        JmI5 = Matrix{R}(ℒ.I(n))
        K25 = zeros(R, size(𝐒ⁱ²ᵉ, 2), n)
        K35 = zeros(R, size(𝐒ⁱ³ᵉ, 2), n)
        Jf5 = zeros(R, n, n)
        r5 = zeros(R, n)
        t5 = zeros(R, n)
        xc5 = zeros(R, n)
        xt5 = zeros(R, n)
        dx5 = zeros(R, n)
        kb2 = zeros(R, size(𝐒ⁱ²ᵉ, 2))
        kb3 = zeros(R, size(𝐒ⁱ³ᵉ, 2))
        u5 = zeros(R, n)
        w5 = zeros(R, n)
        nb5 = ℒ.norm(shock_independent)
        rtol5s = R(1e-12)
        rtol5 = R(1e-10)
        nbrs5 = max(128, 16 * n)
        scales5_3 = (R(10), R(50), R(200), R(400), R(800), R(3e3), R(1e4),
                   R(3e4), R(1e5), R(3e5), R(1e6), R(3e6), R(1e7), R(3e7),
                   R(1e8), R(3e8), R(1e9), R(3e9), R(1e10), R(3e10), R(1e11),
                   R(3e11), R(1e12), R(3e12), R(1e13), R(3e13), R(1e14),
                   R(3e14), R(1e15), R(3e15), R(1e16), R(3e16), R(1e17),
                   R(3e17), R(1e18), R(1e19), R(1e20), R(1e21), R(3e21),
                   R(1e22), R(3e22), R(1e23), R(3e23), R(1e24), R(3e24),
                   R(1e25), R(3e25), R(1e26), R(3e26), R(1e27), R(3e27),
                   R(1e28), R(3e28), R(1e29), R(3e29), R(1e30), R(3e30),
                   R(1e31), R(3e31), R(1e32), R(3e32), R(1e33), R(3e33),
                   R(1e34), R(3e34), R(1e35), R(3e35), R(1e36), R(3e36),
                   R(1e37), R(3e37), R(1e38), R(3e38), R(1e39), R(3e39),
                   R(1e40), R(3e40), R(1e41), R(3e41), R(1e42), R(3e42),
                   R(1e43), R(3e43), R(1e44), R(3e44), R(1e45), R(3e45),
                   R(1e46), R(3e46), R(1e47), R(3e47), R(1e48), R(3e48),
                   R(1e49), R(3e49), R(1e50), R(3e50), R(1e51), R(3e51),
                   R(1e52), R(3e52), R(1e53), R(3e53), R(1e54), R(3e54),
                   R(1e55), R(3e55), R(1e56), R(3e56), R(1e57), R(3e57),
                   R(1e58), R(3e58), R(1e59), R(3e59), R(1e60), R(3e60),
                   R(1e61), R(3e61), R(1e62), R(3e62), R(1e63), R(3e63),
                   R(1e64), R(3e64), R(1e65), R(3e65), R(1e66), R(3e66),
                   R(1e67), R(3e67), R(1e68), R(3e68), R(1e69), R(3e69),
                   R(1e70), R(3e70), R(1e71), R(3e71), R(1e72), R(3e72),
                   R(1e73), R(3e73), R(1e74), R(3e74), R(1e75), R(3e75),
                   R(1e76), R(3e76), R(1e77), R(3e77), R(1e78), R(3e78),
                   R(1e79), R(3e79), R(1e80), R(3e80), R(1e81), R(3e81),
                   R(1e82), R(3e82), R(1e83), R(3e83), R(1e84), R(3e84),
                   R(1e85), R(3e85), R(1e86), R(3e86), R(1e87), R(3e87),
                   R(1e88), R(3e88), R(1e89), R(3e89), R(1e90), R(3e90),
                   R(1e91), R(3e91), R(1e92), R(3e92), R(1e93), R(3e93),
                   R(1e94), R(3e94), R(1e95), R(3e95), R(1e96), R(3e96),
                   R(1e97), R(3e97), R(1e98), R(3e98), R(1e99))
        best_cand5 = zeros(R, n)
        best_cand_n5 = R(Inf)
        best_found5 = false
        best_loose5 = zeros(R, n)
        best_loose_ratio5 = R(Inf)
        best_loose_found5 = false
        for s5 in scales5_3, k5 in 1:nbrs5
            dn5 = Random.randn(n)
            @inbounds for i in 1:n
                xc5[i] = s5 * R(dn5[i])
            end
            all(isfinite, xc5) || continue
            hit_strict5 = false
            for _ in 1:100
                # r5 = 𝐒ⁱ*x + 𝐒ⁱ²ᵉ*ck2p(x) + 𝐒ⁱ³ᵉ*ck3p(x) - shock_independent
                compressed_kron²_power!(kb2, xc5)
                compressed_kron³_power!(kb3, xc5)
                (all(isfinite, kb2) && all(isfinite, kb3)) || break
                ℒ.mul!(r5, 𝐒ⁱ, xc5)
                nu5 = ℒ.norm(r5)
                ℒ.mul!(u5, 𝐒ⁱ²ᵉ, kb2)
                nq25 = ℒ.norm(u5)
                ℒ.axpy!(1, u5, r5)
                ℒ.mul!(w5, 𝐒ⁱ³ᵉ, kb3)
                nq35 = ℒ.norm(w5)
                ℒ.axpy!(1, w5, r5)
                ℒ.axpy!(-1, shock_independent, r5)
                nr5 = ℒ.norm(r5)
                isfinite(nr5) || break
                part5 = max(nu5, nq25, nq35, nb5)
                ratio5 = part5 > 0 ? nr5 / part5 : nr5
                if ratio5 <= rtol5s
                    hit_strict5 = true
                elseif ratio5 <= rtol5 && ratio5 < best_loose_ratio5
                    copyto!(best_loose5, xc5)
                    best_loose_ratio5 = ratio5
                    best_loose_found5 = true
                end
                # Jf5 = 𝐒ⁱ + 2 𝐒ⁱ²ᵉ * ck2p(x, I) + 3 𝐒ⁱ³ᵉ * ck3p(x, x, I)
                compressed_kron²!(K25, xc5, JmI5)
                compressed_kron³!(K35, xc5, xc5, JmI5)
                copyto!(Jf5, 𝐒ⁱ)
                ℒ.mul!(Jf5, 𝐒ⁱ²ᵉ, K25, 2, 1)
                ℒ.mul!(Jf5, 𝐒ⁱ³ᵉ, K35, 3, 1)
                lu5 = ℒ.lu(Jf5, check = false)
                ℒ.issuccess(lu5) || break
                ℒ.ldiv!(dx5, lu5, r5)
                ℒ.rmul!(dx5, -1)
                all(isfinite, dx5) || break
                α5 = one(R)
                accepted5 = false
                for _ls in 1:30
                    @inbounds for i in 1:n
                        xt5[i] = xc5[i] + α5 * dx5[i]
                    end
                    all(isfinite, xt5) || (α5 /= 2; continue)
                    compressed_kron²_power!(kb2, xt5)
                    compressed_kron³_power!(kb3, xt5)
                    ℒ.mul!(t5, 𝐒ⁱ, xt5)
                    ℒ.mul!(t5, 𝐒ⁱ²ᵉ, kb2, 1, 1)
                    ℒ.mul!(t5, 𝐒ⁱ³ᵉ, kb3, 1, 1)
                    ℒ.axpy!(-1, shock_independent, t5)
                    nt5 = ℒ.norm(t5)
                    if isfinite(nt5) && nt5 < nr5
                        copyto!(xc5, xt5)
                        accepted5 = true
                        break
                    end
                    α5 /= 2
                end
                accepted5 || break
            end
            if hit_strict5
                compressed_kron²_power!(kb2, xc5)
                compressed_kron³_power!(kb3, xc5)
                ℒ.mul!(r5, 𝐒ⁱ, xc5)
                nu5 = ℒ.norm(r5)
                ℒ.mul!(u5, 𝐒ⁱ²ᵉ, kb2)
                nq25 = ℒ.norm(u5)
                ℒ.axpy!(1, u5, r5)
                ℒ.mul!(w5, 𝐒ⁱ³ᵉ, kb3)
                nq35 = ℒ.norm(w5)
                ℒ.axpy!(1, w5, r5)
                ℒ.axpy!(-1, shock_independent, r5)
                nr5 = ℒ.norm(r5)
                part5 = max(nu5, nq25, nq35, nb5)
                ratio5 = part5 > 0 ? nr5 / part5 : nr5
                if ratio5 <= rtol5s
                    nk5 = ℒ.norm(xc5)
                    if nk5 < best_cand_n5
                        copyto!(best_cand5, xc5)
                        best_cand_n5 = nk5
                        best_found5 = true
                    end
                end
            end
        end
        if best_found5 || best_loose_found5
            strict5 = best_found5
            cand5 = strict5 ? best_cand5 : best_loose5
            xk5, mk5 = find_shocks(Val(:LagrangeNewton), copy(cand5), kron_buffer, kron_buffer²,
                                   kron_buffer2, kron_buffer3, kron_buffer4,
                                   J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent;
                                   max_iter = restart_max_iter, tol = tol, verbose = false)
            if mk5 && (ℒ.norm(xk5) < ℒ.norm(cand5) || !strict5)
                best_x = xk5
            else
                best_x = cand5
            end
            best_res = res_norm3(best_x, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
            best_n = ℒ.norm(best_x)
            verbose && @info "LagrangeNewton restarts (3rd order): large-radius Newton rescue" newton_certified = mk5 strict_tier = strict5 residual = best_res norm = best_n
        end
    end
    matched = best_x !== nothing
    verbose && @info "LagrangeNewton restarts (3rd order)" starts = length(starts) matched = matched best_residual = best_res
    matched && return best_x, true
    return zeros(R, n), false
end


function find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{R},
                    kron_buffer::Vector{R},
                    kron_buffer²::Vector{R},
                    kron_buffer2::AbstractMatrix{R},
                    kron_buffer3::AbstractMatrix{R},
                    kron_buffer4::AbstractMatrix{R},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{R},
                    𝐒ⁱ²ᵉ::AbstractMatrix{R},
                    𝐒ⁱ³ᵉ::AbstractMatrix{R},
                    shock_independent::Vector{R};
                    max_iter::Int = 1000,
                    tol::AbstractFloat = 1e-13,
                    verbose::Bool = false) where R <: Real
    if size(𝐒ⁱ, 1) == size(𝐒ⁱ, 2) && max_iter > 0
        square_x, matched = find_square_shocks(initial_guess, kron_buffer, kron_buffer²,
            kron_buffer2, kron_buffer3, J, 𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent;
            max_iter = max_iter, tol = tol)
        matched && return square_x, true
    end
    x = copy(initial_guess)

    λ = zeros(R, size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm_part = one(R)
    
    Δnorm = R(1e12)

    x̂ = copy(shock_independent)

    part_buf = zeros(R, size(𝐒ⁱ, 1))
    if !all(iszero, x)
        shock_polynomial_residual!(x̂, part_buf, kron_buffer, kron_buffer², x,
                                   𝐒ⁱ, 𝐒ⁱ²ᵉ, 𝐒ⁱ³ᵉ, shock_independent)
        ℒ.rmul!(x̂, -1)
    end

    x̄ = zeros(R, size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)

    ∂x̂ = zero(𝐒ⁱ)
    
    fxλ = zeros(R, length(xλ))
    
    fxλp = zeros(R, length(xλ), length(xλ))

    n_shock = size(𝐒ⁱ, 2)
    tmp = zeros(R, n_shock, n_shock)
    tmp_coeff = zeros(R, length(kron_buffer))
    tmp3_coeff = zeros(R, size(𝐒ⁱ³ᵉ, 2))

    lI = R(-2) * Matrix(ℒ.I(n_shock))
    
    iter = 0
    @inbounds for i in 1:max_iter
        iter = i
        # Initialize x ⊗ x for the current iterate before using kron_buffer in Jacobian terms.
        compressed_kron²_power!(kron_buffer, x)
        compressed_kron²!(kron_buffer2, x, J)
        compressed_kron³!(kron_buffer3, x, x, J)

        copy!(∂x, 𝐒ⁱ)
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

        ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]
        
        # ∇²[coeff'·compressed_kron²_power(x)] is 2·compressed_pair_hessian!(coeff),
        # so the pair coefficients carry the factor 2 (as in the 2nd-order method).
        ℒ.mul!(tmp_coeff, 𝐒ⁱ²ᵉ', λ, 2, 0)
        compressed_pair_hessian!(tmp, tmp_coeff)
        ℒ.mul!(tmp3_coeff, 𝐒ⁱ³ᵉ', λ, 6, 0)
        compressed_triple_hessian!(tmp, tmp3_coeff, x)
        ℒ.axpy!(1, lI, tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x
        # fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        f̂xλp = ℒ.lu(fxλp, check = false)

        if !ℒ.issuccess(f̂xλp)
            return x, false
        end

        ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        compressed_kron²_power!(kron_buffer, x)

        compressed_kron³_power!(kron_buffer², x)

        # Part-relative residual: ‖F‖ / max(‖A·x‖, ‖Q·k₂‖, ‖Q₃·k₃‖, ‖b‖)
        ℒ.mul!(x̂, 𝐒ⁱ, x)
        norm_a = ℒ.norm(x̂)
        ℒ.mul!(part_buf, 𝐒ⁱ²ᵉ, kron_buffer)
        norm_q2 = ℒ.norm(part_buf)
        ℒ.axpy!(1, part_buf, x̂)
        ℒ.mul!(part_buf, 𝐒ⁱ³ᵉ, kron_buffer²)
        norm_q3 = ℒ.norm(part_buf)
        ℒ.axpy!(1, part_buf, x̂)
        norm_part = max(norm_a, norm_q2, norm_q3, norm1)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        residual = norm_part > 0 ? ℒ.norm(x̂) / norm_part : ℒ.norm(x̂)

        if residual < tol && ℒ.norm(Δxλ) / max(one(R), ℒ.norm(xλ)) < sqrt(tol)
            break
        end
    end

    # Final part-relative residual at the returned x.
    compressed_kron²_power!(kron_buffer, x)
    compressed_kron³_power!(kron_buffer², x)
    ℒ.mul!(x̂, 𝐒ⁱ, x)
    norm_a = ℒ.norm(x̂)
    ℒ.mul!(part_buf, 𝐒ⁱ²ᵉ, kron_buffer)
    norm_q2 = ℒ.norm(part_buf)
    ℒ.axpy!(1, part_buf, x̂)
    ℒ.mul!(part_buf, 𝐒ⁱ³ᵉ, kron_buffer²)
    norm_q3 = ℒ.norm(part_buf)
    ℒ.axpy!(1, part_buf, x̂)
    norm_part = max(norm_a, norm_q2, norm_q3, norm1)
    ℒ.axpby!(1, shock_independent, -1, x̂)
    residual = norm_part > 0 ? ℒ.norm(x̂) / norm_part : ℒ.norm(x̂)

    step_norm = ℒ.norm(Δxλ) / max(one(R), ℒ.norm(xλ))
    matched = residual < tol && step_norm < sqrt(tol)
    verbose && @info "LagrangeNewton solve (3rd order)" iterations = iter residual = residual step_norm = step_norm matched = matched
    return x, matched
end







# @stable default_mode = "disable" begin

# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13,
#                     verbose::Bool = false) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, -2, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))'
#         end

#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     ℒ.kron!(kron_buffer, x, x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron_buffer

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
    
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:SLSQP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13,
#                     verbose::Bool = false) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         if length(grad) > 0
#             copy!(grad, X)

#             ℒ.rmul!(grad, 2)
#             # grad .= 2 .* X
#         end
        
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         if length(jac) > 0
#             ℒ.kron!(kron_buffer2, J, x)

#             ℒ.kron!(kron_buffer3, J, kron_buffer)

#             copy!(jac', 𝐒ⁱ)

#             ℒ.mul!(jac', 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

#             ℒ.mul!(jac', 𝐒ⁱ³ᵉ, kron_buffer3, -3, -1)
#             # jac .= -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(J, x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(J, ℒ.kron(x,x)))'
#         end

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x) - 𝐒ⁱ³ᵉ * ℒ.kron!(kron_buffer², x, kron_buffer)
#     end
    
#     opt = NLopt.Opt(NLopt.:LD_SLSQP, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = try 
#         NLopt.optimize(opt, initial_guess)
#     catch
#         return initial_guess, false
#     end

#     # println("SLSQP - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println(ℒ.norm(x))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end






# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13,
#                     verbose::Bool = false) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X)
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])

#     # println("COBYLA: $(opt.numevals)")

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end



# function find_shocks(::Val{:COBYLA},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 10000,
#                     tol::Float64 = 1e-13,
#                     verbose::Bool = false) # will fail for higher or lower precision
#     function objective_optim_fun(X::Vector{S}, grad::Vector{S}) where S
#         sum(abs2, X)
#     end

#     function constraint_optim(res::Vector{S}, x::Vector{S}, jac::Matrix{S}) where S <: Float64
#         ℒ.kron!(kron_buffer, x, x)

#         ℒ.kron!(kron_buffer², x, kron_buffer)

#         ℒ.mul!(res, 𝐒ⁱ, x)

#         ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

#         ℒ.mul!(res, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

#         ℒ.axpby!(1, shock_independent, -1, res)
#         # res .= shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * ℒ.kron(X,X) - 𝐒ⁱ³ᵉ * ℒ.kron(X, ℒ.kron(X,X))
#     end

#     opt = NLopt.Opt(NLopt.:LN_COBYLA, size(𝐒ⁱ,2))
                    
#     opt.min_objective = objective_optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     # opt.xtol_rel = eps()
#     # opt.ftol_rel = eps()
#     # opt.constrtol_abs = eps() # doesn't work
#     opt.maxeval = max_iter

#     NLopt.equality_constraint!(opt, constraint_optim, fill(eps(),size(𝐒ⁱ,1)))

#     (minf,x,ret) = NLopt.optimize(opt, initial_guess)

#     # println("COBYLA - retcode: $ret, nevals: $(opt.numevals)")

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     solved = ret ∈ Symbol.([
#         NLopt.MAXEVAL_REACHED,
#         NLopt.SUCCESS,
#         NLopt.STOPVAL_REACHED,
#         NLopt.FTOL_REACHED,
#         NLopt.XTOL_REACHED,
#         NLopt.ROUNDOFF_LIMITED,
#     ])
#     # println(ℒ.norm(x))
#     # println(x)
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("COBYLA: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("COBYLA: $(opt.numevals)")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")

#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol && solved
# end

# end # dispatch_doctor





# function find_shocks(::Val{:MadNLP},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(MadNLP.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:Ipopt},
#                     initial_guess::Vector{Float64},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     kron_buffer4::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     max_iter::Int = 500,
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision
#     model = JuMP.Model(Ipopt.Optimizer)

#     JuMP.set_silent(model)

#     JuMP.set_optimizer_attribute(model, "tol", tol)

#     JuMP.@variable(model, x[1:length(initial_guess)])

#     JuMP.set_start_value.(x, initial_guess)

#     JuMP.@objective(model, Min, sum(abs2,x))

#     JuMP.@constraint(model, 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)) .== shock_independent)

#     JuMP.optimize!(model)

#     x = JuMP.value.(x)

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println(ℒ.norm(y - shock_independent) / max(norm1,norm2))
#     # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
#     # println("SLSQP - $ret: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
#     # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")
#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:newton},
#     kron_buffer::Vector{Float64},
#     kron_buffer2::AbstractMatrix{Float64},
#     J::ℒ.Diagonal{Bool, Vector{Bool}},
#     𝐒ⁱ::AbstractMatrix{Float64},
#     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#     shock_independent::Vector{Float64};
#     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

# 	norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 

#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2) < tol)
#             # println(i)
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:newton},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-13) # will fail for higher or lower precision

#     nExo = Int(sqrt(length(kron_buffer)))

#     x = zeros(nExo)

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̂ = zeros(size(𝐒ⁱ²ᵉ,1))

#     x̄ = zeros(size(𝐒ⁱ²ᵉ,1))

#     Δx = zeros(nExo)

#     ∂x = zero(𝐒ⁱ)

#     Ĵ = ℒ.I(nExo)*2

#     max_iter = 1000

#     norm1 = 1

# 	norm2 = ℒ.norm(shock_independent)

#     for i in 1:max_iter
#         ℒ.kron!(kron_buffer, x, x)
#         ℒ.kron!(kron_buffer², x, kron_buffer)
#         ℒ.kron!(kron_buffer2, Ĵ, x)
#         ℒ.kron!(kron_buffer3, Ĵ, kron_buffer)
        
#         ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
#         ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 1 ,1)
#         ℒ.axpy!(1, 𝐒ⁱ, ∂x)
#         # ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(nExo), x) + 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(nExo), ℒ.kron(x,x)))

#         ∂x̂ = try 
#             ℒ.factorize(∂x)
#         catch
#             return x, false
#         end 
							
#         ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)
#         ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)
#         ℒ.mul!(x̄, 𝐒ⁱ, x)
#         ℒ.axpy!(1, x̄, x̂)
# 				norm1 = ℒ.norm(x̂)
#         ℒ.axpby!(1, shock_independent, -1, x̂)
#         try 
#             ℒ.ldiv!(Δx, ∂x̂, x̂)
#         catch
#             return x, false
#         end
#         # Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
#         # println(ℒ.norm(Δx))
#         if i > 6 && (ℒ.norm(x̂) / max(norm1,norm2)) < tol
#             # println("Iters: $i Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#             break
#         end
        
#         ℒ.axpy!(1, Δx, x)
#         # x += Δx

#         if !all(isfinite.(x))
#             return x, false
#         end
#     end

#     # println("Iters: $max_iter Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
#     return x, ℒ.norm(x̂) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end
    

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     opt.xtol_abs = eps()
#     opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:LBFGS},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = eps()) # will fail for higher or lower precision

#     function optim_fun(x::Vector{S}, grad::Vector{S}) where S <: Float64
#         if length(grad) > 0
#             grad .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#         end

#         return sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     opt = NLopt.Opt(NLopt.:LD_LBFGS, size(𝐒ⁱ,2))
                    
#     opt.min_objective = optim_fun

#     # opt.xtol_abs = eps()
#     # opt.ftol_abs = eps()
#     opt.maxeval = 10000

#     (minf,x,ret) = NLopt.optimize(opt, zeros(size(𝐒ⁱ,2)))

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = Optim.optimize(f,g!,
#         # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X)))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end



# function find_shocks(::Val{:LBFGSjl},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = Optim.optimize(f,g!,
#     # X -> sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X))),
#                         zeros(size(𝐒ⁱ,2)), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)))#; 
#                         # autodiff = :forward)

#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end


# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer²::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     kron_buffer3::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     𝐒ⁱ³ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision

#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X) - 𝐒ⁱ³ᵉ * kron(X,kron(X,X))))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * kron(ℒ.I(length(x)),kron(x,x)))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x) - 𝐒ⁱ³ᵉ * kron(x,kron(x,x))))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)
# println(sol)
#     x = sol.minimizer

#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x) + 𝐒ⁱ³ᵉ * kron(x,kron(x,x))

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end




# function find_shocks(::Val{:speedmapping},
#                     kron_buffer::Vector{Float64},
#                     kron_buffer2::AbstractMatrix{Float64},
#                     J::ℒ.Diagonal{Bool, Vector{Bool}},
#                     𝐒ⁱ::AbstractMatrix{Float64},
#                     𝐒ⁱ²ᵉ::AbstractMatrix{Float64},
#                     shock_independent::Vector{Float64};
#                     tol::Float64 = 1e-15) # will fail for higher or lower precision
#     function f(X)
#         sqrt(sum(abs2, shock_independent - 𝐒ⁱ * X - 𝐒ⁱ²ᵉ * kron(X,X)))
#     end

#     function g!(G, x)
#         G .= - (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)) / sqrt(sum(abs2, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron(x,x)))
#     end

#     sol = speedmapping(zeros(size(𝐒ⁱ,2)), f = f, g! = g!, tol = tol, maps_limit = 10000, stabilize = false)

#     x = sol.minimizer
    
#     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * kron(x,x)

#     norm1 = ℒ.norm(y)

# 	norm2 = ℒ.norm(shock_independent)

#     # println("Norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
#     return x, ℒ.norm(y - shock_independent) / max(norm1,norm2) < tol
# end

end # @stable
