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
                                         verbose::Bool = false)

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

    shock_idxs = so.shock_idxs
    shock²_idxs = so.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = so.var_vol²_idxs
    var²_idxs = so.var²_idxs
    shockvar_idxs = sparse(ℒ.kron(so.e_in_s⁺, so.s_in_s)).nzind

    if third_order
        var_vol³_idxs = to.var_vol³_idxs
        shock_idxs2 = to.shock_idxs2
        shock_idxs3 = to.shock_idxs3
        shock³_idxs = to.shock³_idxs
        shockvar1_idxs = to.shockvar1_idxs
        shockvar2_idxs = to.shockvar2_idxs
        shockvar3_idxs = to.shockvar3_idxs
        shockvar³2_idxs = to.shockvar³2_idxs
        shockvar³_idxs = to.shockvar³_idxs
    end

    fixed_shock_idx = setdiff(1:n_exo, free_shock_idx)

    if isempty(cond_var_idx) && isempty(fixed_shock_idx)
        return zeros(length(free_shock_idx)), true
    end

    J = ℒ.I(n_exo)

    nPast = T.nPast_not_future_and_mixed
    third_order_pruning = third_order && pruning
    ensure_find_shocks_state_buffers!(ws, n_exo, nPast;
                                      third_order = third_order,
                                      third_order_pruning = third_order_pruning)
    kron_state_vol = ws.kron_state_vol
    kron_I_state = ws.kron_I_state

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
                𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
            else
                𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
                𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
                𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

                ℒ.kron!(kron_state_vol, state_vol, state_vol)
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
                𝐒ⁱ²ᵉ = zeros(size(𝐒¹ᵉ, 1), n_exo^2)
            else
                𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
                𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
                𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

                ℒ.kron!(kron_state_vol, state_vol, state_vol)
                ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

                ℒ.kron!(kron_I_state, J, state_vol)
                𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_I_state
                𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
            end
        end

        𝐒ⁱ³ᵉ = nothing
    elseif third_order
        # Third order (pruned or non-pruned)
        II = sparse(ℒ.I(n_exo^2))

        if pruning
            state₁ = initial_state[1][T.past_not_future_and_mixed_idx]
            state₂ = initial_state[2][T.past_not_future_and_mixed_idx]
            state₃ = initial_state[3][T.past_not_future_and_mixed_idx]
            state_vol = vcat(state₁, 1)

            𝐒¹⁻ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
            𝐒²⁻ = @views 𝐒₂[cond_var_idx, var²_idxs]
            𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
            𝐒²⁻ᵛᵉ = @views 𝐒₂[cond_var_idx, shockvar_idxs]
            𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

            𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
            𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
            𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
            𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)
            ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
            ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

            ℒ.kron!(kron_state_vol, state_vol, state_vol)
            ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

            kron_state₁₂ = ws.kron_state₁₂
            ℒ.kron!(kron_state₁₂, state₁, state₂)
            ℒ.mul!(shock_independent, 𝐒²⁻, kron_state₁₂, -1, 1)

            kron_state_vol3 = ws.kron_state_vol3
            ℒ.kron!(kron_state_vol3, state_vol, kron_state_vol)
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

            𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
            𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6
        else
            state = initial_state[T.past_not_future_and_mixed_idx]
            state_vol = vcat(state, 1)

            𝐒¹⁻ᵛ = @views 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
            𝐒¹ᵉ = @views 𝐒₁[cond_var_idx, end-n_exo+1:end]

            𝐒²⁻ᵛ = @views 𝐒₂[cond_var_idx, var_vol²_idxs]
            𝐒²⁻ᵉ = @views 𝐒₂[cond_var_idx, shockvar²_idxs]
            𝐒²ᵉ = @views 𝐒₂[cond_var_idx, shock²_idxs]

            𝐒³⁻ᵛ = @views 𝐒₃[cond_var_idx, var_vol³_idxs]
            𝐒³⁻ᵉ² = @views 𝐒₃[cond_var_idx, shockvar³2_idxs]
            𝐒³⁻ᵉ = @views 𝐒₃[cond_var_idx, shockvar³_idxs]
            𝐒³ᵉ = @views 𝐒₃[cond_var_idx, shock³_idxs]

            shock_independent = copy(conditions)
            ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state_vol, -1, 1)

            ℒ.kron!(kron_state_vol, state_vol, state_vol)
            ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kron_state_vol, -1/2, 1)

            kron_state_vol3 = ws.kron_state_vol3
            ℒ.kron!(kron_state_vol3, state_vol, kron_state_vol)
            ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kron_state_vol3, -1/6, 1)

            ℒ.kron!(kron_I_state, J, state_vol)
            kron_I_state_state = ws.kron_I_state_state
            ℒ.kron!(kron_I_state_state, J, kron_state_vol)

            𝐒ⁱ = 𝐒¹ᵉ +
            𝐒²⁻ᵉ * kron_I_state +
            𝐒³⁻ᵉ² * kron_I_state_state / 2

            𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state_vol) / 2
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

"""
Solve for minimum-norm shocks that satisfy linear and quadratic constraints using a
Lagrange-Newton stepper. For higher-order solutions the global minimum-norm problem is
NP-hard because the number of feasible roots grows exponentially; any gradient-based
solver started at the origin (not just LagrangeNewton) will converge to the root whose
basin contains the origin rather than guaranteeing the global optimum.
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
    x = copy(initial_guess)
    
    λ = zeros(R, size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = one(R)
    
    Δnorm = R(1e12)

    x̂ = copy(shock_independent)

    x̄ = zeros(R, size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)
    
    fxλ = zeros(R, length(xλ))
    
    fxλp = zeros(R, length(xλ), length(xλ))

    tmp = zeros(R, size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    lI = R(-2) * vec(ℒ.I(size(𝐒ⁱ, 2)))

    iter = 0
    @inbounds for i in 1:max_iter
        iter = i
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2)
        ℒ.axpby!(1, 𝐒ⁱ, 2, ∂x)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x))]

        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ)
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

        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x, 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol reached, $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
            # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
        # end
    end

    # println(λ)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(Δxλ) / ℒ.norm(xλ))
    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    residual = ℒ.norm(x̂) / max(norm1,norm2)
    step_norm = ℒ.norm(Δxλ) / ℒ.norm(xλ)
    matched = residual < tol && step_norm < sqrt(tol)
    verbose && @info "LagrangeNewton solve (2nd order)" iterations = iter residual = residual step_norm = step_norm matched = matched
    return x, matched
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
    x = copy(initial_guess)

    λ = zeros(R, size(𝐒ⁱ, 1))
    
    xλ = [  x
            λ   ]

    Δxλ = copy(xλ)

    norm1 = ℒ.norm(shock_independent) 

    norm2 = one(R)
    
    Δnorm = R(1e12)

    x̂ = copy(shock_independent)

    x̄ = zeros(R, size(𝐒ⁱ,2))

    ∂x = zero(𝐒ⁱ)

    ∂x̂ = zero(𝐒ⁱ)
    
    fxλ = zeros(R, length(xλ))
    
    fxλp = zeros(R, length(xλ), length(xλ))

    tmp = zeros(R, size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    tmp2 = zeros(R, size(𝐒ⁱ, 1),size(𝐒ⁱ, 2) * size(𝐒ⁱ, 2))

    II = sparse(ℒ.I(length(x)^2))

    lI = R(-2) * vec(ℒ.I(size(𝐒ⁱ, 2)))
    
    iter = 0
    @inbounds for i in 1:max_iter
        iter = i
        # Initialize x ⊗ x for the current iterate before using kron_buffer in Jacobian terms.
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.kron!(kron_buffer3, J, kron_buffer)

        copy!(∂x, 𝐒ⁱ)
        ℒ.mul!(∂x, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)

        ℒ.mul!(∂x, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)

        ℒ.mul!(x̄, ∂x', λ)
        
        ℒ.axpy!(-2, x, x̄)

        copyto!(fxλ, 1, x̄, 1, size(𝐒ⁱ,2))
        copyto!(fxλ, size(𝐒ⁱ,2) + 1, x̂, 1, size(shock_independent,1))
        # fXλ = [(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
                # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]
        
        x_kron_II!(kron_buffer4, x)
        # ℒ.kron!(kron_buffer4, II, x)
        ℒ.mul!(tmp2, 𝐒ⁱ³ᵉ, kron_buffer4)
        ℒ.mul!(tmp, tmp2', λ)
        ℒ.mul!(tmp, 𝐒ⁱ²ᵉ', λ, 2, 6)
        ℒ.axpy!(1,lI,tmp)

        fxλp[1:size(𝐒ⁱ, 2), 1:size(𝐒ⁱ, 2)] = tmp
        
        fxλp[1:size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)+1:end] = ∂x'

        ℒ.rmul!(∂x, -1)
        fxλp[size(𝐒ⁱ, 2)+1:end, 1:size(𝐒ⁱ, 2)] = ∂x
        # fXλp = [reshape((2 * 𝐒ⁱ²ᵉ + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
        #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]
        
        try
            f̂xλp = ℒ.factorize(fxλp)
            ℒ.ldiv!(Δxλ, f̂xλp, fxλ)
        catch
            # ℒ.svd(fxλp)
            # println("factorization fails")
            return x, false
        end
        
        if !all(isfinite,Δxλ) break end
        
        ℒ.axpy!(-1, Δxλ, xλ)
        # xλ -= Δxλ
    
        # x = xλ[1:size(𝐒ⁱ, 2)]
        copyto!(x, 1, xλ, 1, size(𝐒ⁱ,2))

        # λ = xλ[size(𝐒ⁱ, 2)+1:end]
        copyto!(λ, 1, xλ, size(𝐒ⁱ,2) + 1, length(λ))

        ℒ.kron!(kron_buffer, x, x)

        ℒ.kron!(kron_buffer², x, kron_buffer)

        ℒ.mul!(x̂, 𝐒ⁱ, x)

        ℒ.mul!(x̂, 𝐒ⁱ²ᵉ, kron_buffer, 1, 1)

        ℒ.mul!(x̂, 𝐒ⁱ³ᵉ, kron_buffer², 1, 1)

        norm2 = ℒ.norm(x̂)

        ℒ.axpby!(1, shock_independent, -1, x̂)

        if ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol)
            # println("LagrangeNewton: $i, Tol: $(ℒ.norm(Δxλ) / ℒ.norm(xλ)) reached, x: $x")
            break
        end

        # if i > 500 && ℒ.norm(Δxλ) > 1e-11 && ℒ.norm(Δxλ) > Δnorm
        #     # println(ℒ.norm(Δxλ))
        #     # println(ℒ.norm(x̂) / max(norm1,norm2))
        #     # println("LagrangeNewton: $i, Norm increase")
        #     return x, false
        # end
        # if i == max_iter
        #     println("LagrangeNewton: $i, Max iter reached")
        #     # println(ℒ.norm(Δxλ))
        # end
    end

    # λ = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), kron_buffer))' \ x * 2
    # println("LagrangeNewton: $(ℒ.norm([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))]))")

    # println(ℒ.norm(x))
    # println(x)
    # println(λ)
    # println([(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))' * λ - 2 * x
    # shock_independent - (𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x, x)))])
    # println(fxλp)
    # println(reshape(tmp, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2)))
    # println([reshape((2 * 𝐒ⁱ²ᵉ - 2 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(ℒ.I(length(x)),x)))' * λ, size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2*ℒ.I(size(𝐒ⁱ, 2))  (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))'
    #         -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x)), x) - 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(length(x)), ℒ.kron(x, x)))  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))])
    # println(fxλp)
    # println("Norm: $(ℒ.norm(x̂) / max(norm1,norm2))")
    # println(ℒ.norm(Δxλ))
    # println(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < tol)

    # if !(ℒ.norm(x̂) / max(norm1,norm2) < tol && ℒ.norm(Δxλ) / ℒ.norm(xλ) < sqrt(tol))
    #     println("Find shocks failed. Norm 1: $(ℒ.norm(x̂) / max(norm1,norm2)); Norm 2: $(ℒ.norm(Δxλ) / ℒ.norm(xλ))")
    # end

    residual = ℒ.norm(x̂) / max(norm1,norm2)
    step_norm = ℒ.norm(Δxλ) / ℒ.norm(xλ)
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
