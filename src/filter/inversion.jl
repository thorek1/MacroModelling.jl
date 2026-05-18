@stable default_mode = "disable" begin


# ---------------------------------------------------------------------
# Aumann–Shapley shock decomposition (marginal-contribution driver)
# ---------------------------------------------------------------------
#
# Computes per-period Shapley shares for the inversion-filter shock
# decomposition under pruned 2nd / 3rd order solutions via the path-
# integral identity
#     φᵢ(v, t) = ∫₀¹ ∂Ṽ_t(s·𝟙)/∂xᵢ ds
#                ≈ Σ_k w_k · ∂Ṽ_t(s_k·𝟙)/∂xᵢ      (Gauss–Legendre)
# where Ṽ_t is the polynomial extension of `S → ŝ_t(S)[v]`. The production
# drivers start from the low-order Gauss–Legendre rules (2 nodes at 2nd
# order, 3 at 3rd order) and rerun with 4 nodes only when the coarse
# Shapley-efficiency closure residual exceeds `1e-3`.
#
# Per period the driver maintains, for every Gauss–Legendre node s_k:
#   - one primal pruned-state trajectory under shocks scaled by s_k;
#   - one tangent trajectory per shock direction i = 1..nᵉ giving
#     ∂ŝ_t/∂xᵢ at x = s_k·𝟙.
# Each tangent recursion mirrors the primal recursion with derivatives
# threaded by the chain rule; the shock contribution to the tangent's
# augmented vector picks up `εᵢ_t · eᵢ` because ∂(xᵢ·εᵢ_t)/∂xᵢ = εᵢ_t.
# A separate s = 0 primal trajectory is propagated to obtain V(∅).
#
# Each function fills `decomposition[:, 1:nᵉ, :]` with per-shock Aumann–
# Shapley shares of the incremental response `V(N) − V(∅)`. By linearity,
# these columns equal each shock's standalone effect plus its allocated share
# of the cross-shock interaction. The `decomposition[:, nᵉ+1, :]` column keeps
# the zero-shock / initial-values path `V(∅)` plus any tiny numerical closure
# residual, matching the layout the public API expects when
# `marginal_contribution = true`.

# Pruned 2nd-order state update: new_s1, new_s2 = 𝐒₁·aug1, 𝐒₁·aug2 + ½𝐒₂·(aug1⊗aug1).
# Builds augmented vectors via copyto!, computes kron product, and applies solution matrices in-place.
function pruned_state_update_2nd_order!(
        new_s1, new_s2, s1, s2, past_idx, shock_dir, zero_dir,
        aug1, aug2, kk, 𝐒)
    n_past = length(past_idx)
    @views copyto!(aug1[1:n_past], s1[past_idx])
    aug1[n_past + 1] = 1.0
    copyto!(aug1, n_past + 2, shock_dir, 1, length(shock_dir))
    @views copyto!(aug2[1:n_past], s2[past_idx])
    aug2[n_past + 1] = 0.0
    copyto!(aug2, n_past + 2, zero_dir, 1, length(zero_dir))
    ℒ.kron!(kk, aug1, aug1)
    ℒ.mul!(new_s1, 𝐒[1], aug1)
    ℒ.mul!(new_s2, 𝐒[1], aug2)
    ℒ.mul!(new_s2, 𝐒[2], kk, 0.5, 1.0)
    return nothing
end

# Pruned 3rd-order state update: extends 2nd-order with new_s3 = 𝐒₁·aug3 + 𝐒₂·(aug1̂⊗aug2) + ⅙𝐒₃·(aug1⊗aug1⊗aug1).
# aug1̂ is the no-constant variant of aug1 (constant slot = 0).
function pruned_state_update_3rd_order!(
        new_s1, new_s2, new_s3, s1, s2, s3, past_idx, shock_dir, zero_dir,
        aug1, aug1̂, aug2, aug3, k11, k12̂, k111, 𝐒)
    n_past = length(past_idx)
    @views copyto!(aug1[1:n_past], s1[past_idx])
    aug1[n_past + 1] = 1.0
    copyto!(aug1, n_past + 2, shock_dir, 1, length(shock_dir))
    @views copyto!(aug1̂[1:n_past], s1[past_idx])
    aug1̂[n_past + 1] = 0.0
    copyto!(aug1̂, n_past + 2, shock_dir, 1, length(shock_dir))
    @views copyto!(aug2[1:n_past], s2[past_idx])
    aug2[n_past + 1] = 0.0
    copyto!(aug2, n_past + 2, zero_dir, 1, length(zero_dir))
    @views copyto!(aug3[1:n_past], s3[past_idx])
    aug3[n_past + 1] = 0.0
    copyto!(aug3, n_past + 2, zero_dir, 1, length(zero_dir))
    ℒ.kron!(k11,  aug1, aug1)
    ℒ.kron!(k12̂,  aug1̂, aug2)
    ℒ.kron!(k111, k11,  aug1)
    ℒ.mul!(new_s1, 𝐒[1], aug1)
    ℒ.mul!(new_s2, 𝐒[1], aug2)
    ℒ.mul!(new_s2, 𝐒[2], k11, 0.5, 1.0)
    ℒ.mul!(new_s3, 𝐒[1], aug3)
    ℒ.mul!(new_s3, 𝐒[2], k12̂, 1.0, 1.0)
    ℒ.mul!(new_s3, 𝐒[3], k111, 1/6, 1.0)
    return nothing
end

function aumann_shapley_shock_decomposition_pruned_2nd_order!(
        decomposition::AbstractArray{R},
        variables::AbstractMatrix,
        shocks::AbstractMatrix,
        initial_state,
        𝐒,
        T,
        nE::Int;
        verbose::Bool = false) where R <: Real
    n_nodes = 2
    max_error = aumann_shapley_shock_decomposition_pruned_2nd_order!(decomposition,
                                                                      variables,
                                                                      shocks,
                                                                      initial_state,
                                                                      𝐒,
                                                                      T,
                                                                      nE,
                                                                      n_nodes)
    if verbose
        println("Aumann-Shapley second-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
    end
    while max_error > AUMANN_SHAPLEY_REFINEMENT_RTOL && n_nodes < AUMANN_SHAPLEY_REFINEMENT_MAX_NODES
        next_nodes = min(n_nodes + 1, AUMANN_SHAPLEY_REFINEMENT_MAX_NODES)
        if verbose
            println("Aumann-Shapley second-order shock decomposition rerunning with ", next_nodes, " nodes after closure error ", max_error, " at ", n_nodes, " nodes")
        end
        n_nodes = next_nodes
        max_error = aumann_shapley_shock_decomposition_pruned_2nd_order!(decomposition,
                                                                          variables,
                                                                          shocks,
                                                                          initial_state,
                                                                          𝐒,
                                                                          T,
                                                                          nE,
                                                                          n_nodes)
        if verbose
            println("Aumann-Shapley second-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
        end
    end
    return decomposition
end

function aumann_shapley_shock_decomposition_pruned_2nd_order!(
        decomposition::AbstractArray{R},
        variables::AbstractMatrix,
        shocks::AbstractMatrix,
        initial_state,
        𝐒,
        T,
        nE::Int,
        n_nodes::Int) where R <: Real
    nᵥ = T.nVars
    iₚ = T.past_not_future_and_mixed_idx
    nₚ = length(iₚ)
    n_aug = nₚ + 1 + nE
    n_kron = n_aug^2
    nₜ = size(decomposition, 3)

    nodes, weights = gausslegendre_unit_interval(n_nodes)

    # Scratch buffers — one set reused sequentially across quadrature nodes.
    # s₁/s₂ are primal pruned state components; s₁⁺/s₂⁺ are next-period outputs.
    s₁   = zeros(R, nᵥ)
    s₂   = zeros(R, nᵥ)
    s₁⁺  = zeros(R, nᵥ)
    s₂⁺  = zeros(R, nᵥ)
    # ds* buffers hold tangent states ∂s/∂xᵢ for each shock direction i.
    ds₁ᵢ   = [zeros(R, nᵥ) for _ in 1:nE]
    ds₂ᵢ   = [zeros(R, nᵥ) for _ in 1:nE]
    ds₁ᵢ⁺  = [zeros(R, nᵥ) for _ in 1:nE]
    ds₂ᵢ⁺  = [zeros(R, nᵥ) for _ in 1:nE]

    # Augmented primal/tangent vectors [past state; constant; shocks].
    a₁  = Vector{R}(undef, n_aug)
    a₂  = Vector{R}(undef, n_aug)
    da₁ = Vector{R}(undef, n_aug)
    da₂ = Vector{R}(undef, n_aug)
    # Kronecker workspaces for a₁⊗a₁ and its directional derivative.
    k₁₁  = Vector{R}(undef, n_kron)
    dk₁₁ = Vector{R}(undef, n_kron)
    dk₁₁′ = Vector{R}(undef, n_kron)

    # Shock-direction vectors: scaled node shocks, basis shock i, and zero shocks.
    ε̄ₜ = zeros(R, nE)
    εᵢₜ = zeros(R, nE)
    ε₀ = zeros(R, nE)

    # --- Pass 1: V(∅) trajectory (zero shocks) → store in decomposition[:, nE+1, :]. ---
    s₁ .= initial_state[1]
    s₂ .= initial_state[2]
    # Propagate the baseline path with all shocks set to zero.
    for t in 1:nₜ
        pruned_state_update_2nd_order!(s₁⁺, s₂⁺, s₁, s₂, iₚ, ε₀, ε₀, a₁, a₂, k₁₁, 𝐒)
        @inbounds for v in 1:nᵥ
            decomposition[v, nE + 1, t] = s₁⁺[v] + s₂⁺[v]
        end
        # Swap current and next buffers instead of allocating a fresh state.
        s₁, s₁⁺ = s₁⁺, s₁
        s₂, s₂⁺ = s₂⁺, s₂
    end

    # --- Pass 2: one node at a time, accumulate weighted tangents. ---
    @views fill!(decomposition[:, 1:nE, :], zero(R))

    # Quadrature over shock scaling nodes; each node contributes one weighted path.
    for k in 1:n_nodes
        sₖ = nodes[k]
        wₖ = weights[k]
        s₁ .= initial_state[1]
        s₂ .= initial_state[2]
        # Reset the tangent trajectories for this quadrature node.
        for i in 1:nE
            fill!(ds₁ᵢ[i], 0.0)
            fill!(ds₂ᵢ[i], 0.0)
        end

        # March forward one period at a time, updating the primal path and all tangents.
        for t in 1:nₜ
            εₜ = @view shocks[:, t]
            ε̄ₜ .= sₖ .* εₜ
            pruned_state_update_2nd_order!(s₁⁺, s₂⁺, s₁, s₂, iₚ, ε̄ₜ, ε₀, a₁, a₂, k₁₁, 𝐒)

            # For each shock direction i, propagate tangent recursions and
            # accumulate node-weighted directional derivatives.
            for i in 1:nE
                fill!(εᵢₜ, 0.0)
                εᵢₜ[i] = εₜ[i]

                @views copyto!(da₁[1:nₚ], ds₁ᵢ[i][iₚ])
                da₁[nₚ + 1] = 0.0
                copyto!(da₁, nₚ + 2, εᵢₜ, 1, nE)

                @views copyto!(da₂[1:nₚ], ds₂ᵢ[i][iₚ])
                da₂[nₚ + 1] = 0.0
                copyto!(da₂, nₚ + 2, ε₀, 1, nE)

                # d(aug1 ⊗ aug1) = (d aug1 ⊗ aug1) + (aug1 ⊗ d aug1)
                ℒ.kron!(dk₁₁,  da₁, a₁)
                ℒ.kron!(dk₁₁′, a₁, da₁)
                dk₁₁ .+= dk₁₁′

                # Plain form: ds₁ᵢ⁺ = S1 * da₁
                ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
                # Plain form: ds₂ᵢ⁺ = S1 * da₂ + 0.5 * S2 * d(a₁⊗a₁)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 0.5, 1.0)

                @inbounds for v in 1:nᵥ
                    decomposition[v, i, t] += wₖ * (ds₁ᵢ⁺[i][v] + ds₂ᵢ⁺[i][v])
                end
            end

            # Advance the primal and tangent buffers to the next period.
            s₁, s₁⁺ = s₁⁺, s₁
            s₂, s₂⁺ = s₂⁺, s₂
            for i in 1:nE
                ds₁ᵢ[i], ds₁ᵢ⁺[i] = ds₁ᵢ⁺[i], ds₁ᵢ[i]
                ds₂ᵢ[i], ds₂ᵢ⁺[i] = ds₂ᵢ⁺[i], ds₂ᵢ[i]
            end
        end
    end

    max_residual = zero(R)
    max_reference = zero(R)
    @inbounds for t in 1:nₜ, v in 1:nᵥ
        ϕsum = zero(R)
        for i in 1:nE
            ϕsum += decomposition[v, i, t]
        end
        residual = variables[v, t] - (decomposition[v, nE + 1, t] + ϕsum)
        max_residual = max(max_residual, abs(residual))
        max_reference = max(max_reference, abs(variables[v, t]))
    end

    T = float(R)
    scale = max(T(max_reference), sqrt(eps(T)))
    return T(max_residual) / scale
end


function aumann_shapley_shock_decomposition_pruned_3rd_order!(
        decomposition::AbstractArray{R},
        variables::AbstractMatrix,
        shocks::AbstractMatrix,
        initial_state,
        𝐒,
        T,
        nE::Int;
        verbose::Bool = false) where R <: Real
    n_nodes = 3
    max_error = _aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition,
                                                                      variables,
                                                                      shocks,
                                                                      initial_state,
                                                                      𝐒,
                                                                      T,
                                                                      nE,
                                                                      n_nodes)
    if verbose
        println("Aumann-Shapley third-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
    end
    while max_error > AUMANN_SHAPLEY_REFINEMENT_RTOL && n_nodes < AUMANN_SHAPLEY_REFINEMENT_MAX_NODES
        next_nodes = min(n_nodes + 1, AUMANN_SHAPLEY_REFINEMENT_MAX_NODES)
        if verbose
            println("Aumann-Shapley third-order shock decomposition rerunning with ", next_nodes, " nodes after closure error ", max_error, " at ", n_nodes, " nodes")
        end
        n_nodes = next_nodes
        max_error = _aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition,
                                                                          variables,
                                                                          shocks,
                                                                          initial_state,
                                                                          𝐒,
                                                                          T,
                                                                          nE,
                                                                          n_nodes)
        if verbose
            println("Aumann-Shapley third-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
        end
    end
    return decomposition
end

function _aumann_shapley_shock_decomposition_pruned_3rd_order!(
        decomposition::AbstractArray{R},
        variables::AbstractMatrix,
        shocks::AbstractMatrix,
        initial_state,
        𝐒,
        T,
        nE::Int,
        n_nodes::Int) where R <: Real
    nᵥ = T.nVars
    iₚ = T.past_not_future_and_mixed_idx
    nₚ = length(iₚ)
    n_aug = nₚ + 1 + nE
    n_kron2 = n_aug^2
    n_kron3 = n_aug^3
    nₜ = size(decomposition, 3)

    nodes, weights = gausslegendre_unit_interval(n_nodes)

    # Scratch buffers — one set reused sequentially across quadrature nodes.
    # s₁/s₂/s₃ are primal pruned state components; s*⁺ are next-period outputs.
    s₁  = zeros(R, nᵥ)
    s₂  = zeros(R, nᵥ)
    s₃  = zeros(R, nᵥ)
    s₁⁺ = zeros(R, nᵥ)
    s₂⁺ = zeros(R, nᵥ)
    s₃⁺ = zeros(R, nᵥ)
    # ds* buffers hold tangent states ∂s/∂xᵢ for each shock direction i.
    ds₁ᵢ  = [zeros(R, nᵥ) for _ in 1:nE]
    ds₂ᵢ  = [zeros(R, nᵥ) for _ in 1:nE]
    ds₃ᵢ  = [zeros(R, nᵥ) for _ in 1:nE]
    ds₁ᵢ⁺ = [zeros(R, nᵥ) for _ in 1:nE]
    ds₂ᵢ⁺ = [zeros(R, nᵥ) for _ in 1:nE]
    ds₃ᵢ⁺ = [zeros(R, nᵥ) for _ in 1:nE]

    # Augmented primal/tangent vectors [past state; constant; shocks].
    # a₁⁰ is a₁ with zero constant slot for the third-order cross term.
    a₁  = Vector{R}(undef, n_aug)
    a₁⁰ = Vector{R}(undef, n_aug)
    a₂  = Vector{R}(undef, n_aug)
    a₃  = Vector{R}(undef, n_aug)
    da₁ = Vector{R}(undef, n_aug)
    da₂ = Vector{R}(undef, n_aug)
    da₃ = Vector{R}(undef, n_aug)

    # Kronecker workspaces for primal terms and directional derivatives:
    # k₁₁=a₁⊗a₁, k₁₂⁰=a₁⁰⊗a₂, k₁₁₁=(a₁⊗a₁)⊗a₁ and their d/dxᵢ variants.
    k₁₁   = Vector{R}(undef, n_kron2)
    k₁₂⁰  = Vector{R}(undef, n_kron2)
    dk₁₁  = Vector{R}(undef, n_kron2)
    dk₁₂⁰ = Vector{R}(undef, n_kron2)
    k₂tmp = Vector{R}(undef, n_kron2)
    k₁₁₁  = Vector{R}(undef, n_kron3)
    dk₁₁₁ = Vector{R}(undef, n_kron3)
    k₃tmp = Vector{R}(undef, n_kron3)

    # Shock-direction vectors: scaled node shocks, basis shock i, and zero shocks.
    ε̄ₜ = zeros(R, nE)
    εᵢₜ = zeros(R, nE)
    ε₀ = zeros(R, nE)

    # --- Pass 1: V(∅) trajectory (zero shocks) → store in decomposition[:, nE+1, :]. ---
    s₁ .= initial_state[1]
    s₂ .= initial_state[2]
    s₃ .= initial_state[3]
    # Propagate the baseline path with all shocks set to zero.
    for t in 1:nₜ
        pruned_state_update_3rd_order!(s₁⁺, s₂⁺, s₃⁺, s₁, s₂, s₃, iₚ, ε₀, ε₀,
                                       a₁, a₁⁰, a₂, a₃, k₁₁, k₁₂⁰, k₁₁₁, 𝐒)
        @inbounds for v in 1:nᵥ
            decomposition[v, nE + 1, t] = s₁⁺[v] + s₂⁺[v] + s₃⁺[v]
        end
        # Swap current and next buffers instead of allocating a fresh state.
        s₁, s₁⁺ = s₁⁺, s₁
        s₂, s₂⁺ = s₂⁺, s₂
        s₃, s₃⁺ = s₃⁺, s₃
    end

    # --- Pass 2: one node at a time, accumulate weighted tangents. ---
    @views fill!(decomposition[:, 1:nE, :], zero(R))

    # Quadrature over shock scaling nodes; each node contributes one weighted path.
    for k in 1:n_nodes
        sₖ = nodes[k]
        wₖ = weights[k]
        s₁ .= initial_state[1]
        s₂ .= initial_state[2]
        s₃ .= initial_state[3]
        # Reset the tangent trajectories for this quadrature node.
        for i in 1:nE
            fill!(ds₁ᵢ[i], 0.0)
            fill!(ds₂ᵢ[i], 0.0)
            fill!(ds₃ᵢ[i], 0.0)
        end

        # March forward one period at a time, updating the primal path and all tangents.
        for t in 1:nₜ
            εₜ = @view shocks[:, t]
            ε̄ₜ .= sₖ .* εₜ

            pruned_state_update_3rd_order!(s₁⁺, s₂⁺, s₃⁺, s₁, s₂, s₃, iₚ, ε̄ₜ, ε₀,
                                           a₁, a₁⁰, a₂, a₃, k₁₁, k₁₂⁰, k₁₁₁, 𝐒)

            # For each shock direction i, propagate first/second/third-order
            # tangents and accumulate the node-weighted contribution.
            for i in 1:nE
                fill!(εᵢₜ, 0.0)
                εᵢₜ[i] = εₜ[i]

                @views copyto!(da₁[1:nₚ], ds₁ᵢ[i][iₚ])
                da₁[nₚ + 1] = 0.0
                copyto!(da₁, nₚ + 2, εᵢₜ, 1, nE)

                @views copyto!(da₂[1:nₚ], ds₂ᵢ[i][iₚ])
                da₂[nₚ + 1] = 0.0
                copyto!(da₂, nₚ + 2, ε₀, 1, nE)

                @views copyto!(da₃[1:nₚ], ds₃ᵢ[i][iₚ])
                da₃[nₚ + 1] = 0.0
                copyto!(da₃, nₚ + 2, ε₀, 1, nE)

                # d(aug1 ⊗ aug1) = (d aug1 ⊗ aug1) + (aug1 ⊗ d aug1)
                ℒ.kron!(dk₁₁, da₁, a₁)
                ℒ.kron!(k₂tmp, a₁, da₁)
                dk₁₁ .+= k₂tmp

                # d(aug1_no_const ⊗ aug2) = (d aug1 ⊗ aug2) + (aug1_no_const ⊗ d aug2)
                ℒ.kron!(dk₁₂⁰, da₁, a₂)
                ℒ.kron!(k₂tmp, a₁⁰, da₂)
                dk₁₂⁰ .+= k₂tmp

                # d(k11 ⊗ aug1) = (d k11 ⊗ aug1) + (k11 ⊗ d aug1)
                ℒ.kron!(dk₁₁₁, dk₁₁, a₁)
                ℒ.kron!(k₃tmp, k₁₁, da₁)
                dk₁₁₁ .+= k₃tmp

                # Plain form: ds₁ᵢ⁺ = S1 * da₁
                ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
                # Plain form: ds₂ᵢ⁺ = S1 * da₂ + 0.5 * S2 * d(a₁⊗a₁)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 0.5, 1.0)
                # Plain form: ds₃ᵢ⁺ = S1 * da₃ + S2 * d(a₁⁰⊗a₂) + (1/6) * S3 * d(a₁⊗a₁⊗a₁)
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[1], da₃)
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[2], dk₁₂⁰, 1.0, 1.0)
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[3], dk₁₁₁, 1/6, 1.0)

                @inbounds for v in 1:nᵥ
                    decomposition[v, i, t] += wₖ * (
                        ds₁ᵢ⁺[i][v] +
                        ds₂ᵢ⁺[i][v] +
                        ds₃ᵢ⁺[i][v]
                    )
                end
            end

            # Advance the primal and tangent buffers to the next period.
            s₁, s₁⁺ = s₁⁺, s₁
            s₂, s₂⁺ = s₂⁺, s₂
            s₃, s₃⁺ = s₃⁺, s₃
            for i in 1:nE
                ds₁ᵢ[i], ds₁ᵢ⁺[i] = ds₁ᵢ⁺[i], ds₁ᵢ[i]
                ds₂ᵢ[i], ds₂ᵢ⁺[i] = ds₂ᵢ⁺[i], ds₂ᵢ[i]
                ds₃ᵢ[i], ds₃ᵢ⁺[i] = ds₃ᵢ⁺[i], ds₃ᵢ[i]
            end
        end
    end
    max_residual = zero(R)
    max_reference = zero(R)
    @inbounds for t in 1:nₜ, v in 1:nᵥ
        ϕsum = zero(R)
        for i in 1:nE
            ϕsum += decomposition[v, i, t]
        end
        residual = variables[v, t] - (decomposition[v, nE + 1, t] + ϕsum)
        max_residual = max(max_residual, abs(residual))
        max_reference = max(max_reference, abs(variables[v, t]))
    end

    T = float(R)
    scale = max(T(max_reference), sqrt(eps(T)))
    return T(max_residual) / scale
end

"""
Compute log-likelihood using the inversion filter, which calls the find_shocks function
to recover shocks that match the observables. For higher-order solutions the global
minimum-norm shocks search is NP-hard because feasible roots grow exponentially; starting
from the origin with gradient-based solvers (including the default LagrangeNewton)
returns the root whose basin contains the origin rather than guaranteeing the global
minimum.
"""
function calculate_loglikelihood(::Val{:inversion},
                                                    ::Val{:first_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Matrix{R}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    constants::constants,
                                                    state, 
                                                    workspaces::workspaces; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real,U <: AbstractFloat}
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    ensure_inversion_buffers!(ws, T.nExo, T.nPast_not_future_and_mixed; third_order = false)
    ensure_inversion_estimation_buffers!(ws, T.nExo, length(observables_index))

    # @timeit_debug timer "Inversion filter" begin    
    # first order
    # Reduce state to past_not_future_and_mixed_idx — the only rows that are
    # ever read downstream. Pre-slice 𝐒 to the same row set so the per-period state
    # update touches the minimum number of rows.
    past_idx = T.past_not_future_and_mixed_idx
    n_past = length(past_idx)
    state = convert(Vector{R}, state[1][past_idx])
    𝐒past = 𝐒[past_idx, :]

    precision_factor = one(R)

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    # Use workspace buffers for observation and shock vectors
    state_concat = ws.state_concat

    shocks² = zero(R)
    logabsdets = zero(R)
    jac = zeros(R, 0, 0)

    if warmup_iterations > 0
        if warmup_iterations >= 1
            jac = 𝐒[cond_var_idx,end-T.nExo+1:end]
            if warmup_iterations >= 2
                jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                if warmup_iterations >= 3
                    Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    for e in 1:warmup_iterations-2
                        jac = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], jac)
                        Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                    end
                end
            end
        end

        # Warmup linear solve: LU instead of SVD so ForwardDiff Duals work.
        warmup_rhs = data_in_deviations[:,1]
        if size(jac,1) == size(jac,2)
            warmup_lu = ℒ.lu(jac, check = false)
            if !ℒ.issuccess(warmup_lu)
                if opts.verbose println("Inversion filter failed") end
                return on_failure_loglikelihood
            end
            x = warmup_lu \ warmup_rhs
        else
            JJt_w = jac * jac'
            JJt_w_lu = ℒ.lu(JJt_w, check = false)
            if !ℒ.issuccess(JJt_w_lu)
                if opts.verbose println("Inversion filter failed") end
                return on_failure_loglikelihood
            end
            x = jac' * (JJt_w_lu \ warmup_rhs)
        end

        warmup_shocks = reshape(x, T.nExo, warmup_iterations)
    
        for i in 1:warmup_iterations-1
            # `state` was reduced to T.past_not_future_and_mixed_idx rows above
            # (length T.nPast_not_future_and_mixed), so positions 1:n_past hold
            # the past-not-future-and-mixed entries directly; no view-indexing
            # by past_not_future_and_mixed_idx is needed.
            copyto!(state_concat, 1, state, 1, T.nPast_not_future_and_mixed)
            copyto!(state_concat, T.nPast_not_future_and_mixed + 1, view(warmup_shocks, :, i), 1, T.nExo)
            ℒ.mul!(state, 𝐒past, state_concat)
            # state = state_update(state, warmup_shocks[:,i])
        end

        for i in 1:warmup_iterations
            jac_i = jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor
            if size(jac_i,1) == size(jac_i,2)
                logabsdets += ℒ.logabsdet(jac_i)[1]
            else
                logabsdets += ℒ.logabsdet(jac_i * jac_i')[1] / 2
            end
        end
    
        shocks² += sum(abs2,x)
    end

    y = ws.y_obs
    x = ws.x_shocks
    fill!(y, zero(R))
    fill!(x, zero(R))
    jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

    if T.nExo == length(observables_index)
        if R <: AbstractFloat
            lu_ws = FastLapackInterface.LUWs(jac)
            lu_ws, _, ok, lu_handle = factorize_lu!(Val(:FastLapack), jac, lu_ws, size(jac))

            if !ok
                if opts.verbose println("Inversion filter failed") end
                return on_failure_loglikelihood
            end

            # logabsdet from U-factor diagonal (jac now holds LU factors in place)
            logabsdets = zero(R)
            @inbounds for k in 1:size(jac,1)
                logabsdets += log(abs(jac[k,k]))
            end
            invjac = Matrix{R}(ℒ.I, size(jac))
            solve_lu_left!(jac, invjac, lu_ws, lu_handle)
        else
            jacdecomp = ℒ.lu(jac, check = false)

            if !ℒ.issuccess(jacdecomp)
                if opts.verbose println("Inversion filter failed") end
                return on_failure_loglikelihood
            end

            logabsdets = ℒ.logabsdet(jacdecomp)[1]
            invjac = inv(jacdecomp)
        end
    else
        # Fat jac (n_obs < n_exo): right pseudo-inverse via normal equations.
        # LU is AD-friendly; original SVD/pinv have no ForwardDiff.Dual method.
        JJt = jac * jac'
        JJt_lu = ℒ.lu(JJt, check = false)
        if !ℒ.issuccess(JJt_lu)
            if opts.verbose println("Inversion filter failed") end
            return on_failure_loglikelihood
        end
        logabsdets = ℒ.logabsdet(JJt_lu)[1] / 2
        invjac = jac' / JJt_lu
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods
    
    if !isfinite(logabsdets) return on_failure_loglikelihood end

    𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

    # @timeit_debug timer "Loop" begin    
    for i in axes(data_in_deviations,2)
        ℒ.mul!(y, 𝐒obs, state)
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x, invjac, y)

        # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,x)
            if !isfinite(shocks²) return on_failure_loglikelihood end
        end

        # Use pre-allocated state_concat instead of vcat
        copyto!(state_concat, 1, state, 1, n_past)
        copyto!(state_concat, n_past + 1, x, 1, T.nExo)
        ℒ.mul!(state, 𝐒past, state_concat)
        # state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
    end

    # end # timeit_debug
    # end # timeit_debug

    return -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
    # return -(logabsdets + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood(::Val{:inversion},
                                                    ::Val{:pruned_second_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    constants::constants,
                                                    state, 
                                                    workspaces::workspaces; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    warmup_iterations::Int = 0,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real,U <: AbstractFloat}
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    # @timeit_debug timer "Pruned 2nd - Inversion filter" begin
    # @timeit_debug timer "Preallocation" begin
    
    # Ensure workspaces are properly sized
    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = false)
    ensure_inversion_estimation_buffers!(ws, n_exo, length(observables_index))

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = zero(R)
    logabsdets = zero(R)

    cc = ensure_computational_constants!(constants)
    s_in_s⁺  = cc.s_in_s
    sv_in_s⁺ = cc.s_in_s⁺
    e_in_s⁺  = cc.e_in_s⁺
    
    so = ensure_conditional_forecast_constants!(constants)
    shock_idxs = so.shock_idxs
    shock²_idxs = so.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = so.var_vol²_idxs
    var²_idxs = so.var²_idxs

    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state₁ = convert(Vector{R}, state[1][T.past_not_future_and_mixed_idx])
    state₂ = convert(Vector{R}, state[2][T.past_not_future_and_mixed_idx])

    n_state_vol = n_past + 1
    n_aug = n_past + 1 + n_exo
    n_cond = length(cond_var_idx)

    if R === Float64
        # Use workspaces for model-constant allocations
        state¹⁻_vol = ws.state_vol
        copyto!(state¹⁻_vol, 1, state₁, 1)
        state¹⁻_vol[end] = 1

        aug_state₁ = ws.aug_state₁
        copyto!(aug_state₁, 1, state₁, 1)
        aug_state₁[length(state₁) + 1] = 1
        fill!(view(aug_state₁, length(state₁) + 2:length(aug_state₁)), 1)
        
        aug_state₂ = ws.aug_state₂
        copyto!(aug_state₂, 1, state₂, 1)
        aug_state₂[length(state₂) + 1] = 0
        fill!(view(aug_state₂, length(state₂) + 2:length(aug_state₂)), 0)

        kronaug_state₁ = ws.kronaug_state

        kron_buffer = ws.kron_buffer
        kron_buffer2 = ws.kron_buffer2
        kron_buffer3 = ws.kron_buffer_state
        kronstate¹⁻_vol = ws.kronstate_vol

        shock_independent = ws.shock_independent
        fill!(shock_independent, zero(R))

        𝐒ⁱ = ws.Si_buffer
        copyto!(𝐒ⁱ, 𝐒¹ᵉ)

        jacc = ws.jacc_buffer
        copyto!(jacc, 𝐒¹ᵉ)

        init_guess = ws.init_guess
        fill!(init_guess, zero(R))
    else
        # Allocate R-typed buffers for AD compatibility (e.g. ForwardDiff Dual)
        state¹⁻_vol = vcat(state₁, one(R))
        aug_state₁ = vcat(state₁, one(R), ones(R, n_exo))
        aug_state₂ = vcat(state₂, zero(R), zeros(R, n_exo))
        kronaug_state₁ = zeros(R, n_aug^2)
        kron_buffer = zeros(R, n_exo^2)
        kron_buffer2 = zeros(R, n_exo^2, n_exo)
        kron_buffer3 = zeros(R, n_exo * n_state_vol, n_exo)
        kronstate¹⁻_vol = zeros(R, n_state_vol^2)
        shock_independent = zeros(R, n_cond)
        𝐒ⁱ = Matrix{R}(𝐒¹ᵉ)
        jacc = Matrix{R}(𝐒¹ᵉ)
        init_guess = zeros(R, n_exo)
    end

    J = ℒ.I(T.nExo)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin

    for i in axes(data_in_deviations, 2)
        # state¹⁻ = state₁
        # state¹⁻_vol = vcat(state¹⁻, 1)
        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        copyto!(state¹⁻_vol, 1, state₁, 1)

        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒¹⁻ * state²⁻ + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)  
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        # end # timeit_debug
                     
        # if matched println("$filter_algorithm: $matched; current x: $x") end      
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    if opts.verbose println("Inversion filter failed at step $i") end
                    return on_failure_loglikelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += ℒ.logabsdet(jacc * jacc')[1] / 2
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood
            end
        end

        # aug_state₁ = [state₁; 1; x]
        # aug_state₂ = [state₂; 0; zero(x)]
        copyto!(aug_state₁, 1, state₁, 1)
        copyto!(aug_state₁, length(state₁) + 2, x, 1)
        copyto!(aug_state₂, 1, state₂, 1)

        # state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁)

        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂)
        ℒ.kron!(kronaug_state₁, aug_state₁, aug_state₁)
        ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf and Fair and Taylor (1983)
    return -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood(::Val{:inversion},
                                                    ::Val{:second_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    constants::constants,
                                                    state, 
                                                    workspaces::workspaces; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    on_failure_loglikelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    # @timeit_debug timer "2nd - Inversion filter" begin
    # @timeit_debug timer "Preallocation" begin

    # Ensure workspaces are properly sized
    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = false)
    ensure_inversion_estimation_buffers!(ws, n_exo, length(observables_index))

    precision_factor = one(R)

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = zero(R)
    logabsdets = zero(R)

    # s_in_s⁺ = computational_constants.s_in_s
    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants)
    shock_idxs = cc.shock_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    
    # tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    # var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    # 𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    # 𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    # 𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = convert(Vector{R}, state[T.past_not_future_and_mixed_idx])

    # Use workspaces for model-constant allocations
    state¹⁻_vol = ws.state_vol
    copyto!(state¹⁻_vol, 1, state, 1)
    state¹⁻_vol[end] = 1

    aug_state = ws.aug_state₁
    fill!(aug_state, 0)
    aug_state[n_past + 1] = 1

    kronaug_state = ws.kronaug_state

    kron_buffer = ws.kron_buffer

    J = ℒ.I(T.nExo)

    kron_buffer2 = ws.kron_buffer2

    kron_buffer3 = ws.kron_buffer_state

    # Use workspace buffers instead of fresh allocations
    shock_independent = ws.shock_independent
    fill!(shock_independent, zero(R))

    kronstate¹⁻_vol = ws.kronstate_vol

    𝐒ⁱ = ws.Si_buffer
    copyto!(𝐒ⁱ, 𝐒¹ᵉ)

    jacc = ws.jacc_buffer
    copyto!(jacc, 𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    init_guess = ws.init_guess
    fill!(init_guess, zero(R))

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state#[T.past_not_future_and_mixed_idx]
        # state¹⁻_vol = vcat(state¹⁻, 1)
        
        copyto!(state¹⁻_vol, 1, state, 1)

        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )  
        # end # timeit_debug

        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    if opts.verbose println("Inversion filter failed at step $i") end
                    return on_failure_loglikelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc)[1] # ./ precision_factor
            else
                logabsdets += ℒ.logabsdet(jacc * jacc')[1] / 2 # ./ precision_factor
            end

            shocks² += sum(abs2,x)

            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood
            end
        end

        # aug_state = [state; 1; x]
        # aug_state[1:T.nPast_not_future_and_mixed] = state
        # aug_state[end-T.nExo+1:end] = x
        copyto!(aug_state, 1, state, 1)
        copyto!(aug_state, length(state) + 2, x, 1)

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.mul!(state, 𝐒⁻¹, aug_state)
        ℒ.mul!(state, 𝐒⁻², kronaug_state, 1/2 ,1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

function calculate_loglikelihood(::Val{:inversion},
                                                    ::Val{:pruned_third_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    constants::constants,
                                                    state, 
                                                    workspaces::workspaces;
                                                    # timer::TimerOutput = TimerOutput(), 
                                                    on_failure_loglikelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    # @timeit_debug timer "Inversion filter" begin

    # Ensure workspaces are properly sized
    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, length(observables_index); third_order = true)

    precision_factor = one(R)

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = zero(R)
    logabsdets = zero(R)

    cc = ensure_computational_constants!(constants)
    s_in_s⁺ = cc.s_in_s
    e_in_s⁺ = cc.e_in_s⁺

    so = ensure_conditional_forecast_constants!(constants; third_order = true)
    shockvar_idxs = cc.shockvar_idxs
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = so.var²_idxs
    to = constants.third_order

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx,shockvar_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ   = nnz(𝐒²⁻ᵛᵉ)   / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ)   : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    var_vol³_idxs = to.var_vol³_idxs
    shock³_idxs = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs = to.shockvar³_idxs

    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs] |> collect
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    # Shadow the input `state` with R-typed local copies so the kernel can
    # be driven by ForwardDiff Duals (the input may be Vector{Vector{Float64}}).
    state = Vector{R}[
        convert(Vector{R}, state[1][T.past_not_future_and_mixed_idx]),
        convert(Vector{R}, state[2][T.past_not_future_and_mixed_idx]),
        convert(Vector{R}, state[3][T.past_not_future_and_mixed_idx]),
    ]

    # Use workspace buffers
    kron_buffer = ws.kron_buffer
    kron_buffer² = ws.kron_buffer²
    J = ℒ.I(T.nExo)
    II = ℒ.I(T.nExo^2)
    kron_buffer2 = ws.kron_buffer2
    kron_buffer3 = ws.kron_buffer3
    kron_buffer4 = ws.kron_buffer4
    kron_buffer_state = ws.kron_buffer_state
    𝐒ⁱ = ws.Si_buffer
    jacc = ws.jacc_buffer
    shock_independent = ws.shock_independent
    init_guess = ws.init_guess
    state_vol = ws.state_vol
    kronstate_vol = ws.kronstate_vol
    kronstate_vol³ = ws.kronstate_vol³
    state²⁻_vol = ws.state²⁻_vol

    # Pruned-third specific kron buffers (not in ws, allocated once per call)
    kron_buffer4sv = ℒ.kron(II, vcat(1,state[1]))
    kron_buffer2ss = ℒ.kron(state[1], state[1])
    kron_buffer3sv = ℒ.kron(ℒ.kron(J, vcat(1,state[1])), vcat(1,state[1]))
    
    # Use workspaces for augmented state kron operations
    kron_aug_state₁ = ws.kronaug_state
    
    kron_kron_aug_state₁ = ws.kron_kron_aug_state

    aug_state₁ = ws.aug_state₁
    aug_state₁̂ = ws.aug_state₁̂
    aug_state₂ = ws.aug_state₂
    aug_state₃ = ws.aug_state₃

    state¹⁻ = state[1]

    state²⁻ = state[2]#[T.past_not_future_and_mixed_idx]

    state³⁻ = state[3]#[T.past_not_future_and_mixed_idx]

    # @timeit_debug timer "Loop" begin

    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    # Hoisted: 𝐒ⁱ²ᵉ assembled in-place per period (parity with missing variant).
    𝐒ⁱ²ᵉ = similar(𝐒²ᵉ)

    fill!(init_guess, zero(R))
    
    for i in axes(data_in_deviations,2)
        # state¹⁻_vol = [state¹⁻; 1]
        copyto!(state_vol, 1, state¹⁻, 1, n_past)
        state_vol[end] = 1
        state¹⁻_vol = state_vol

        copyto!(shock_independent, view(data_in_deviations, :, i))

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)

        ℒ.kron!(kronstate_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate_vol, -1/2, 1)

        ℒ.kron!(kron_buffer2ss, state¹⁻, state²⁻)

        ℒ.mul!(shock_independent, 𝐒²⁻, kron_buffer2ss, -1, 1)

        ℒ.kron!(kronstate_vol³, kronstate_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kronstate_vol³, -1/6, 1)   
        
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵛᵉ * kron(J, s2_vol) + 𝐒²⁻ᵉ * kron(J, sv) + 𝐒³⁻ᵉ² * kron(kron(J, sv), sv) / 2
        
        copyto!(state²⁻_vol, 1, state²⁻, 1)
        state²⁻_vol[end] = 0
        ℒ.kron!(kron_buffer_state, J, state²⁻_vol)
    
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵛᵉ, kron_buffer_state)

        ℒ.kron!(kron_buffer_state, J, state¹⁻_vol)
    
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer_state, 1, 1)
    
        ℒ.kron!(kron_buffer3sv, kron_buffer_state, state¹⁻_vol)
        
        ℒ.mul!(𝐒ⁱ, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        x_kron_II!(kron_buffer4sv, state¹⁻_vol)

        copyto!(𝐒ⁱ²ᵉ, 𝐒²ᵉ)
        ℒ.rdiv!(𝐒ⁱ²ᵉ, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess *= 0

        # x² , matched = find_shocks(Val(filter_algorithm), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
        #                         println(x²)

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
        # end # timeit_debug
                                
                                # println(x)
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && (!matched || ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂) || (matched && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x)))
        #             # println("$i - $filter_algorithm restart - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             # println("$i - $backup_solver - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         # else
        #         #     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #         #     norm1 = ℒ.norm(y)

        #         #     norm2 = ℒ.norm(shock_independent)

        #             # println("$i - $filter_algorithm - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")#, residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     # else
        #     #     println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
                # println("$filter_algorithm: $matched; current x: $x")
                # if !matched
                #     x, matched = find_shocks(Val(:COBYLA), 
                #                             x,
                #                             kron_buffer,
                #                             kron_buffer²,
                #                             kron_buffer2,
                #                             kron_buffer3,
                #                             J,
                #                             𝐒ⁱ,
                #                             𝐒ⁱ²ᵉ,
                #                             𝐒ⁱ³ᵉ,
                #                             shock_independent)
                # end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x))), $(ℒ.norm(x2)-ℒ.norm(x))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x))), $(ℒ.norm(x3)-ℒ.norm(x))")
        # end
        
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.kron!(kron_buffer3, kron_buffer2, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.mul!(jacc, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 2)

        ℒ.axpby!(-1, 𝐒ⁱ, -1, jacc)

        # jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron_buffer2 + 3 * 𝐒ⁱ³ᵉ * kron_buffer3)

        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += ℒ.logabsdet(jacc * jacc')[1] / 2
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood
            end
        end

        # aug_state₁ = [state¹⁻; 1; x]
        copyto!(aug_state₁, 1, state¹⁻, 1, n_past)
        aug_state₁[n_past + 1] = 1
        copyto!(aug_state₁, n_past + 2, x, 1, n_exo)

        # aug_state₁̂ = [state¹⁻; 0; x]
        copyto!(aug_state₁̂, 1, state¹⁻, 1, n_past)
        aug_state₁̂[n_past + 1] = 0
        copyto!(aug_state₁̂, n_past + 2, x, 1, n_exo)

        # aug_state₂ = [state²⁻; 0; zero(x)]
        copyto!(aug_state₂, 1, state²⁻, 1, n_past)
        aug_state₂[n_past + 1] = 0
        fill!(view(aug_state₂, n_past + 2:n_past + 1 + n_exo), zero(R))

        # aug_state₃ = [state³⁻; 0; zero(x)]
        copyto!(aug_state₃, 1, state³⁻, 1, n_past)
        aug_state₃[n_past + 1] = 0
        fill!(view(aug_state₃, n_past + 2:n_past + 1 + n_exo), zero(R))
        
        # kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)
        ℒ.kron!(kron_aug_state₁, aug_state₁, aug_state₁)

        ℒ.kron!(kron_kron_aug_state₁, kron_aug_state₁, aug_state₁)
        # res = 𝐒[1][cond_var_idx,:] * aug_state₁   +   𝐒[1][cond_var_idx,:] * aug_state₂ + 𝐒[2][cond_var_idx,:] * kron_aug_state₁ / 2   +   𝐒[1][cond_var_idx,:] * aug_state₃ + 𝐒[2][cond_var_idx,:] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3][cond_var_idx,:] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")
        
        # println(ℒ.norm(x))

        # state[1] = 𝐒⁻¹ * aug_state₁
        # state[2] = 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2
        # state[3] = 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6
        
        ℒ.mul!(state¹⁻, 𝐒⁻¹, aug_state₁)

        ℒ.mul!(state²⁻, 𝐒⁻¹, aug_state₂)
        ℒ.mul!(state²⁻, 𝐒⁻², kron_aug_state₁, 1/2, 1)

        ℒ.mul!(state³⁻, 𝐒⁻¹, aug_state₃)

        ℒ.kron!(kron_aug_state₁, aug_state₁̂, aug_state₂)
        
        ℒ.mul!(state³⁻, 𝐒⁻², kron_aug_state₁, 1, 1)
        ℒ.mul!(state³⁻, 𝐒⁻³, kron_kron_aug_state₁, 1/6, 1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood(::Val{:inversion},
                                                    ::Val{:third_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}}, 
                                                    data_in_deviations::Matrix{R}, 
                                                    constants::constants,
                                                    state, 
                                                    workspaces::workspaces; 
                                                    # timer::TimerOutput = TimerOutput(),
                                                    on_failure_loglikelihood::U = -Inf,
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real,U <: AbstractFloat}
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    # @timeit_debug timer "3rd - Inversion filter" begin
    # @timeit_debug timer "Preallocation" begin

    # Ensure workspaces are properly sized
    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, length(observables_index); third_order = true)

    precision_factor = one(R)

    n_obs = size(data_in_deviations,2)

    cond_var_idx = observables_index

    shocks² = zero(R)
    logabsdets = zero(R)

    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants; third_order = true)
    shock²_idxs = cc.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = cc.var_vol²_idxs
    var²_idxs = so.var²_idxs
    to = constants.third_order

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = convert(Vector{R}, state[T.past_not_future_and_mixed_idx])

    var_vol³_idxs = to.var_vol³_idxs
    shock³_idxs = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs = to.shockvar³_idxs

    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    # Use workspaces for shock-related kron operations
    kron_buffer = ws.kron_buffer

    kron_buffer² = ws.kron_buffer²

    J = ℒ.I(T.nExo)

    kron_buffer2 = ws.kron_buffer2

    kron_buffer3 = ws.kron_buffer3

    kron_buffer4 = ws.kron_buffer4

    II = sparse(ℒ.I(T.nExo^2))

    # Use workspace buffers for state/estimation temporaries
    state_vol = ws.state_vol
    kronstate_vol = ws.kronstate_vol
    kronstate_vol³ = ws.kronstate_vol³
    kron_buffer_state = ws.kron_buffer_state
    shock_independent = ws.shock_independent
    init_guess = ws.init_guess
    𝐒ⁱ = ws.Si_buffer
    jacc = ws.jacc_buffer
    aug_state = ws.aug_state₁
    kronaug_state = ws.kronaug_state
    kron_kron_aug_state = ws.kron_kron_aug_state
    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    # Hoisted per-period buffers (parity with missing variant).
    kron_buffer3sv = zeros(R, T.nExo * (n_past+1)^2, T.nExo)
    kron_buffer4sv = zeros(R, T.nExo^2 * (n_past+1), T.nExo^2)
    𝐒ⁱ²ᵉ = similar(𝐒²ᵉ)

    # end # timeit_debug
    # @timeit_debug timer "Loop" begin
    
    for i in axes(data_in_deviations,2)
        # Build state_vol = [state; 1]
        copyto!(state_vol, 1, state, 1, n_past)
        state_vol[end] = 1
        state¹⁻_vol = state_vol

        copyto!(shock_independent, view(data_in_deviations, :, i))

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.kron!(kronstate_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate_vol, -1/2, 1)
        
        ℒ.kron!(kronstate_vol³, state¹⁻_vol, kronstate_vol)
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kronstate_vol³, -1/6, 1)   

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron(I, sv) + 𝐒³⁻ᵉ² * kron(kron(I, sv), sv) / 2
        ℒ.kron!(kron_buffer_state, J, state¹⁻_vol)
        copyto!(𝐒ⁱ, 𝐒¹ᵉ)
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer_state, 1, 1)
        ℒ.kron!(kron_buffer3sv, kron_buffer_state, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)
    
        x_kron_II!(kron_buffer4sv, state¹⁻_vol)
        copyto!(𝐒ⁱ²ᵉ, 𝐒²ᵉ); ℒ.rdiv!(𝐒ⁱ²ᵉ, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        fill!(init_guess, zero(R))

        # @timeit_debug timer "Find shocks" begin
        x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
        # end # timeit_debug
                                
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂)
        #             println("$i - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             println("$i - $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         else
        #             y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #             norm1 = ℒ.norm(y)

        #             norm2 = ℒ.norm(shock_independent)

        #             println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     else
        #         println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end

        if !matched
            if opts.verbose println("Inversion filter failed at step $i") end
            return on_failure_loglikelihood # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive 
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
            #     println("$filter_algorithm: $matched; current x: $x")
            #     if !matched
            #         x, matched = find_shocks(Val(:COBYLA), 
            #                                 x,
            #                                 kron_buffer,
            #                                 kron_buffer²,
            #                                 kron_buffer2,
            #                                 kron_buffer3,
            #                                 J,
            #                                 𝐒ⁱ,
            #                                 𝐒ⁱ²ᵉ,
            #                                 𝐒ⁱ³ᵉ,
            #                                 shock_independent)
            #         println("COBYLA: $matched; current x: $x")
            #     end
            # end
        # end

        # x2, mat = find_shocks(Val(:COBYLA), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
            
        # x3, mat2 = find_shocks(Val(filter_algorithm), 
        #                         x2,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # # if mat
        #     println("COBYLA - $mat: $x2, $(ℒ.norm(x2))")
        # # end
        # # if mat2
        #     println("LagrangeNewton restart - $mat2: $x3, $(ℒ.norm(x3))")
        # # end

        # jacc = -(𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron(I,x) + 3 * 𝐒ⁱ³ᵉ * kron(I, kron(x,x)))
        ℒ.kron!(kron_buffer2, J, x)
        ℒ.kron!(kron_buffer, x, x)
        ℒ.kron!(kron_buffer3, J, kron_buffer)
        copyto!(jacc, 𝐒ⁱ)
        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2, 2, 1)
        ℒ.mul!(jacc, 𝐒ⁱ³ᵉ, kron_buffer3, 3, 1)
        ℒ.rmul!(jacc, -1)
    
        if i > presample_periods
            # due to change of variables: jacobian determinant adjustment
            if T.nExo == length(observables_index)
                logabsdets += ℒ.logabsdet(jacc)[1]
            else
                logabsdets += ℒ.logabsdet(jacc * jacc')[1] / 2
            end

            shocks² += sum(abs2,x)
            
            if !isfinite(logabsdets) || !isfinite(shocks²)
                return on_failure_loglikelihood
            end
        end

        # aug_state = [state; 1; x]
        copyto!(aug_state, 1, state, 1, n_past)
        aug_state[n_past + 1] = 1
        copyto!(aug_state, n_past + 2, x, 1, n_exo)

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * kron(aug,aug)/2 + 𝐒⁻³ * kron(kron(aug,aug),aug)/6
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.kron!(kron_kron_aug_state, kronaug_state, aug_state)
        ℒ.mul!(state, 𝐒⁻¹, aug_state)
        ℒ.mul!(state, 𝐒⁻², kronaug_state, 1/2, 1)
        ℒ.mul!(state, 𝐒⁻³, kron_kron_aug_state, 1/6, 1)
    end

    # end # timeit_debug
    # end # timeit_debug

    # See: https://pcubaborda.net/documents/CGIZ-final.pdf
    return -(logabsdets + shocks² + (length(observables_index) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

@unstable function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:first_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    
    decomposition = zeros(T.nVars, T.nExo + 2, size(data_in_deviations, 2))

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    state = zeros(T.nVars)

    initial_state = zeros(T.nVars)

    ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        opts = opts,
                                                        parameter_values = 𝓂.parameter_values)
    
    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    if !solved 
        @error "No solution for these parameters."
        return variables, shocks, zeros(0,0), decomposition
    end

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    observables = get_and_check_observables(T, data_in_deviations)

    cond_var_idx = indexin(observables, sort(union(T.aux,T.var,T.exo_present)))

    data_arr = collect(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_arr)

    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)
    jac = 𝐒₁[cond_var_idx, end-T.nExo+1:end]

    if warmup_iterations > 0
        if has_missing
            n_warm = min(warmup_iterations - 1, size(data_in_deviations, 2))
            @inbounds for i in 1:n_warm
                @views ℒ.mul!(y, 𝐒₁[cond_var_idx,1:end-T.nExo], state[T.past_not_future_and_mixed_idx])
                @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
                idx = obs_idx_per_t[i]
                if isempty(idx)
                    fill!(x, 0)
                else
                    jac_v = jac[idx, :]
                    y_v = y[idx]
                    x .= jac_v \ y_v
                end
                ℒ.mul!(state, 𝐒₁, vcat(state[T.past_not_future_and_mixed_idx], x))
            end
        else
            if warmup_iterations >= 1
                warmup_jac = jac
                if warmup_iterations >= 2
                    warmup_jac = hcat(𝐒₁[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒₁[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], warmup_jac)
                    if warmup_iterations >= 3
                        Sᵉ = 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                        for e in 1:warmup_iterations-2
                            warmup_jac = hcat(𝐒₁[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒₁[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], warmup_jac)
                            Sᵉ *= 𝐒₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                        end
                    end
                end
                jacdecomp = ℒ.svd(warmup_jac)

                x_warmup = jacdecomp \ data_in_deviations[:,1]
            
                warmup_shocks = reshape(x_warmup, T.nExo, warmup_iterations)
            
                for i in 1:warmup_iterations-1
                    ℒ.mul!(state, 𝐒₁, vcat(state[T.past_not_future_and_mixed_idx], warmup_shocks[:,i]))
                    # state = state_update(state, warmup_shocks[:,i])
                end
            end
        end
    end

    if T.nExo == length(observables)
        if eltype(jac) <: AbstractFloat
            lu_ws = FastLapackInterface.LUWs(jac)
            lu_ws, _, ok, lu_handle = factorize_lu!(Val(:FastLapack), jac, lu_ws, size(jac))

            if !ok
                @error "Inversion filter failed"
                return variables, shocks, zeros(0,0), decomposition
            end

            invjac = Matrix{eltype(jac)}(ℒ.I, size(jac))
            solve_lu_left!(jac, invjac, lu_ws, lu_handle)
        else
            jacdecomp = ℒ.lu(jac, check = false)

            if !ℒ.issuccess(jacdecomp)
                @error "Inversion filter failed"
                return variables, shocks, zeros(0,0), decomposition
            end

            invjac = inv(jacdecomp)
        end
    else
        # jacdecomp = ℒ.svd(jac)
        
        invjac = ℒ.pinv(jac)
    end

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒₁[cond_var_idx,1:end-T.nExo], state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)

        if has_missing
            idx = obs_idx_per_t[i]
            m = length(idx)
            if m == 0
                fill!(x, 0)
            else
                jac_v = jac[idx, :]
                y_v   = y[idx]
                x .= jac_v \ y_v
            end
        else
            ℒ.mul!(x, invjac, y)
        end

        ℒ.mul!(state, 𝐒₁, vcat(state[T.past_not_future_and_mixed_idx], x))

        shocks[:,i] .= x
        variables[:,i] .= state
        # state = 𝐒₁ * vcat(state[T.past_not_future_and_mixed_idx], x)
    end

    decomposition[:,end,:] .= variables

    for i in 1:T.nExo
        sck = zeros(T.nExo)
        sck[i] = shocks[i, 1]
        decomposition[:,i,1] .= 𝐒₁ * vcat(initial_state[T.past_not_future_and_mixed_idx], sck) # state_update(initial_state , sck)
    end

    decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end-2, 1], dims=2)

    for i in 2:size(data_in_deviations,2)
        for ii in 1:T.nExo
            sck = zeros(T.nExo)
            sck[ii] = shocks[ii, i]
            decomposition[:, ii, i] .= 𝐒₁ * vcat(decomposition[T.past_not_future_and_mixed_idx, ii, i-1], sck) # state_update(decomposition[:,ii, i-1], sck)
        end

        decomposition[:, end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:, 1:end-2, i], dims=2)
    end
    
    return variables, shocks, zeros(0,0), decomposition
end


@unstable function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:second_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:second_order), 𝓂.parameter_values, 𝓂, opts = opts)

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        @error "Could not find 2nd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    ms = constants.post_complete_parameters
    all_SS = expand_steady_state(SS_and_pars, ms)

    full_state = collect(sss) - all_SS

    observables = get_and_check_observables(T, data_in_deviations)

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    computational_constants = ensure_computational_constants!(𝓂.constants)
    so = ensure_conditional_forecast_constants!(𝓂.constants)
    # s_in_s⁺ = computational_constants.s_in_s
    shock²_idxs = computational_constants.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = computational_constants.var_vol²_idxs
    
    # tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
    # var²_idxs = tmp.nzind
    
    𝐒⁻¹ = 𝐒₁[T.past_not_future_and_mixed_idx,:]
    # 𝐒¹⁻ = 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒₁[cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒₁[cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒₂[cond_var_idx,var_vol²_idxs]
    # 𝐒²⁻ = 𝐒₂[cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒₂[cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒₂[cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒₂[T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    # 𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = full_state[T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state, 1)

    aug_state = [zeros(T.nPast_not_future_and_mixed); 1; zeros(T.nExo)]

    kronaug_state = zeros((T.nPast_not_future_and_mixed + 1 + T.nExo)^2)

    kron_buffer = zeros(T.nExo^2)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    shock_independent = zeros(size(data_in_deviations,1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    init_guess = zeros(size(𝐒ⁱ, 2))

    data_arr = collect(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_arr)
    n_exo = T.nExo

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state#[T.past_not_future_and_mixed_idx]
        # state¹⁻_vol = vcat(state¹⁻, 1)
        
        copyto!(state¹⁻_vol, 1, state, 1)

        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)
        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        if has_missing
            idx = obs_idx_per_t[i]
            m = length(idx)
            if m == 0
                x = zeros(n_exo)
                matched = true
            else
                if m > n_exo
                    @error "Inversion filter (2nd) failed at step $i: m=$m > n_exo=$n_exo"
                    return variables, shocks, zeros(0,0), zeros(0,0)
                end
                𝐒ⁱ_v   = 𝐒ⁱ[idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
                si_v   = shock_independent[idx]
                x, matched = find_shocks(Val(filter_algorithm),
                                        zeros(n_exo),
                                        kron_buffer,
                                        kron_buffer2,
                                        J,
                                        𝐒ⁱ_v,
                                        𝐒ⁱ²ᵉ_v,
                                        si_v)
            end
        else
            x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        end

        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    @error "Inversion filter failed at step $i"
                    return variables, shocks, zeros(0,0), zeros(0,0)
                end 
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)
        ℒ.kron!(kron_buffer2, J, x)

        ℒ.mul!(jacc, 𝐒ⁱ²ᵉ, kron_buffer2)

        ℒ.axpby!(1, 𝐒ⁱ, 2, jacc)

        # aug_state = [state; 1; x]
        # aug_state[1:T.nPast_not_future_and_mixed] = state
        # aug_state[end-T.nExo+1:end] = x
        copyto!(aug_state, 1, state, 1)
        copyto!(aug_state, length(state) + 2, x, 1)

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.mul!(full_state, 𝐒₁, aug_state)
        ℒ.mul!(full_state, 𝐒₂, kronaug_state, 1/2 ,1)

        shocks[:,i] .= x
        variables[:,i] .= full_state

        state .= full_state[T.past_not_future_and_mixed_idx]
    end

    return variables, shocks, zeros(0,0), zeros(0,0)
end


@unstable function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:pruned_second_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                marginal_contribution::Bool = false,
                                opts::CalculationOptions = merge_calculation_options())
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    ms = constants.post_complete_parameters

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    decomposition = zeros(T.nVars, marginal_contribution ? T.nExo + 2 : T.nExo + 3, size(data_in_deviations, 2))

    observables = get_and_check_observables(T, data_in_deviations)
    
    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_stochastic_steady_state(Val(:pruned_second_order), 𝓂.parameter_values, 𝓂, opts = opts)

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        @error "Could not find pruned 2nd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end
    
    𝐒 = [𝐒₁, 𝐒₂]

    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(𝓂.constants.post_model_macro.nVars), collect(sss) - all_SS]
     
    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    computational_constants = ensure_computational_constants!(𝓂.constants)
    so = ensure_conditional_forecast_constants!(𝓂.constants)
    sv_in_s⁺ = computational_constants.s_in_s⁺
    
    shock²_idxs = computational_constants.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = computational_constants.var_vol²_idxs
    var²_idxs = so.var²_idxs
    
    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    initial_state = deepcopy(state)

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]

    state¹⁻_vol = vcat(state₁, 1)

    aug_state₁ = [state₁; 1; ones(T.nExo)]
    aug_state₂ = [state₂; 0; zeros(T.nExo)]

    kronaug_state₁ = zeros(length(aug_state₁)^2)

    J = ℒ.I(T.nExo)

    kron_buffer = zeros(T.nExo^2)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, zeros(T.nPast_not_future_and_mixed + 1))

    kronstate¹⁻_vol = zeros((T.nPast_not_future_and_mixed + 1)^2)

    shock_independent = zeros(size(data_in_deviations,1))

    𝐒ⁱ = copy(𝐒¹ᵉ)

    jacc = copy(𝐒¹ᵉ)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
        
    init_guess = zeros(size(𝐒ⁱ, 2))

    data_arr = collect(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_arr)
    n_exo = T.nExo

    for i in axes(data_in_deviations, 2)
        # state¹⁻ = state₁
        # state¹⁻_vol = vcat(state¹⁻, 1)
        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        copyto!(state¹⁻_vol, 1, state₁, 1)

        # shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒¹⁻ * state²⁻ + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
        copyto!(shock_independent, data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)  
        # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * kron_buffer3
        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)

        ℒ.mul!(𝐒ⁱ, 𝐒²⁻ᵉ, kron_buffer3)

        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ)

        init_guess *= 0

        if has_missing
            idx = obs_idx_per_t[i]
            m = length(idx)
            if m == 0
                x = zeros(n_exo)
                matched = true
            else
                if m > n_exo
                    @error "Inversion filter (pruned 2nd) failed at step $i: m=$m > n_exo=$n_exo"
                    return variables, shocks, zeros(0,0), decomposition
                end
                𝐒ⁱ_v   = 𝐒ⁱ[idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
                si_v   = shock_independent[idx]
                x, matched = find_shocks(Val(filter_algorithm),
                                        zeros(n_exo),
                                        kron_buffer,
                                        kron_buffer2,
                                        J,
                                        𝐒ⁱ_v,
                                        𝐒ⁱ²ᵉ_v,
                                        si_v)
            end
        else
            x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent,
                                # max_iter = 100
                                )
        end
                     
        # if matched println("$filter_algorithm: $matched; current x: $x") end      
        # if !matched
        #     x, matched = find_shocks(Val(:COBYLA), 
        #                             zeros(size(𝐒ⁱ, 2)),
        #                             kron_buffer,
        #                             kron_buffer2,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             shock_independent,
        #                             # max_iter = 500
        #                             )
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer2,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             shock_independent)
                if !matched
                    @error "Inversion filter failed at step $i"
                    return variables, shocks, zeros(0,0), decomposition
                end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x)))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x)))")
        # end

        # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x)

        # aug_state₁ = [state₁; 1; x]
        # aug_state₂ = [state₂; 0; zero(x)]
        copyto!(aug_state₁, 1, state₁, 1)
        copyto!(aug_state₁, length(state₁) + 2, x, 1)
        copyto!(aug_state₂, 1, state₂, 1)

        # state₁, state₂ = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        # ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁)
        ℒ.mul!(state[1], 𝐒[1], aug_state₁)
        state₁ .= state[1][T.past_not_future_and_mixed_idx]

        ℒ.kron!(kronaug_state₁, aug_state₁, aug_state₁)
        # ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂)
        # ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
        ℒ.mul!(state[2], 𝐒[1], aug_state₂)
        ℒ.mul!(state[2], 𝐒[2], kronaug_state₁, 1/2, 1)
        state₂ .= state[2][T.past_not_future_and_mixed_idx]

        variables[:,i] .= sum(state)
        shocks[:,i] .= x
    end

    states = [initial_state for _ in 1:𝓂.constants.post_model_macro.nExo + 1]

    decomposition[:, end, :] .= variables

    if marginal_contribution
        nE = 𝓂.constants.post_model_macro.nExo
        aumann_shapley_shock_decomposition_pruned_2nd_order!(decomposition,
                                                      variables,
                                                      shocks,
                                                      initial_state,
                                                      𝐒,
                                                      T,
                                                      nE;
                                                      verbose = opts.verbose)
        return variables, shocks, zeros(0,0), decomposition
    end

    for i in 1:𝓂.constants.post_model_macro.nExo
        sck = zeros(𝓂.constants.post_model_macro.nExo)
        sck[i] = shocks[i, 1]

        aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; sck]
        aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]

        states[i] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(initial_state , sck)
        decomposition[:,i,1] = sum(states[i])
    end

    aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; shocks[:, 1]]
    aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]

    states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(initial_state, shocks[:, 1])

    decomposition[:, end - 2, 1] = sum(states[end]) - sum(decomposition[:, 1:end - 3, 1], dims = 2)
    decomposition[:, end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:, 1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.constants.post_model_macro.nExo
            sck = zeros(𝓂.constants.post_model_macro.nExo)
            sck[ii] = shocks[ii, i]
            
            aug_state₁ = [states[ii][1][T.past_not_future_and_mixed_idx]; 1; sck]
            aug_state₂ = [states[ii][2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
    
            states[ii] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(states[ii] , sck)
            decomposition[:, ii, i] = sum(states[ii])
        end

        aug_state₁ = [states[end][1][T.past_not_future_and_mixed_idx]; 1; shocks[:, i]]
        aug_state₂ = [states[end][2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
    
        states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * ℒ.kron(aug_state₁, aug_state₁) / 2] # state_update(states[end] , shocks[:, i])

        decomposition[:, end - 2, i] = sum(states[end]) - sum(decomposition[:, 1:end - 3, i], dims = 2)
        decomposition[:, end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:, 1:end - 2, i], dims = 2)
    end

    return variables, shocks, zeros(0,0), decomposition
end

@unstable function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:third_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                opts::CalculationOptions = merge_calculation_options())
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    ms = constants.post_complete_parameters

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    
    observables = get_and_check_observables(T, data_in_deviations)

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:third_order), 𝓂.parameter_values, 𝓂, opts = opts) # timer = timer,

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        @error "Could not find 3rd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    𝐒 = [𝐒₁, 𝐒₂, 𝐒₃]
    
    all_SS = expand_steady_state(SS_and_pars, ms)

    state = collect(sss) - all_SS


    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    computational_constants = ensure_computational_constants!(𝓂.constants)
    so = ensure_conditional_forecast_constants!(𝓂.constants; third_order = true)
    shock²_idxs = computational_constants.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = computational_constants.var_vol²_idxs
    var²_idxs = so.var²_idxs
    to = 𝓂.constants.third_order

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    state = state[T.past_not_future_and_mixed_idx]

    var_vol³_idxs = to.var_vol³_idxs
    shock³_idxs = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs = to.shockvar³_idxs

    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    II = sparse(ℒ.I(T.nExo^2))

    data_arr = collect(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_arr)
    n_exo = T.nExo

    for i in axes(data_in_deviations,2)
        state¹⁻ = state

        state¹⁻_vol = vcat(state¹⁻, 1)
        
        shock_independent = collect(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))

        if has_missing
            idx = obs_idx_per_t[i]
            m = length(idx)
            if m == 0
                x = zeros(n_exo)
                matched = true
            else
                if m > n_exo
                    @error "Inversion filter (3rd) failed at step $i: m=$m > n_exo=$n_exo"
                    return variables, shocks, zeros(0,0), zeros(0,0)
                end
                𝐒ⁱ_v   = 𝐒ⁱ[idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
                𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ[idx, :]
                si_v   = shock_independent[idx]
                x, matched = find_shocks(Val(filter_algorithm),
                                        zeros(n_exo),
                                        kron_buffer, kron_buffer², kron_buffer2,
                                        kron_buffer3, kron_buffer4, J,
                                        𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            end
        else
            x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
        end
                                
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂)
        #             println("$i - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             println("$i - $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         else
        #             y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #             norm1 = ℒ.norm(y)

        #             norm2 = ℒ.norm(shock_independent)

        #             println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     else
        #         println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end
        if !matched
            @error "Inversion filter failed at step $i"
            return variables, shocks, zeros(0,0), zeros(0,0)
        end 
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
            #     println("$filter_algorithm: $matched; current x: $x")
            #     if !matched
            #         x, matched = find_shocks(Val(:COBYLA), 
            #                                 x,
            #                                 kron_buffer,
            #                                 kron_buffer²,
            #                                 kron_buffer2,
            #                                 kron_buffer3,
            #                                 J,
            #                                 𝐒ⁱ,
            #                                 𝐒ⁱ²ᵉ,
            #                                 𝐒ⁱ³ᵉ,
            #                                 shock_independent)
            #         println("COBYLA: $matched; current x: $x")
            #     end
            # end
        # end

        # x2, mat = find_shocks(Val(:COBYLA), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
            
        # x3, mat2 = find_shocks(Val(filter_algorithm), 
        #                         x2,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # # if mat
        #     println("COBYLA - $mat: $x2, $(ℒ.norm(x2))")
        # # end
        # # if mat2
        #     println("LagrangeNewton restart - $mat2: $x3, $(ℒ.norm(x3))")
        # # end

        aug_state = [state; 1; x]

        # res = 𝐒[1][cond_var_idx, :] * aug_state + 𝐒[2][cond_var_idx, :] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3][cond_var_idx, :] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")

        # state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        full_state = 𝐒[1] * aug_state + 𝐒[2] * ℒ.kron(aug_state, aug_state) / 2 + 𝐒[3] * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        # state = state_update(state, x)
        
        shocks[:,i] .= x
        variables[:,i] .= full_state

        state .= full_state[T.past_not_future_and_mixed_idx]
    end
    
    return variables, shocks, zeros(0,0), zeros(0,0)
end


@unstable function filter_data_with_model(𝓂::ℳ,
                                data_in_deviations::KeyedArray{Float64},
                                ::Val{:pruned_third_order}, # algo
                                ::Val{:inversion}; # filter
                                warmup_iterations::Int = 0,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                smooth::Bool = true,
                                marginal_contribution::Bool = false,
                                opts::CalculationOptions = merge_calculation_options())
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)
    T = constants.post_model_macro
    ensure_model_structure_constants!(constants, 𝓂.equations.calibration_parameters)
    ms = constants.post_complete_parameters

    variables = zeros(T.nVars, size(data_in_deviations,2))
    shocks = zeros(T.nExo, size(data_in_deviations,2))
    decomposition = zeros(T.nVars, marginal_contribution ? T.nExo + 2 : T.nExo + 3, size(data_in_deviations, 2))
    
    observables = get_and_check_observables(T, data_in_deviations)

    sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_stochastic_steady_state(Val(:pruned_third_order), 𝓂.parameter_values, 𝓂, opts = opts) # timer = timer,

    if !converged || solution_error > opts.tol.nsss.acceptance_tol
        @error "Could not find pruned 3rd order stochastic steady state"
        return variables, shocks, zeros(0,0), zeros(0,0)
    end

    𝐒 = [𝐒₁, 𝐒₂, 𝐒₃]

    all_SS = expand_steady_state(SS_and_pars, ms)

    state = [zeros(T.nVars), collect(sss) - all_SS, zeros(T.nVars)]

    precision_factor = 1.0

    n_obs = size(data_in_deviations,2)

    cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

    computational_constants = ensure_computational_constants!(𝓂.constants)
    so = ensure_conditional_forecast_constants!(𝓂.constants; third_order = true)
    s_in_s⁺ = computational_constants.s_in_s
    e_in_s⁺ = computational_constants.e_in_s⁺

    shockvar_idxs = so.shockvar_idxs
    
    shock²_idxs = computational_constants.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs = computational_constants.var_vol²_idxs

    var²_idxs = so.var²_idxs
    to = 𝓂.constants.third_order

    𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
    𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
    𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
    𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx,shockvar_idxs]
    𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
    𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

    𝐒²⁻ᵛ    = nnz(𝐒²⁻ᵛ)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
    𝐒²⁻     = nnz(𝐒²⁻)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
    𝐒²⁻ᵉ    = nnz(𝐒²⁻ᵉ)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ   = nnz(𝐒²⁻ᵛᵉ)   / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ)   : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ     = nnz(𝐒²ᵉ)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
    𝐒⁻²     = nnz(𝐒⁻²)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

    var_vol³_idxs = to.var_vol³_idxs
    shock³_idxs = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs = to.shockvar³_idxs

    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = nnz(𝐒³⁻ᵛ)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = nnz(𝐒³⁻ᵉ)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = nnz(𝐒³ᵉ)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = nnz(𝐒⁻³)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³

    initial_state = deepcopy(state)

    state₁ = state[1][T.past_not_future_and_mixed_idx]
    state₂ = state[2][T.past_not_future_and_mixed_idx]
    state₃ = state[3][T.past_not_future_and_mixed_idx]

    kron_buffer = zeros(T.nExo^2)

    kron_buffer² = zeros(T.nExo^3)

    II = sparse(ℒ.I(T.nExo^2))
    
    J = ℒ.I(T.nExo)

    kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

    kron_buffer3 = ℒ.kron(J, kron_buffer)

    kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

    data_arr = collect(data_in_deviations)
    obs_idx_per_t, has_missing = build_obs_index(data_arr)
    n_exo = T.nExo

    for i in axes(data_in_deviations,2)
        # state¹⁻ = state₁

        state¹⁻_vol = vcat(state₁, 1)
        state²⁻_vol = vcat(state₂, 0)

        # state²⁻ = state₂#[T.past_not_future_and_mixed_idx]

        # state³⁻ = state₃#[T.past_not_future_and_mixed_idx]

        shock_independent = collect(data_in_deviations[:,i])

        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)

        ℒ.mul!(shock_independent, 𝐒¹⁻, state₃, -1, 1)

        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state₁, state₂), -1, 1)
        
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒²⁻ᵛᵉ * ℒ.kron(ℒ.I(T.nExo), state²⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

        𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

        # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)

        init_guess = zeros(size(𝐒ⁱ, 2))


        # x² , matched = find_shocks(Val(filter_algorithm), 
        #                         init_guess,
        #                         kron_buffer,
        #                         kron_buffer2,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         shock_independent,
        #                         # max_iter = 200
        #                         )
        #                         println(x²)

        if has_missing
            idx = obs_idx_per_t[i]
            m = length(idx)
            if m == 0
                x = zeros(n_exo)
                matched = true
            else
                if m > n_exo
                    @error "Inversion filter (pruned 3rd) failed at step $i: m=$m > n_exo=$n_exo"
                    return variables, shocks, zeros(0,0), zeros(0,0)
                end
                𝐒ⁱ_v   = 𝐒ⁱ[idx, :]
                𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
                𝐒ⁱ³ᵉ_v = 𝐒ⁱ³ᵉ[idx, :]
                si_v   = shock_independent[idx]
                x, matched = find_shocks(Val(filter_algorithm),
                                        zeros(n_exo),
                                        kron_buffer, kron_buffer², kron_buffer2,
                                        kron_buffer3, kron_buffer4, J,
                                        𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            end
        else
            x, matched = find_shocks(Val(filter_algorithm), 
                                init_guess,
                                kron_buffer,
                                kron_buffer²,
                                kron_buffer2,
                                kron_buffer3,
                                kron_buffer4,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                𝐒ⁱ³ᵉ,
                                shock_independent,
                                # max_iter = 200
                                )
        end
                                
                                # println(x)
        # println("$filter_algorithm: $matched; current x: $x, $(ℒ.norm(x))")
        # if !matched

        # backup_solver = :COBYLA

        # if filter_algorithm ≠ backup_solver
        #     x̂, matched2 = find_shocks(Val(backup_solver), 
        #                         zeros(size(𝐒ⁱ, 2)),
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 5000
        #                         )
        #     if ℒ.norm(x̂) * (1 - eps(Float32)) < ℒ.norm(x)
        #         x̄, matched3 = find_shocks(Val(filter_algorithm), 
        #                             x̂,
        #                             kron_buffer,
        #                             kron_buffer²,
        #                             kron_buffer2,
        #                             kron_buffer3,
        #                             kron_buffer4,
        #                             J,
        #                             𝐒ⁱ,
        #                             𝐒ⁱ²ᵉ,
        #                             𝐒ⁱ³ᵉ,
        #                             shock_independent,
        #                             # max_iter = 200
        #                             )
                              
        #         if matched3 && (!matched || ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x̂) || (matched && ℒ.norm(x̄) * (1 - eps(Float32)) < ℒ.norm(x)))
        #             # println("$i - $filter_algorithm restart - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̄
        #             matched = matched3
        #         elseif matched2
        #             # println("$i - $backup_solver - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")
        #             x = x̂
        #             matched = matched2
        #         # else
        #         #     y = 𝐒ⁱ * x + 𝐒ⁱ²ᵉ * ℒ.kron(x,x) + 𝐒ⁱ³ᵉ * ℒ.kron(x, ℒ.kron(x,x))

        #         #     norm1 = ℒ.norm(y)

        #         #     norm2 = ℒ.norm(shock_independent)

        #             # println("$i - $filter_algorithm - $filter_algorithm restart ($matched3) - $(ℒ.norm(x̄)), $backup_solver ($matched2) - $(ℒ.norm(x̂)), $filter_algorithm ($matched) - $(ℒ.norm(x))")#, residual norm: $(ℒ.norm(y - shock_independent) / max(norm1,norm2))")
        #         end
        #     # else
        #     #     println("$i - $filter_algorithm ($matched) - $(ℒ.norm(x)), $backup_solver ($matched2) - $(ℒ.norm(x̂))")
        #     end
        # end
        if !matched
            @error "Inversion filter failed at step $i"
            return variables, shocks, zeros(0,0), decomposition
        end
            # println("COBYLA: $matched; current x: $x")
            # if !matched
            #     x, matched = find_shocks(Val(filter_algorithm), 
            #                             x,
            #                             kron_buffer,
            #                             kron_buffer²,
            #                             kron_buffer2,
            #                             kron_buffer3,
            #                             J,
            #                             𝐒ⁱ,
            #                             𝐒ⁱ²ᵉ,
            #                             𝐒ⁱ³ᵉ,
            #                             shock_independent)
                # println("$filter_algorithm: $matched; current x: $x")
                # if !matched
                #     x, matched = find_shocks(Val(:COBYLA), 
                #                             x,
                #                             kron_buffer,
                #                             kron_buffer²,
                #                             kron_buffer2,
                #                             kron_buffer3,
                #                             J,
                #                             𝐒ⁱ,
                #                             𝐒ⁱ²ᵉ,
                #                             𝐒ⁱ³ᵉ,
                #                             shock_independent)
                # end
            # end
        # end

        # x2, mat = find_shocks(Val(:SLSQP), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
            
        # x3, mat2 = find_shocks(Val(:COBYLA), 
        #                         x,
        #                         kron_buffer,
        #                         kron_buffer²,
        #                         kron_buffer2,
        #                         kron_buffer3,
        #                         kron_buffer4,
        #                         J,
        #                         𝐒ⁱ,
        #                         𝐒ⁱ²ᵉ,
        #                         𝐒ⁱ³ᵉ,
        #                         shock_independent,
        #                         # max_iter = 500
        #                         )
        # if mat
        #     println("SLSQP: $(ℒ.norm(x2-x) / max(ℒ.norm(x2), ℒ.norm(x))), $(ℒ.norm(x2)-ℒ.norm(x))")
        # elseif mat2
        #     println("COBYLA: $(ℒ.norm(x3-x) / max(ℒ.norm(x3), ℒ.norm(x))), $(ℒ.norm(x3)-ℒ.norm(x))")
        # end

        aug_state₁ = [state₁; 1; x]
        aug_state₁̂ = [state₁; 0; x]
        aug_state₂ = [state₂; 0; zero(x)]
        aug_state₃ = [state₃; 0; zero(x)]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        # res = 𝐒[1][cond_var_idx,:] * aug_state₁   +   𝐒[1][cond_var_idx,:] * aug_state₂ + 𝐒[2][cond_var_idx,:] * kron_aug_state₁ / 2   +   𝐒[1][cond_var_idx,:] * aug_state₃ + 𝐒[2][cond_var_idx,:] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3][cond_var_idx,:] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - data_in_deviations[:,i]
        # println("Match with data: $res")
        
        # println(ℒ.norm(x))

        # state = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2, 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        state = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        
        state₁ .= state[1][T.past_not_future_and_mixed_idx]
        state₂ .= state[2][T.past_not_future_and_mixed_idx]
        state₃ .= state[3][T.past_not_future_and_mixed_idx]

        variables[:,i] .= sum(state)
        shocks[:,i] .= x
    end

    states = [initial_state for _ in 1:𝓂.constants.post_model_macro.nExo + 1]

    decomposition[:, end, :] .= variables

    if marginal_contribution
        nE = 𝓂.constants.post_model_macro.nExo
        aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition,
                                                      variables,
                                                      shocks,
                                                      initial_state,
                                                      𝐒,
                                                      T,
                                                      nE;
                                                      verbose = opts.verbose)
        return variables, shocks, zeros(0,0), decomposition
    end

    for i in 1:𝓂.constants.post_model_macro.nExo
        sck = zeros(𝓂.constants.post_model_macro.nExo)
        sck[i] = shocks[i, 1]

        aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; sck]
        aug_state₁̂ = [initial_state[1][T.past_not_future_and_mixed_idx]; 0; sck]
        aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
        aug_state₃ = [initial_state[3][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        states[i] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(initial_state , sck)

        decomposition[:,i,1] = sum(states[i])
    end

    aug_state₁ = [initial_state[1][T.past_not_future_and_mixed_idx]; 1; shocks[:, 1]]
    aug_state₁̂ = [initial_state[1][T.past_not_future_and_mixed_idx]; 0; shocks[:, 1]]
    aug_state₂ = [initial_state[2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]
    aug_state₃ = [initial_state[3][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, 1])]
    
    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

    states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(initial_state, shocks[:, 1])

    decomposition[:,end - 2, 1] = sum(states[end]) - sum(decomposition[:,1:end - 3, 1], dims = 2)
    decomposition[:,end - 1, 1] .= decomposition[:, end, 1] - sum(decomposition[:,1:end - 2, 1], dims = 2)

    for i in 2:size(data_in_deviations, 2)
        for ii in 1:𝓂.constants.post_model_macro.nExo
            sck = zeros(𝓂.constants.post_model_macro.nExo)
            sck[ii] = shocks[ii, i]

            aug_state₁ = [states[ii][1][T.past_not_future_and_mixed_idx]; 1; sck]
            aug_state₁̂ = [states[ii][1][T.past_not_future_and_mixed_idx]; 0; sck]
            aug_state₂ = [states[ii][2][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
            aug_state₃ = [states[ii][3][T.past_not_future_and_mixed_idx]; 0; zero(sck)]
            
            kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

            states[ii] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(states[ii] , sck)

            decomposition[:, ii, i] = sum(states[ii])
        end

        aug_state₁ = [states[end][1][T.past_not_future_and_mixed_idx]; 1; shocks[:, i]]
        aug_state₁̂ = [states[end][1][T.past_not_future_and_mixed_idx]; 0; shocks[:, i]]
        aug_state₂ = [states[end][2][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
        aug_state₃ = [states[end][3][T.past_not_future_and_mixed_idx]; 0; zero(shocks[:, i])]
        
        kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

        states[end] = [𝐒[1] * aug_state₁, 𝐒[1] * aug_state₂ + 𝐒[2] * kron_aug_state₁ / 2, 𝐒[1] * aug_state₃ + 𝐒[2] * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒[3] * ℒ.kron(kron_aug_state₁,aug_state₁) / 6] # state_update(states[end] , shocks[:, i])
        
        decomposition[:,end - 2, i] = sum(states[end]) - sum(decomposition[:,1:end - 3, i], dims = 2)
        decomposition[:,end - 1, i] .= decomposition[:, end, i] - sum(decomposition[:,1:end - 2, i], dims = 2)
    end

    return variables, shocks, zeros(0,0), decomposition
end

# ===========================================================================
# Missing-value variants of `calculate_loglikelihood(::Val{:inversion}, ...)`.
# Routed to from the dense entry points when `build_obs_index` reports any
# non-finite entry. These functions never run when the data are dense (zero
# overhead for existing callers).
# ===========================================================================

function calculate_loglikelihood_with_missing(::Val{:inversion}, ::Val{:first_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Matrix{R},
                                                    data_in_deviations::Matrix{R},
                                                    constants::constants,
                                                    state,
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    eff_presample_periods = presample_periods + warmup_iterations
    T = constants.post_model_macro
    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)

    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)
    t⁻ = T.past_not_future_and_mixed_idx

    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = false)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond)

    # Reduce state to past_not_future_and_mixed_idx rows (matches higher-order
    # methods); pre-slice 𝐒 to the same rows so the per-period state update
    # only touches the minimum number of rows.
    state = convert(Vector{R}, state[1][t⁻])
    𝐒past = 𝐒[t⁻, :]

    shocks² = zero(R)
    logabsdets = zero(R)
    n_obs_total = 0

    jac_full = 𝐒[cond_var_idx, end-n_exo+1:end]
    𝐒obs    = 𝐒[cond_var_idx, 1:end-n_exo]

    state_concat = ws.state_concat
    y_full = ws.y_obs
    x_buf  = ws.x_shocks
    jac_v_buf = ws.jacc_v_buf
    y_v_buf   = ws.obs_sub_buf
    JJt_buf   = ws.JJt_buf
    fill!(x_buf, zero(R))

    for i in axes(data_in_deviations, 2)
        idx = obs_idx_per_t[i]
        m = length(idx)

        if m == 0
            fill!(x_buf, zero(R))
            copyto!(state_concat, 1, state, 1, n_past)
            copyto!(state_concat, n_past + 1, x_buf, 1, n_exo)
            ℒ.mul!(state, 𝐒past, state_concat)
            continue
        end

        ℒ.mul!(y_full, 𝐒obs, state)
        jac_v = view(jac_v_buf, 1:m, :)
        y_v   = view(y_v_buf, 1:m)
        @inbounds for k in 1:m
            ii = idx[k]
            y_v[k] = data_in_deviations[ii, i] - y_full[ii]
            for j in 1:n_exo
                jac_v[k, j] = jac_full[ii, j]
            end
        end

        if !all(isfinite, jac_v)
            if opts.verbose println("Inversion filter failed at step $i (non-finite Jacobian)") end
            return on_failure_loglikelihood
        end

        if m == n_exo
            jacdecomp = ℒ.lu(jac_v, check = false)
            if !ℒ.issuccess(jacdecomp)
                if opts.verbose println("Inversion filter failed at step $i (LU singular)") end
                return on_failure_loglikelihood
            end
            x_v = jacdecomp \ y_v
            if i > eff_presample_periods
                logabsdets += ℒ.logabsdet(jacdecomp)[1]
            end
            copyto!(x_buf, x_v)
        else
            JJt = view(JJt_buf, 1:m, 1:m)
            ℒ.mul!(JJt, jac_v, jac_v')
            JJt_lu = ℒ.lu(JJt, check = false)
            if !ℒ.issuccess(JJt_lu)
                if opts.verbose println("Inversion filter failed at step $i (LU singular)") end
                return on_failure_loglikelihood
            end
            z = JJt_lu \ y_v
            ℒ.mul!(x_buf, jac_v', z)
            if i > eff_presample_periods
                logabsdets += ℒ.logabsdet(JJt_lu)[1] / 2
            end
        end

        if i > eff_presample_periods
            shocks² += sum(abs2, x_buf)
            n_obs_total += m
            if !isfinite(shocks²) || !isfinite(logabsdets)
                return on_failure_loglikelihood
            end
        end

        copyto!(state_concat, 1, state, 1, n_past)
        copyto!(state_concat, n_past + 1, x_buf, 1, n_exo)
        ℒ.mul!(state, 𝐒past, state_concat)
    end

    return -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood_with_missing(::Val{:inversion}, ::Val{:pruned_second_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}},
                                                    data_in_deviations::Matrix{R},
                                                    constants::constants,
                                                    state,
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    eff_presample_periods = presample_periods + warmup_iterations
    T = constants.post_model_macro
    n_exo  = T.nExo
    n_past = T.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)

    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = false)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond)

    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants)
    shock²_idxs    = cc.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs  = cc.var_vol²_idxs
    var²_idxs      = so.var²_idxs

    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:n_past]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-n_exo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻  = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ = nnz(𝐒²⁻ᵛ) / length(𝐒²⁻ᵛ) > .1 ? collect(𝐒²⁻ᵛ) : 𝐒²⁻ᵛ
    𝐒²⁻  = nnz(𝐒²⁻)  / length(𝐒²⁻)  > .1 ? collect(𝐒²⁻)  : 𝐒²⁻
    𝐒²⁻ᵉ = nnz(𝐒²⁻ᵉ) / length(𝐒²⁻ᵉ) > .1 ? collect(𝐒²⁻ᵉ) : 𝐒²⁻ᵉ
    𝐒²ᵉ  = nnz(𝐒²ᵉ)  / length(𝐒²ᵉ)  > .1 ? collect(𝐒²ᵉ)  : 𝐒²ᵉ
    𝐒⁻²  = nnz(𝐒⁻²)  / length(𝐒⁻²)  > .1 ? collect(𝐒⁻²)  : 𝐒⁻²

    state₁ = convert(Vector{R}, state[1][T.past_not_future_and_mixed_idx])
    state₂ = convert(Vector{R}, state[2][T.past_not_future_and_mixed_idx])

    J = ℒ.I(n_exo)
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2

    state¹⁻_vol      = ws.state_vol
    aug_state₁       = ws.aug_state₁
    aug_state₂       = ws.aug_state₂
    kronaug_state₁   = ws.kronaug_state
    kron_buffer      = ws.kron_buffer
    kron_buffer2     = ws.kron_buffer2
    kron_buffer3     = ws.kron_buffer_state
    kronstate¹⁻_vol  = ws.kronstate_vol
    shock_independent = ws.shock_independent
    𝐒ⁱ_full          = ws.Si_buffer
    jacc_v_buf       = ws.jacc_v_buf
    init_guess       = ws.init_guess
    x_zero           = ws.x_shocks
    fill!(x_zero, zero(R))

    shocks² = zero(R)
    logabsdets = zero(R)
    n_obs_total = 0

    for i in axes(data_in_deviations, 2)
        idx = obs_idx_per_t[i]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, state₁, 1)
        state¹⁻_vol[end] = one(R)

        copyto!(shock_independent, view(data_in_deviations, :, i))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, state₂, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer3)
        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ_full)

        if m == 0
            x = x_zero
            fill!(x, zero(R))
        else
            if m > n_exo
                if opts.verbose println("Inversion filter (pruned 2nd) failed at step $i: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, zero(R))
            x, matched = find_shocks(Val(filter_algorithm),
                                    init_guess, kron_buffer, kron_buffer2, J,
                                    𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter (pruned 2nd) failed at step $i") end
                return on_failure_loglikelihood
            end
            if i > eff_presample_periods
                jacc_v = view(jacc_v_buf, 1:m, :)
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.mul!(jacc_v, 𝐒ⁱ²ᵉ_v, kron_buffer2)
                ℒ.axpby!(1, 𝐒ⁱ_v, 2, jacc_v)
                logabsdets += m == n_exo ? ℒ.logabsdet(jacc_v)[1] : ℒ.logabsdet(jacc_v * jacc_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood
                end
            end
        end

        copyto!(aug_state₁, 1, state₁, 1)
        aug_state₁[length(state₁) + 1] = one(R)
        copyto!(aug_state₁, length(state₁) + 2, x, 1)
        copyto!(aug_state₂, 1, state₂, 1)
        aug_state₂[length(state₂) + 1] = zero(R)
        fill!(view(aug_state₂, length(state₂) + 2:length(aug_state₂)), zero(R))

        ℒ.mul!(state₁, 𝐒⁻¹, aug_state₁)
        ℒ.mul!(state₂, 𝐒⁻¹, aug_state₂)
        ℒ.kron!(kronaug_state₁, aug_state₁, aug_state₁)
        ℒ.mul!(state₂, 𝐒⁻², kronaug_state₁, 1/2, 1)
    end

    return -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood_with_missing(::Val{:inversion}, ::Val{:second_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}},
                                                    data_in_deviations::Matrix{R},
                                                    constants::constants,
                                                    state,
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    eff_presample_periods = presample_periods + warmup_iterations
    T = constants.post_model_macro
    n_exo  = T.nExo
    n_past = T.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)

    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = false)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond)

    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants)
    shock²_idxs    = cc.shock²_idxs
    shockvar²_idxs = so.shockvar²_idxs
    var_vol²_idxs  = cc.var_vol²_idxs

    𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ  = 𝐒[1][cond_var_idx, end-n_exo+1:end]

    𝐒²⁻ᵛ = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻ᵉ = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ  = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ = nnz(𝐒²⁻ᵛ) / length(𝐒²⁻ᵛ) > .1 ? collect(𝐒²⁻ᵛ) : 𝐒²⁻ᵛ
    𝐒²⁻ᵉ = nnz(𝐒²⁻ᵉ) / length(𝐒²⁻ᵉ) > .1 ? collect(𝐒²⁻ᵉ) : 𝐒²⁻ᵉ
    𝐒²ᵉ  = nnz(𝐒²ᵉ)  / length(𝐒²ᵉ)  > .1 ? collect(𝐒²ᵉ)  : 𝐒²ᵉ
    𝐒⁻²  = nnz(𝐒⁻²)  / length(𝐒⁻²)  > .1 ? collect(𝐒⁻²)  : 𝐒⁻²

    st = convert(Vector{R}, state[T.past_not_future_and_mixed_idx])

    J = ℒ.I(n_exo)
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2

    state¹⁻_vol      = ws.state_vol
    aug_state        = ws.aug_state₁
    kronaug_state    = ws.kronaug_state
    kron_buffer      = ws.kron_buffer
    kron_buffer2     = ws.kron_buffer2
    kron_buffer3     = ws.kron_buffer_state
    kronstate¹⁻_vol  = ws.kronstate_vol
    shock_independent = ws.shock_independent
    𝐒ⁱ_full          = ws.Si_buffer
    jacc_v_buf       = ws.jacc_v_buf
    init_guess       = ws.init_guess
    x_zero           = ws.x_shocks
    fill!(x_zero, zero(R))

    shocks² = zero(R)
    logabsdets = zero(R)
    n_obs_total = 0

    for i in axes(data_in_deviations, 2)
        idx = obs_idx_per_t[i]
        m = length(idx)

        copyto!(state¹⁻_vol, 1, st, 1)
        state¹⁻_vol[end] = one(R)

        copyto!(shock_independent, view(data_in_deviations, :, i))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.kron!(kronstate¹⁻_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate¹⁻_vol, -1/2, 1)

        ℒ.kron!(kron_buffer3, J, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer3)
        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ_full)

        if m == 0
            x = x_zero
            fill!(x, zero(R))
        else
            if m > n_exo
                if opts.verbose println("Inversion filter (2nd) failed at step $i: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood
            end
            𝐒ⁱ_v   = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v = 𝐒ⁱ²ᵉ[idx, :]
            si_v   = shock_independent[idx]
            fill!(init_guess, zero(R))
            x, matched = find_shocks(Val(filter_algorithm),
                                    init_guess, kron_buffer, kron_buffer2, J,
                                    𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter (2nd) failed at step $i") end
                return on_failure_loglikelihood
            end
            if i > eff_presample_periods
                jacc_v = view(jacc_v_buf, 1:m, :)
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.mul!(jacc_v, 𝐒ⁱ²ᵉ_v, kron_buffer2)
                ℒ.axpby!(1, 𝐒ⁱ_v, 2, jacc_v)
                logabsdets += m == n_exo ? ℒ.logabsdet(jacc_v)[1] : ℒ.logabsdet(jacc_v * jacc_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood
                end
            end
        end

        copyto!(aug_state, 1, st, 1)
        aug_state[length(st) + 1] = one(R)
        copyto!(aug_state, length(st) + 2, x, 1)

        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.mul!(st, 𝐒⁻¹, aug_state)
        ℒ.mul!(st, 𝐒⁻², kronaug_state, 1/2, 1)
    end

    return -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood_with_missing(::Val{:inversion}, ::Val{:pruned_third_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}},
                                                    data_in_deviations::Matrix{R},
                                                    constants::constants,
                                                    state,
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    eff_presample_periods = presample_periods + warmup_iterations
    T = constants.post_model_macro
    n_exo  = T.nExo
    n_past = T.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)

    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond; third_order = true)

    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants; third_order = true)
    shockvar_idxs   = cc.shockvar_idxs
    shock²_idxs     = cc.shock²_idxs
    shockvar²_idxs  = so.shockvar²_idxs
    var_vol²_idxs   = cc.var_vol²_idxs
    var²_idxs       = so.var²_idxs
    to = constants.third_order
    var_vol³_idxs   = to.var_vol³_idxs
    shock³_idxs     = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs  = to.shockvar³_idxs

    𝐒⁻¹   = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻   = 𝐒[1][cond_var_idx, 1:n_past]
    𝐒¹⁻ᵛ  = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ   = 𝐒[1][cond_var_idx, end-n_exo+1:end]

    𝐒²⁻ᵛ  = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻   = 𝐒[2][cond_var_idx, var²_idxs]
    𝐒²⁻ᵉ  = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²⁻ᵛᵉ = 𝐒[2][cond_var_idx, shockvar_idxs]
    𝐒²ᵉ   = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²   = 𝐒[2][T.past_not_future_and_mixed_idx, :]
    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx, var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx, shockvar³2_idxs] |> collect
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx, shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx, shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ  = nnz(𝐒²⁻ᵛ)  / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)  : 𝐒²⁻ᵛ
    𝐒²⁻   = nnz(𝐒²⁻)   / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)   : 𝐒²⁻
    𝐒²⁻ᵉ  = nnz(𝐒²⁻ᵉ)  / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)  : 𝐒²⁻ᵉ
    𝐒²⁻ᵛᵉ = nnz(𝐒²⁻ᵛᵉ) / length(𝐒²⁻ᵛᵉ) > .1 ? collect(𝐒²⁻ᵛᵉ) : 𝐒²⁻ᵛᵉ
    𝐒²ᵉ   = nnz(𝐒²ᵉ)   / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)   : 𝐒²ᵉ
    𝐒⁻²   = nnz(𝐒⁻²)   / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)   : 𝐒⁻²
    𝐒³⁻ᵛ  = nnz(𝐒³⁻ᵛ)  / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)  : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ  = nnz(𝐒³⁻ᵉ)  / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)  : 𝐒³⁻ᵉ
    𝐒³ᵉ   = nnz(𝐒³ᵉ)   / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)   : 𝐒³ᵉ
    𝐒⁻³   = nnz(𝐒⁻³)   / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)   : 𝐒⁻³

    st1 = convert(Vector{R}, state[1][T.past_not_future_and_mixed_idx])
    st2 = convert(Vector{R}, state[2][T.past_not_future_and_mixed_idx])
    st3 = convert(Vector{R}, state[3][T.past_not_future_and_mixed_idx])

    J  = ℒ.I(n_exo)
    II = ℒ.I(n_exo^2)
    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    state_vol         = ws.state_vol
    kronstate_vol     = ws.kronstate_vol
    kronstate_vol³    = ws.kronstate_vol³
    state²⁻_vol       = ws.state²⁻_vol
    kron_buffer_state = ws.kron_buffer_state
    kron_buffer       = ws.kron_buffer
    kron_buffer²      = ws.kron_buffer²
    kron_buffer2      = ws.kron_buffer2
    kron_buffer3      = ws.kron_buffer3
    kron_buffer4      = ws.kron_buffer4
    shock_independent = ws.shock_independent
    𝐒ⁱ_full           = ws.Si_buffer
    𝐒ⁱ²ᵉ_full         = ws.Si2e_buffer
    jacc_v_buf        = ws.jacc_v_buf
    init_guess        = ws.init_guess
    x_zero            = ws.x_shocks
    fill!(x_zero, zero(R))
    aug_state₁        = ws.aug_state₁
    aug_state₁̂       = ws.aug_state₁̂
    aug_state₂        = ws.aug_state₂
    aug_state₃        = ws.aug_state₃
    kron_aug_state₁   = ws.kronaug_state
    kron_kron_aug_state₁ = ws.kron_kron_aug_state

    # Pruned-third specific kron buffers (not in ws, allocated once per call)
    kron_buffer3sv = zeros(R, n_exo * (n_past+1)^2, n_exo)
    kron_buffer4sv = zeros(R, n_exo^2 * (n_past+1), n_exo^2)
    kron_buffer2ss = zeros(R, n_past^2)

    shocks² = zero(R)
    logabsdets = zero(R)
    n_obs_total = 0

    for i in axes(data_in_deviations, 2)
        idx = obs_idx_per_t[i]
        m = length(idx)

        copyto!(state_vol, 1, st1, 1, n_past); state_vol[end] = one(R)
        state¹⁻_vol = state_vol

        copyto!(shock_independent, view(data_in_deviations, :, i))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, st2, -1, 1)
        ℒ.mul!(shock_independent, 𝐒¹⁻, st3, -1, 1)
        ℒ.kron!(kronstate_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate_vol, -1/2, 1)
        ℒ.kron!(kron_buffer2ss, st1, st2)
        ℒ.mul!(shock_independent, 𝐒²⁻, kron_buffer2ss, -1, 1)
        ℒ.kron!(kronstate_vol³, kronstate_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kronstate_vol³, -1/6, 1)

        copyto!(state²⁻_vol, 1, st2, 1); state²⁻_vol[end] = zero(R)
        ℒ.kron!(kron_buffer_state, J, state²⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵛᵉ, kron_buffer_state)
        ℒ.kron!(kron_buffer_state, J, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer_state, 1, 1)
        ℒ.kron!(kron_buffer3sv, kron_buffer_state, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)
        ℒ.axpy!(1, 𝐒¹ᵉ, 𝐒ⁱ_full)

        x_kron_II!(kron_buffer4sv, state¹⁻_vol)
        copyto!(𝐒ⁱ²ᵉ_full, 𝐒²ᵉ); ℒ.rdiv!(𝐒ⁱ²ᵉ_full, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ_full, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        if m == 0
            x = x_zero
            fill!(x, zero(R))
        else
            if m > n_exo
                if opts.verbose println("Inversion filter (pruned 3rd) failed at step $i: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood
            end
            𝐒ⁱ_v    = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v  = 𝐒ⁱ²ᵉ_full[idx, :]
            𝐒ⁱ³ᵉ_v  = 𝐒ⁱ³ᵉ[idx, :]
            si_v    = shock_independent[idx]
            fill!(init_guess, zero(R))
            x, matched = find_shocks(Val(filter_algorithm),
                                    init_guess, kron_buffer, kron_buffer², kron_buffer2,
                                    kron_buffer3, kron_buffer4, J,
                                    𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter (pruned 3rd) failed at step $i") end
                return on_failure_loglikelihood
            end
            if i > eff_presample_periods
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.kron!(kron_buffer3, kron_buffer2, x)
                jacc_v = view(jacc_v_buf, 1:m, :)
                ℒ.mul!(jacc_v, 𝐒ⁱ²ᵉ_v, kron_buffer2)
                ℒ.mul!(jacc_v, 𝐒ⁱ³ᵉ_v, kron_buffer3, 3, 2)
                ℒ.axpby!(-1, 𝐒ⁱ_v, -1, jacc_v)
                logabsdets += m == n_exo ? ℒ.logabsdet(jacc_v)[1] : ℒ.logabsdet(jacc_v * jacc_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood
                end
            end
        end

        copyto!(aug_state₁, 1, st1, 1, n_past); aug_state₁[n_past+1] = one(R); copyto!(aug_state₁, n_past+2, x, 1, n_exo)
        copyto!(aug_state₁̂, 1, st1, 1, n_past); aug_state₁̂[n_past+1] = zero(R); copyto!(aug_state₁̂, n_past+2, x, 1, n_exo)
        copyto!(aug_state₂, 1, st2, 1, n_past); aug_state₂[n_past+1] = zero(R); fill!(view(aug_state₂, n_past+2:n_past+1+n_exo), zero(R))
        copyto!(aug_state₃, 1, st3, 1, n_past); aug_state₃[n_past+1] = zero(R); fill!(view(aug_state₃, n_past+2:n_past+1+n_exo), zero(R))

        ℒ.kron!(kron_aug_state₁, aug_state₁, aug_state₁)
        ℒ.kron!(kron_kron_aug_state₁, kron_aug_state₁, aug_state₁)

        ℒ.mul!(st1, 𝐒⁻¹, aug_state₁)
        ℒ.mul!(st2, 𝐒⁻¹, aug_state₂); ℒ.mul!(st2, 𝐒⁻², kron_aug_state₁, 1/2, 1)
        ℒ.mul!(st3, 𝐒⁻¹, aug_state₃)
        ℒ.kron!(kron_aug_state₁, aug_state₁̂, aug_state₂)
        ℒ.mul!(st3, 𝐒⁻², kron_aug_state₁, 1, 1)
        ℒ.mul!(st3, 𝐒⁻³, kron_kron_aug_state₁, 1/6, 1)
    end

    return -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2
end


function calculate_loglikelihood_with_missing(::Val{:inversion}, ::Val{:third_order},
                                                    observables_index::Vector{Int},
                                                    𝐒::Vector{AbstractMatrix{R}},
                                                    data_in_deviations::Matrix{R},
                                                    constants::constants,
                                                    state,
                                                    workspaces::workspaces,
                                                    obs_idx_per_t::Vector{Vector{Int}};
                                                    warmup_iterations::Int = 0,
                                                    presample_periods::Int = 0,
                                                    initial_covariance::Symbol = :theoretical,
                                                    on_failure_loglikelihood::U = -Inf,
                                                    opts::CalculationOptions = merge_calculation_options(),
                                                    filter_algorithm::Symbol = :LagrangeNewton)::R where {R <: Real, U <: AbstractFloat}
    eff_presample_periods = presample_periods + warmup_iterations
    T = constants.post_model_macro
    n_exo  = T.nExo
    n_past = T.nPast_not_future_and_mixed
    cond_var_idx = observables_index
    n_cond = length(cond_var_idx)

    ws = R === Float64 ? workspaces.inversion : Inversion_workspace(R)
    ensure_inversion_buffers!(ws, n_exo, n_past; third_order = true)
    ensure_inversion_estimation_buffers!(ws, n_exo, n_cond; third_order = true)

    cc = ensure_computational_constants!(constants)
    so = ensure_conditional_forecast_constants!(constants; third_order = true)
    shock²_idxs     = cc.shock²_idxs
    shockvar²_idxs  = so.shockvar²_idxs
    var_vol²_idxs   = cc.var_vol²_idxs
    to = constants.third_order
    var_vol³_idxs   = to.var_vol³_idxs
    shock³_idxs     = to.shock³_idxs
    shockvar³2_idxs = to.shockvar³2_idxs
    shockvar³_idxs  = to.shockvar³_idxs

    𝐒⁻¹   = 𝐒[1][T.past_not_future_and_mixed_idx, :]
    𝐒¹⁻ᵛ  = 𝐒[1][cond_var_idx, 1:n_past+1]
    𝐒¹ᵉ   = 𝐒[1][cond_var_idx, end-n_exo+1:end]

    𝐒²⁻ᵛ  = 𝐒[2][cond_var_idx, var_vol²_idxs]
    𝐒²⁻ᵉ  = 𝐒[2][cond_var_idx, shockvar²_idxs]
    𝐒²ᵉ   = 𝐒[2][cond_var_idx, shock²_idxs]
    𝐒⁻²   = 𝐒[2][T.past_not_future_and_mixed_idx, :]
    𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx, var_vol³_idxs]
    𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx, shockvar³2_idxs] |> collect
    𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx, shockvar³_idxs]
    𝐒³ᵉ   = 𝐒[3][cond_var_idx, shock³_idxs]
    𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx, :]

    𝐒²⁻ᵛ  = nnz(𝐒²⁻ᵛ)  / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)  : 𝐒²⁻ᵛ
    𝐒²⁻ᵉ  = nnz(𝐒²⁻ᵉ)  / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)  : 𝐒²⁻ᵉ
    𝐒²ᵉ   = nnz(𝐒²ᵉ)   / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)   : 𝐒²ᵉ
    𝐒⁻²   = nnz(𝐒⁻²)   / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)   : 𝐒⁻²
    𝐒³⁻ᵛ  = nnz(𝐒³⁻ᵛ)  / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)  : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ  = nnz(𝐒³⁻ᵉ)  / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)  : 𝐒³⁻ᵉ
    𝐒³ᵉ   = nnz(𝐒³ᵉ)   / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)   : 𝐒³ᵉ
    𝐒⁻³   = nnz(𝐒⁻³)   / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)   : 𝐒⁻³

    st = convert(Vector{R}, state[T.past_not_future_and_mixed_idx])

    J  = ℒ.I(n_exo)
    II = sparse(ℒ.I(n_exo^2))
    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

    state_vol         = ws.state_vol
    kronstate_vol     = ws.kronstate_vol
    kronstate_vol³    = ws.kronstate_vol³
    kron_buffer_state = ws.kron_buffer_state
    kron_buffer       = ws.kron_buffer
    kron_buffer²      = ws.kron_buffer²
    kron_buffer2      = ws.kron_buffer2
    kron_buffer3      = ws.kron_buffer3
    kron_buffer4      = ws.kron_buffer4
    shock_independent = ws.shock_independent
    𝐒ⁱ_full           = ws.Si_buffer
    𝐒ⁱ²ᵉ_full         = ws.Si2e_buffer
    jacc_v_buf        = ws.jacc_v_buf
    init_guess        = ws.init_guess
    x_zero            = ws.x_shocks
    fill!(x_zero, zero(R))
    aug_state         = ws.aug_state₁
    kronaug_state     = ws.kronaug_state
    kron_kron_aug_state = ws.kron_kron_aug_state

    # Per-call third-order kron scratch buffers (not in ws)
    kron_buffer3sv = zeros(R, n_exo * (n_past+1)^2, n_exo)
    kron_buffer4sv = zeros(R, n_exo^2 * (n_past+1), n_exo^2)

    shocks² = zero(R)
    logabsdets = zero(R)
    n_obs_total = 0

    for i in axes(data_in_deviations, 2)
        idx = obs_idx_per_t[i]
        m = length(idx)

        copyto!(state_vol, 1, st, 1, n_past); state_vol[end] = one(R)
        state¹⁻_vol = state_vol

        copyto!(shock_independent, view(data_in_deviations, :, i))
        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ℒ.kron!(kronstate_vol, state¹⁻_vol, state¹⁻_vol)
        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, kronstate_vol, -1/2, 1)
        ℒ.kron!(kronstate_vol³, state¹⁻_vol, kronstate_vol)
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, kronstate_vol³, -1/6, 1)

        ℒ.kron!(kron_buffer_state, J, state¹⁻_vol)
        copyto!(𝐒ⁱ_full, 𝐒¹ᵉ)
        ℒ.mul!(𝐒ⁱ_full, 𝐒²⁻ᵉ, kron_buffer_state, 1, 1)
        ℒ.kron!(kron_buffer3sv, kron_buffer_state, state¹⁻_vol)
        ℒ.mul!(𝐒ⁱ_full, 𝐒³⁻ᵉ², kron_buffer3sv, 1/2, 1)

        x_kron_II!(kron_buffer4sv, state¹⁻_vol)
        copyto!(𝐒ⁱ²ᵉ_full, 𝐒²ᵉ); ℒ.rdiv!(𝐒ⁱ²ᵉ_full, 2)
        ℒ.mul!(𝐒ⁱ²ᵉ_full, 𝐒³⁻ᵉ, kron_buffer4sv, 1/2, 1)

        if m == 0
            x = x_zero
            fill!(x, zero(R))
        else
            if m > n_exo
                if opts.verbose println("Inversion filter (3rd) failed at step $i: m=$m > n_exo=$n_exo") end
                return on_failure_loglikelihood
            end
            𝐒ⁱ_v    = 𝐒ⁱ_full[idx, :]
            𝐒ⁱ²ᵉ_v  = 𝐒ⁱ²ᵉ_full[idx, :]
            𝐒ⁱ³ᵉ_v  = 𝐒ⁱ³ᵉ[idx, :]
            si_v    = shock_independent[idx]
            fill!(init_guess, zero(R))
            x, matched = find_shocks(Val(filter_algorithm),
                                    init_guess, kron_buffer, kron_buffer², kron_buffer2,
                                    kron_buffer3, kron_buffer4, J,
                                    𝐒ⁱ_v, 𝐒ⁱ²ᵉ_v, 𝐒ⁱ³ᵉ_v, si_v)
            if !matched
                if opts.verbose println("Inversion filter (3rd) failed at step $i") end
                return on_failure_loglikelihood
            end
            if i > eff_presample_periods
                ℒ.kron!(kron_buffer2, J, x)
                ℒ.kron!(kron_buffer, x, x)
                ℒ.kron!(kron_buffer3, J, kron_buffer)
                jacc_v = view(jacc_v_buf, 1:m, :)
                copyto!(jacc_v, 𝐒ⁱ_v)
                ℒ.mul!(jacc_v, 𝐒ⁱ²ᵉ_v, kron_buffer2, 2, 1)
                ℒ.mul!(jacc_v, 𝐒ⁱ³ᵉ_v, kron_buffer3, 3, 1)
                ℒ.rmul!(jacc_v, -1)
                logabsdets += m == n_exo ? ℒ.logabsdet(jacc_v)[1] : ℒ.logabsdet(jacc_v * jacc_v')[1] / 2
                shocks² += sum(abs2, x)
                n_obs_total += m
                if !isfinite(logabsdets) || !isfinite(shocks²)
                    return on_failure_loglikelihood
                end
            end
        end

        copyto!(aug_state, 1, st, 1, n_past); aug_state[n_past+1] = one(R); copyto!(aug_state, n_past+2, x, 1, n_exo)
        ℒ.kron!(kronaug_state, aug_state, aug_state)
        ℒ.kron!(kron_kron_aug_state, kronaug_state, aug_state)
        ℒ.mul!(st, 𝐒⁻¹, aug_state)
        ℒ.mul!(st, 𝐒⁻², kronaug_state, 1/2, 1)
        ℒ.mul!(st, 𝐒⁻³, kron_kron_aug_state, 1/6, 1)
    end

    return -(logabsdets + shocks² + n_obs_total * log(2 * 3.141592653589793)) / 2
end

end # @stable
