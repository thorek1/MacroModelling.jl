@stable default_mode = "disable" begin

# Historical shock decomposition, shared by the inversion and the particle
# filters. Both produce a shock path and then attribute the observed trajectory
# to the individual shocks, and that attribution belongs to neither filter, so it
# lives here: `inversion.jl` calls in from its `calculate_*` routines,
# `particle.jl` from `run_particle_estimates`.
#
# Not to be confused with `src/aumann_shapley.jl`, which applies the same
# cooperative-game idea to the *variance* decomposition of the moments.
#
# ---------------------------------------------------------------------
# Aumann–Shapley shock decomposition (marginal-contribution driver)
# ---------------------------------------------------------------------
#
# Computes per-period Shapley shares for the inversion- and particle-filter shock
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
    compressed_kron²_power!(kk, aug1)
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
    compressed_kron²_power!(k11, aug1)
    compressed_kron²!(k12̂, aug1̂, aug2)
    compressed_kron³_power!(k111, aug1)
    ℒ.mul!(new_s1, 𝐒[1], aug1)
    ℒ.mul!(new_s2, 𝐒[1], aug2)
    ℒ.mul!(new_s2, 𝐒[2], k11, 0.5, 1.0)
    ℒ.mul!(new_s3, 𝐒[1], aug3)
    ℒ.mul!(new_s3, 𝐒[2], k12̂, 1.0, 1.0)
    ℒ.mul!(new_s3, 𝐒[3], k111, 1/6, 1.0)
    return nothing
end

function advance_aumann_shapley_pruned_2nd_warmup!(
        s₁, s₂, s₁⁺, s₂⁺,
        ds₁ᵢ, ds₂ᵢ, ds₁ᵢ⁺, ds₂ᵢ⁺,
        warmup_shocks::AbstractMatrix,
        sₖ,
        iₚ,
        a₁, a₂, da₁, da₂,
        k₁₁, dk₁₁,
        ε̄ₜ, εᵢₜ, ε₀,
        𝐒)
    nₚ = length(iₚ)

    for w in axes(warmup_shocks, 2)
        εₜ = @view warmup_shocks[:, w]
        ε̄ₜ .= sₖ .* εₜ
        pruned_state_update_2nd_order!(s₁⁺, s₂⁺, s₁, s₂, iₚ, ε̄ₜ, ε₀, a₁, a₂, k₁₁, 𝐒)

        for i in eachindex(ds₁ᵢ)
            fill!(εᵢₜ, 0.0)
            εᵢₜ[i] = εₜ[i]

            @views copyto!(da₁[1:nₚ], ds₁ᵢ[i][iₚ])
            da₁[nₚ + 1] = 0.0
            copyto!(da₁, nₚ + 2, εᵢₜ, 1, size(warmup_shocks, 1))

            @views copyto!(da₂[1:nₚ], ds₂ᵢ[i][iₚ])
            da₂[nₚ + 1] = 0.0
            copyto!(da₂, nₚ + 2, ε₀, 1, size(warmup_shocks, 1))

            # C₂ is symmetric, so d C₂(a₁, a₁) = 2 C₂(da₁, a₁).
            compressed_kron²!(dk₁₁, da₁, a₁)

            ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
            ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
            # The outer 1/2 cancels the derivative's factor of 2.
            ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 1.0, 1.0)

            copyto!(ds₁ᵢ[i], ds₁ᵢ⁺[i])
            copyto!(ds₂ᵢ[i], ds₂ᵢ⁺[i])
        end

        copyto!(s₁, s₁⁺)
        copyto!(s₂, s₂⁺)
    end

    return nothing
end

function advance_aumann_shapley_pruned_3rd_warmup!(
        s₁, s₂, s₃, s₁⁺, s₂⁺, s₃⁺,
        ds₁ᵢ, ds₂ᵢ, ds₃ᵢ, ds₁ᵢ⁺, ds₂ᵢ⁺, ds₃ᵢ⁺,
        warmup_shocks::AbstractMatrix,
        sₖ,
        iₚ,
        a₁, a₁⁰, a₂, a₃, da₁, da₂, da₃,
        k₁₁, k₁₂⁰, k₁₁₁, dk₁₁, dk₁₂⁰, dk₁₁₁,
        k₂tmp,
        ε̄ₜ, εᵢₜ, ε₀,
        𝐒)
    nₚ = length(iₚ)

    for w in axes(warmup_shocks, 2)
        εₜ = @view warmup_shocks[:, w]
        ε̄ₜ .= sₖ .* εₜ
        pruned_state_update_3rd_order!(s₁⁺, s₂⁺, s₃⁺, s₁, s₂, s₃, iₚ, ε̄ₜ, ε₀,
                                       a₁, a₁⁰, a₂, a₃, k₁₁, k₁₂⁰, k₁₁₁, 𝐒)

        for i in eachindex(ds₁ᵢ)
            fill!(εᵢₜ, 0.0)
            εᵢₜ[i] = εₜ[i]

            @views copyto!(da₁[1:nₚ], ds₁ᵢ[i][iₚ])
            da₁[nₚ + 1] = 0.0
            copyto!(da₁, nₚ + 2, εᵢₜ, 1, size(warmup_shocks, 1))

            @views copyto!(da₂[1:nₚ], ds₂ᵢ[i][iₚ])
            da₂[nₚ + 1] = 0.0
            copyto!(da₂, nₚ + 2, ε₀, 1, size(warmup_shocks, 1))

            @views copyto!(da₃[1:nₚ], ds₃ᵢ[i][iₚ])
            da₃[nₚ + 1] = 0.0
            copyto!(da₃, nₚ + 2, ε₀, 1, size(warmup_shocks, 1))

            # C₂ is symmetric, so d C₂(a₁, a₁) = 2 C₂(da₁, a₁).
            compressed_kron²!(dk₁₁, da₁, a₁)

            compressed_kron²!(dk₁₂⁰, da₁, a₂)
            compressed_kron²!(k₂tmp, a₁⁰, da₂)
            dk₁₂⁰ .+= k₂tmp

            # C₃ is symmetric, so d C₃(a₁, a₁, a₁) = 3 C₃(da₁, a₁, a₁).
            compressed_kron³!(dk₁₁₁, da₁, a₁, a₁)

            ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
            ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
            # The outer 1/2 cancels the pair derivative's factor of 2.
            ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 1.0, 1.0)
            ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[1], da₃)
            ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[2], dk₁₂⁰, 1.0, 1.0)
            # The outer 1/6 times the cubic derivative's factor of 3 is 1/2.
            ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[3], dk₁₁₁, 1/2, 1.0)

            copyto!(ds₁ᵢ[i], ds₁ᵢ⁺[i])
            copyto!(ds₂ᵢ[i], ds₂ᵢ⁺[i])
            copyto!(ds₃ᵢ[i], ds₃ᵢ⁺[i])
        end

        copyto!(s₁, s₁⁺)
        copyto!(s₂, s₂⁺)
        copyto!(s₃, s₃⁺)
    end

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
        verbose::Bool = false,
        warmup_shocks::Union{Nothing,AbstractMatrix} = nothing) where R <: Real
    n_nodes = 2
    max_error = aumann_shapley_shock_decomposition_pruned_2nd_order!(decomposition,
                                                                      variables,
                                                                      shocks,
                                                                      initial_state,
                                                                      𝐒,
                                                                      T,
                                                                      nE,
                                                                      n_nodes;
                                                                      warmup_shocks = warmup_shocks)
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
                                                                          n_nodes;
                                                                          warmup_shocks = warmup_shocks)
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
    n_nodes::Int;
    warmup_shocks::Union{Nothing,AbstractMatrix} = nothing) where R <: Real
    nᵥ = T.nVars
    iₚ = T.past_not_future_and_mixed_idx
    nₚ = length(iₚ)
    n_aug = nₚ + 1 + nE
    n_kron = n_aug * (n_aug + 1) ÷ 2
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

    # Shock-direction vectors: scaled node shocks, basis shock i, and zero shocks.
    ε̄ₜ = zeros(R, nE)
    εᵢₜ = zeros(R, nE)
    ε₀ = zeros(R, nE)

    # --- Pass 1: V(∅) trajectory (zero shocks) → store in decomposition[:, nE+1, :]. ---
    s₁ .= initial_state[1]
    s₂ .= initial_state[2]
    if !isnothing(warmup_shocks)
        for _ in axes(warmup_shocks, 2)
            pruned_state_update_2nd_order!(s₁⁺, s₂⁺, s₁, s₂, iₚ, ε₀, ε₀, a₁, a₂, k₁₁, 𝐒)
            s₁, s₁⁺ = s₁⁺, s₁
            s₂, s₂⁺ = s₂⁺, s₂
        end
    end
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
        if !isnothing(warmup_shocks)
            advance_aumann_shapley_pruned_2nd_warmup!(s₁, s₂, s₁⁺, s₂⁺,
                                                      ds₁ᵢ, ds₂ᵢ, ds₁ᵢ⁺, ds₂ᵢ⁺,
                                                      warmup_shocks,
                                                      sₖ,
                                                      iₚ,
                                                      a₁, a₂, da₁, da₂,
                                                      k₁₁, dk₁₁,
                                                      ε̄ₜ, εᵢₜ, ε₀,
                                                      𝐒)
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

                # C₂ is symmetric, so d C₂(a₁, a₁) = 2 C₂(da₁, a₁).
                compressed_kron²!(dk₁₁, da₁, a₁)

                # Plain form: ds₁ᵢ⁺ = S1 * da₁
                ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
                # The outer 1/2 cancels the derivative's factor of 2.
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 1.0, 1.0)

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
    verbose::Bool = false,
    warmup_shocks::Union{Nothing,AbstractMatrix} = nothing) where R <: Real
    n_nodes = 3
    max_error = aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition,
                                                                      variables,
                                                                      shocks,
                                                                      initial_state,
                                                                      𝐒,
                                                                      T,
                                                                      nE,
                                      n_nodes;
                                      warmup_shocks = warmup_shocks)
    if verbose
        println("Aumann-Shapley third-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
    end
    while max_error > AUMANN_SHAPLEY_REFINEMENT_RTOL && n_nodes < AUMANN_SHAPLEY_REFINEMENT_MAX_NODES
        next_nodes = min(n_nodes + 1, AUMANN_SHAPLEY_REFINEMENT_MAX_NODES)
        if verbose
            println("Aumann-Shapley third-order shock decomposition rerunning with ", next_nodes, " nodes after closure error ", max_error, " at ", n_nodes, " nodes")
        end
        n_nodes = next_nodes
        max_error = aumann_shapley_shock_decomposition_pruned_3rd_order!(decomposition,
                                                                          variables,
                                                                          shocks,
                                                                          initial_state,
                                                                          𝐒,
                                                                          T,
                                                                          nE,
                                                                          n_nodes;
                                                                          warmup_shocks = warmup_shocks)
        if verbose
            println("Aumann-Shapley third-order shock decomposition closure error with ", n_nodes, " nodes: ", max_error)
        end
    end
    return decomposition
end

function aumann_shapley_shock_decomposition_pruned_3rd_order!(
        decomposition::AbstractArray{R},
        variables::AbstractMatrix,
        shocks::AbstractMatrix,
        initial_state,
        𝐒,
        T,
        nE::Int,
    n_nodes::Int;
    warmup_shocks::Union{Nothing,AbstractMatrix} = nothing) where R <: Real
    nᵥ = T.nVars
    iₚ = T.past_not_future_and_mixed_idx
    nₚ = length(iₚ)
    n_aug = nₚ + 1 + nE
    n_kron2 = n_aug * (n_aug + 1) ÷ 2
    n_kron3 = n_aug * (n_aug + 1) * (n_aug + 2) ÷ 6
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

    # Shock-direction vectors: scaled node shocks, basis shock i, and zero shocks.
    ε̄ₜ = zeros(R, nE)
    εᵢₜ = zeros(R, nE)
    ε₀ = zeros(R, nE)

    # --- Pass 1: V(∅) trajectory (zero shocks) → store in decomposition[:, nE+1, :]. ---
    s₁ .= initial_state[1]
    s₂ .= initial_state[2]
    s₃ .= initial_state[3]
    if !isnothing(warmup_shocks)
        for _ in axes(warmup_shocks, 2)
            pruned_state_update_3rd_order!(s₁⁺, s₂⁺, s₃⁺, s₁, s₂, s₃, iₚ, ε₀, ε₀,
                                           a₁, a₁⁰, a₂, a₃, k₁₁, k₁₂⁰, k₁₁₁, 𝐒)
            s₁, s₁⁺ = s₁⁺, s₁
            s₂, s₂⁺ = s₂⁺, s₂
            s₃, s₃⁺ = s₃⁺, s₃
        end
    end
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
        if !isnothing(warmup_shocks)
            advance_aumann_shapley_pruned_3rd_warmup!(s₁, s₂, s₃, s₁⁺, s₂⁺, s₃⁺,
                                                      ds₁ᵢ, ds₂ᵢ, ds₃ᵢ, ds₁ᵢ⁺, ds₂ᵢ⁺, ds₃ᵢ⁺,
                                                      warmup_shocks,
                                                      sₖ,
                                                      iₚ,
                                                      a₁, a₁⁰, a₂, a₃, da₁, da₂, da₃,
                                                      k₁₁, k₁₂⁰, k₁₁₁, dk₁₁, dk₁₂⁰, dk₁₁₁,
                                                      k₂tmp,
                                                      ε̄ₜ, εᵢₜ, ε₀,
                                                      𝐒)
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

                # C₂ is symmetric, so d C₂(a₁, a₁) = 2 C₂(da₁, a₁).
                compressed_kron²!(dk₁₁, da₁, a₁)

                # d(aug1_no_const ⊗ aug2) = (d aug1 ⊗ aug2) + (aug1_no_const ⊗ d aug2)
                compressed_kron²!(dk₁₂⁰, da₁, a₂)
                compressed_kron²!(k₂tmp, a₁⁰, da₂)
                dk₁₂⁰ .+= k₂tmp

                # C₃ is symmetric, so d C₃(a₁, a₁, a₁) = 3 C₃(da₁, a₁, a₁).
                compressed_kron³!(dk₁₁₁, da₁, a₁, a₁)

                # Plain form: ds₁ᵢ⁺ = S1 * da₁
                ℒ.mul!(ds₁ᵢ⁺[i], 𝐒[1], da₁)
                # Plain form: ds₂ᵢ⁺ = S1 * da₂ + 0.5 * S2 * d(a₁⊗a₁)
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[1], da₂)
                # The outer 1/2 cancels the pair derivative's factor of 2.
                ℒ.mul!(ds₂ᵢ⁺[i], 𝐒[2], dk₁₁, 1.0, 1.0)
                # Plain form: ds₃ᵢ⁺ = S1 * da₃ + S2 * d(a₁⁰⊗a₂) + (1/6) * S3 * d(a₁⊗a₁⊗a₁)
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[1], da₃)
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[2], dk₁₂⁰, 1.0, 1.0)
                # The outer 1/6 times the cubic derivative's factor of 3 is 1/2.
                ℒ.mul!(ds₃ᵢ⁺[i], 𝐒[3], dk₁₁₁, 1/2, 1.0)

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

end  # @stable
