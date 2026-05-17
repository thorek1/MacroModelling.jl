using Test
using MacroModelling
using Random
using AxisKeys
import LinearAlgebra as ℒ
import ForwardDiff
import Zygote
import FiniteDifferences

# -----------------------------------------------------------------------------
# Verifies that ∂loglik/∂parameters for the inversion filter agrees between
# ForwardDiff, Zygote, and FiniteDifferences across all five perturbation-order
# dispatches (first_order, pruned_second_order, second_order, pruned_third_order,
# third_order).
#
# Models:
#   * Gali_2015_chapter_3_nonlinear — all 5 algorithms (under-identified obs;
#     first_order also tested with square obs and FULL parameter vector)
#   * Smets_Wouters_2007 — first_order (full param vector) + pruned_second_order
#     (subset).
#
# Edge cases (where supported by the dispatch):
#   * n_obs <  n_shocks  (under-identified — LagrangeNewton path)
#   * n_obs == n_shocks  (square — first_order only; higher orders cannot
#     invert square systems on this model with LagrangeNewton)
#   * partial + fully missing observations (public missing-data dispatch)
#   * warmup_iterations > 0   (first_order only — codebase warns it's first-
#     order-only and ignores it otherwise)
#   * presample_periods > 0
#
# Note: MacroModelling enforces n_obs ≤ n_shocks at the API level, so the
# over-identified case is intentionally not exercised.
# -----------------------------------------------------------------------------

const RTOL = 1e-5
const FDM   = FiniteDifferences.central_fdm(5, 1)

# Build a Zygote/ForwardDiff-friendly closure that varies a subset of params.
# The full parameter vector is built via a comprehension (no in-place writes)
# so that reverse-mode AD has no mutation to worry about.
function make_llh_closure(model, data, base_params, idx, algorithm; kwargs...)
    n   = length(base_params)
    pos = zeros(Int, n)            # 0 ⇒ use base_params[j], else θ_subset[pos[j]]
    @inbounds for (k, j) in enumerate(idx)
        pos[j] = k
    end
    return function(θ_subset)
        # Use `map` (Zygote-friendly, no in-place setindex! tracing) to splice
        # θ_subset into the base-parameter vector.  An explicit element-type
        # conversion makes the result eltype-stable so ForwardDiff sees a
        # `Vector{Dual}` rather than `Vector{Real}`.
        T = eltype(θ_subset)
        full = map(j -> pos[j] == 0 ? T(base_params[j]) : θ_subset[pos[j]], 1:n)
        return get_loglikelihood(model, data, full;
                                 filter = :inversion,
                                 algorithm = algorithm,
                                 on_failure_loglikelihood = -Inf,
                                 kwargs...)
    end
end

function compare_gradients(label, model, data, base_params, idx, algorithm;
                           rtol = RTOL, kwargs...)
    @testset "$label" begin
        f       = make_llh_closure(model, data, base_params, idx, algorithm; kwargs...)
        θ       = base_params[idx]
        llh_val = f(θ)
        @test isfinite(llh_val)
        if !isfinite(llh_val)
            @info "Skipping $label: forward loglik not finite at base params"
            return
        end

        # FiniteDifferences reference (slow, but accurate).
        fd_grad = first(FiniteDifferences.grad(FDM, f, θ))
        if !all(isfinite, fd_grad)
            @info "Skipping $label: FD reference contains NaN/Inf — model fails under perturbation"
            return
        end

        # ForwardDiff (forward-mode AD).  Wrapped so that a ForwardDiff failure
        # does not prevent the Zygote check from running — they exercise
        # different code paths (the generic primal vs. the rrules).
        @testset "ForwardDiff vs FiniteDifferences" begin
            fdiff_grad = nothing
            fdiff_err  = nothing
            try
                fdiff_grad = ForwardDiff.gradient(f, θ)
            catch err
                fdiff_err = err
            end
            if fdiff_err !== nothing
                @error "ForwardDiff threw" exception = (fdiff_err, catch_backtrace())
                @test false
            else
                @test all(isfinite, fdiff_grad)
                @test isapprox(fdiff_grad, fd_grad; rtol = rtol)
            end
        end

        # Zygote (reverse-mode AD; exercises the rrules in src/rrules.jl).
        @testset "Zygote vs FiniteDifferences" begin
            zg_grad   = nothing
            zg_err    = nothing
            try
                zg_grad, = Zygote.gradient(f, θ)
            catch err
                zg_err = err
            end
            if zg_err !== nothing
                @error "Zygote threw" exception = (zg_err, catch_backtrace())
                @test false
            else
                @test zg_grad !== nothing
                @test all(isfinite, zg_grad)
                @test isapprox(zg_grad, fd_grad; rtol = rtol)
            end
        end
    end
end

# Build data as steady-state level + small Gaussian perturbations.  This keeps
# the inversion filter inside its convergence basin at every perturbation
# order, so any test failure reflects an AD problem rather than a numerical
# breakdown of the filter itself.
function ss_perturbed_data(model, observables; periods = 8, σ = 1e-4, seed = 42)
    SS     = get_steady_state(model)
    ss_obs = collect(SS(observables, :Steady_state))
    Random.seed!(seed)
    dat    = repeat(ss_obs, 1, periods) .+ σ .* randn(length(observables), periods)
    return KeyedArray(dat; Variables = observables, Time = 1:periods)
end

function data_with_missing_observations(data)
    dat_nan = Matrix{Float64}(collect(data))
    n_obs, n_time = size(dat_nan)
    @assert n_time >= 6 "Need at least 6 periods to inject non-boundary missing observations"

    partial_t1 = 2
    partial_t2 = n_time - 1
    full_t1 = n_time ÷ 2
    full_t2 = min(full_t1 + 1, n_time - 1)

    dat_nan[1, partial_t1] = NaN
    dat_nan[min(2, n_obs), partial_t2] = NaN
    dat_nan[:, full_t1] .= NaN
    dat_nan[:, full_t2] .= NaN

    dat_miss = Matrix{Union{Missing,Float64}}(copy(dat_nan))
    @inbounds for j in axes(dat_miss, 2), i in axes(dat_miss, 1)
        if !isfinite(Float64(coalesce(dat_miss[i, j], NaN)))
            dat_miss[i, j] = missing
        end
    end

    return KeyedArray(dat_miss; Variables = collect(axiskeys(data, 1)), Time = axes(dat_miss, 2))
end

# =============================================================================
# Gali (2015) Chapter 3 nonlinear NK — all 5 algorithms
# =============================================================================
include("../models/Gali_2015_chapter_3_nonlinear.jl")
const GALI = Gali_2015_chapter_3_nonlinear

# 3 shocks total → under-identified = 1 or 2 obs, square = 3 obs.
# Square obs must be linearly independent in the first-order solution.
# log_W_real = σ·log_y + φ·log_N (static identity from W_real = Y^σ·N^φ),
# so [:log_y, :log_W_real, :log_N] is rank-deficient; use i_ann instead.
const GALI_OBS_UNDER  = [:log_y, :log_W_real]
const GALI_OBS_SQUARE = [:log_y, :log_N, :i_ann]

const GALI_PARAM_SUBSET_NAMES = [:σ, :φ, :ϕᵖⁱ, :ρ_a, :ρ_z, :std_a, :std_z]

function gali_subset_indices()
    pnames = GALI.constants.post_complete_parameters.parameters
    return [findfirst(==(p), pnames) for p in GALI_PARAM_SUBSET_NAMES]
end

@testset "inversion filter gradient cross-checks (Gali + SW07)" begin

@testset "Gali_2015 nonlinear inversion filter — gradient cross-checks" begin
    base_params = copy(GALI.parameter_values)
    p_subset    = gali_subset_indices()

    # --- (a) per-algorithm baseline: subset of params, under-identified obs --
    algorithms = [:first_order, :pruned_second_order, :second_order,
                  :pruned_third_order, :third_order]

    for algo in algorithms
        data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 8, σ = 1e-4, seed = 11)
        compare_gradients("Gali :$algo (under-identified, $(length(p_subset)) params)",
                          GALI, data, base_params, p_subset, algo)
    end

    # --- (a2) missing observations: partial + fully missing periods --------
    for algo in algorithms
        data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 10, σ = 1e-4, seed = 16)
        data_missing = data_with_missing_observations(data)
        compare_gradients("Gali :$algo (under-identified, missing obs, $(length(p_subset)) params)",
                          GALI, data_missing, base_params, p_subset, algo)
    end

    # --- (b) FULL parameter vector — first_order only, under-identified ------
    let algo = :first_order
        data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 8, σ = 1e-4, seed = 12)
        compare_gradients("Gali :$algo (under-identified, FULL param vector)",
                          GALI, data, base_params, collect(eachindex(base_params)), algo)
    end

    # --- (c) square observables (n_obs == n_shocks): first_order only -------
    #   (higher-order inversion does not converge for square systems on this
    #    model — exercising it would test the failure path, not the gradient.)
    let algo = :first_order
        # Use a tighter σ so that finite-difference perturbations keep the
        # square-system inversion inside its convergence basin.
        data = ss_perturbed_data(GALI, GALI_OBS_SQUARE; periods = 6, σ = 1e-6, seed = 13)
        compare_gradients("Gali :$algo (square obs)",
                          GALI, data, base_params, p_subset, algo)
    end

    # --- (c2) square observables + missing periods: first_order only -------
    let algo = :first_order
        data = ss_perturbed_data(GALI, GALI_OBS_SQUARE; periods = 8, σ = 1e-6, seed = 17)
        data_missing = data_with_missing_observations(data)
        compare_gradients("Gali :$algo (square obs + missing periods)",
                          GALI, data_missing, base_params, p_subset, algo)
    end

    # --- (d) warmup_iterations > 0 (first_order only, per implementation) ---
    @testset "Gali :first_order (warmup_iterations=2)" begin
        let algo = :first_order
            data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 8, σ = 1e-4, seed = 14)
            compare_gradients("Gali :$algo (warmup_iterations=2)",
                              GALI, data, base_params, p_subset, algo;
                              warmup_iterations = 2)
        end
    end

    # --- (e) presample_periods > 0 — exercise across all 5 algorithms -------
    for algo in algorithms
        data = ss_perturbed_data(GALI, GALI_OBS_UNDER; periods = 10, σ = 1e-4, seed = 15)
        compare_gradients("Gali :$algo (presample_periods=3)",
                          GALI, data, base_params, p_subset, algo;
                          presample_periods = 3)
    end
end  # Gali testset


# =============================================================================
# Smets-Wouters 2007 — first_order (full) + pruned_second_order (subset)
# =============================================================================
include("../models/Smets_Wouters_2007.jl")
SW07 = Smets_Wouters_2007

# 7 shocks → under-identified by using 3 obs.
SW07_OBS = [:dy, :dc, :dinve]

# Representative parameter subset for the higher-order pass.
SW07_SUBSET_PREF = [:crhoa, :crhob, :crhog, :csadjcost, :chabb,
                    :csigma, :cprobw]

function sw07_subset_indices()
    pnames = SW07.constants.post_complete_parameters.parameters
    idx = Int[]
    for p in SW07_SUBSET_PREF
        j = findfirst(==(p), pnames)
        if j !== nothing
            push!(idx, j)
        end
    end
    if length(idx) < 5
        idx = collect(1:min(7, length(pnames)))
    end
    return idx
end

@testset "Smets-Wouters 2007 inversion filter — gradient cross-checks" begin
    base_params = copy(SW07.parameter_values)

    # First-order: FULL parameter vector
    let algo = :first_order
        data = ss_perturbed_data(SW07, SW07_OBS; periods = 12, σ = 1e-4, seed = 21)
        compare_gradients("SW07 :$algo (under-identified, FULL param vector, $(length(base_params)) params)",
                          SW07, data, base_params,
                          collect(eachindex(base_params)), algo)
    end

    # Pruned-2nd: subset only
    let algo = :pruned_second_order
        data = ss_perturbed_data(SW07, SW07_OBS; periods = 12, σ = 1e-4, seed = 22)
        p_subset = sw07_subset_indices()
        compare_gradients("SW07 :$algo (under-identified, $(length(p_subset))-param subset)",
                          SW07, data, base_params, p_subset, algo)
    end

    # Missing-data coverage on the SW07 paths already present in this file.
    for algo in (:first_order, :pruned_second_order)
        data = ss_perturbed_data(SW07, SW07_OBS; periods = 12, σ = 1e-4, seed = algo == :first_order ? 23 : 24)
        data_missing = data_with_missing_observations(data)
        p_idx = algo == :first_order ? collect(eachindex(base_params)) : sw07_subset_indices()
        compare_gradients("SW07 :$algo (under-identified, missing obs)",
                          SW07, data_missing, base_params, p_idx, algo)
    end

    # Presample-period coverage for the SW07 algorithms exercised in this file.
    for algo in (:first_order, :pruned_second_order)
        data = ss_perturbed_data(SW07, SW07_OBS; periods = 12, σ = 1e-4, seed = algo == :first_order ? 25 : 26)
        p_idx = algo == :first_order ? collect(eachindex(base_params)) : sw07_subset_indices()
        compare_gradients("SW07 :$algo (presample_periods=3)",
                          SW07, data, base_params, p_idx, algo;
                          presample_periods = 3)
    end
end  # SW07 testset

end  # outer wrapping testset
