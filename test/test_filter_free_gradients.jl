using Test
using MacroModelling
using Random
using AxisKeys
import LinearAlgebra as ℒ
import ChainRulesCore as CRC
import ForwardDiff
import Zygote
import FiniteDifferences

# -----------------------------------------------------------------------------
# Filter-free log-likelihood gradient cross-checks.
#
# Mirrors test_inversion_filter_gradients.jl: builds a closure that varies a
# subset of parameters (optionally together with the latent shock path and the
# measurement-error std), and checks that ForwardDiff and Zygote agree with
# FiniteDifferences across all five perturbation algorithms:
#
#   :first_order, :second_order, :pruned_second_order,
#   :third_order, :pruned_third_order
#
# Both scalar and per-observable (Vector{Real} of length n_obs) `me_std` are
# exercised. Each algorithm is tested twice:
#   1. Gradient wrt parameter subset only — fixed shocks / me_std.
#   2. Gradient wrt the joint (parameter subset, shocks, me_std) vector —
#      this is the actual quantity NUTS asks for inside the Turing model.
# -----------------------------------------------------------------------------

const RTOL_FF = 1e-5
const FDM_FF  = FiniteDifferences.central_fdm(5, 1)

function ss_perturbed_data_ff(model, observables; periods = 6, σ = 1e-4, seed = 42)
    SS     = get_steady_state(model)
    ss_obs = collect(SS(observables, :Steady_state))
    Random.seed!(seed)
    dat    = repeat(ss_obs, 1, periods) .+ σ .* randn(length(observables), periods)
    return KeyedArray(dat; Variables = observables, Time = 1:periods)
end

function boundary_missing_data_ff(data; n_leading = 2, n_trailing = 1)
    dat = Matrix{Union{Missing,Float64}}(collect(data))
    n_leading > 0 && (dat[:, 1:n_leading] .= missing)
    n_trailing > 0 && (dat[:, end - n_trailing + 1:end] .= missing)
    dat[min(2, size(dat, 1)), 4] = missing
    return KeyedArray(dat; Variables = collect(axiskeys(data, 1)), Time = axes(dat, 2))
end

trim_visible_shocks_ff(shocks, first_t, last_t, n_warm) = shocks[:, vcat(collect(1:n_warm), n_warm .+ collect(first_t:last_t))]
manual_centered_diff_ff(f, x; h = 1e-6) = (f(x + h) - f(x - h)) / (2h)

# Closure that varies a subset of parameters; shocks / me_std are captured.
function make_ff_param_closure(model, data, base_params, idx, shocks, me_std, algorithm;
                               initial_state = nothing, kwargs...)
    n   = length(base_params)
    pos = zeros(Int, n)
    @inbounds for (k, j) in enumerate(idx)
        pos[j] = k
    end
    return function(θ_subset)
        T    = eltype(θ_subset)
        full = map(j -> pos[j] == 0 ? T(base_params[j]) : θ_subset[pos[j]], 1:n)
        if isnothing(initial_state)
            return get_filter_free_loglikelihood(model, data, full, shocks, me_std;
                                                  algorithm = algorithm,
                                                  on_failure_loglikelihood = -Inf,
                                                  kwargs...)
        end
        return get_filter_free_loglikelihood(model, data, full, shocks, me_std, initial_state;
                                              algorithm = algorithm,
                                              on_failure_loglikelihood = -Inf,
                                              kwargs...)
    end
end

# Closure that varies (parameter subset, shocks_vec, me_std) jointly. The
# layout of z is: [θ_subset; vec(shocks); me_std_part], where me_std_part is
# either a length-1 view (scalar) or length-n_obs view (vector).
function make_ff_joint_closure(model, data, base_params, idx,
                                nExo, nT, nObs, algorithm; vec_me::Bool,
                                initial_state = nothing, kwargs...)
    n   = length(base_params)
    pos = zeros(Int, n)
    @inbounds for (k, j) in enumerate(idx)
        pos[j] = k
    end
    nP  = length(idx)
    nSh = nExo * nT
    nMe = vec_me ? nObs : 1
    return (
        n_inputs = nP + nSh + nMe,
        f = function(z)
            T       = eltype(z)
            θ       = z[1:nP]
            sh_vec  = z[nP+1:nP+nSh]
            me_part = z[nP+nSh+1:nP+nSh+nMe]
            full    = map(j -> pos[j] == 0 ? T(base_params[j]) : θ[pos[j]], 1:n)
            shocks  = reshape(sh_vec, nExo, nT)
            me_std  = vec_me ? me_part : me_part[1]
            if isnothing(initial_state)
                return get_filter_free_loglikelihood(model, data, full, shocks, me_std;
                                                      algorithm = algorithm,
                                                      on_failure_loglikelihood = -Inf,
                                                      kwargs...)
            end
            return get_filter_free_loglikelihood(model, data, full, shocks, me_std, initial_state;
                                                  algorithm = algorithm,
                                                  on_failure_loglikelihood = -Inf,
                                                  kwargs...)
        end,
    )
end

function compare_ff_gradients(label, f, x0; rtol = RTOL_FF)
    @testset "$label" begin
        llh_val = f(x0)
        @test isfinite(llh_val)
        if !isfinite(llh_val)
            @info "Skipping $label: forward loglik not finite at base point"
            return
        end

        fd_grad = first(FiniteDifferences.grad(FDM_FF, f, x0))
        if !all(isfinite, fd_grad)
            @info "Skipping $label: FD reference contains NaN/Inf"
            return
        end

        @testset "ForwardDiff vs FiniteDifferences" begin
            fdiff_grad, fdiff_err = nothing, nothing
            try
                fdiff_grad = ForwardDiff.gradient(f, x0)
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

        @testset "Zygote vs FiniteDifferences" begin
            zg_grad, zg_err = nothing, nothing
            try
                zg_grad, = Zygote.gradient(f, x0)
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

# =============================================================================
# Gali (2015) Chapter 3 nonlinear NK — all 5 algorithms
# =============================================================================
include("../models/Gali_2015_chapter_3_nonlinear.jl")
const GALI_FF = Gali_2015_chapter_3_nonlinear

const GALI_FF_OBS  = [:log_y, :log_W_real]
const GALI_FF_SUBSET = [:σ, :φ, :ϕᵖⁱ, :ρ_a, :ρ_z, :std_a, :std_z]

function gali_ff_subset_indices()
    pnames = GALI_FF.constants.post_complete_parameters.parameters
    return [findfirst(==(p), pnames) for p in GALI_FF_SUBSET]
end

@testset "filter-free log-likelihood gradient cross-checks (Gali_2015)" begin
    base_params = copy(GALI_FF.parameter_values)
    p_subset    = gali_ff_subset_indices()
    nObs        = length(GALI_FF_OBS)
    nExo        = length(get_shocks(GALI_FF))
    nT          = 6
    data        = ss_perturbed_data_ff(GALI_FF, GALI_FF_OBS; periods = nT, σ = 1e-4, seed = 91)

    Random.seed!(101)
    shocks      = 1e-3 .* randn(nExo, nT)
    me_scalar   = 0.05
    me_vector   = collect(range(0.03, 0.07, length = nObs))

    algorithms = [:first_order, :pruned_second_order, :second_order,
                  :pruned_third_order, :third_order]

    for algo in algorithms
        # (a) parameter-subset gradient, scalar me_std
        f1 = make_ff_param_closure(GALI_FF, data, base_params, p_subset,
                                    shocks, me_scalar, algo)
        compare_ff_gradients("Gali :$algo  (param subset, scalar me_std)",
                              f1, base_params[p_subset])

        # (b) parameter-subset gradient, vector me_std
        f2 = make_ff_param_closure(GALI_FF, data, base_params, p_subset,
                                    shocks, me_vector, algo)
        compare_ff_gradients("Gali :$algo  (param subset, vector me_std)",
                              f2, base_params[p_subset])

        # (c) joint (params, shocks, me_std) gradient — scalar me_std
        joint_s = make_ff_joint_closure(GALI_FF, data, base_params, p_subset,
                                         nExo, nT, nObs, algo; vec_me = false)
        z0_s    = vcat(base_params[p_subset], vec(shocks), [me_scalar])
        compare_ff_gradients("Gali :$algo  (joint params+shocks+me, scalar me_std)",
                              joint_s.f, z0_s)

        # (d) joint gradient — vector me_std
        joint_v = make_ff_joint_closure(GALI_FF, data, base_params, p_subset,
                                         nExo, nT, nObs, algo; vec_me = true)
        z0_v    = vcat(base_params[p_subset], vec(shocks), me_vector)
        compare_ff_gradients("Gali :$algo  (joint params+shocks+me, vector me_std)",
                              joint_v.f, z0_v)
    end

    @testset "Gali filter-free warmup gradients" begin
        warmup_iterations = 3
        n_warm = max(warmup_iterations - 1, 0)
        data_warm = ss_perturbed_data_ff(GALI_FF, GALI_FF_OBS; periods = nT, σ = 1e-4, seed = 92)

        Random.seed!(102)
        shocks_warm = 1e-3 .* randn(nExo, nT + n_warm)

        for algo in algorithms
            joint_warm = make_ff_joint_closure(GALI_FF, data_warm, base_params, p_subset,
                                               nExo, nT + n_warm, nObs, algo;
                                               vec_me = false,
                                               warmup_iterations = warmup_iterations)
            z0_warm = vcat(base_params[p_subset], vec(shocks_warm), [me_scalar])
            compare_ff_gradients("Gali :$algo  (joint params+shocks+me, scalar me_std, warmup_iterations=3)",
                                  joint_warm.f, z0_warm)
        end
    end

    @testset "Gali filter-free boundary trimming" begin
        nT_boundary = 8
        data_boundary = boundary_missing_data_ff(ss_perturbed_data_ff(GALI_FF, GALI_FF_OBS; periods = nT_boundary, σ = 1e-4, seed = 93))
        data_trimmed = data_boundary[:, 3:7]

        Random.seed!(103)
        shocks_full = 1e-3 .* randn(nExo, nT_boundary)
        shocks_trimmed = trim_visible_shocks_ff(shocks_full, 3, 7, 0)

        me_matrix = [0.025 + 0.01 * (i - 1) + 0.0025 * (t - 1) for i in 1:nObs, t in 1:nT_boundary]
        me_trimmed = me_matrix[:, 3:7]
        me_scalar_boundary = 0.05

        for algo in algorithms
            ll_full = get_filter_free_loglikelihood(GALI_FF,
                                                    data_boundary,
                                                    base_params,
                                                    shocks_full,
                                                    me_matrix;
                                                    algorithm = algo,
                                                    on_failure_loglikelihood = -Inf)
            ll_trimmed = get_filter_free_loglikelihood(GALI_FF,
                                                       data_trimmed,
                                                       base_params,
                                                       shocks_trimmed,
                                                       me_trimmed;
                                                       algorithm = algo,
                                                       on_failure_loglikelihood = -Inf)

            @test isfinite(ll_full)
            @test isapprox(ll_full, ll_trimmed; rtol = 1e-12, atol = 1e-12)

            ll_rrule, pb = CRC.rrule(get_filter_free_loglikelihood,
                                     GALI_FF,
                                     data_boundary,
                                     base_params,
                                     shocks_full,
                                     me_scalar_boundary;
                                     algorithm = algo,
                                     on_failure_loglikelihood = -Inf)
            @test isapprox(ll_rrule,
                           get_filter_free_loglikelihood(GALI_FF,
                                                         data_boundary,
                                                         base_params,
                                                         shocks_full,
                                                         me_scalar_boundary;
                                                         algorithm = algo,
                                                         on_failure_loglikelihood = -Inf);
                           rtol = 1e-12,
                           atol = 1e-12)

            _, _, _, _, dshocks, dme = pb(1.0)

            dropped_shock_fd = manual_centered_diff_ff(x -> begin
                shocks_local = copy(shocks_full)
                shocks_local[1, 1] = x
                get_filter_free_loglikelihood(GALI_FF,
                                              data_boundary,
                                              base_params,
                                              shocks_local,
                                              me_scalar_boundary;
                                              algorithm = algo,
                                              on_failure_loglikelihood = -Inf)
            end, shocks_full[1, 1])

            kept_shock_fd = manual_centered_diff_ff(x -> begin
                shocks_local = copy(shocks_full)
                shocks_local[1, 3] = x
                get_filter_free_loglikelihood(GALI_FF,
                                              data_boundary,
                                              base_params,
                                              shocks_local,
                                              me_scalar_boundary;
                                              algorithm = algo,
                                              on_failure_loglikelihood = -Inf)
            end, shocks_full[1, 3])

            me_fd = manual_centered_diff_ff(x -> get_filter_free_loglikelihood(GALI_FF,
                                                                                data_boundary,
                                                                                base_params,
                                                                                shocks_full,
                                                                                x;
                                                                                algorithm = algo,
                                                                                on_failure_loglikelihood = -Inf),
                                            me_scalar_boundary)

            @test isapprox(dshocks[1, 1], dropped_shock_fd; atol = 1e-10, rtol = 1e-8)
            @test isapprox(dshocks[1, 1], 0.0; atol = 1e-10, rtol = 0.0)
            @test isapprox(dshocks[1, 3], kept_shock_fd; atol = 1e-8, rtol = RTOL_FF)
            @test isapprox(dme, me_fd; atol = 1e-8, rtol = RTOL_FF)
        end
    end
end

# =============================================================================
# Caldara et al. (2012) — filter-free rrule consistency check
#
# Verifies that for the higher-order Caldara model the analytical rrule from
# `src/rrules.jl` agrees with Mooncake reverse-mode AD (through the same
# rrule) on (parameters, latent shocks, me_std).  This previously lived in
# `test_3rd_order_estimation.jl` and `test_pruned_3rd_order_estimation.jl`;
# it is consolidated here so the estimation scripts only contain end-to-end
# sampler runs.
# =============================================================================
import DifferentiationInterface
import ADTypes: AutoMooncake
import Mooncake
import DelimitedFiles

dat, header = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "usmodel.csv"), ',', header = true)
dat = Float64.(dat)
names = vec(Symbol.(strip.(header)))
full_data = KeyedArray(dat', Variable = names, Time = axes(dat, 1))
data_caldara = full_data([:dy], 75:230)

include(joinpath(@__DIR__, "models", "Caldara_et_al_2012_estim.jl"))

const CALDARA_FF = Caldara_et_al_2012_estim
const CALDARA_FF_T = 10
const CALDARA_FF_DATA = data_caldara[:, 1:CALDARA_FF_T]
const CALDARA_FF_NEXO = length(get_shocks(CALDARA_FF))

@testset "Filter-free rrule consistency (Caldara_et_al_2012, $algo, $melabel)" for
        (algo, melabel, me) in [
            (:third_order,         "scalar me_std", 0.05),
            (:third_order,         "vector me_std", [0.05]),
            (:pruned_third_order,  "scalar me_std", 0.05),
            (:pruned_third_order,  "vector me_std", [0.05]),
        ]
    Random.seed!(42)
    pars = copy(CALDARA_FF.parameter_values)
    shocks = 0.01 .* randn(CALDARA_FF_NEXO, CALDARA_FF_T)

    llh_fwd = get_filter_free_loglikelihood(CALDARA_FF, CALDARA_FF_DATA,
                                            pars, shocks, me;
                                            algorithm = algo)
    @test isfinite(llh_fwd)

    llh_r, pb = CRC.rrule(get_filter_free_loglikelihood, CALDARA_FF,
                          CALDARA_FF_DATA, pars, shocks, me; algorithm = algo)
    @test isapprox(llh_r, llh_fwd; rtol = 1e-12)
    _, _, _, dpars_a, dshk_a, dme_a = pb(1.0)

    nP  = length(pars)
    nSh = CALDARA_FF_NEXO * CALDARA_FF_T
    nMe = me isa AbstractVector ? length(me) : 1
    z0  = vcat(pars, vec(shocks), me isa AbstractVector ? me : [me])
    obj = function(z)
        p     = z[1:nP]
        s     = reshape(z[nP+1:nP+nSh], CALDARA_FF_NEXO, CALDARA_FF_T)
        mpart = z[nP+nSh+1:nP+nSh+nMe]
        mloc  = me isa AbstractVector ? mpart : mpart[1]
        get_filter_free_loglikelihood(CALDARA_FF, CALDARA_FF_DATA, p, s, mloc;
                                      algorithm = algo)
    end
    g_mc       = DifferentiationInterface.gradient(obj, AutoMooncake(config = nothing), z0)
    dpars_mc   = g_mc[1:nP]
    dshk_mc    = reshape(g_mc[nP+1:nP+nSh], CALDARA_FF_NEXO, CALDARA_FF_T)
    dme_mc_raw = g_mc[nP+nSh+1:nP+nSh+nMe]
    dme_mc     = me isa AbstractVector ? dme_mc_raw : dme_mc_raw[1]

    @test isapprox(dpars_mc, dpars_a; rtol = 1e-8)
    @test isapprox(dshk_mc,  dshk_a;  rtol = 1e-8)
    @test isapprox(dme_mc,   dme_a;   rtol = 1e-8)
end

@testset "Filter-free initial_state AD gradients (Gali_2015)" begin
    MacroModelling.solve!(GALI_FF, silent = true)
    ss_vec_gali = copy(GALI_FF.caches.non_stochastic_steady_state)
    state_idx_gali = GALI_FF.constants.post_model_macro.past_not_future_and_mixed_idx
    nVars_gali = GALI_FF.constants.post_model_macro.nVars

    base_params_gali = copy(GALI_FF.parameter_values)
    p_subset_gali    = gali_ff_subset_indices()
    nObs_gali        = length(GALI_FF_OBS)
    nExo_gali        = length(get_shocks(GALI_FF))
    nT_gali          = 6
    data_gali        = ss_perturbed_data_ff(GALI_FF, GALI_FF_OBS; periods = nT_gali, σ = 1e-4, seed = 91)

    Random.seed!(101)
    shocks_gali = 1e-3 .* randn(nExo_gali, nT_gali)
    me_scalar_gali = 0.05

    # Perturbed initial state (levels)
    init_state_levels = copy(ss_vec_gali)
    init_state_levels[state_idx_gali[1]] += 0.1

    # Vector-of-vectors initial state (deviations) for pruned variants
    init_state_vv = [zeros(nVars_gali), zeros(nVars_gali)]
    init_state_vv[1][state_idx_gali[1]] = 0.1

    init_state_vv3 = [zeros(nVars_gali), zeros(nVars_gali), zeros(nVars_gali)]
    init_state_vv3[1][state_idx_gali[1]] = 0.1

    for algo in [:first_order, :pruned_second_order, :second_order,
                 :pruned_third_order, :third_order]
        f_lvl = make_ff_param_closure(GALI_FF, data_gali, base_params_gali, p_subset_gali,
                                      shocks_gali, me_scalar_gali, algo;
                                      initial_state = init_state_levels)
        compare_ff_gradients("Gali :$algo  (initial_state levels)",
                              f_lvl, base_params_gali[p_subset_gali])
    end

    # Vector{Vector{Float64}} (deviations) input for pruned variants
    for (algo, init_vv) in [(:pruned_second_order, init_state_vv),
                             (:pruned_third_order, init_state_vv3)]
        f_vv = make_ff_param_closure(GALI_FF, data_gali, base_params_gali, p_subset_gali,
                                     shocks_gali, me_scalar_gali, algo;
                                     initial_state = init_vv)
        compare_ff_gradients("Gali :$algo  (initial_state Vector{Vector}, deviations)",
                              f_vv, base_params_gali[p_subset_gali])
    end
end
