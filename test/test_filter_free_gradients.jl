using Test
using MacroModelling
using Random
using AxisKeys
import LinearAlgebra as ℒ
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

# Closure that varies a subset of parameters; shocks / me_std are captured.
function make_ff_param_closure(model, data, base_params, idx, shocks, me_std, algorithm)
    n   = length(base_params)
    pos = zeros(Int, n)
    @inbounds for (k, j) in enumerate(idx)
        pos[j] = k
    end
    return function(θ_subset)
        T    = eltype(θ_subset)
        full = map(j -> pos[j] == 0 ? T(base_params[j]) : θ_subset[pos[j]], 1:n)
        return get_filter_free_loglikelihood(model, data, full, shocks, me_std;
                                              algorithm = algorithm,
                                              on_failure_loglikelihood = -Inf)
    end
end

# Closure that varies (parameter subset, shocks_vec, me_std) jointly. The
# layout of z is: [θ_subset; vec(shocks); me_std_part], where me_std_part is
# either a length-1 view (scalar) or length-n_obs view (vector).
function make_ff_joint_closure(model, data, base_params, idx,
                                nExo, nT, nObs, algorithm; vec_me::Bool)
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
            return get_filter_free_loglikelihood(model, data, full, shocks, me_std;
                                                  algorithm = algorithm,
                                                  on_failure_loglikelihood = -Inf)
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
end
