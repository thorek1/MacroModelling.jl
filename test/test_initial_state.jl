using Test
using MacroModelling
using Random
using AxisKeys
using DelimitedFiles
import LinearAlgebra as ℒ
import ADTypes
import DifferentiationInterface
import FiniteDifferences
import Mooncake
import ForwardDiff

include("test_helpers.jl")

# ── Model setup ──────────────────────────────────────────────────────────────
#
# This file contains ONLY tests that involve derivatives (Mooncake reverse
# mode, ForwardDiff, and FiniteDifferences cross-checks). Non-derivative
# value-equivalence checks are integrated into `functionality_test` in
# `functionality_tests.jl` so they run for every (model, algorithm) pair
# exercised by the full test suite.

include("../models/FS2000.jl")

dat, header = readdlm("data/FS2000_data.csv", ',', header = true)
dat = Float64.(dat)
names = vec(header)
data = KeyedArray(dat', Variable = Symbol.("log_".*names), Time = axes(dat, 1))
data = log.(data)
observables = sort(Symbol.("log_".*names))
data = data(observables, :)

params = FS2000.parameter_values
T = FS2000.constants.post_model_macro
state_idx = T.past_not_future_and_mixed_idx

MacroModelling.solve!(FS2000, silent = true)
ss_vec = copy(FS2000.caches.non_stochastic_steady_state)

perturbed = copy(ss_vec)
perturbed[state_idx[1]] += 0.5

# ── 1. AD gradients flow through initial_state (∂/∂params) ───────────────────

@testset "AD gradients w.r.t. params with custom initial_state" begin
    for (filt, algo) in [(:kalman, :first_order), (:inversion, :first_order)]
        @testset "$filt / $algo" begin
            f = x -> get_loglikelihood(FS2000, data, x, perturbed; filter = filt, algorithm = algo)

            back_grad = DifferentiationInterface.gradient(
                f, ADTypes.AutoMooncake(config = nothing), params)
            @test all(isfinite, back_grad)

            fin_grad = FiniteDifferences.grad(
                FiniteDifferences.central_fdm(4, 1), f, params)[1]
            @test isapprox(back_grad, fin_grad, rtol = 1e-4)
        end
    end
end

# ── 2. Derivatives w.r.t. initial_state — FD vs ForwardDiff vs Mooncake ──────
#
# The positional `initial_state` method is differentiable through the rrule
# (Mooncake) and through type-generic dispatch (ForwardDiff Duals). Compare
# both against central finite differences.

@testset "Derivatives w.r.t. initial_state — FD vs ForwardDiff vs Mooncake" begin
    nVars = T.nVars
    Random.seed!(42)
    init = ss_vec .+ 1e-3 .* randn(nVars)

    fdm = FiniteDifferences.central_fdm(4, 1)
    backend_mc = ADTypes.AutoMooncake(config = nothing)

    flatten_nested_initial_state(vv) = reduce(vcat, vv)
    function rebuild_nested_initial_state(v, template)
        rebuilt = Vector{Vector{eltype(v)}}(undef, length(template))
        offset = 0
        for i in eachindex(template)
            n = length(template[i])
            rebuilt[i] = collect(@view v[offset + 1:offset + n])
            offset += n
        end
        return rebuilt
    end

    @testset "kalman / first_order — d/dinit" begin
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :kalman, algorithm = :first_order)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        @test isapprox(g_fd, g_fin, rtol = 1e-5)
        @test isapprox(g_mc, g_fin, rtol = 1e-5)
    end

    @testset "kalman / first_order — d/dparams (custom init)" begin
        f = x -> get_loglikelihood(FS2000, data, x, init; filter = :kalman, algorithm = :first_order)
        g_fin = FiniteDifferences.grad(fdm, f, params)[1]
        g_fd  = ForwardDiff.gradient(f, params)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, params)
        @test isapprox(g_fd, g_fin, rtol = 1e-4)
        @test isapprox(g_mc, g_fin, rtol = 1e-4)
    end

    @testset "inversion / first_order — d/dinit" begin
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :first_order)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        @test isapprox(g_fd, g_fin, rtol = 1e-5)
        @test isapprox(g_mc, g_fin, rtol = 1e-5)
    end

    @testset "inversion / first_order — d/dnested init" begin
        nested_init = [init .- ss_vec]
        flat_init = flatten_nested_initial_state(nested_init)
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :first_order)
        f_flat = y -> f(rebuild_nested_initial_state(y, nested_init))
        g_fin = FiniteDifferences.grad(fdm, f_flat, flat_init)[1]
        g_mc = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        @test length(g_mc) == length(nested_init)
        @test isapprox(flatten_nested_initial_state(g_mc), g_fin, rtol = 1e-5)
    end

    @testset "inversion / pruned_second_order — d/dnested init" begin
        nested_init = [init .- ss_vec, zeros(nVars)]
        flat_init = flatten_nested_initial_state(nested_init)
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :pruned_second_order)
        f_flat = y -> f(rebuild_nested_initial_state(y, nested_init))
        h = 1e-5
        check_idxs = (1, nVars + 1, 2nVars)
        g_dir_fin = map(check_idxs) do idx
            direction = zeros(length(flat_init))
            direction[idx] = 1
            (f_flat(flat_init .+ h .* direction) - f_flat(flat_init .- h .* direction)) / (2h)
        end
        g_mc = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        @test length(g_mc) == length(nested_init)
        g_mc_flat = flatten_nested_initial_state(g_mc)
        for (idx, g_fin) in zip(check_idxs, g_dir_fin)
            @test isapprox(g_mc_flat[idx], g_fin, rtol = 1e-4, atol = 1e-5)
        end
    end

    nExo_fw = length(get_shocks(FS2000))
    nT_fw   = size(data, 2)
    shks_fw = 1e-3 .* randn(nExo_fw, nT_fw)
    me_fw   = 0.05
    @testset "filter-free / first_order — d/dinit" begin
        f = y -> get_filter_free_loglikelihood(FS2000, data, params, shks_fw, me_fw, y; algorithm = :first_order)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        @test isapprox(g_fd, g_fin, rtol = 1e-5)
        @test isapprox(g_mc, g_fin, rtol = 1e-5)
    end

    @testset "filter-free / first_order — d/dnested init" begin
        nested_init = [init .- ss_vec]
        flat_init = flatten_nested_initial_state(nested_init)
        f = y -> get_filter_free_loglikelihood(FS2000, data, params, shks_fw, me_fw, y; algorithm = :first_order)
        f_flat = y -> f(rebuild_nested_initial_state(y, nested_init))
        g_fin = FiniteDifferences.grad(fdm, f_flat, flat_init)[1]
        g_mc = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        @test length(g_mc) == length(nested_init)
        @test isapprox(flatten_nested_initial_state(g_mc), g_fin, rtol = 1e-5)
    end
end

# ── 3. Default equivalence: gradient w.r.t. params is unchanged when the user
#       supplies an initial_state value-equivalent to the internal default ────
#
# The override `state[1] = user - SS_and_pars[1:nVars]` introduces no new
# tangent path for `Vec{Vec}` zeros (zeros are AD constants → `d_state[1]` is
# the same as in the no-override branch). For non-pruned higher-order algos
# the equivalent input is the SSS levels themselves, recomputed inside the
# closure so the closure's gradient threads through `initial_state` (only
# possible because `initial_state` is positional).

@testset "Gradient equivalence — value-equivalent initial_state override" begin
    nVars = T.nVars
    zero_vv = [zeros(nVars)]
    backend = ADTypes.AutoMooncake(config = nothing)

    for (filt, algo) in [(:kalman, :first_order), (:inversion, :first_order),
                          (:inversion, :pruned_second_order), (:inversion, :pruned_third_order)]
        @testset "$filt / $algo" begin
            f_base = x -> get_loglikelihood(FS2000, data, x; filter = filt, algorithm = algo)
            f_ovr  = x -> get_loglikelihood(FS2000, data, x, zero_vv; filter = filt, algorithm = algo)
            g_base = DifferentiationInterface.gradient(f_base, backend, params)
            g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
            @test isapprox(g_base, g_ovr, rtol = 1e-10, atol = 1e-12)
        end
    end

    @testset "filter-free / first_order" begin
        Random.seed!(123)
        nExo = length(get_shocks(FS2000))
        nT   = size(data, 2)
        shks = 1e-3 .* randn(nExo, nT)
        me   = 0.05

        f_base = x -> get_filter_free_loglikelihood(FS2000, data, x, shks, me; algorithm = :first_order)
        f_ovr  = x -> get_filter_free_loglikelihood(FS2000, data, x, shks, me, zero_vv; algorithm = :first_order)
        g_base = DifferentiationInterface.gradient(f_base, backend, params)
        g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
        @test isapprox(g_base, g_ovr, rtol = 1e-10, atol = 1e-12)
    end

    # Non-pruned higher-order: recompute SSS levels inside the closure so
    # AD threads the tangent back through `sss_levels_fn(x, algo)`.
    opts_cached = MacroModelling.merge_calculation_options()
    sss_levels_fn = (x, algo) -> begin
        _, sap, _, sd, _ = MacroModelling.get_relevant_steady_state_and_state_update(
            Val(algo), x, FS2000; opts = opts_cached, estimation = true)
        sd .+ sap[1:nVars]
    end

    for algo in (:second_order, :third_order)
        @testset "inversion / $algo" begin
            f_base = x -> get_loglikelihood(FS2000, data, x; filter = :inversion, algorithm = algo)
            f_ovr  = x -> get_loglikelihood(FS2000, data, x, sss_levels_fn(x, algo);
                                             filter = :inversion, algorithm = algo)
            g_base = DifferentiationInterface.gradient(f_base, backend, params)
            g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
            @test isapprox(g_base, g_ovr, rtol = 1e-6)
        end
    end

    Random.seed!(123)
    nExo_hf = length(get_shocks(FS2000))
    nT_hf   = size(data, 2)
    shks_hf = 1e-3 .* randn(nExo_hf, nT_hf)
    me_hf   = 0.05
    for algo in (:second_order, :third_order)
        @testset "filter-free / $algo" begin
            f_base = x -> get_filter_free_loglikelihood(FS2000, data, x, shks_hf, me_hf; algorithm = algo)
            f_ovr  = x -> get_filter_free_loglikelihood(FS2000, data, x, shks_hf, me_hf, sss_levels_fn(x, algo);
                                                        algorithm = algo)
            g_base = DifferentiationInterface.gradient(f_base, backend, params)
            g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
            @test isapprox(g_base, g_ovr, rtol = 1e-6)
        end
    end
end
