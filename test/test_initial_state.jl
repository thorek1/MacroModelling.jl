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
    backend_mc = ADTypes.AutoMooncake(config = nothing)
    fdm        = FiniteDifferences.central_fdm(4, 1)

    # First-order: cross-check Mooncake + ForwardDiff against finite differences.
    for (filt, algo) in [(:kalman, :first_order), (:inversion, :first_order)]
        @testset "$filt / $algo" begin
            f = x -> get_loglikelihood(FS2000, data, x, perturbed; filter = filt, algorithm = algo)

            back_grad = DifferentiationInterface.gradient(f, backend_mc, params)
            fwd_grad  = ForwardDiff.gradient(f, params)
            @test all(isfinite, back_grad)
            @test all(isfinite, fwd_grad)

            fin_grad = FiniteDifferences.grad(fdm, f, params)[1]
            @test isapprox(back_grad, fin_grad, rtol = 1e-4)
            @test isapprox(fwd_grad,  fin_grad, rtol = 1e-4)
            @test isapprox(back_grad, fwd_grad, rtol = 1e-6)
        end
    end

    # Higher-order inversion: cross-check Mooncake + ForwardDiff against
    # FiniteDifferences for both the levels (Vector{Float64}) and the
    # pruned-state (Vector{Vector{Float64}}) forms of `initial_state`. FD
    # is more expensive at higher order but `params` is only 9-D for FS2000
    # so the total cost stays tolerable.
    nVars_pp        = T.nVars
    perturbed_dev   = perturbed .- ss_vec
    nested_init_2   = [perturbed_dev, zeros(nVars_pp)]
    nested_init_3   = [perturbed_dev, zeros(nVars_pp), zeros(nVars_pp)]

    higher_specs = [
        (:second_order,         perturbed),
        (:third_order,          perturbed),
        (:pruned_second_order,  perturbed),
        (:pruned_third_order,   perturbed),
        (:pruned_second_order,  nested_init_2),
        (:pruned_third_order,   nested_init_3),
    ]

    for (algo, init_val) in higher_specs
        init_label = init_val isa AbstractVector{<:AbstractVector} ? "nested" : "levels"
        @testset "inversion / $algo ($init_label)" begin
            f = x -> get_loglikelihood(FS2000, data, x, init_val;
                                       filter = :inversion, algorithm = algo)

            back_grad = DifferentiationInterface.gradient(f, backend_mc, params)
            fwd_grad  = ForwardDiff.gradient(f, params)
            fin_grad  = FiniteDifferences.grad(fdm, f, params)[1]
            @test all(isfinite, back_grad)
            @test all(isfinite, fwd_grad)
            @test isapprox(back_grad, fwd_grad, rtol = 1e-6)
            @test isapprox(back_grad, fin_grad, rtol = 1e-4)
            @test isapprox(fwd_grad,  fin_grad, rtol = 1e-4)
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
        g_fd  = ForwardDiff.gradient(f_flat, flat_init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        @test length(g_mc) == length(nested_init)
        @test isapprox(flatten_nested_initial_state(g_mc), g_fin, rtol = 1e-5)
        @test isapprox(g_fd, g_fin, rtol = 1e-5)
    end

    @testset "inversion / pruned_second_order — d/dnested init" begin
        nested_init = [init .- ss_vec, zeros(nVars)]
        flat_init = flatten_nested_initial_state(nested_init)
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :pruned_second_order)
        f_flat = y -> f(rebuild_nested_initial_state(y, nested_init))
        g_fin     = FiniteDifferences.grad(fdm, f_flat, flat_init)[1]
        g_mc      = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        g_fd_flat = ForwardDiff.gradient(f_flat, flat_init)
        @test length(g_mc) == length(nested_init)
        g_mc_flat = flatten_nested_initial_state(g_mc)
        @test isapprox(g_mc_flat, g_fin,     rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_fd_flat, g_fin,     rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_mc_flat, g_fd_flat, rtol = 1e-6)
    end

    # Higher-order inversion: d/dinit cross-checks for the remaining algos.
    # We cover both the levels form (Vector{Float64}) and the pruned nested
    # form (Vector{Vector{Float64}}). For nested forms we flatten/rebuild so
    # ForwardDiff and FiniteDifferences (which both consume flat
    # Vector{<:Real} inputs) can still be exercised.
    @testset "inversion / second_order — d/dinit (levels)" begin
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :second_order)
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        @test all(isfinite, g_fd)
        @test all(isfinite, g_mc)
        @test isapprox(g_mc, g_fd,  rtol = 1e-6)
        @test isapprox(g_mc, g_fin, rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_fd, g_fin, rtol = 1e-4, atol = 1e-6)
    end

    @testset "inversion / third_order — d/dinit (levels)" begin
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :third_order)
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        @test all(isfinite, g_fd)
        @test all(isfinite, g_mc)
        @test isapprox(g_mc, g_fd,  rtol = 1e-6)
        @test isapprox(g_mc, g_fin, rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_fd, g_fin, rtol = 1e-4, atol = 1e-6)
    end

    @testset "inversion / pruned_third_order — d/dnested init" begin
        nested_init = [init .- ss_vec, zeros(nVars), zeros(nVars)]
        flat_init   = flatten_nested_initial_state(nested_init)
        f = y -> get_loglikelihood(FS2000, data, params, y; filter = :inversion, algorithm = :pruned_third_order)
        f_flat = y -> f(rebuild_nested_initial_state(y, nested_init))
        g_fd_flat = ForwardDiff.gradient(f_flat, flat_init)
        g_mc      = DifferentiationInterface.gradient(f, backend_mc, nested_init)
        g_fin     = FiniteDifferences.grad(fdm, f_flat, flat_init)[1]
        @test length(g_mc) == length(nested_init)
        g_mc_flat = flatten_nested_initial_state(g_mc)
        @test isapprox(g_mc_flat, g_fd_flat, rtol = 1e-6)
        @test isapprox(g_mc_flat, g_fin,     rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_fd_flat, g_fin,     rtol = 1e-4, atol = 1e-6)
    end

    nExo_fw = length(get_shocks(FS2000))
    nT_fw   = size(data, 2)
    shks_fw = 1e-3 .* randn(nExo_fw, nT_fw)
    me_fw   = 0.05
    @testset "filter-free / first_order — d/dinit" begin
        f = y -> get_loglikelihood(FS2000, data, params, shks_fw, me_fw, y; algorithm = :first_order)
        g_fin = FiniteDifferences.grad(fdm, f, init)[1]
        g_fd  = ForwardDiff.gradient(f, init)
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, init)
        @test isapprox(g_fd, g_fin, rtol = 1e-5)
        @test isapprox(g_mc, g_fin, rtol = 1e-5)
    end

    @testset "filter-free / first_order — d/dnested init" begin
        nested_init = [init .- ss_vec]
        flat_init = flatten_nested_initial_state(nested_init)
        f = y -> get_loglikelihood(FS2000, data, params, shks_fw, me_fw, y; algorithm = :first_order)
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
    fdm     = FiniteDifferences.central_fdm(4, 1)

    for (filt, algo) in [(:kalman, :first_order), (:inversion, :first_order),
                          (:inversion, :pruned_second_order), (:inversion, :pruned_third_order)]
        @testset "$filt / $algo" begin
            f_base = x -> get_loglikelihood(FS2000, data, x; filter = filt, algorithm = algo)
            f_ovr  = x -> get_loglikelihood(FS2000, data, x, zero_vv; filter = filt, algorithm = algo)
            g_base = DifferentiationInterface.gradient(f_base, backend, params)
            g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
            @test isapprox(g_base, g_ovr, rtol = 1e-10, atol = 1e-12)

            g_base_fd = ForwardDiff.gradient(f_base, params)
            g_ovr_fd  = ForwardDiff.gradient(f_ovr,  params)
            @test isapprox(g_base_fd, g_ovr_fd, rtol = 1e-10, atol = 1e-12)
            @test isapprox(g_base_fd, g_base,   rtol = 1e-6)

            g_base_fin = FiniteDifferences.grad(fdm, f_base, params)[1]
            g_ovr_fin  = FiniteDifferences.grad(fdm, f_ovr,  params)[1]
            @test isapprox(g_base_fin, g_ovr_fin, rtol = 1e-6)
            @test isapprox(g_base,     g_base_fin, rtol = 1e-4)
            @test isapprox(g_base_fd,  g_base_fin, rtol = 1e-4)
        end
    end

    @testset "filter-free / first_order" begin
        Random.seed!(123)
        nExo = length(get_shocks(FS2000))
        nT   = size(data, 2)
        shks = 1e-3 .* randn(nExo, nT)
        me   = 0.05

        f_base = x -> get_loglikelihood(FS2000, data, x, shks, me; algorithm = :first_order)
        f_ovr  = x -> get_loglikelihood(FS2000, data, x, shks, me, zero_vv; algorithm = :first_order)
        g_base = DifferentiationInterface.gradient(f_base, backend, params)
        g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
        @test isapprox(g_base, g_ovr, rtol = 1e-10, atol = 1e-12)

        g_base_fd = ForwardDiff.gradient(f_base, params)
        g_ovr_fd  = ForwardDiff.gradient(f_ovr,  params)
        @test isapprox(g_base_fd, g_ovr_fd, rtol = 1e-10, atol = 1e-12)
        @test isapprox(g_base_fd, g_base,   rtol = 1e-6)

        g_base_fin = FiniteDifferences.grad(fdm, f_base, params)[1]
        g_ovr_fin  = FiniteDifferences.grad(fdm, f_ovr,  params)[1]
        @test isapprox(g_base_fin, g_ovr_fin, rtol = 1e-6)
        @test isapprox(g_base,     g_base_fin, rtol = 1e-4)
        @test isapprox(g_base_fd,  g_base_fin, rtol = 1e-4)
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

            g_base_fd = ForwardDiff.gradient(f_base, params)
            g_ovr_fd  = ForwardDiff.gradient(f_ovr,  params)
            @test isapprox(g_base_fd, g_ovr_fd, rtol = 1e-6)
            @test isapprox(g_base_fd, g_base,   rtol = 1e-6)

            g_base_fin = FiniteDifferences.grad(fdm, f_base, params)[1]
            g_ovr_fin  = FiniteDifferences.grad(fdm, f_ovr,  params)[1]
            @test isapprox(g_base_fin, g_ovr_fin, rtol = 1e-4)
            @test isapprox(g_base,     g_base_fin, rtol = 1e-4)
            @test isapprox(g_base_fd,  g_base_fin, rtol = 1e-4)
        end
    end

    Random.seed!(123)
    nExo_hf = length(get_shocks(FS2000))
    nT_hf   = size(data, 2)
    shks_hf = 1e-3 .* randn(nExo_hf, nT_hf)
    me_hf   = 0.05
    for algo in (:second_order, :third_order)
        @testset "filter-free / $algo" begin
            f_base = x -> get_loglikelihood(FS2000, data, x, shks_hf, me_hf; algorithm = algo)
            f_ovr  = x -> get_loglikelihood(FS2000, data, x, shks_hf, me_hf, sss_levels_fn(x, algo);
                                                        algorithm = algo)
            g_base = DifferentiationInterface.gradient(f_base, backend, params)
            g_ovr  = DifferentiationInterface.gradient(f_ovr,  backend, params)
            @test isapprox(g_base, g_ovr, rtol = 1e-6)

            g_base_fd = ForwardDiff.gradient(f_base, params)
            g_ovr_fd  = ForwardDiff.gradient(f_ovr,  params)
            @test isapprox(g_base_fd, g_ovr_fd, rtol = 1e-6)
            @test isapprox(g_base_fd, g_base,   rtol = 1e-6)

            g_base_fin = FiniteDifferences.grad(fdm, f_base, params)[1]
            g_ovr_fin  = FiniteDifferences.grad(fdm, f_ovr,  params)[1]
            @test isapprox(g_base_fin, g_ovr_fin, rtol = 1e-4)
            @test isapprox(g_base,     g_base_fin, rtol = 1e-4)
            @test isapprox(g_base_fd,  g_base_fin, rtol = 1e-4)
        end
    end
end

# ── 4. Cross-AD coverage — filter-free `get_loglikelihood` signature ─────────
#
# §1 and §2 cover the kalman/inversion-filter signature across algorithms and
# AD modes (ForwardDiff + Mooncake, with FD as reference at :first_order and
# AD-vs-AD comparison at higher orders). This section extends the same
# coverage to the filter-free signature (`get_loglikelihood(model, data, p,
# shocks, me, init; algorithm)`) — both ∂/∂params with a custom init and
# ∂/∂init — and supplements the higher-order inversion testsets in §1 with
# cheap directional FD spot-checks on the params gradient (params are only
# 9-D, so 3 directions per algo is tolerable).

@testset "Cross-AD coverage — filter-free signature & FD reference" begin
    nVars = T.nVars
    Random.seed!(7)
    init_lvl = ss_vec .+ 5e-4 .* randn(nVars)
    init_dev = init_lvl .- ss_vec

    backend_mc = ADTypes.AutoMooncake(config = nothing)
    fdm        = FiniteDifferences.central_fdm(4, 1)

    init_for(algo) = algo === :pruned_second_order ? [copy(init_dev), zeros(nVars)] :
                     algo === :pruned_third_order  ? [copy(init_dev), zeros(nVars), zeros(nVars)] :
                                                      copy(init_lvl)

    flatten_nested(vv) = reduce(vcat, vv)
    function rebuild_nested(v, template)
        rebuilt = Vector{Vector{eltype(v)}}(undef, length(template))
        offset = 0
        for i in eachindex(template)
            n = length(template[i])
            rebuilt[i] = collect(@view v[offset + 1:offset + n])
            offset += n
        end
        return rebuilt
    end

    # ── 4a. FD reference for §1's inversion higher-order params gradient ────
    @testset "inversion / $algo — d/dparams FD reference" for algo in
            (:pruned_second_order, :second_order, :pruned_third_order, :third_order)
        init = init_for(algo)
        f = x -> get_loglikelihood(FS2000, data, x, init;
                                   filter = :inversion, algorithm = algo)
        @test isfinite(f(params))
        g_mc  = DifferentiationInterface.gradient(f, backend_mc, params)
        g_fd  = ForwardDiff.gradient(f, params)
        g_fin = FiniteDifferences.grad(fdm, f, params)[1]
        @test isapprox(g_mc, g_fin, rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_fd, g_fin, rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_mc, g_fd,  rtol = 1e-6)
    end

    # ── 4b. Filter-free signature — d/dparams across all 5 algorithms ───────
    Random.seed!(11)
    nExo = length(get_shocks(FS2000))
    nT   = size(data, 2)
    shks = 1e-3 .* randn(nExo, nT)
    me   = 0.05

    for algo in (:first_order, :pruned_second_order, :second_order,
                  :pruned_third_order, :third_order)
        @testset "filter-free / $algo — d/dparams (custom init)" begin
            init = init_for(algo)
            f = x -> get_loglikelihood(FS2000, data, x, shks, me, init;
                                       algorithm = algo)
            @test isfinite(f(params))

            g_mc  = DifferentiationInterface.gradient(f, backend_mc, params)
            g_fd  = ForwardDiff.gradient(f, params)
            g_fin = FiniteDifferences.grad(fdm, f, params)[1]
            @test all(isfinite, g_mc)
            @test all(isfinite, g_fd)
            @test isapprox(g_mc, g_fd,  rtol = 1e-5, atol = 1e-8)
            @test isapprox(g_mc, g_fin, rtol = 1e-4, atol = 1e-6)
            @test isapprox(g_fd, g_fin, rtol = 1e-4, atol = 1e-6)
        end
    end

    # ── 4c. Filter-free signature — d/dinit (flat levels) for non-pruned ────
    for algo in (:second_order, :third_order)
        @testset "filter-free / $algo — d/dinit (flat)" begin
            f = y -> get_loglikelihood(FS2000, data, params, shks, me, y;
                                       algorithm = algo)
            @test isfinite(f(init_lvl))
            g_fd  = ForwardDiff.gradient(f, init_lvl)
            g_mc  = DifferentiationInterface.gradient(f, backend_mc, init_lvl)
            g_fin = FiniteDifferences.grad(fdm, f, init_lvl)[1]
            @test isapprox(g_fd, g_mc,  rtol = 1e-5, atol = 1e-8)
            @test isapprox(g_mc, g_fin, rtol = 1e-4, atol = 1e-6)
            @test isapprox(g_fd, g_fin, rtol = 1e-4, atol = 1e-6)
        end
    end

    # ── 4d. Filter-free signature — d/dinit (nested deviations) for pruned ──
    for (algo, k) in ((:pruned_second_order, 2), (:pruned_third_order, 3))
        @testset "filter-free / $algo — d/dnested init" begin
            nested = Vector{Vector{Float64}}(undef, k)
            nested[1] = copy(init_dev)
            for i in 2:k; nested[i] = zeros(nVars); end
            flat = flatten_nested(nested)

            f = y -> get_loglikelihood(FS2000, data, params, shks, me, y;
                                       algorithm = algo)
            f_flat = v -> f(rebuild_nested(v, nested))
            @test isfinite(f(nested))

            g_fd      = ForwardDiff.gradient(f_flat, flat)
            g_mc      = DifferentiationInterface.gradient(f, backend_mc, nested)
            g_mc_flat = flatten_nested(g_mc)
            g_fin     = FiniteDifferences.grad(fdm, f_flat, flat)[1]
            @test isapprox(g_fd, g_mc_flat, rtol = 1e-5, atol = 1e-8)
            @test isapprox(g_mc_flat, g_fin, rtol = 1e-4, atol = 1e-6)
            @test isapprox(g_fd,      g_fin, rtol = 1e-4, atol = 1e-6)
        end
    end
end
