using Test
using MacroModelling
import Zygote
import ForwardDiff
import FiniteDifferences
import LinearAlgebra as ℒ

using Random, AxisKeys

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Zygote-compatible full parameter vector builder
# ──────────────────────────────────────────────────────────────────────────────
function make_param_builder(true_params::Vector{Float64}, est_idx::Vector{Int})
    n = length(true_params)
    ep = zeros(Int, n)
    for (j, i) in enumerate(est_idx)
        ep[i] = j
    end
    fp = copy(true_params)
    return function(x)
        T = eltype(x)
        map(1:n) do i
            ep[i] > 0 ? x[ep[i]] : T(fp[i])
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────────────────────
include("../models/RBC_baseline.jl")
include("../models/FS2000.jl")
include("../models/Ireland_2004.jl")

# ══════════════════════════════════════════════════════════════════════════════
# Test 1: IRF rrule gradient across all 5 algorithms (RBC_baseline)
# ══════════════════════════════════════════════════════════════════════════════
@testset "IRF rrule gradient - all algorithms (RBC_baseline)" begin
    m = RBC_baseline
    # Parameters: σᶻ(1), σᵍ(2), σ(3), i_y(4), k_y(5), ρᶻ(6), ρᵍ(7), g_y(8), α(9)
    est_idx = [1, 2, 6, 7]  # σᶻ, σᵍ, ρᶻ, ρᵍ
    build_params = make_param_builder(m.parameter_values, est_idx)
    test_point = m.parameter_values[est_idx]

    for alg in [:first_order, :pruned_second_order, :pruned_third_order,
                :second_order, :third_order]
        @testset "$alg" begin
            # Invalidate caches to prevent Dual contamination between algorithms
            MacroModelling.invalidate_cache_validity!(m)

            f = x -> begin
                all_p = build_params(x)
                irf_v = get_irf(m, all_p, algorithm = alg, periods = 3)
                return sum(irf_v)
            end

            zy_grad = Zygote.gradient(f, test_point)[1]
            MacroModelling.invalidate_cache_validity!(m)
            fd_grad = ForwardDiff.gradient(f, test_point)
            MacroModelling.invalidate_cache_validity!(m)
            fi_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, test_point)[1]

            @test all(isfinite, zy_grad)
            @test all(isfinite, fd_grad)
            @test all(isfinite, fi_grad)

            zy_fd = maximum(abs.(zy_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
            zy_fi = maximum(abs.(zy_grad .- fi_grad) ./ max.(abs.(fi_grad), 1e-10))
            fd_fi = maximum(abs.(fd_grad .- fi_grad) ./ max.(abs.(fi_grad), 1e-10))
            println("  IRF $alg: Zy-FD=$zy_fd  Zy-FI=$zy_fi  FD-FI=$fd_fi")
            @test zy_fd < 1e-6
            @test zy_fi < 1e-4  # FiniteDiff has lower precision
            @test fd_fi < 1e-4
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Combined objective gradient - pruned_second_order on multiple models
# ══════════════════════════════════════════════════════════════════════════════
model_configs = [
    (
        name = "RBC_baseline",
        model = RBC_baseline,
        observables = [:y, :c],
        # σᶻ(1), σᵍ(2), ρᶻ(6), ρᵍ(7)
        est_idx = [1, 2, 6, 7],
        nsss_vars = [:y, :c],
        moment_vars = [:y, :c],
    ),
    (
        name = "FS2000",
        model = FS2000,
        observables = [:log_gy_obs, :log_gp_obs],
        # rho(5), z_e_a(8), z_e_m(9)
        est_idx = [5, 8, 9],
        nsss_vars = [:y, :c],
        moment_vars = [:y, :c],
    ),
    (
        name = "Ireland_2004",
        model = Ireland_2004,
        observables = [:ĝ, :ŷ, :π̂, :r̂],
        # ρᵃ(9), ρᵉ(10), σʳ(11), σᵃ(12), σᵉ(13), σᶻ(14)
        est_idx = [9, 10, 11, 12, 13, 14],
        nsss_vars = [:ŷ, :π̂],
        moment_vars = [:ŷ, :π̂],
    ),
]

@testset "Combined objective gradient - pruned_second_order" begin
    alg = :pruned_second_order

    for cfg in model_configs
        @testset "$(cfg.name)" begin
            m = cfg.model
            MacroModelling.invalidate_cache_validity!(m)
            build_params = make_param_builder(m.parameter_values, cfg.est_idx)
            test_point = m.parameter_values[cfg.est_idx]

            # Simulate data at true parameter values
            Random.seed!(42)
            sim = simulate(m, algorithm = alg)
            data = sim(cfg.observables, :, :simulate)

            # Compute targets at true parameters
            target_stats = get_statistics(m, m.parameter_values,
                non_stochastic_steady_state = cfg.nsss_vars,
                mean = cfg.moment_vars,
                standard_deviation = cfg.moment_vars,
                algorithm = alg)
            target_nsss = target_stats[:non_stochastic_steady_state]
            target_mean = target_stats[:mean]
            target_std  = target_stats[:standard_deviation]

            target_irf_full = get_irf(m, m.parameter_values, algorithm = alg, periods = 3)
            irf_var_idx = sort(MacroModelling.parse_variables_input_to_index(cfg.observables[1], m))
            target_irf = target_irf_full[irf_var_idx, 1, 1]

            # Combined objective exercising all 4 differentiable functions
            f = x -> begin
                all_p = build_params(x)

                llh = get_loglikelihood(m, data, all_p,
                    algorithm = alg, on_failure_loglikelihood = -Inf)

                stats_n = get_statistics(m, all_p,
                    non_stochastic_steady_state = cfg.nsss_vars, algorithm = alg)
                llh -= sum((stats_n[:non_stochastic_steady_state] .- target_nsss).^2)

                stats_m = get_statistics(m, all_p,
                    mean = cfg.moment_vars, standard_deviation = cfg.moment_vars,
                    algorithm = alg)
                llh -= sum((stats_m[:mean] .- target_mean).^2)
                llh -= sum((stats_m[:standard_deviation] .- target_std).^2)

                irf_v = get_irf(m, all_p, algorithm = alg, periods = 3)
                llh -= sum((irf_v[irf_var_idx, 1, 1] .- target_irf).^2)

                return llh
            end

            zy_grad = Zygote.gradient(f, test_point)[1]
            MacroModelling.invalidate_cache_validity!(m)
            fd_grad = ForwardDiff.gradient(f, test_point)
            MacroModelling.invalidate_cache_validity!(m)
            fi_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, test_point)[1]

            @test !isnothing(zy_grad)
            @test all(isfinite, zy_grad)
            @test all(isfinite, fd_grad)
            @test all(isfinite, fi_grad)

            zy_fd = maximum(abs.(zy_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
            zy_fi = maximum(abs.(zy_grad .- fi_grad) ./ max.(abs.(fi_grad), 1e-10))
            fd_fi = maximum(abs.(fd_grad .- fi_grad) ./ max.(abs.(fi_grad), 1e-10))
            println("  $(cfg.name): Zy-FD=$zy_fd  Zy-FI=$zy_fi  FD-FI=$fd_fi")
            @test zy_fd < 1e-6
            @test zy_fi < 1e-4
            @test fd_fi < 1e-4
        end
    end
end
