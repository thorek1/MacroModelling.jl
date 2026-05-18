using Test
using MacroModelling
import Turing
import Zygote
import Mooncake
import ForwardDiff
import ADTypes
import ADTypes: AutoZygote, AutoForwardDiff, AutoMooncake
import DifferentiationInterface
import FiniteDifferences
import Turing: NUTS, sample
import LinearAlgebra as ℒ
using Random, AxisKeys

using FlexiChains
include("test_helpers.jl")

include("../models/Gali_2015_chapter_3_nonlinear.jl")

# Gali model parameter order (from @parameters block):
# σ=1, φ=5, ϕᵖⁱ=1.5, ϕʸ=0.125, θ=0.75, ρ_ν=0.5, ρ_z=0.5, ρ_a=0.9,
# β=0.99, η=3.77, α=0.25, ϵ=9, τ=0, std_a=0.01, std_z=0.05, std_nu=0.0025

# Simulate data from the model at true parameter values
Random.seed!(42)
simulated_data = simulate(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)

# Observables: log_y, pi_ann, i_ann (3 observables matching 3 shocks)
observables = [:log_y, :pi_ann, :i_ann]
data = simulated_data(observables, :, :simulate)

# Compute target values at true parameters for NSSS, moments, and IRF targeting
true_params = Gali_2015_chapter_3_nonlinear.parameter_values

nsss_vars = [:Y, :Pi]
moment_vars = [:Y, :Pi]

target_stats = get_statistics(Gali_2015_chapter_3_nonlinear, true_params,
                              non_stochastic_steady_state = nsss_vars,
                              mean = moment_vars,
                              standard_deviation = moment_vars,
                              algorithm = :pruned_second_order)

target_nsss = target_stats[:non_stochastic_steady_state]
target_mean = target_stats[:mean]
target_std  = target_stats[:standard_deviation]

# IRF targets: first period response of log_y to first shock (eps_a)
target_irf_full = get_irf(Gali_2015_chapter_3_nonlinear, true_params,
                          algorithm = :pruned_second_order, periods = 5)
irf_var_idx = sort(MacroModelling.parse_variables_input_to_index(:log_y, Gali_2015_chapter_3_nonlinear))
target_irf = target_irf_full[irf_var_idx, 1, 1]

# Prior distributions for estimated parameters (6 out of 16)
# Estimated: ρ_ν (idx 6), ρ_z (idx 7), ρ_a (idx 8), std_a (idx 14), std_z (idx 15), std_nu (idx 16)
estimated_param_indices = [6, 7, 8, 14, 15, 16]

# Build index mapping to reconstruct full parameter vector (Zygote-compatible, no mutation)
# Fixed params: indices 1-5 (σ,φ,ϕᵖⁱ,ϕʸ,θ) and 9-13 (β,η,α,ϵ,τ)
# Estimated params: indices 6-8 (ρ_ν,ρ_z,ρ_a) and 14-16 (std_a,std_z,std_nu)
function build_full_params(estimated_vals::AbstractVector{T}) where T
    return vcat(
        T.(true_params[1:5]),       # σ, φ, ϕᵖⁱ, ϕʸ, θ
        estimated_vals[1:3],        # ρ_ν, ρ_z, ρ_a
        T.(true_params[9:13]),      # β, η, α, ϵ, τ
        estimated_vals[4:6],        # std_a, std_z, std_nu
    )
end

dists = [
    Beta(0.5, 0.15, μσ = true),               # ρ_ν
    Beta(0.5, 0.15, μσ = true),               # ρ_z
    Beta(0.9, 0.05, μσ = true),               # ρ_a
    InverseGamma(0.01, Inf, μσ = true),       # std_a
    InverseGamma(0.05, Inf, μσ = true),       # std_z
    InverseGamma(0.0025, Inf, μσ = true),     # std_nu
]

Turing.@model function Gali_estimation(data, m, algorithm, on_failure_llh,
                                        target_nsss, target_mean, target_std, target_irf,
                                        nsss_vars, moment_vars, irf_var_idx;
                                        verbose = false)
    estimated_params ~ Turing.product_distribution(dists)
    all_params = build_full_params(estimated_params)

    # 1. Log-likelihood from data
    llh = get_loglikelihood(m, data, all_params,
                            algorithm = algorithm,
                            on_failure_loglikelihood = on_failure_llh)
    maybe_print_loglikelihood(verbose, llh, dists, estimated_params)
    Turing.@addlogprob! llh

    # 2. NSSS targeting via get_statistics (positional params)
    stats_nsss = get_statistics(m, all_params,
                                non_stochastic_steady_state = nsss_vars,
                                algorithm = algorithm)
    nsss_vals = stats_nsss[:non_stochastic_steady_state]
    Turing.@addlogprob! sum(Turing.logpdf.(Turing.Normal.(target_nsss, 0.1), nsss_vals))

    # 3. Moment targeting via get_statistics (positional params)
    stats_moments = get_statistics(m, all_params,
                                    mean = moment_vars,
                                    standard_deviation = moment_vars,
                                    algorithm = algorithm)
    mean_vals = stats_moments[:mean]
    std_vals = stats_moments[:standard_deviation]
    Turing.@addlogprob! sum(Turing.logpdf.(Turing.Normal.(target_mean, 0.1), mean_vals))
    Turing.@addlogprob! sum(Turing.logpdf.(Turing.Normal.(target_std, 0.05), std_vals))

    # 4. IRF targeting via get_irf (positional params with algorithm)
    irf_vals = get_irf(m, all_params, algorithm = algorithm, periods = 5)
    Turing.@addlogprob! sum(Turing.logpdf.(Turing.Normal.(target_irf, 0.1), irf_vals[irf_var_idx, 1, 1]))
end


# Instantiate the Turing model
gali_model = Gali_estimation(data, Gali_2015_chapter_3_nonlinear, :pruned_second_order, -Inf,
                              target_nsss, target_mean, target_std, target_irf,
                              nsss_vars, moment_vars, irf_var_idx)

Random.seed!(123)

n_samples = 1000

samps = @time sample(gali_model,
                     NUTS(adtype = AutoForwardDiff()),
                     n_samples,
                     progress = true,
                     initial_params = Turing.InitFromParams((estimated_params = true_params[estimated_param_indices],)))

posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean estimated values (ForwardDiff): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

sample_means = collect(values(FlexiChains.mean(samps); parameters_only = true))

@testset "Gali pruned 2nd order estimation results" begin
    @test length(sample_means) == 6
    @test all(isfinite, sample_means)
    # Means should be in the right ballpark of true values
    @test isapprox(sample_means, true_params[estimated_param_indices], rtol = 0.5)
end

# ---------------------------------------------------------------------------
# Mooncake NUTS sampling
# ---------------------------------------------------------------------------
Random.seed!(123)

samps_mc = @time sample(gali_model,
                     NUTS(adtype = AutoMooncake(; config=nothing)),
                     n_samples,
                     progress = true,
                     initial_params = Turing.InitFromParams((estimated_params = true_params[estimated_param_indices],)))

posterior_summary_mc = FlexiChains.summarystats(samps_mc)
show(stdout, MIME"text/plain"(), posterior_summary_mc)
println()

sample_means_mc = collect(values(FlexiChains.mean(samps_mc); parameters_only = true))
println("Mean estimated values (Mooncake): $(sample_means_mc)")

@testset "Gali pruned 2nd order estimation results (Mooncake)" begin
    @test length(sample_means_mc) == 6
    @test all(isfinite, sample_means_mc)
    @test isapprox(sample_means_mc, true_params[estimated_param_indices], rtol = 0.5)
end

@testset "Zygote vs FiniteDifferences gradient (Gali pruned 2nd order)" begin
    # Test gradient of combined objective at true parameter values
    function combined_objective(x)
        all_p = build_full_params(x)
        m = Gali_2015_chapter_3_nonlinear
        alg = :pruned_second_order

        llh = get_loglikelihood(m, data, all_p, algorithm = alg, on_failure_loglikelihood = -Inf)

        stats_n = get_statistics(m, all_p, non_stochastic_steady_state = nsss_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_nsss, 0.1), stats_n[:non_stochastic_steady_state]))

        stats_m = get_statistics(m, all_p, mean = moment_vars, standard_deviation = moment_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_mean, 0.1), stats_m[:mean]))
        llh += sum(Turing.logpdf.(Turing.Normal.(target_std, 0.05), stats_m[:standard_deviation]))

        irf_v = get_irf(m, all_p, algorithm = alg, periods = 5)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_irf, 0.1), irf_v[irf_var_idx, 1, 1]))

        return llh
    end

    test_point = true_params[estimated_param_indices]

    back_grad = Zygote.gradient(combined_objective, test_point)[1]
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), combined_objective, test_point)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end

# Test Zygote gradient of the full log posterior (likelihood + priors)
# NUTS with AutoZygote fails due to DynamicPPL's bijector using setindex!,
# so we test Zygote differentiation of the combined objective directly.
@testset "Zygote log posterior gradient (Gali pruned 2nd order)" begin
    function turing_logjoint(x)
        all_p = build_full_params(x)
        m = Gali_2015_chapter_3_nonlinear
        alg = :pruned_second_order

        llh = get_loglikelihood(m, data, all_p, algorithm = alg, on_failure_loglikelihood = -Inf)

        stats_n = get_statistics(m, all_p, non_stochastic_steady_state = nsss_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_nsss, 0.1), stats_n[:non_stochastic_steady_state]))

        stats_m = get_statistics(m, all_p, mean = moment_vars, standard_deviation = moment_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_mean, 0.1), stats_m[:mean]))
        llh += sum(Turing.logpdf.(Turing.Normal.(target_std, 0.05), stats_m[:standard_deviation]))

        irf_v = get_irf(m, all_p, algorithm = alg, periods = 5)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_irf, 0.1), irf_v[irf_var_idx, 1, 1]))

        # Add prior log density
        for (i, d) in enumerate(dists)
            llh += Turing.logpdf(d, x[i])
        end

        return llh
    end

    test_pt = true_params[estimated_param_indices]

    zy_grad = Zygote.gradient(turing_logjoint, test_pt)[1]
    fd_grad = ForwardDiff.gradient(turing_logjoint, test_pt)

    @test all(isfinite, zy_grad)
    @test all(isfinite, fd_grad)

    rel_err = maximum(abs.(zy_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
    println("Zygote vs ForwardDiff gradient rel err on log posterior: $rel_err")
    @test rel_err < 1e-6
end

@testset "Mooncake vs ForwardDiff gradient (Gali pruned 2nd order)" begin
    function combined_objective_mc(x)
        all_p = build_full_params(x)
        m = Gali_2015_chapter_3_nonlinear
        alg = :pruned_second_order

        llh = get_loglikelihood(m, data, all_p, algorithm = alg, on_failure_loglikelihood = -Inf)

        stats_n = get_statistics(m, all_p, non_stochastic_steady_state = nsss_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_nsss, 0.1), stats_n[:non_stochastic_steady_state]))

        stats_m = get_statistics(m, all_p, mean = moment_vars, standard_deviation = moment_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_mean, 0.1), stats_m[:mean]))
        llh += sum(Turing.logpdf.(Turing.Normal.(target_std, 0.05), stats_m[:standard_deviation]))

        irf_v = get_irf(m, all_p, algorithm = alg, periods = 5)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_irf, 0.1), irf_v[irf_var_idx, 1, 1]))

        return llh
    end

    test_point = true_params[estimated_param_indices]

    mc_grad = DifferentiationInterface.gradient(combined_objective_mc, AutoMooncake(config = nothing), test_point)
    fd_grad = ForwardDiff.gradient(combined_objective_mc, test_point)

    @test all(isfinite, mc_grad)
    @test all(isfinite, fd_grad)

    rel_err = maximum(abs.(mc_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
    println("Mooncake vs ForwardDiff gradient rel err: $rel_err")
    @test rel_err < 1e-4
end

@testset "Mooncake log posterior gradient (Gali pruned 2nd order)" begin
    function turing_logjoint_mc(x)
        all_p = build_full_params(x)
        m = Gali_2015_chapter_3_nonlinear
        alg = :pruned_second_order

        llh = get_loglikelihood(m, data, all_p, algorithm = alg, on_failure_loglikelihood = -Inf)

        stats_n = get_statistics(m, all_p, non_stochastic_steady_state = nsss_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_nsss, 0.1), stats_n[:non_stochastic_steady_state]))

        stats_m = get_statistics(m, all_p, mean = moment_vars, standard_deviation = moment_vars, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_mean, 0.1), stats_m[:mean]))
        llh += sum(Turing.logpdf.(Turing.Normal.(target_std, 0.05), stats_m[:standard_deviation]))

        irf_v = get_irf(m, all_p, algorithm = alg, periods = 5)
        llh += sum(Turing.logpdf.(Turing.Normal.(target_irf, 0.1), irf_v[irf_var_idx, 1, 1]))

        for (i, d) in enumerate(dists)
            llh += Turing.logpdf(d, x[i])
        end

        return llh
    end

    test_pt = true_params[estimated_param_indices]

    mc_grad = DifferentiationInterface.gradient(turing_logjoint_mc, AutoMooncake(config = nothing), test_pt)
    fd_grad = ForwardDiff.gradient(turing_logjoint_mc, test_pt)

    @test all(isfinite, mc_grad)
    @test all(isfinite, fd_grad)

    rel_err = maximum(abs.(mc_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
    println("Mooncake vs ForwardDiff gradient rel err on log posterior: $rel_err")
    @test rel_err < 1e-4
end


# ---------------------------------------------------------------------------
# Replicate the system-prior estimation problem on data with missing
# observations.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data)

gali_model_missing = Gali_estimation(data_missing, Gali_2015_chapter_3_nonlinear, :pruned_second_order, -Inf,
                                     target_nsss, target_mean, target_std, target_irf,
                                     nsss_vars, moment_vars, irf_var_idx)

Random.seed!(123)

samps_missing = @time sample(gali_model_missing,
                     NUTS(adtype = AutoForwardDiff()),
                     n_samples,
                     progress = true,
                     initial_params = Turing.InitFromParams((estimated_params = true_params[estimated_param_indices],)))

posterior_summary_missing = FlexiChains.summarystats(samps_missing)
show(stdout, MIME"text/plain"(), posterior_summary_missing)
println()
println("Mean estimated values (ForwardDiff, missing data): $(collect(values(FlexiChains.mean(samps_missing); parameters_only = true)))")

sample_means_missing = collect(values(FlexiChains.mean(samps_missing); parameters_only = true))

@testset "Gali pruned 2nd order estimation results (missing data)" begin
    @test length(sample_means_missing) == 6
    @test all(isfinite, sample_means_missing)
    @test isapprox(sample_means_missing, true_params[estimated_param_indices], rtol = 0.5)
end

Random.seed!(123)

samps_mc_missing = @time sample(gali_model_missing,
                     NUTS(adtype = AutoMooncake(; config=nothing)),
                     n_samples,
                     progress = true,
                     initial_params = Turing.InitFromParams((estimated_params = true_params[estimated_param_indices],)))

posterior_summary_mc_missing = FlexiChains.summarystats(samps_mc_missing)
show(stdout, MIME"text/plain"(), posterior_summary_mc_missing)
println()

sample_means_mc_missing = collect(values(FlexiChains.mean(samps_mc_missing); parameters_only = true))
println("Mean estimated values (Mooncake, missing data): $(sample_means_mc_missing)")

@testset "Gali pruned 2nd order estimation results (Mooncake, missing data)" begin
    @test length(sample_means_mc_missing) == 6
    @test all(isfinite, sample_means_mc_missing)
    @test isapprox(sample_means_mc_missing, true_params[estimated_param_indices], rtol = 0.5)
end

@testset "Mooncake vs ForwardDiff gradient (Gali pruned 2nd order, missing data)" begin
    # `data_missing` is a KeyedArray containing NaN entries (injected by
    # `inject_missing_observations`). If it is referenced inside the objective
    # as a (non-`const`) `GlobalRef`, Mooncake emits a
    # `__verify_const(global_ref, stored_value)` check which asserts
    # `global_ref == primal(stored_value)`; that comparison is false for
    # NaN-bearing arrays (`NaN != NaN`) and aborts the gradient. Pass
    # `data_missing` (and the other captured values) as
    # `DifferentiationInterface.Constant` contexts so they are threaded through
    # as ordinary primals instead of GlobalRefs.
    function combined_objective_mc_missing(x, data_m, model, nsss_v, moment_v,
                                           tgt_nsss, tgt_mean, tgt_std,
                                           tgt_irf, irf_idx)
        all_p = build_full_params(x)
        alg = :pruned_second_order

        llh = get_loglikelihood(model, data_m, all_p, algorithm = alg, on_failure_loglikelihood = -Inf)

        stats_n = get_statistics(model, all_p, non_stochastic_steady_state = nsss_v, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(tgt_nsss, 0.1), stats_n[:non_stochastic_steady_state]))

        stats_m = get_statistics(model, all_p, mean = moment_v, standard_deviation = moment_v, algorithm = alg)
        llh += sum(Turing.logpdf.(Turing.Normal.(tgt_mean, 0.1), stats_m[:mean]))
        llh += sum(Turing.logpdf.(Turing.Normal.(tgt_std, 0.05), stats_m[:standard_deviation]))

        irf_v = get_irf(model, all_p, algorithm = alg, periods = 5)
        llh += sum(Turing.logpdf.(Turing.Normal.(tgt_irf, 0.1), irf_v[irf_idx, 1, 1]))

        return llh
    end

    test_point = true_params[estimated_param_indices]

    ctx = (
        DifferentiationInterface.Constant(data_missing),
        DifferentiationInterface.Constant(Gali_2015_chapter_3_nonlinear),
        DifferentiationInterface.Constant(nsss_vars),
        DifferentiationInterface.Constant(moment_vars),
        DifferentiationInterface.Constant(target_nsss),
        DifferentiationInterface.Constant(target_mean),
        DifferentiationInterface.Constant(target_std),
        DifferentiationInterface.Constant(target_irf),
        DifferentiationInterface.Constant(irf_var_idx),
    )

    mc_grad = DifferentiationInterface.gradient(combined_objective_mc_missing, AutoMooncake(config = nothing), test_point, ctx...)
    fd_grad = DifferentiationInterface.gradient(combined_objective_mc_missing, AutoForwardDiff(), test_point, ctx...)

    @test all(isfinite, mc_grad)
    @test all(isfinite, fd_grad)

    rel_err = maximum(abs.(mc_grad .- fd_grad) ./ max.(abs.(fd_grad), 1e-10))
    println("Mooncake vs ForwardDiff gradient rel err (missing data): $rel_err")
    @test rel_err < 1e-4
end
