using MacroModelling
using Test
import Random
import Statistics
import AxisKeys: KeyedArray
import ForwardDiff
import Zygote

# A small RBC model with two shocks and two observables, so the (bootstrap)
# particle filter is non-degenerate and can be validated against the exact
# Kalman likelihood on the linear (first-order) solution.
@model RBC_pf begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
    z[0] = ρz * z[-1] + std_z * eps_z[x]
    g[0] = ρg * g[-1] + std_g * eps_g[x]
end

@parameters RBC_pf begin
    std_z = 0.01
    std_g = 0.01
    ρz = 0.4
    ρg = 0.6
    δ = 0.02
    α = 0.5
    β = 0.95
end

Random.seed!(12345)
sim = simulate(RBC_pf, periods = 40)
data = sim([:c, :q], :, :simulate)
p = RBC_pf.parameter_values
me = 0.002

threw(f) = try; f(); false; catch; true; end

@testset "Particle filter" begin

    @testset "Measurement error on the Kalman filter" begin
        llk_no = get_loglikelihood(RBC_pf, data, p; filter = :kalman)
        # measurement_error_std = 0 must reduce exactly to the no-ME likelihood
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = 0.0) == llk_no
        # a positive measurement error changes the likelihood and stays finite
        llk_me = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = me)
        @test isfinite(llk_me)
        @test llk_me != llk_no
        # scalar broadcast equals the equivalent per-observable vector
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = [me, me]) ≈ llk_me
        # ForwardDiff flows through the Kalman likelihood with measurement error
        g = ForwardDiff.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :kalman, measurement_error_std = me), p)
        @test all(isfinite, g)
    end

    kal = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = me)

    @testset "Bootstrap PF converges to the Kalman likelihood" begin
        # The bootstrap particle-filter likelihood estimator is unbiased for the
        # true likelihood, so log L̂ is downward biased by ≈ Var(log L̂)/2 and both
        # the bias and the variance shrink with the number of particles.
        pf(N, s) = get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                     measurement_error_std = me, n_particles = N, particle_rng = Random.Xoshiro(s))
        nseeds = 24
        ll_small = [pf(2_000, 100 + s) for s in 1:nseeds]
        ll_large = [pf(16_000, 200 + s) for s in 1:nseeds]

        m_small, v_small = Statistics.mean(ll_small), Statistics.var(ll_small)
        m_large, v_large = Statistics.mean(ll_large), Statistics.var(ll_large)

        @test all(isfinite, ll_small)
        @test all(isfinite, ll_large)
        # variance decreases with the number of particles
        @test v_large < v_small
        # bias ≈ Var/2: the bias-corrected estimate is close to the Kalman value
        @test isapprox(m_large + v_large / 2, kal, atol = 1.0)
        # the estimate itself lands near the Kalman value at the larger N
        @test abs(kal - m_large) < 2.0
    end

    @testset "Variants: correct and ordered by efficiency" begin
        variant(pf_filter, N, s) = get_loglikelihood(RBC_pf, data, p; filter = pf_filter, algorithm = :first_order,
                                                     measurement_error_std = me,
                                                     n_particles = N, particle_rng = Random.Xoshiro(s))
        nseeds = 16
        boot = [variant(:bootstrap_particle, 3_000, 300 + s) for s in 1:nseeds]
        aux  = [variant(:auxiliary_particle, 3_000, 300 + s) for s in 1:nseeds]
        temp = [variant(:tempered_particle, 3_000, 300 + s) for s in 1:nseeds]

        for v in (boot, aux, temp)
            @test all(isfinite, v)
            @test abs(kal - Statistics.mean(v)) < 6.0   # all centred near the truth
        end
        # the tempered filter has markedly lower variance than the bootstrap filter
        @test Statistics.std(temp) < Statistics.std(boot)
    end

    @testset "Resampling schemes" begin
        rng = Random.Xoshiro(1)
        W = rand(rng, 200); W ./= sum(W)
        for scheme in (:systematic, :stratified, :multinomial, :residual)
            idx = MacroModelling.particle_resample_indices(rng, W, scheme)
            @test length(idx) == length(W)
            @test all(i -> 1 <= i <= length(W), idx)
        end
        # a degenerate weight vector (all mass on one particle) selects only it
        Wdeg = zeros(50); Wdeg[7] = 1.0
        for scheme in (:systematic, :stratified, :multinomial, :residual)
            @test all(==(7), MacroModelling.particle_resample_indices(rng, Wdeg, scheme))
        end
        @test MacroModelling.effective_sample_size(fill(1 / 100, 100)) ≈ 100.0
        @test threw(() -> MacroModelling.particle_resample_indices(rng, W, :nonexistent))
    end

    @testset "Higher-order algorithms run" begin
        for algo in (:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order)
            llh = get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = algo,
                                    measurement_error_std = me, n_particles = 3_000, particle_rng = Random.Xoshiro(7))
            @test isfinite(llh)
        end
        # every variant runs at a pruned nonlinear order
        for pf_filter in (:bootstrap_particle, :auxiliary_particle, :tempered_particle)
            llh = get_loglikelihood(RBC_pf, data, p; filter = pf_filter, algorithm = :pruned_second_order,
                                    measurement_error_std = me,
                                    n_particles = 3_000, particle_rng = Random.Xoshiro(7))
            @test isfinite(llh)
        end
    end

    @testset "Missing observations" begin
        raw = Array{Union{Missing,Float64}}(collect(data))
        raw[1, 3:5] .= missing
        raw[2, 20] = missing
        datam = KeyedArray(raw, Variable = [:c, :q], Time = 1:size(raw, 2))
        kal_m = get_loglikelihood(RBC_pf, datam, p; filter = :kalman, measurement_error_std = me)
        pf_m = get_loglikelihood(RBC_pf, datam, p; filter = :bootstrap_particle, algorithm = :first_order,
                                 measurement_error_std = me, n_particles = 16_000, particle_rng = Random.Xoshiro(9))
        @test isfinite(kal_m)
        @test isfinite(pf_m)
        @test abs(kal_m - pf_m) < 3.0
    end

    @testset "Filtered estimates from the particle filters" begin
        # the filtered particle estimates should track the Kalman estimates closely
        kal_v = get_estimated_variables(RBC_pf, data; filter = :kalman)
        for pf_filter in (:bootstrap_particle, :tempered_particle)
            v = get_estimated_variables(RBC_pf, data; filter = pf_filter, algorithm = :first_order,
                                        measurement_error_std = me, n_particles = 20_000,
                                        particle_rng = Random.Xoshiro(1))
            @test size(v) == size(kal_v)
            @test all(isfinite, collect(v))
            @test maximum(abs, collect(v) .- collect(kal_v)) < 0.1
        end
        s = get_estimated_shocks(RBC_pf, data; filter = :bootstrap_particle, algorithm = :first_order,
                                 measurement_error_std = me, n_particles = 10_000,
                                 particle_rng = Random.Xoshiro(1))
        @test all(isfinite, collect(s))
        # nonlinear orders and the combined estimates entry point
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :pruned_second_order, measurement_error_std = me,
                        n_particles = 5_000, particle_rng = Random.Xoshiro(1))))
        @test all(isfinite, collect(get_model_estimates(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, measurement_error_std = me,
                        n_particles = 5_000, particle_rng = Random.Xoshiro(1))))
    end

    @testset "Particle smoothing" begin
        kal_sm  = collect(get_estimated_variables(RBC_pf, data; filter = :kalman, smooth = true))
        pf_filt = collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = false, measurement_error_std = me,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1)))
        pf_sm   = collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = true, measurement_error_std = me,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1)))
        @test all(isfinite, pf_sm)
        @test size(pf_sm) == size(kal_sm)
        # using the whole sample must move the estimates closer to the Kalman smoother
        @test maximum(abs, pf_sm .- kal_sm) < maximum(abs, pf_filt .- kal_sm)
        # smoothing works for the other variants and at nonlinear orders
        for pf_filter in (:auxiliary_particle, :tempered_particle)
            @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = pf_filter,
                            algorithm = :first_order, smooth = true, measurement_error_std = me,
                            n_particles = 5_000, particle_rng = Random.Xoshiro(2))))
        end
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :pruned_second_order, smooth = true, measurement_error_std = me,
                        n_particles = 3_000, particle_rng = Random.Xoshiro(3))))
        @test all(isfinite, collect(get_estimated_shocks(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = true, measurement_error_std = me,
                        n_particles = 10_000, particle_rng = Random.Xoshiro(4))))
        # the inversion filter still has no smoother
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :inversion,
                        algorithm = :first_order, smooth = true)))
    end

    @testset "Full measurement-error covariance" begin
        Hdiag = [me^2 0.0; 0.0 me^2]
        # a diagonal covariance reproduces the equivalent per-observable stds
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_covariance = Hdiag) ≈
              get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = me)
        # the Kalman filter accepts genuinely correlated measurement error
        Hfull = [me^2 0.6me^2; 0.6me^2 me^2]
        llf = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_covariance = Hfull)
        @test isfinite(llf)
        @test llf != get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = me)
        # the particle filters take a diagonal covariance but reject an off-diagonal one
        @test isfinite(get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                         measurement_error_covariance = Hdiag, n_particles = 2_000,
                                         particle_rng = Random.Xoshiro(1)))
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                            measurement_error_covariance = Hfull, n_particles = 500,
                                            particle_rng = Random.Xoshiro(1)))
        # a covariance must be symmetric, positive definite and correctly sized
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_covariance = [1.0 2.0; 0.0 1.0]))
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_covariance = -Hdiag))
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_covariance = fill(me^2, 1, 1)))
    end

    @testset "Filter selection and automatic measurement error" begin
        # `:particle` is an alias for the bootstrap filter: same RNG ⇒ same value
        @test get_loglikelihood(RBC_pf, data, p; filter = :particle, algorithm = :first_order,
                                measurement_error_std = me, n_particles = 2_000, particle_rng = Random.Xoshiro(5)) ==
              get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                measurement_error_std = me, n_particles = 2_000, particle_rng = Random.Xoshiro(5))
        # `:auto` leaves the Kalman filter without measurement error
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman) ==
              get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error_std = :auto)
        # `:auto` gives the particle filters a workable measurement error
        @test isfinite(get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                         n_particles = 2_000, particle_rng = Random.Xoshiro(5)))
        # an unknown filter name is rejected
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :not_a_filter))
    end

    @testset "Error guards" begin
        # measurement error is not available for the inversion filter
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :inversion, measurement_error_std = me))
        # the particle filter requires measurement error
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                            measurement_error_std = 0.0))
        # the particle filter is not differentiable (forward or reverse mode)
        @test threw(() -> ForwardDiff.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :bootstrap_particle,
                          algorithm = :first_order, measurement_error_std = me, n_particles = 500,
                          particle_rng = Random.Xoshiro(1)), p))
        @test threw(() -> Zygote.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :bootstrap_particle,
                          algorithm = :first_order, measurement_error_std = me, n_particles = 500,
                          particle_rng = Random.Xoshiro(1)), p))
        # reverse-mode AD of the Kalman likelihood with measurement error is guarded
        @test threw(() -> Zygote.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :kalman,
                          measurement_error_std = me), p))
    end

end
