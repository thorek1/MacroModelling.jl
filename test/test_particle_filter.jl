using MacroModelling
using Test
import Random
import Statistics
import AxisKeys: KeyedArray
import ForwardDiff
import Zygote
using StatsPlots

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
        # measurement_error = 0 must reduce exactly to the no-ME likelihood
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = 0.0) == llk_no
        # a positive measurement error changes the likelihood and stays finite
        llk_me = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = me^2)
        @test isfinite(llk_me)
        @test llk_me != llk_no
        # scalar broadcast equals the equivalent per-observable vector
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = [me^2, me^2]) ≈ llk_me
        # ForwardDiff flows through the Kalman likelihood with measurement error
        g = ForwardDiff.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :kalman, measurement_error = me^2), p)
        @test all(isfinite, g)
    end

    kal = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = me^2)

    @testset "Bootstrap PF converges to the Kalman likelihood" begin
        # The bootstrap particle-filter likelihood estimator is unbiased for the
        # true likelihood, so log L̂ is downward biased by ≈ Var(log L̂)/2 and both
        # the bias and the variance shrink with the number of particles.
        pf(N, s) = get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                     measurement_error = me^2, n_particles = N, particle_rng = Random.Xoshiro(s))
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
                                                     measurement_error = me^2,
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
                                    measurement_error = me^2, n_particles = 3_000, particle_rng = Random.Xoshiro(7))
            @test isfinite(llh)
        end
        # every variant runs at a pruned nonlinear order
        for pf_filter in (:bootstrap_particle, :auxiliary_particle, :tempered_particle)
            llh = get_loglikelihood(RBC_pf, data, p; filter = pf_filter, algorithm = :pruned_second_order,
                                    measurement_error = me^2,
                                    n_particles = 3_000, particle_rng = Random.Xoshiro(7))
            @test isfinite(llh)
        end
    end

    @testset "Missing observations" begin
        raw = Array{Union{Missing,Float64}}(collect(data))
        raw[1, 3:5] .= missing
        raw[2, 20] = missing
        datam = KeyedArray(raw, Variable = [:c, :q], Time = 1:size(raw, 2))
        kal_m = get_loglikelihood(RBC_pf, datam, p; filter = :kalman, measurement_error = me^2)
        pf_m = get_loglikelihood(RBC_pf, datam, p; filter = :bootstrap_particle, algorithm = :first_order,
                                 measurement_error = me^2, n_particles = 16_000, particle_rng = Random.Xoshiro(9))
        @test isfinite(kal_m)
        @test isfinite(pf_m)
        @test abs(kal_m - pf_m) < 3.0
    end

    @testset "Filtered estimates from the particle filters" begin
        # the filtered particle estimates should track the Kalman estimates closely
        kal_v = get_estimated_variables(RBC_pf, data; filter = :kalman)
        for pf_filter in (:bootstrap_particle, :tempered_particle)
            v = get_estimated_variables(RBC_pf, data; filter = pf_filter, algorithm = :first_order,
                                        measurement_error = me^2, n_particles = 20_000,
                                        particle_rng = Random.Xoshiro(1))
            @test size(v) == size(kal_v)
            @test all(isfinite, collect(v))
            @test maximum(abs, collect(v) .- collect(kal_v)) < 0.1
        end
        s = get_estimated_shocks(RBC_pf, data; filter = :bootstrap_particle, algorithm = :first_order,
                                 measurement_error = me^2, n_particles = 10_000,
                                 particle_rng = Random.Xoshiro(1))
        @test all(isfinite, collect(s))
        # nonlinear orders and the combined estimates entry point
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :pruned_second_order, measurement_error = me^2,
                        n_particles = 5_000, particle_rng = Random.Xoshiro(1))))
        @test all(isfinite, collect(get_model_estimates(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, measurement_error = me^2,
                        n_particles = 5_000, particle_rng = Random.Xoshiro(1))))
    end

    @testset "Particle smoothing" begin
        kal_sm  = collect(get_estimated_variables(RBC_pf, data; filter = :kalman, smooth = true))
        pf_filt = collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = false, measurement_error = me^2,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1)))
        pf_sm   = collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = true, measurement_error = me^2,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1)))
        @test all(isfinite, pf_sm)
        @test size(pf_sm) == size(kal_sm)
        # using the whole sample must move the estimates closer to the Kalman smoother
        @test maximum(abs, pf_sm .- kal_sm) < maximum(abs, pf_filt .- kal_sm)
        # smoothing works for the other variants and at nonlinear orders
        for pf_filter in (:auxiliary_particle, :tempered_particle)
            @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = pf_filter,
                            algorithm = :first_order, smooth = true, measurement_error = me^2,
                            n_particles = 5_000, particle_rng = Random.Xoshiro(2))))
        end
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :pruned_second_order, smooth = true, measurement_error = me^2,
                        n_particles = 3_000, particle_rng = Random.Xoshiro(3))))
        @test all(isfinite, collect(get_estimated_shocks(RBC_pf, data; filter = :bootstrap_particle,
                        algorithm = :first_order, smooth = true, measurement_error = me^2,
                        n_particles = 10_000, particle_rng = Random.Xoshiro(4))))
        # the inversion filter still has no smoother
        @test all(isfinite, collect(get_estimated_variables(RBC_pf, data; filter = :inversion,
                        algorithm = :first_order, smooth = true)))

        # The tempered filter rejuvenates the cloud within each period, so its
        # estimates path runs a different recursion — it must still land on the
        # same smoothing distribution the bootstrap filter targets, and hence
        # close to the Kalman smoother on this linear model.
        tp_sm = collect(get_estimated_variables(RBC_pf, data; filter = :tempered_particle,
                        algorithm = :first_order, smooth = true, measurement_error = me^2,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1)))
        @test size(tp_sm) == size(kal_sm)
        @test maximum(abs, tp_sm .- kal_sm) < maximum(abs, pf_filt .- kal_sm)
        # ... and the tempering controls must actually reach the recursion
        tp_coarse = collect(get_estimated_variables(RBC_pf, data; filter = :tempered_particle,
                        algorithm = :first_order, smooth = true, measurement_error = me^2,
                        n_particles = 20_000, particle_rng = Random.Xoshiro(1),
                        particle_target_ratio = 50.0, particle_mh_steps = 3,
                        particle_mh_scale = 0.9))
        @test tp_coarse != tp_sm
        @test all(isfinite, tp_coarse)
    end

    @testset "Shock decomposition" begin
        # `get_shock_decomposition` returns [contributions..., (interaction,) residual];
        # the residual carries whatever the shocks do not explain, i.e. the
        # contribution of the initial state.
        nE = length(get_shocks(RBC_pf))
        dec(; kw...) = get_shock_decomposition(RBC_pf, data; filter = :bootstrap_particle,
                            measurement_error = me^2, n_particles = 6_000,
                            particle_rng = Random.Xoshiro(1), kw...)

        # available for the filtered *and* the smoothed shock estimates
        for sm in (false, true)
            d = dec(algorithm = :first_order, smooth = sm)
            @test size(d, 2) == nE + 1
            @test all(isfinite, collect(d))
            @test !all(iszero, collect(d))
            # first order is additive, so the shocks explain most of the movement
            A = collect(d)
            @test maximum(abs, A[:, end, :]) < 0.25 * maximum(abs, A[:, 1:end-1, :])
        end
        # filtered and smoothed shock paths give different decompositions
        @test collect(dec(algorithm = :first_order, smooth = false)) !=
              collect(dec(algorithm = :first_order, smooth = true))

        # pruned orders, both attributions, filtered and smoothed
        for algo in (:pruned_second_order, :pruned_third_order), sm in (false, true)
            # sequential: an explicit interaction column for the non-additive part
            ds = dec(algorithm = algo, smooth = sm, marginal_contribution = false)
            @test size(ds, 2) == nE + 2
            @test all(isfinite, collect(ds))
            @test !all(iszero, collect(ds))
            # Aumann-Shapley: the interaction is distributed across the shocks
            dm = dec(algorithm = algo, smooth = sm, marginal_contribution = true)
            @test size(dm, 2) == nE + 1
            @test all(isfinite, collect(dm))
            @test !all(iszero, collect(dm))
        end
    end

    @testset "Plotting with the particle filters" begin
        tmp = mktempdir()
        for (pf_filter, algo, kw) in ((:bootstrap_particle, :first_order, (;)),
                                      (:tempered_particle, :first_order, (; smooth = true)),
                                      (:bootstrap_particle, :pruned_second_order,
                                       (; smooth = true, shock_decomposition = true, marginal_contribution = true)))
            p = plot_model_estimates(RBC_pf, data; filter = pf_filter, algorithm = algo,
                                     measurement_error = me^2, n_particles = 2_000,
                                     particle_rng = Random.Xoshiro(1), show_plots = false,
                                     save_plots = true, save_plots_path = tmp,
                                     save_plots_format = :png, kw...)
            @test p !== nothing
        end
        @test !isempty(readdir(tmp))
    end

    @testset "Full measurement-error covariance" begin
        Hdiag = [me^2 0.0; 0.0 me^2]
        # a diagonal covariance reproduces the equivalent per-observable stds
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = Hdiag) ≈
              get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = me^2)
        # the Kalman filter accepts genuinely correlated measurement error
        Hfull = [me^2 0.6me^2; 0.6me^2 me^2]
        llf = get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = Hfull)
        @test isfinite(llf)
        @test llf != get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = me^2)
        # a diagonal covariance is equivalent to the variance vector for the
        # particle filters too (same RNG ⇒ bit-identical, since a diagonal matrix
        # is reduced to the elementwise fast path)
        @test get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                measurement_error = Hdiag, n_particles = 2_000,
                                particle_rng = Random.Xoshiro(1)) ==
              get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                measurement_error = me^2, n_particles = 2_000,
                                particle_rng = Random.Xoshiro(1))
        # the particle filters also handle genuinely correlated measurement error,
        # and on a linear model they must still converge to the Kalman value
        pf_full = [get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                     measurement_error = Hfull, n_particles = 40_000,
                                     particle_rng = Random.Xoshiro(100 + s)) for s in 1:4]
        @test all(isfinite, pf_full)
        @test isapprox(Statistics.mean(pf_full), llf, atol = 1.0)
        # ... and must differ from the diagonal answer, i.e. the off-diagonal
        # entries are genuinely used rather than silently dropped
        @test get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                measurement_error = Hfull, n_particles = 2_000,
                                particle_rng = Random.Xoshiro(1)) !=
              get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                measurement_error = Hdiag, n_particles = 2_000,
                                particle_rng = Random.Xoshiro(1))
        # every variant accepts a correlated covariance
        for f in (:auxiliary_particle, :tempered_particle)
            @test isfinite(get_loglikelihood(RBC_pf, data, p; filter = f, algorithm = :first_order,
                                             measurement_error = Hfull, n_particles = 2_000,
                                             particle_rng = Random.Xoshiro(3)))
        end
        # a covariance must be symmetric, positive definite and correctly sized
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = [1.0 2.0; 0.0 1.0]))
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = -Hdiag))
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = fill(me^2, 1, 1)))
    end

    @testset "Estimated variable standard deviations" begin
        sk = get_estimated_variable_standard_deviations(RBC_pf, data)
        @test size(sk, 2) == size(data, 2)
        @test all(isfinite, sk) && all(>=(0), sk)
        # the particle filters report the spread of the cloud, at any order
        for (f, algo) in ((:bootstrap_particle, :first_order),
                          (:tempered_particle, :pruned_second_order))
            sp = get_estimated_variable_standard_deviations(RBC_pf, data; filter = f, algorithm = algo,
                                                            measurement_error = me^2, n_particles = 3_000,
                                                            particle_rng = Random.Xoshiro(11))
            @test size(sp) == size(sk)
            @test all(isfinite, sp) && all(>=(0), sp)
            @test any(>(0), sp)
        end
        # the inversion filter identifies the state exactly and has no spread
        @test threw(() -> get_estimated_variable_standard_deviations(RBC_pf, data; filter = :inversion))
    end

    @testset "Linear model: every perturbation order reproduces the Kalman likelihood" begin
        # A *linear* model has no higher-order solution terms, so every
        # perturbation order describes the same system and the exact likelihood is
        # the Kalman one — at every order. That turns the Kalman filter into a
        # reference for the higher-order particle machinery (the pruned and
        # non-pruned second- and third-order transitions, their Kronecker scratch,
        # and the pruned `Vector{Vector}` particle layout), which otherwise has
        # nothing exact to be checked against. Deviations beyond Monte-Carlo error
        # are a bug in the transition code, not a property of the model.
        @model LIN_pf begin
            zs[0] = rho_l * zs[-1] + sig_l * e1[x]
            ys[0] = zs[0] + 0 * ys[1]
        end
        @parameters LIN_pf begin
            rho_l = 0.5
            sig_l = 0.01
        end

        Random.seed!(4242)
        dlin = simulate(LIN_pf, periods = 60)([:ys], :, :simulate)
        plin = LIN_pf.parameter_values
        mel  = 5e-6   # variance

        # Premise: the model really is linear. The inversion filter is
        # deterministic, so it must return the identical value at every order.
        inv_lls = [get_loglikelihood(LIN_pf, dlin, plin; filter = :inversion, algorithm = a)
                   for a in (:first_order, :pruned_second_order, :second_order,
                             :pruned_third_order, :third_order)]
        @test all(isapprox.(inv_lls, inv_lls[1], rtol = 1e-10))

        kal_lin = get_loglikelihood(LIN_pf, dlin, plin; filter = :kalman,
                                    measurement_error = mel)
        @test isfinite(kal_lin)

        for algo in (:first_order, :pruned_second_order, :second_order,
                     :pruned_third_order, :third_order)
            lls = [get_loglikelihood(LIN_pf, dlin, plin; filter = :bootstrap_particle,
                                     algorithm = algo, measurement_error = mel,
                                     n_particles = 20_000,
                                     particle_rng = Random.Xoshiro(600 + s)) for s in 1:3]
            @test all(isfinite, lls)
            @test abs(kal_lin - Statistics.mean(lls)) < 8.0
        end
    end

    @testset "Higher order: particle filter approaches the inversion filter" begin
        # At higher order there is no Kalman filter to check against, but there is
        # still an exact reference. As H -> 0 the measurement density collapses onto
        # the change of variables y -> eps, so
        #     p(y_t | x_{t-1}) -> N(eps_hat; 0, I) / |det Z(x_{t-1})|,
        # which is precisely the inversion filter's per-period contribution. Giving
        # the particle filter a degenerate initial cloud (`initial_covariance = 0`)
        # matches the inversion filter's other assumption — that x_0 is known
        # exactly — so the two must agree in that limit, at any perturbation order.
        #
        # The limit is numerically hostile: shrinking H is exactly what makes the
        # importance weights degenerate, so the approach stalls at a floor set by
        # particle noise rather than continuing to zero. The test therefore checks
        # the *direction* — a moderate H is far from the inversion value, a small
        # one is close — instead of pinning a single tolerance.
        nV = length(get_variables(RBC_pf))
        Z0 = zeros(nV, nV)
        algo = :pruned_second_order
        inv_ll = get_loglikelihood(RBC_pf, data, p; filter = :inversion, algorithm = algo)
        @test isfinite(inv_ll)

        pf(h) = Statistics.mean(get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle,
                    algorithm = algo, initial_covariance = Z0, measurement_error = h,
                    n_particles = 25_000, particle_rng = Random.Xoshiro(900 + s)) for s in 1:2)

        far   = pf(1e-4)   # too much measurement error: a different problem
        close = pf(1e-5)   # small enough to approach the zero-measurement-error limit
        @test isfinite(far) && isfinite(close)
        @test abs(close - inv_ll) < 12
        @test abs(close - inv_ll) < abs(far - inv_ll)
    end

    @testset "Filter selection and automatic measurement error" begin
        # `:particle` is an alias for the guided filter: same RNG ⇒ same value
        @test get_loglikelihood(RBC_pf, data, p; filter = :particle, algorithm = :first_order,
                                measurement_error = me^2, n_particles = 2_000, particle_rng = Random.Xoshiro(5)) ==
              get_loglikelihood(RBC_pf, data, p; filter = :guided_particle, algorithm = :first_order,
                                measurement_error = me^2, n_particles = 2_000, particle_rng = Random.Xoshiro(5))
        # `:auto` leaves the Kalman filter without measurement error
        @test get_loglikelihood(RBC_pf, data, p; filter = :kalman) ==
              get_loglikelihood(RBC_pf, data, p; filter = :kalman, measurement_error = :auto)
        # `:auto` gives the particle filters a workable measurement error
        @test isfinite(get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                         n_particles = 2_000, particle_rng = Random.Xoshiro(5)))
        # an unknown filter name is rejected
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :not_a_filter))
    end

    @testset "Error guards" begin
        # measurement error is not available for the inversion filter
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :inversion, measurement_error = me^2))
        # the particle filter requires measurement error
        @test threw(() -> get_loglikelihood(RBC_pf, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                                            measurement_error = 0.0))
        # the particle filter is not differentiable (forward or reverse mode)
        @test threw(() -> ForwardDiff.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :bootstrap_particle,
                          algorithm = :first_order, measurement_error = me^2, n_particles = 500,
                          particle_rng = Random.Xoshiro(1)), p))
        @test threw(() -> Zygote.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :bootstrap_particle,
                          algorithm = :first_order, measurement_error = me^2, n_particles = 500,
                          particle_rng = Random.Xoshiro(1)), p))
        # reverse-mode AD of the Kalman likelihood with measurement error is guarded
        @test threw(() -> Zygote.gradient(x -> get_loglikelihood(RBC_pf, data, x; filter = :kalman,
                          measurement_error = me^2), p))
    end

end
