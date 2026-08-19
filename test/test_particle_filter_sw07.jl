using MacroModelling
using Test
import Random
import Statistics
using DelimitedFiles, AxisKeys
import LinearAlgebra as ℒ

# Validate the particle filter on the Smets & Wouters (2007) linear model and the
# US data used by the estimation tests: in the linear (first-order) case every
# particle-filter variant must reproduce the exact Kalman log-likelihood.
#
# SW07 has 7 observables and 7 shocks, so with small measurement error the
# importance weights are very peaked (curse of dimensionality). We therefore use
# a moderately large measurement error (2·data-std), which keeps the estimator
# variance low so the Monte-Carlo mean lands on the Kalman value (up to the
# expected Var/2 finite-particle bias). The generous tolerance is meant to catch
# an incorrect filter (which would be off by hundreds of log-points), not to pin
# the value to the last digit.

@testset "SW07 linear: particle filter matches Kalman" begin
    dat, header = readdlm(joinpath(@__DIR__, "data", "usmodel.csv"), ',', header = true)
    dat = Float64.(dat)
    csv_names = vec(Symbol.(strip.(header)))
    data = KeyedArray(dat', Variable = csv_names, Time = axes(dat, 1))
    data = data([:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs], 47:230)
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
    data = rekey(data, :Variable => observables)

    include("../models/Smets_Wouters_2007_linear.jl")
    SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])
    m = Smets_Wouters_2007_linear
    p = m.parameter_values

    me = 2.0 .* [Statistics.std(collect(data(o))) for o in observables]

    # Compare both filters from the same initial state distribution: the particle
    # cloud represents the ergodic distribution, matching `initial_covariance = :theoretical`.
    kal = get_loglikelihood(m, data(observables), p; filter = :kalman,
                            presample_periods = 4, initial_covariance = :theoretical,
                            measurement_error = me .^ 2)
    @test isfinite(kal)

    for (pf_filter, N) in ((:bootstrap_particle, 20_000), (:auxiliary_particle, 20_000), (:tempered_particle, 8_000))
        lls = [get_loglikelihood(m, data(observables), p; filter = pf_filter, algorithm = :first_order,
                                 presample_periods = 4, initial_covariance = :theoretical,
                                 measurement_error = me .^ 2, n_particles = N, particle_rng = Random.Xoshiro(1000 + s)) for s in 1:6]
        @test all(isfinite, lls)
        # the Monte-Carlo mean matches the Kalman value up to the (downward) Var/2 bias
        @test abs(kal - Statistics.mean(lls)) < 15
    end
end

# -----------------------------------------------------------------------------
# Filter equivalences on SW07, expressed through the initial state covariance.
#
# `initial_covariance` is the prior on the *state* at the start of the sample —
# where the economy was before the first observation. It is a different object
# from `measurement_error`, which is noise on the *observation* and enters every
# period forever. P₁ is transient: its influence decays at the rate of the
# filter's own error dynamics, which is exactly why the three filters agree only
# once their initial conditions are made to match.
#
#   inversion  <=>  Kalman with P₁ = BB'
#       The inversion filter assumes the state is known exactly at t = 0, so the
#       only uncertainty about x₁ is the first period's shocks: Var(x₁) = BB'.
#       Given at least as many observables as shocks, the update then drives the
#       posterior covariance to exactly zero and it stays there, so the two
#       filters coincide period by period, not merely asymptotically.
#
#   particle   <=>  Kalman with P₁ = ergodic
#       The initial cloud is drawn around the mean with the ergodic covariance Σ,
#       so Var(x₁) = AΣA' + BB' = Σ, matching `initial_covariance = :theoretical`.
#       (Covered by the measurement-error testset above.)
# -----------------------------------------------------------------------------
@testset "SW07: inversion equals Kalman started at BB'" begin
    dat, header = readdlm(joinpath(@__DIR__, "data", "usmodel.csv"), ',', header = true)
    dat = Float64.(dat)
    csv_names = vec(Symbol.(strip.(header)))
    data = KeyedArray(dat', Variable = csv_names, Time = axes(dat, 1))
    data = data([:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs], 47:230)
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
    data = rekey(data, :Variable => observables)

    include("../models/Smets_Wouters_2007_linear.jl")
    SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])
    m = Smets_Wouters_2007_linear
    p = m.parameter_values

    # populate the first-order solution cache at these parameters
    get_loglikelihood(m, data, p)

    # Var(x₁ | x₀ known) = BB', over the Kalman filter's state ordering
    # (`union(past states, observables)`, sorted).
    T = m.constants.post_model_macro
    S1 = m.caches.first_order_solution_matrix
    nP = T.nPast_not_future_and_mixed
    past = T.past_not_future_and_mixed_idx
    ssn = m.constants.post_complete_parameters.SS_and_pars_names
    obs_idx = convert(Vector{Int}, indexin(observables, ssn))
    oas = sort(union(past, obs_idx))
    B = S1[oas, nP+1:end]
    P1 = B * B'

    llh_inv = get_loglikelihood(m, data, p; filter = :inversion)
    llh_kal = get_loglikelihood(m, data, p; filter = :kalman, initial_covariance = P1)

    @test isfinite(llh_inv)
    # 7 shocks, 7 observables, no measurement error: the two are the same filter
    @test isapprox(llh_inv, llh_kal, rtol = 1e-9)

    # The ergodic prior is a genuinely different starting point, and on SW07 the
    # difference is large and persistent — this guards against the test passing
    # for the trivial reason that every initial covariance gives the same answer.
    llh_kal_erg = get_loglikelihood(m, data, p; filter = :kalman)
    @test abs(llh_kal_erg - llh_inv) > 100

    # Dropping the first observations does not reconcile them either: SW07's
    # inverse-system dynamics are slow, so the initial condition is still felt.
    @test !isapprox(get_loglikelihood(m, data, p; filter = :inversion, presample_periods = 20),
                    get_loglikelihood(m, data, p; filter = :kalman,    presample_periods = 20),
                    rtol = 1e-3)

    # The particle filters' `initial_covariance` is Var(x₀) — the cloud is drawn
    # around the initial state and *then* propagated — whereas the Kalman filter's
    # is Var(x₁), the covariance of the first predicted state. The two therefore
    # correspond as  P₁ = A·Var(x₀)·A' + BB'. This is invisible at the
    # `:theoretical` default, because the ergodic Σ is the fixed point of
    # Σ = AΣA' + BB' and the shift maps it to itself, which is why the
    # measurement-error testset above can pass the same symbol to both. Pin both
    # ends of the correspondence so the distinction cannot drift.
    Ak = S1[oas, 1:nP] * Matrix(1.0 * ℒ.I, T.nVars, T.nVars)[past, oas]
    me = 2.0 .* [Statistics.std(collect(data(o))) for o in observables]

    # Var(x₀) = 0  ⇒  P₁ = BB'
    kal_BB = get_loglikelihood(m, data, p; filter = :kalman,
                               initial_covariance = P1, measurement_error = me .^ 2)
    pf_0 = [get_loglikelihood(m, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                              initial_covariance = zeros(T.nVars, T.nVars),
                              measurement_error = me .^ 2, n_particles = 20_000,
                              particle_rng = Random.Xoshiro(300 + s)) for s in 1:4]
    @test all(isfinite, pf_0)
    @test abs(kal_BB - Statistics.mean(pf_0)) < 8

    # ---- does the equivalence carry to the states and shocks? ----
    # Theory: with P₁ = BB' the Kalman gain is P C'F⁻¹ = BB'C'(CBB'C')⁻¹ = B Z⁺,
    # which is exactly the inversion filter's state recursion, so the two must
    # track the same path. The estimates path works in the full nVars basis
    # (unlike the likelihood path, which uses union(past, observables)), so the
    # prior has to be built there.
    Bf = S1[:, nP+1:end]
    P1_est = Bf * Bf'

    inv_v = collect(get_estimated_variables(m, data; filter = :inversion))
    kal_v = collect(get_estimated_variables(m, data; filter = :kalman, smooth = true,
                                            initial_covariance = P1_est))
    inv_s = collect(get_estimated_shocks(m, data; filter = :inversion))
    kal_s = collect(get_estimated_shocks(m, data; filter = :kalman, smooth = true,
                                         initial_covariance = P1_est))

    reldev(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())

    # states and shocks agree to machine precision — a far sharper check on the
    # inversion filter's implementation than any likelihood comparison, since it
    # pins the whole path rather than one scalar
    @test reldev(inv_v, kal_v) < 1e-8
    @test reldev(inv_s, kal_s) < 1e-8

    # the shocks also match the Kalman filter's *filtered* estimates; the states
    # do not, because a single period's seven observations do not pin all forty
    # variables contemporaneously even though the full sample does
    @test reldev(inv_s, collect(get_estimated_shocks(m, data; filter = :kalman, smooth = false,
                                                     initial_covariance = P1_est))) < 1e-8

    # ... and "the full sample pins the state exactly" is directly checkable:
    # under P₁ = BB' the smoothed dispersion collapses, which is precisely the
    # assumption the inversion filter makes.
    sd_sm = collect(get_estimated_variable_standard_deviations(m, data; filter = :kalman,
                                                               smooth = true, initial_covariance = P1_est))
    @test maximum(sd_sm) < 1e-3

    # with the ergodic prior the two are far apart, so the agreement above is not
    # an artefact of every initial covariance giving the same answer
    @test reldev(inv_v, collect(get_estimated_variables(m, data; filter = :kalman, smooth = true))) > 0.1

    # Var(x₀) = BB'  ⇒  P₁ = A BB' A' + BB'
    Bfull = S1[:, nP+1:end]
    kal_shift = get_loglikelihood(m, data, p; filter = :kalman,
                                  initial_covariance = Ak * P1 * Ak' + P1,
                                  measurement_error = me .^ 2)
    pf_BB = [get_loglikelihood(m, data, p; filter = :bootstrap_particle, algorithm = :first_order,
                               initial_covariance = Bfull * Bfull',
                               measurement_error = me .^ 2, n_particles = 20_000,
                               particle_rng = Random.Xoshiro(400 + s)) for s in 1:4]
    @test all(isfinite, pf_BB)
    @test abs(kal_shift - Statistics.mean(pf_BB)) < 8
end

# -----------------------------------------------------------------------------
# Higher-order equivalence for the particle filters.
#
# Smets-Wouters (2007) in its log-linearised form is *linear*, so every
# perturbation order yields the same solution — asserted below rather than
# assumed. That makes it a rare thing: a model complex enough to be a real test
# (40 variables, 7 shocks, 7 observables, 184 periods) on which the exact
# likelihood is known, yet which exercises the higher-order particle machinery —
# the pruned and non-pruned second-order transitions, their augmented Kronecker
# scratch, and the pruned `Vector{Vector}` particle layout. (Third order is
# covered on a small linear model in `test_particle_filter.jl` — same idea, but
# seconds rather than minutes per evaluation on 40 variables.)
#
# Any deviation from the Kalman value beyond Monte-Carlo error is therefore a
# bug in the higher-order transition code, not a property of the model.
# -----------------------------------------------------------------------------
@testset "SW07 linear: particle filter at higher order matches Kalman" begin
    dat, header = readdlm(joinpath(@__DIR__, "data", "usmodel.csv"), ',', header = true)
    dat = Float64.(dat)
    csv_names = vec(Symbol.(strip.(header)))
    data = KeyedArray(dat', Variable = csv_names, Time = axes(dat, 1))
    data = data([:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs], 47:230)
    observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
    data = rekey(data, :Variable => observables)

    include("../models/Smets_Wouters_2007_linear.jl")
    SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])
    m = Smets_Wouters_2007_linear
    p = m.parameter_values

    # Premise: the model is linear. The inversion filter is deterministic, so if
    # the higher-order solution terms vanish it must return the identical value
    # at every order.
    inv_lls = [get_loglikelihood(m, data, p; filter = :inversion, algorithm = a)
               for a in (:first_order, :pruned_second_order, :second_order,
                         :pruned_third_order, :third_order)]
    @test all(isapprox.(inv_lls, inv_lls[1], rtol = 1e-10))

    me = 2.0 .* [Statistics.std(collect(data(o))) for o in observables]
    kal = get_loglikelihood(m, data, p; filter = :kalman, presample_periods = 4,
                            initial_covariance = :theoretical, measurement_error = me .^ 2)

    # Second order only: third order on a 40-variable model costs minutes per
    # evaluation, which is not worth it here. The third-order transitions are
    # covered against the same exact reference on a small linear model in
    # `test_particle_filter.jl`, where the whole sweep runs in seconds.
    for (algo, N, tol) in ((:pruned_second_order, 20_000, 15.0),
                           (:second_order,        20_000, 15.0))
        lls = [get_loglikelihood(m, data, p; filter = :bootstrap_particle, algorithm = algo,
                                 presample_periods = 4, initial_covariance = :theoretical,
                                 measurement_error = me .^ 2, n_particles = N,
                                 particle_rng = Random.Xoshiro(700 + s)) for s in 1:2]
        @test all(isfinite, lls)
        @test abs(kal - Statistics.mean(lls)) < tol
    end
end
