using MacroModelling
using Test
import Random
import Statistics
using DelimitedFiles, AxisKeys

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
end
