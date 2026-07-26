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
                            measurement_error_std = me)
    @test isfinite(kal)

    for (pf_filter, N) in ((:bootstrap_particle, 20_000), (:auxiliary_particle, 20_000), (:tempered_particle, 8_000))
        lls = [get_loglikelihood(m, data(observables), p; filter = pf_filter, algorithm = :first_order,
                                 presample_periods = 4, initial_covariance = :theoretical,
                                 measurement_error_std = me, n_particles = N, particle_rng = Random.Xoshiro(1000 + s)) for s in 1:6]
        @test all(isfinite, lls)
        # the Monte-Carlo mean matches the Kalman value up to the (downward) Var/2 bias
        @test abs(kal - Statistics.mean(lls)) < 15
    end
end
