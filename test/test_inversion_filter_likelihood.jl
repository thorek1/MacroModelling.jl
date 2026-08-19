using Test
using MacroModelling
using Random
using AxisKeys
import LinearAlgebra as ℒ

# -----------------------------------------------------------------------------
# Level checks for the inversion-filter loglikelihood.
#
# The inversion filter recovers the shocks that reproduce the data exactly and
# scores them under their own standard normal prior, with a Jacobian term for the
# change of variables y -> eps:
#
#   log p(y_t | x_{t-1}) = -1/2 (eps' eps + n log 2pi) - log|det Z|,   Z = C B
#
# The Jacobian enters with weight one, not one half. Getting that weight wrong is
# invisible in a gradient cross-check (every path shares the same primal) and
# invisible in any test that only asserts finiteness, but it biases estimation:
# log|det Z| carries the shock standard deviations, so halving it halves the
# likelihood's penalty against large shocks and inflates their estimates by
# sqrt(2). These tests pin the level against closed-form answers instead.
# -----------------------------------------------------------------------------

@testset "inversion filter loglikelihood level" begin

    # An AR(1) observed directly: the conditional likelihood is available in
    # closed form, so this pins the absolute level rather than a comparison.
    @model AR1_lik begin
        z[0] = rho * z[-1] + sig * e_z[x]
        y[0] = z[0] + 0 * y[1]
    end

    @parameters AR1_lik begin
        rho = 0.5
        sig = 0.01
    end

    Random.seed!(7)
    nT = 300
    rho_true, sig_true = 0.5, 0.01
    z = zeros(nT)
    for t in 2:nT
        z[t] = rho_true * z[t-1] + sig_true * randn()
    end
    data = KeyedArray(reshape(z, 1, nT); Variable = [:y], Time = 1:nT)

    # p(y_2..y_T | y_1); presample_periods = 1 drops the first observation, whose
    # treatment differs between the filters (ergodic prior vs. known initial state)
    exact = -0.5 * sum(((z[t] - rho_true * z[t-1]) / sig_true)^2 + log(sig_true^2) + log(2π)
                       for t in 2:nT)

    llh_inv = get_loglikelihood(AR1_lik, data, AR1_lik.parameter_values;
                                filter = :inversion, presample_periods = 1)
    llh_kal = get_loglikelihood(AR1_lik, data, AR1_lik.parameter_values;
                                filter = :kalman, presample_periods = 1)

    @test isapprox(llh_inv, exact, rtol = 1e-8)
    @test isapprox(llh_kal, exact, rtol = 1e-8)

    # The shock standard deviation must be identified correctly. With the
    # Jacobian at half weight the profile likelihood peaks at sqrt(2) times the
    # true value, so this is the sharpest guard against that regression.
    grid = range(0.006, 0.020, length = 701)
    prof = [get_loglikelihood(AR1_lik, data, [rho_true, s];
                              filter = :inversion, presample_periods = 1) for s in grid]
    Q = sum((z[t] - rho_true * z[t-1])^2 for t in 2:nT)
    @test isapprox(grid[argmax(prof)], sqrt(Q / (nT - 1)), rtol = 2e-3)

    # On a linear model with as many shocks as observables and no measurement
    # error the inversion filter and the Kalman filter identify the same state,
    # so their likelihoods agree once the initial-condition transient has died.
    @model RBC_lik begin
        1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
        z[0] = ρz * z[-1] + std_z * eps_z[x]
        g[0] = ρg * g[-1] + std_g * eps_g[x]
    end

    @parameters RBC_lik begin
        std_z = 0.01
        std_g = 0.01
        ρz = 0.4
        ρg = 0.6
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    Random.seed!(12345)
    sim = simulate(RBC_lik, periods = 200)
    d2 = sim([:c, :q], :, :simulate)   # 2 observables, 2 shocks
    p = RBC_lik.parameter_values

    k2 = get_loglikelihood(RBC_lik, d2, p; filter = :kalman,    presample_periods = 50)
    i2 = get_loglikelihood(RBC_lik, d2, p; filter = :inversion, presample_periods = 50)
    @test isapprox(i2, k2, atol = 0.5)

    # Sharper: the whole difference between the two filters on a square system is
    # the initial state covariance. The inversion filter assumes x₀ is known, so
    # Var(x₁) = BB'; hand the Kalman filter that same prior and the two agree to
    # machine precision, period by period, with no presample needed.
    get_loglikelihood(RBC_lik, d2, p)          # populate the solution cache
    Tc = RBC_lik.constants.post_model_macro
    S1 = RBC_lik.caches.first_order_solution_matrix
    nP = Tc.nPast_not_future_and_mixed
    ssn = RBC_lik.constants.post_complete_parameters.SS_and_pars_names
    oas = sort(union(Tc.past_not_future_and_mixed_idx,
                     convert(Vector{Int}, indexin([:c, :q], ssn))))
    Bm = S1[oas, nP+1:end]

    @test isapprox(get_loglikelihood(RBC_lik, d2, p; filter = :inversion),
                   get_loglikelihood(RBC_lik, d2, p; filter = :kalman,
                                     initial_covariance = Bm * Bm'),
                   rtol = 1e-10)

    # With more shocks than observables the state is no longer pinned down by the
    # data. The inversion filter clamps the state covariance to zero, so it uses
    # a strictly smaller innovation covariance than the Kalman filter and the two
    # part company. They stay in the same ballpark here only because little of the
    # unidentified subspace propagates into the next period's observables.
    d1 = sim([:c], :, :simulate)       # 1 observable, 2 shocks
    k1 = get_loglikelihood(RBC_lik, d1, p; filter = :kalman,    presample_periods = 50)
    i1 = get_loglikelihood(RBC_lik, d1, p; filter = :inversion, presample_periods = 50)
    @test isfinite(i1) && isfinite(k1)
    @test isapprox(i1, k1, atol = 5.0)
end
