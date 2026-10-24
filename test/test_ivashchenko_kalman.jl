using MacroModelling
using Test
using Random
using ForwardDiff
using Zygote
import LinearAlgebra as ℒ
import AxisKeys: KeyedArray

@testset "Ivashchenko unpruned Gaussian filter" begin
    @model RBC_ivashchenko begin
        1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
        z[0] = ρz * z[-1] + std_z * eps_z[x]
        g[0] = ρg * g[-1] + std_g * eps_g[x]
    end

    @parameters RBC_ivashchenko begin
        std_z = 0.02
        std_g = 0.02
        ρz = 0.4
        ρg = 0.6
        δ = 0.02
        α = 0.5
        β = 0.95
    end

    opts = MacroModelling.merge_calculation_options()
    obs = [:c, :q]
    data = KeyedArray(zeros(2, 10); Variable = obs, Time = 1:10)
    missing_values = Matrix{Float64}(collect(data))
    missing_values[1, 3] = NaN
    missing_values[:, 5] .= NaN
    missing_values[:, 6] .= NaN
    missing_data = KeyedArray(missing_values; Variable = obs, Time = 1:10)

    for order in (:second_order, :third_order)
        MacroModelling.solve!(RBC_ivashchenko, algorithm = order, dynamics = true, opts = opts)
        parameters = RBC_ivashchenko.parameter_values
        _, _, solution, state, solved = MacroModelling.get_relevant_steady_state_and_state_update(
            Val(order), parameters, RBC_ivashchenko, opts = opts)
        @test solved

        constants = RBC_ivashchenko.constants
        names = constants.post_complete_parameters.SS_and_pars_names
        obs_idx = convert(Vector{Int}, indexin(obs, names))
        sys = MacroModelling.build_ivashchenko_kalman_system_from_constants(
            constants, solution, obs_idx, order)
        @test sys.order == order
        @test size(solution[2], 2) == sys.n_pair
        if order == :third_order
            @test size(sys.S3, 2) == sys.dv * (sys.dv + 1) * (sys.dv + 2) ÷ 6
        end

        if order == :second_order
            # A likelihood needs only the state recursion and observed rows;
            # dropping unobserved jumpers must preserve both the moments and
            # the analytical reverse rule.
            single_obs = [:c]
            single_obs_idx = convert(Vector{Int}, indexin(single_obs, names))
            full_single = MacroModelling.build_ivashchenko_kalman_system_from_constants(
                constants, solution, single_obs_idx, order; keep_all_rows = true)
            compact_single = MacroModelling.build_ivashchenko_kalman_system_from_constants(
                constants, solution, single_obs_idx, order; keep_all_rows = false)
            single_data = KeyedArray(zeros(1, 10); Variable = single_obs, Time = 1:10)
            full_single_llh = MacroModelling.run_ivashchenko_kalman(
                full_single, collect(single_data), state[full_single.past];
                initial_covariance = :diagonal, measurement_error = 1e-4)
            compact_single_llh = MacroModelling.run_ivashchenko_kalman(
                compact_single, collect(single_data), state[compact_single.past];
                initial_covariance = :diagonal, measurement_error = 1e-4)
            @test compact_single.nout < full_single.nout
            @test compact_single_llh ≈ full_single_llh

            single_likelihood(p) = get_loglikelihood(
                RBC_ivashchenko, single_data, p;
                algorithm = :second_order,
                filter = :ivashchenko_kalman,
                initial_covariance = :diagonal,
                measurement_error = 1e-4)
            single_reverse = Zygote.gradient(single_likelihood,
                                              RBC_ivashchenko.parameter_values)[1]
            single_forward = ForwardDiff.gradient(single_likelihood,
                                                   RBC_ivashchenko.parameter_values)
            @test isapprox(single_reverse, single_forward; rtol = 1e-5, atol = 1e-5)
        end

        # The closed moments of the raw polynomial map are checked against a
        # direct Monte-Carlo evaluation, rather than against another filter.
        Random.seed!(17 + (order == :third_order))
        mean_state = copy(state[sys.past])
        covariance_state = 0.002 .* Matrix{Float64}(ℒ.I(sys.nPast))
        scalar_type = promote_type(eltype(sys.S1), Float64)
        ws = MacroModelling.ivashchenko_kalman_workspace(sys, scalar_type)
        closed_mean, closed_covariance = MacroModelling.ivashchenko_polynomial_moments!(
            sys, mean_state, covariance_state, ws)
        closed_mean = copy(closed_mean)
        closed_covariance = copy(closed_covariance)

        if order == :third_order
            # The compressed Hermite factorisation must reproduce the direct
            # sixth-Gaussian contraction exactly, without relying on Monte Carlo.
            factorized = zeros(size(closed_covariance))
            MacroModelling.ivashchenko_third_order_covariance!(
                factorized, sys, ws.third_derivative, ws.third_linear,
                ws.covariance_input, ws)
            direct = zeros(size(closed_covariance))
            for p in eachindex(sys.random_triple_indices), q in eachindex(sys.random_triple_indices)
                i, j, k = sys.random_triple_indices[p]
                l, m, n = sys.random_triple_indices[q]
                weight = sys.random_triple_multiplicities[p] *
                         sys.random_triple_multiplicities[q] *
                         MacroModelling.ivashchenko_gaussian_sixth(
                             (i, j, k, l, m, n), ws.covariance_input) / 36
                direct .+= weight .* (ws.third_derivative[:, p] *
                                      ws.third_derivative[:, q]')
            end
            @test factorized ≈ direct rtol = 1e-10 atol = 1e-10
        end

        selected_S1 = sys.S1
        selected_S2 = sys.S2
        selected_S3 = order == :third_order ? sys.S3 : nothing
        nmc = 120_000
        sample_mean = zeros(length(sys.output_rows))
        sample_second = zeros(length(sys.output_rows), length(sys.output_rows))
        for _ in 1:nmc
            x = mean_state + ℒ.cholesky(covariance_state).L * randn(sys.nPast)
            ε = randn(sys.nExo)
            v = vcat(x, 1.0, ε)
            value = selected_S1 * v + selected_S2 *
                    MacroModelling.compressed_kron²_power(v) / 2
            if selected_S3 !== nothing
                value += selected_S3 * MacroModelling.compressed_kron³_power(v) / 6
            end
            sample_mean .+= value
            sample_second .+= value * value'
        end
        sample_mean ./= nmc
        sample_covariance = sample_second ./ nmc - sample_mean * sample_mean'
        relative(a, b) = maximum(abs, a - b) / max(1e-10, maximum(abs, b))
        @test relative(sample_mean, closed_mean) < 0.08
        @test relative(sample_covariance, closed_covariance) < 0.08

        # Public dispatch reaches the separate filter for both raw solution
        # orders and uses the coupled stationary Gaussian initialization.
        ll = get_loglikelihood(RBC_ivashchenko, data, parameters;
                               algorithm = order,
                               filter = :ivashchenko_kalman,
                               measurement_error = 1e-4)
        @test isfinite(ll)
        ll_diagonal = get_loglikelihood(RBC_ivashchenko, data, parameters;
                                        algorithm = order,
                                        filter = :ivashchenko_kalman,
                                        initial_covariance = :diagonal,
                                        measurement_error = 1e-4)
        @test isfinite(ll_diagonal)

        # Partial observations use the observed sub-block of the innovation
        # covariance; fully missing periods are prediction-only steps.
        ll_missing = get_loglikelihood(RBC_ivashchenko, missing_data, parameters;
                                       algorithm = order,
                                       filter = :ivashchenko_kalman,
                                       initial_covariance = :diagonal,
                                       measurement_error = 1e-4)
        @test isfinite(ll_missing)

        estimates = get_estimated_variables(RBC_ivashchenko, missing_data;
                                            algorithm = order,
                                            filter = :ivashchenko_kalman,
                                            initial_covariance = :diagonal,
                                            measurement_error = 1e-4,
                                            levels = false,
                                            smooth = true)
        @test size(estimates) == (sys.nVars, size(data, 2))
        @test all(isfinite, collect(estimates))

        shocks = get_estimated_shocks(RBC_ivashchenko, missing_data;
                                      algorithm = order,
                                      filter = :ivashchenko_kalman,
                                      initial_covariance = :diagonal,
                                      measurement_error = 1e-4,
                                      smooth = true)
        @test size(shocks) == (sys.nExo, size(data, 2))
        @test all(isfinite, collect(shocks))

        standard_deviations = get_estimated_variable_standard_deviations(
            RBC_ivashchenko, missing_data;
            algorithm = order,
            filter = :ivashchenko_kalman,
            initial_covariance = :diagonal,
            measurement_error = 1e-4,
            smooth = true)
        @test size(standard_deviations) == size(estimates)
        @test all(isfinite, collect(standard_deviations))
    end

    forward_likelihood(p) = get_loglikelihood(RBC_ivashchenko, data, p;
                                              algorithm = :second_order,
                                              filter = :ivashchenko_kalman,
                                              measurement_error = 1e-4)
    forward_gradient = ForwardDiff.gradient(forward_likelihood, RBC_ivashchenko.parameter_values)
    @test all(isfinite, forward_gradient)

    # The custom reverse rule covers both dense and missing-data paths.  Use
    # the explicit diagonal prior here so this test isolates the filter and
    # measurement-update adjoints from the nonlinear stationary fixed point.
    for order in (:second_order, :third_order)
        likelihood(p) = get_loglikelihood(RBC_ivashchenko, missing_data, p;
                                          algorithm = order,
                                          filter = :ivashchenko_kalman,
                                          initial_covariance = :diagonal,
                                          measurement_error = 1e-4)
        reverse_gradient = Zygote.gradient(likelihood, RBC_ivashchenko.parameter_values)[1]
        forward_gradient = ForwardDiff.gradient(likelihood, RBC_ivashchenko.parameter_values)
        @test all(isfinite, reverse_gradient)
        @test isapprox(reverse_gradient, forward_gradient; rtol = 1e-5, atol = 1e-5)
    end

    # Cheaper Gaussian covariance closures retain the exact polynomial mean but
    # make an explicit covariance approximation. Their analytical pullbacks
    # must agree with forward mode on the same second-order likelihood.
    for closure in (:linearized, :diagonal)
        closure_likelihood(p) = get_loglikelihood(
            RBC_ivashchenko, data, p;
            algorithm = :second_order,
            filter = :ivashchenko_kalman,
            initial_covariance = :diagonal,
            measurement_error = 1e-4,
            ivashchenko_gaussian_closure = closure)
        closure_value = closure_likelihood(RBC_ivashchenko.parameter_values)
        @test isfinite(closure_value)
        closure_reverse = Zygote.gradient(closure_likelihood,
                                          RBC_ivashchenko.parameter_values)[1]
        closure_forward = ForwardDiff.gradient(closure_likelihood,
                                               RBC_ivashchenko.parameter_values)
        @test all(isfinite, closure_reverse)
        @test isapprox(closure_reverse, closure_forward; rtol = 1e-5, atol = 1e-5)
    end

    linearized_third_likelihood(p) = get_loglikelihood(
        RBC_ivashchenko, data, p;
        algorithm = :third_order,
        filter = :ivashchenko_kalman,
        initial_covariance = :diagonal,
        measurement_error = 1e-4,
        ivashchenko_gaussian_closure = :linearized)
    @test isfinite(linearized_third_likelihood(RBC_ivashchenko.parameter_values))
    linearized_third_reverse = Zygote.gradient(
        linearized_third_likelihood, RBC_ivashchenko.parameter_values)[1]
    linearized_third_forward = ForwardDiff.gradient(
        linearized_third_likelihood, RBC_ivashchenko.parameter_values)
    @test all(isfinite, linearized_third_reverse)
    @test isapprox(linearized_third_reverse, linearized_third_forward;
                   rtol = 1e-5, atol = 1e-5)

    # First-order solutions remain outside the Ivashchenko moment closure and
    # continue to fall back to the inversion filter.
    @test get_loglikelihood(RBC_ivashchenko, data, RBC_ivashchenko.parameter_values;
                            algorithm = :first_order,
                            filter = :ivashchenko_kalman) ==
          get_loglikelihood(RBC_ivashchenko, data, RBC_ivashchenko.parameter_values;
                            algorithm = :first_order,
                            filter = :inversion)

    # The RTS cross-covariance is PₜAₜ₊₁', not Aₜ₊₁Pₜ.  Use a non-commuting
    # transition/covariance pair so the orientation error cannot be hidden by
    # symmetric toy matrices.
    smoother_sys = (nVars = 2, nExo = 1, state_position = 1:2)
    post_covariance = [2.0 0.3; 0.3 1.0]
    predicted_covariance = [3.0 0.1; 0.1 2.0]
    transition = [0.4 0.8; -0.2 0.3]
    post_mean = [0.0, 0.0]
    predicted_mean = [0.5, -0.3]
    next_post_mean = [1.2, 0.4]
    smoother_tape = (state_position = 1:2,
                     post_means = [post_mean, next_post_mean],
                     post_covariances = [post_covariance, predicted_covariance],
                     transitions = [zeros(2, 2), transition],
                     predicted_means = [zeros(2), predicted_mean],
                     predicted_covariances = [Matrix{Float64}(ℒ.I, 2, 2), predicted_covariance],
                     output_means = [zeros(2), zeros(2)],
                     output_covariances = [Matrix{Float64}(ℒ.I, 2, 2), predicted_covariance],
                     shock_loadings = [[0.2; 0.4], [0.2; 0.4]])
    smoothed, = MacroModelling.ivashchenko_smooth_pass(smoother_sys, smoother_tape)
    expected = post_mean + post_covariance * transition' /
               predicted_covariance * (next_post_mean - predicted_mean)
    @test smoothed[:, 1] ≈ expected
end
