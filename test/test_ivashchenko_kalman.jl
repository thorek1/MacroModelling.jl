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
        @test size(solution[2], 2) == sys.dv^2
        if order == :third_order
            @test size(sys.third_derivative, 4) == sys.dv
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

        selected_S1 = sys.S1
        selected_S2 = Matrix(solution[2][sys.output_rows, :])
        selected_S3 = order == :third_order ? Matrix(solution[3][sys.output_rows, :]) : nothing
        nmc = 120_000
        sample_mean = zeros(length(sys.output_rows))
        sample_second = zeros(length(sys.output_rows), length(sys.output_rows))
        for _ in 1:nmc
            x = mean_state + ℒ.cholesky(covariance_state).L * randn(sys.nPast)
            ε = randn(sys.nExo)
            v = vcat(x, 1.0, ε)
            value = selected_S1 * v + selected_S2 * ℒ.kron(v, v) / 2
            if selected_S3 !== nothing
                value += selected_S3 * ℒ.kron(ℒ.kron(v, v), v) / 6
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

    # The filter is deliberately gated away from pruned solutions: those have a
    # different state-space representation and belong to the Kollmann filters.
    @test get_loglikelihood(RBC_ivashchenko, data, RBC_ivashchenko.parameter_values;
                            algorithm = :first_order,
                            filter = :ivashchenko_kalman) ==
          get_loglikelihood(RBC_ivashchenko, data, RBC_ivashchenko.parameter_values;
                            algorithm = :first_order,
                            filter = :inversion)
end
