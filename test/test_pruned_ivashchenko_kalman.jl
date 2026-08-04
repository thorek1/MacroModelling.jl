using MacroModelling
using Test
using Random
using LinearAlgebra
using ForwardDiff
using Zygote
import AxisKeys: KeyedArray

@model RBC_pruned_ivashchenko_test begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
    z[0] = ρz * z[-1] + std_z * eps_z[x]
    g[0] = ρg * g[-1] + std_g * eps_g[x]
end

@parameters RBC_pruned_ivashchenko_test begin
    std_z = 0.02
    std_g = 0.02
    ρz = 0.4
    ρg = 0.6
    δ = 0.02
    α = 0.5
    β = 0.95
end

function direct_pruned_output(solution, states, shock, past, keep_rows, observables)
    if length(solution) == 2
        stages = MacroModelling.pruned_second_order_state_update(
            states, shock, past, length(states[1]), solution[1], solution[2])
    else
        stages = MacroModelling.pruned_third_order_state_update(
            states, shock, past, length(states[1]), solution[1], solution[2], solution[3])
    end
    stage_rows = reduce(vcat, (stage[keep_rows] for stage in stages))
    measurement_rows = [sum(stage[row] for stage in stages) for row in observables]
    return vcat(stage_rows, measurement_rows)
end

@testset "Pruned Ivashchenko Gaussian closure" begin
    opts = MacroModelling.merge_calculation_options()
    observables = [:c, :q]
    data = KeyedArray(zeros(2, 12); Variable = observables, Time = 1:12)
    parameters = RBC_pruned_ivashchenko_test.parameter_values

    for (algorithm, stages) in ((:pruned_second_order, 2), (:pruned_third_order, 3))
        MacroModelling.solve!(RBC_pruned_ivashchenko_test, algorithm = algorithm,
                              dynamics = true, opts = opts)
        _, _, solution, state, solved =
            MacroModelling.get_relevant_steady_state_and_state_update(
                Val(algorithm), parameters, RBC_pruned_ivashchenko_test, opts = opts)
        @test solved
        names = RBC_pruned_ivashchenko_test.constants.post_complete_parameters.SS_and_pars_names
        observables_index = convert(Vector{Int}, indexin(observables, names))
        sys = MacroModelling.build_pruned_ivashchenko_kalman_system_from_constants(
            RBC_pruned_ivashchenko_test.constants, solution, observables_index, algorithm)
        @test sys.pruned
        @test sys.past == collect(1:sys.nPast)
        @test sys.nStages == stages

        if stages == 2
            initial_mean = MacroModelling.pruned_ivashchenko_initial_mean(sys, state)
            moment_ws = MacroModelling.ivashchenko_kalman_workspace(sys, Float64)
            stationary_mean, stationary_covariance, initialized =
                MacroModelling.ivashchenko_stationary_initialization(
                    sys, initial_mean, :theoretical, moment_ws;
                    workspaces = RBC_pruned_ivashchenko_test.workspaces)
            @test initialized
            MacroModelling.ivashchenko_polynomial_moments!(
                sys, stationary_mean, stationary_covariance, moment_ws)
            @test moment_ws.mean[sys.state_position] ≈ stationary_mean atol = 1e-9
            @test moment_ws.covariance[sys.state_position, sys.state_position] ≈
                  stationary_covariance atol = 1e-9

            kollmann_sys = MacroModelling.build_quadratic_kalman_system_from_constants(
                RBC_pruned_ivashchenko_test.constants, solution[1], solution[2],
                observables_index)
            zbar = (Matrix{Float64}(I, kollmann_sys.nz, kollmann_sys.nz) -
                    kollmann_sys.𝒜) \ kollmann_sys.c
            kollmann_mean = vcat(kollmann_sys.P * zbar[kollmann_sys.r1],
                                 kollmann_sys.P * zbar[kollmann_sys.r2])
            @test stationary_mean ≈ kollmann_mean atol = 1e-9
        end

        rng = MersenneTwister(20260803 + stages)
        random_states = [randn(rng, length(state[1])) for _ in 1:stages]
        random_shock = randn(rng, sys.nExo)
        effective_input = vcat((random_states[stage][sys.original_past] for stage in 1:stages)...,
                               1.0, random_shock)
        effective_output = sys.effective_solution[1] * effective_input +
                           sys.effective_solution[2] *
                           MacroModelling.compressed_kron²_power(effective_input) / 2
        if stages == 3
            effective_output += sys.effective_solution[3] *
                                MacroModelling.compressed_kron³_power(effective_input) / 6
        end
        @test effective_output ≈ direct_pruned_output(
            solution, random_states, random_shock, sys.original_past,
            sys.keep_rows, observables_index)

        ivashchenko_ll = get_loglikelihood(
            RBC_pruned_ivashchenko_test, data, parameters;
            algorithm = algorithm, filter = :ivashchenko_kalman,
            initial_covariance = :diagonal, measurement_error = 1e-4)
        kollmann_filter = stages == 2 ? :quadratic_kalman : :cubic_kalman
        kollmann_ll = get_loglikelihood(
            RBC_pruned_ivashchenko_test, data, parameters;
            algorithm = algorithm, filter = kollmann_filter,
            initial_covariance = :diagonal, measurement_error = 1e-4)
        @test isfinite(ivashchenko_ll)
        @test isfinite(kollmann_ll)
    end

    second_objective(p) = get_loglikelihood(
        RBC_pruned_ivashchenko_test, data, p;
        algorithm = :pruned_second_order, filter = :ivashchenko_kalman,
        initial_covariance = :diagonal, measurement_error = 1e-4)
    forward_gradient = ForwardDiff.gradient(second_objective, parameters)
    reverse_gradient = Zygote.gradient(second_objective, parameters)[1]
    @test all(isfinite, reverse_gradient)
    @test isapprox(reverse_gradient, forward_gradient; rtol = 1e-8, atol = 1e-4)

    for closure in (:linearized, :diagonal)
        closure_objective(p) = get_loglikelihood(
            RBC_pruned_ivashchenko_test, data, p;
            algorithm = :pruned_second_order, filter = :ivashchenko_kalman,
            initial_covariance = :diagonal, measurement_error = 1e-4,
            ivashchenko_gaussian_closure = closure)
        closure_forward = ForwardDiff.gradient(closure_objective, parameters)
        closure_reverse = Zygote.gradient(closure_objective, parameters)[1]
        @test all(isfinite, closure_reverse)
        @test isapprox(closure_reverse, closure_forward; rtol = 1e-8, atol = 1e-4)
    end

    third_objective(p) = get_loglikelihood(
        RBC_pruned_ivashchenko_test, data, p;
        algorithm = :pruned_third_order, filter = :ivashchenko_kalman,
        initial_covariance = :diagonal, measurement_error = 1e-4)
    third_reverse = Zygote.gradient(third_objective, parameters)[1]
    h = 1e-5
    index = 6
    plus = copy(parameters); plus[index] += h
    minus = copy(parameters); minus[index] -= h
    third_finite_difference = (third_objective(plus) - third_objective(minus)) / (2h)
    @test all(isfinite, third_reverse)
    @test isapprox(third_reverse[index], third_finite_difference; rtol = 1e-4, atol = 1e-3)

    linearized_third_objective(p) = get_loglikelihood(
        RBC_pruned_ivashchenko_test, data, p;
        algorithm = :pruned_third_order, filter = :ivashchenko_kalman,
        initial_covariance = :diagonal, measurement_error = 1e-4,
        ivashchenko_gaussian_closure = :linearized)
    linearized_third_reverse = Zygote.gradient(
        linearized_third_objective, parameters)[1]
    @test all(isfinite, linearized_third_reverse)
    linearized_third_plus = copy(parameters); linearized_third_plus[index] += h
    linearized_third_minus = copy(parameters); linearized_third_minus[index] -= h
    linearized_third_finite_difference =
        (linearized_third_objective(linearized_third_plus) -
         linearized_third_objective(linearized_third_minus)) / (2h)
    @test isapprox(linearized_third_reverse[index], linearized_third_finite_difference;
                   rtol = 1e-4, atol = 1e-3)

    estimates = get_estimated_variables(
        RBC_pruned_ivashchenko_test, data;
        algorithm = :pruned_second_order, filter = :ivashchenko_kalman,
        initial_covariance = :diagonal, measurement_error = 1e-4,
        levels = false, smooth = false)
    @test size(estimates) == (5, size(data, 2))
    @test all(isfinite, collect(estimates))
    smooth_estimates = get_estimated_variables(
        RBC_pruned_ivashchenko_test, data;
        algorithm = :pruned_second_order, filter = :ivashchenko_kalman,
        initial_covariance = :diagonal, measurement_error = 1e-4,
        levels = false, smooth = true)
    @test size(smooth_estimates) == (5, size(data, 2))
    @test all(isfinite, collect(smooth_estimates))
end
