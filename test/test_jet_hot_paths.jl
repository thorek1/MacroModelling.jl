using Test
using MacroModelling

if VERSION < v"1.13"
    using JET
end

import MacroModelling: get_NSSS_and_parameters, calculate_jacobian, calculate_hessian,
    calculate_third_order_derivatives, calculate_first_order_solution,
    calculate_second_order_solution, calculate_third_order_solution,
    calculate_covariance, calculate_mean,
    calculate_second_order_moments, calculate_third_order_moments,
    calculate_second_order_moments_with_covariance,
    calculate_third_order_moments_with_autocorrelation,
    get_relevant_steady_state_and_state_update, irf_initial_state,
    calculate_stochastic_steady_state,
    solve_lyapunov_equation, solve_sylvester_equation, solve_quadratic_matrix_equation,
    filter_and_smooth,
    merge_calculation_options, initialise_constants!, CalculationOptions, Tolerances,
    lyapunov_workspace, sylvester_workspace, ensure_lyapunov_workspace!

using AxisKeys
using SparseArrays
import LinearAlgebra as ℒ

# ---------------------------------------------------------------------------
# Set up a small RBC model so we have concrete arguments for @report_call.
# This avoids the OOM problem of JET.test_package on the whole package.
# ---------------------------------------------------------------------------

@model RBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Solve at third order to force compilation of all derivative/solution functions
get_solution(RBC, algorithm = :third_order, silent = true)

params = copy(RBC.parameter_values)
opts   = merge_calculation_options()

# Populate NSSS so downstream calls have valid state
SS_and_pars, _ = get_NSSS_and_parameters(RBC, params, opts = opts)

# Jacobian
∇₁ = calculate_jacobian(params, SS_and_pars, RBC.caches, RBC.functions.jacobian, RBC.workspaces)

# First-order solution
constants_obj = initialise_constants!(RBC)
𝐒₁, qme_sol, _ = calculate_first_order_solution(∇₁, constants_obj, RBC.workspaces, RBC.caches;
    opts = opts, initial_guess = RBC.caches.qme_solution, parameter_values = params)

# Hessian (available after second+ order solve)
∇₂ = calculate_hessian(params, SS_and_pars, RBC.caches, RBC.functions.hessian, RBC.workspaces)

# Second-order solution
𝐒₂, _ = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, RBC.constants, RBC.workspaces, RBC.caches;
    opts = opts, parameter_values = params)

# Third-order derivatives (available after third order solve)
∇₃ = calculate_third_order_derivatives(params, SS_and_pars, RBC.caches,
    RBC.functions.third_order_derivatives, RBC.workspaces)

# ---------------------------------------------------------------------------
# JET analysis targets
# ---------------------------------------------------------------------------

# Collect ignored modules: filter out packages whose internals we do not own
const JET_TARGET_MODULES = (MacroModelling,)

# Helper: run JET.@report_call with target_modules filtering.
# On Julia < 1.11 the kwarg was `target_defined_modules`.
function jet_test_call(@nospecialize(f), @nospecialize(argtypes);
                        broken::Bool = false)
    if VERSION >= v"1.13"
        @test_skip "JET not supported on Julia ≥ 1.13 yet"
        return
    end
    result = if VERSION < v"1.11"
        JET.report_call(f, argtypes;
            target_defined_modules = true,
            toplevel_logger = nothing)
    else
        JET.report_call(f, argtypes;
            target_modules = JET_TARGET_MODULES,
            toplevel_logger = nothing)
    end
    reports = JET.get_reports(result)
    if broken
        @test_broken isempty(reports)
    else
        @test isempty(reports)
    end
    if !isempty(reports) && !broken
        @warn "JET reports for $(f)" reports
    end
end

@testset verbose = true "JET hot-path analysis" begin

    # ------------------------------------------------------------------
    @testset "get_NSSS_and_parameters" begin
        jet_test_call(get_NSSS_and_parameters,
            Tuple{typeof(RBC), Vector{Float64}})
    end

    # ------------------------------------------------------------------
    @testset "calculate_jacobian" begin
        jet_test_call(calculate_jacobian,
            Tuple{typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.jacobian),
                  typeof(RBC.workspaces)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_first_order_solution" begin
        jet_test_call(calculate_first_order_solution,
            Tuple{typeof(∇₁), typeof(constants_obj),
                  typeof(RBC.workspaces), typeof(RBC.caches)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_hessian" begin
        jet_test_call(calculate_hessian,
            Tuple{typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.hessian),
                  typeof(RBC.workspaces)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_second_order_solution" begin
        jet_test_call(calculate_second_order_solution,
            Tuple{typeof(∇₁), typeof(∇₂), typeof(𝐒₁),
                  typeof(RBC.constants), typeof(RBC.workspaces),
                  typeof(RBC.caches)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_third_order_derivatives" begin
        jet_test_call(calculate_third_order_derivatives,
            Tuple{typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.third_order_derivatives),
                  typeof(RBC.workspaces)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_covariance" begin
        jet_test_call(calculate_covariance,
            Tuple{typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_mean" begin
        jet_test_call(calculate_mean,
            Tuple{typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "calculate_second_order_moments" begin
        jet_test_call(calculate_second_order_moments,
            Tuple{typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "get_relevant_steady_state_and_state_update (first_order)" begin
        jet_test_call(get_relevant_steady_state_and_state_update,
            Tuple{Val{:first_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "get_relevant_steady_state_and_state_update (second_order)" begin
        jet_test_call(get_relevant_steady_state_and_state_update,
            Tuple{Val{:second_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "get_relevant_steady_state_and_state_update (pruned_second_order)" begin
        jet_test_call(get_relevant_steady_state_and_state_update,
            Tuple{Val{:pruned_second_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "get_relevant_steady_state_and_state_update (third_order)" begin
        jet_test_call(get_relevant_steady_state_and_state_update,
            Tuple{Val{:third_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    @testset "get_relevant_steady_state_and_state_update (pruned_third_order)" begin
        jet_test_call(get_relevant_steady_state_and_state_update,
            Tuple{Val{:pruned_third_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # High-level user-facing functions (estimation-style calls with parameters vector)
    # ------------------------------------------------------------------
    @testset "get_solution (parameters, first_order)" begin
        jet_test_call(MacroModelling.get_solution,
            Tuple{typeof(RBC), typeof(params)})
    end

    @testset "get_irf (parameters)" begin
        jet_test_call(MacroModelling.get_irf,
            Tuple{typeof(RBC), typeof(params)})
    end

    @testset "get_loglikelihood" begin
        # Construct minimal fake data with the right shape
        data = KeyedArray(randn(1, 40); Variables = [RBC.constants.post_model_macro.var[1]], Periods = 1:40)
        jet_test_call(MacroModelling.get_loglikelihood,
            Tuple{typeof(RBC), typeof(data), typeof(params)})
    end

    # ------------------------------------------------------------------
    # Third-order solution
    # ------------------------------------------------------------------
    @testset "calculate_third_order_solution" begin
        jet_test_call(calculate_third_order_solution,
            Tuple{typeof(∇₁), typeof(∇₂), typeof(∇₃),
                  typeof(𝐒₁), typeof(𝐒₂),
                  typeof(RBC.constants), typeof(RBC.workspaces),
                  typeof(RBC.caches)})
    end

    # ------------------------------------------------------------------
    # Second-order moments with covariance
    # ------------------------------------------------------------------
    @testset "calculate_second_order_moments_with_covariance" begin
        jet_test_call(calculate_second_order_moments_with_covariance,
            Tuple{typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # Third-order moments
    # ------------------------------------------------------------------
    @testset "calculate_third_order_moments" begin
        jet_test_call(calculate_third_order_moments,
            Tuple{typeof(params), Symbol, typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # Third-order moments with autocorrelation
    # ------------------------------------------------------------------
    @testset "calculate_third_order_moments_with_autocorrelation" begin
        jet_test_call(calculate_third_order_moments_with_autocorrelation,
            Tuple{typeof(params), Symbol, typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # Stochastic steady state (4 algorithm variants)
    # ------------------------------------------------------------------
    @testset "calculate_stochastic_steady_state (second_order)" begin
        jet_test_call(calculate_stochastic_steady_state,
            Tuple{Val{:second_order}, typeof(params), typeof(RBC)})
    end

    @testset "calculate_stochastic_steady_state (pruned_second_order)" begin
        jet_test_call(calculate_stochastic_steady_state,
            Tuple{Val{:pruned_second_order}, typeof(params), typeof(RBC)})
    end

    @testset "calculate_stochastic_steady_state (third_order)" begin
        jet_test_call(calculate_stochastic_steady_state,
            Tuple{Val{:third_order}, typeof(params), typeof(RBC)})
    end

    @testset "calculate_stochastic_steady_state (pruned_third_order)" begin
        jet_test_call(calculate_stochastic_steady_state,
            Tuple{Val{:pruned_third_order}, typeof(params), typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # Lyapunov equation solver (top-level dispatcher, dense doubling path)
    # ------------------------------------------------------------------
    @testset "solve_lyapunov_equation (dense, doubling)" begin
        lyap_ws = ensure_lyapunov_workspace!(RBC.workspaces, RBC.constants.post_model_macro.nVars, :first_order)
        n = RBC.constants.post_model_macro.nVars
        A_test = randn(n, n) * 0.5
        C_test = let X = randn(n, n); X * X'; end
        jet_test_call(solve_lyapunov_equation,
            Tuple{typeof(A_test), typeof(C_test), typeof(lyap_ws)})
    end

    # ------------------------------------------------------------------
    # Sylvester equation solver (top-level dispatcher)
    # ------------------------------------------------------------------
    @testset "solve_sylvester_equation" begin
        sylv_ws = RBC.workspaces.sylvester_1st_order
        n = 3
        A_sylv = randn(n, n) * 0.5
        B_sylv = randn(n, n) * 0.5
        C_sylv = randn(n, n)
        jet_test_call(solve_sylvester_equation,
            Tuple{typeof(A_sylv), typeof(B_sylv), typeof(C_sylv), typeof(sylv_ws)})
    end

    # ------------------------------------------------------------------
    # Quadratic matrix equation solver
    # ------------------------------------------------------------------
    @testset "solve_quadratic_matrix_equation" begin
        constants_qme = initialise_constants!(RBC)
        n = RBC.constants.post_model_macro.nVars - RBC.constants.post_model_macro.nPresent_only
        A_qme = randn(n, n)
        B_qme = randn(n, n)
        C_qme = randn(n, n)
        jet_test_call(solve_quadratic_matrix_equation,
            Tuple{typeof(A_qme), typeof(B_qme), typeof(C_qme),
                  typeof(RBC.constants), typeof(RBC.workspaces), typeof(RBC.caches)})
    end

    # ------------------------------------------------------------------
    # Kalman filter loglikelihood
    # ------------------------------------------------------------------
    @testset "calculate_loglikelihood (Kalman)" begin
        calculate_loglikelihood_fn = MacroModelling.calculate_loglikelihood
        obs_idx = [1]
        data_dev = randn(1, 40)
        state_vec = [zeros(RBC.constants.post_model_macro.nVars)]
        jet_test_call(calculate_loglikelihood_fn,
            Tuple{Val{:kalman}, Val{:first_order}, typeof(obs_idx),
                  Matrix{Float64}, typeof(data_dev),
                  typeof(RBC.constants), typeof(state_vec), typeof(RBC.workspaces)})
    end

    # ------------------------------------------------------------------
    # Inversion filter loglikelihood (first order)
    # ------------------------------------------------------------------
    @testset "calculate_loglikelihood (Inversion, first_order)" begin
        calculate_loglikelihood_fn = MacroModelling.calculate_loglikelihood
        obs_idx = [1]
        data_dev = randn(1, 40)
        state_vec = [zeros(RBC.constants.post_model_macro.nVars)]
        jet_test_call(calculate_loglikelihood_fn,
            Tuple{Val{:inversion}, Val{:first_order}, typeof(obs_idx),
                  Matrix{Float64}, typeof(data_dev),
                  typeof(RBC.constants), typeof(state_vec), typeof(RBC.workspaces)})
    end

    # ------------------------------------------------------------------
    # Filter and smooth (Durbin-Koopman)
    # ------------------------------------------------------------------
    @testset "filter_and_smooth" begin
        obs_syms = [RBC.constants.post_model_macro.var[1]]
        data_fs = randn(1, 40)
        jet_test_call(filter_and_smooth,
            Tuple{typeof(RBC), typeof(data_fs), typeof(obs_syms)})
    end

    # ------------------------------------------------------------------
    # Conditional variance decomposition (user-facing)
    # ------------------------------------------------------------------
    @testset "get_conditional_variance_decomposition" begin
        jet_test_call(MacroModelling.get_conditional_variance_decomposition,
            Tuple{typeof(RBC)})
    end

    # ------------------------------------------------------------------
    # Shock decomposition (user-facing)
    # ------------------------------------------------------------------
    @testset "get_shock_decomposition" begin
        data_sd = KeyedArray(randn(1, 40); Variables = [RBC.constants.post_model_macro.var[1]], Periods = 1:40)
        jet_test_call(MacroModelling.get_shock_decomposition,
            Tuple{typeof(RBC), typeof(data_sd)})
    end

    # ------------------------------------------------------------------
    # Conditional forecast (user-facing)
    # ------------------------------------------------------------------
    @testset "get_conditional_forecast" begin
        cond_mat = Matrix{Union{Nothing,Float64}}(nothing, RBC.constants.post_model_macro.nVars, 5)
        cond_mat[1, 1] = 0.01
        jet_test_call(MacroModelling.get_conditional_forecast,
            Tuple{typeof(RBC), typeof(cond_mat)})
    end

    # ------------------------------------------------------------------
    # get_solution with higher-order algorithms
    # ------------------------------------------------------------------
    @testset "get_solution (parameters, second_order)" begin
        jet_test_call(MacroModelling.get_solution,
            Tuple{typeof(RBC), typeof(params)})
    end

    # ------------------------------------------------------------------
    # ChainRules rrules for key functions (AD hot paths)
    # ------------------------------------------------------------------
    @testset "rrule: calculate_jacobian" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_jacobian), typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.jacobian),
                  typeof(RBC.workspaces)})
    end

    @testset "rrule: calculate_hessian" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_hessian), typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.hessian),
                  typeof(RBC.workspaces)})
    end

    @testset "rrule: calculate_third_order_derivatives" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_third_order_derivatives), typeof(params), typeof(SS_and_pars),
                  typeof(RBC.caches), typeof(RBC.functions.third_order_derivatives),
                  typeof(RBC.workspaces)})
    end

    @testset "rrule: get_NSSS_and_parameters" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(get_NSSS_and_parameters), typeof(RBC), typeof(params)})
    end

    @testset "rrule: calculate_first_order_solution" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_first_order_solution), typeof(∇₁),
                  typeof(constants_obj), typeof(RBC.workspaces), typeof(RBC.caches)})
    end

    @testset "rrule: calculate_second_order_solution" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_second_order_solution), typeof(∇₁), typeof(∇₂), typeof(𝐒₁),
                  typeof(RBC.constants), typeof(RBC.workspaces), typeof(RBC.caches)})
    end

    @testset "rrule: calculate_covariance" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_covariance), typeof(params), typeof(RBC)})
    end

    @testset "rrule: calculate_mean" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_mean), typeof(params), typeof(RBC)})
    end

    @testset "rrule: calculate_second_order_moments" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(calculate_second_order_moments), typeof(params), typeof(RBC)})
    end

    @testset "rrule: get_relevant_steady_state_and_state_update (first_order)" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(get_relevant_steady_state_and_state_update),
                  Val{:first_order}, typeof(params), typeof(RBC)})
    end

    @testset "rrule: get_irf" begin
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(MacroModelling.get_irf), typeof(RBC), typeof(params)})
    end

    @testset "rrule: get_loglikelihood" begin
        data_rl = KeyedArray(randn(1, 40); Variables = [RBC.constants.post_model_macro.var[1]], Periods = 1:40)
        jet_test_call(MacroModelling.rrule,
            Tuple{typeof(MacroModelling.get_loglikelihood), typeof(RBC), typeof(data_rl), typeof(params)})
    end

end
