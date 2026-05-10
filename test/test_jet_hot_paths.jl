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
    get_relevant_steady_state_and_state_update, irf_initial_state,
    merge_calculation_options, initialise_constants!, CalculationOptions, Tolerances

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
        using AxisKeys
        data = KeyedArray(randn(1, 40); Variables = [RBC.constants.post_model_macro.var[1]], Periods = 1:40)
        jet_test_call(MacroModelling.get_loglikelihood,
            Tuple{typeof(RBC), typeof(data), typeof(params)})
    end

end
