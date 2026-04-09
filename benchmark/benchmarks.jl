using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.evals = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Define a parent BenchmarkGroup to contain our SUITE
const SUITE = BenchmarkGroup()

import LinearAlgebra as ℒ
using MacroModelling
import MatrixEquations
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters, ℳ, merge_calculation_options

# ──────────────────────────────────────────────────────────────────────────────
# Version detection
# ──────────────────────────────────────────────────────────────────────────────
# Three API levels:
#   OLD_API       — v0.1.39-41: keyword-based, 𝓂.timings, no workspaces
#   INTERMEDIATE  — v0.1.46: positional args, qme_ws/sylv_ws, 4-arg jacobian
#   LATEST        — current HEAD: positional args, workspaces/caches, 5-arg jacobian
const HAS_WORKSPACE_API  = isdefined(MacroModelling, :Lyapunov_workspace)
const HAS_QME_WS         = isdefined(MacroModelling, :ensure_qme_workspace!)

# ──────────────────────────────────────────────────────────────────────────────
# Version-branched imports and wrapper functions
# ──────────────────────────────────────────────────────────────────────────────
# Each wrapper performs the SAME computation regardless of version,
# just calling through the appropriate internal API.

if HAS_WORKSPACE_API
    import MacroModelling: Lyapunov_workspace, lyapunov_workspace, solve_lyapunov_equation
end

if HAS_QME_WS
    # v0.1.46: has ensure_qme/sylvester workspace helpers
    import MacroModelling: ensure_qme_workspace!, ensure_sylvester_1st_order_workspace!
end

# --- get_timings: extract the model timing/sizing info ---
if HAS_WORKSPACE_API
    # v0.1.46+ stores timings in constants.post_model_macro
    get_timings(𝓂::ℳ) = 𝓂.constants.post_model_macro
else
    # v0.1.39-41 stores timings directly on the model
    get_timings(𝓂::ℳ) = 𝓂.timings
end

# --- calculate_jacobian_for_bench ---
if !HAS_WORKSPACE_API
    # v0.1.39-41: calculate_jacobian(params, ss, 𝓂)
    function calculate_jacobian_for_bench(parameters, SS_and_pars, 𝓂::ℳ)
        return calculate_jacobian(parameters, SS_and_pars, 𝓂)
    end
elseif HAS_QME_WS
    # v0.1.46: calculate_jacobian(params, ss, caches, jacobian_funcs)  — 4 args
    function calculate_jacobian_for_bench(parameters, SS_and_pars, 𝓂::ℳ)
        return calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)
    end
else
    # Current: calculate_jacobian(params, ss, caches, jacobian_funcs, workspaces)  — 5 args
    function calculate_jacobian_for_bench(parameters, SS_and_pars, 𝓂::ℳ)
        return calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces; caching = false)
    end
end

# --- first_order_solution_for_bench ---
if !HAS_WORKSPACE_API
    # v0.1.39-41: keyword-based API
    function first_order_solution_for_bench(∇₁::AbstractMatrix, 𝓂::ℳ; opts = merge_calculation_options())
        T = get_timings(𝓂)
        return calculate_first_order_solution(∇₁; T = T, opts = opts)
    end
elseif HAS_QME_WS
    # v0.1.46: positional (∇₁, constants, qme_ws, sylv_ws)
    function first_order_solution_for_bench(∇₁::AbstractMatrix, 𝓂::ℳ; opts = merge_calculation_options())
        qme_ws  = ensure_qme_workspace!(𝓂)
        sylv_ws = ensure_sylvester_1st_order_workspace!(𝓂)
        return calculate_first_order_solution(∇₁, 𝓂.constants, qme_ws, sylv_ws; opts = opts)
    end
else
    # Current: positional (∇₁, constants, workspaces, caches)
    function first_order_solution_for_bench(∇₁::AbstractMatrix, 𝓂::ℳ; opts = merge_calculation_options())
        return calculate_first_order_solution(∇₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches; opts = opts, caching = false)
    end
end

# --- solve_lyapunov_for_bench ---
if HAS_WORKSPACE_API
    function solve_lyapunov_for_bench(A, C, lyap_ws; lyapunov_algorithm::Symbol = :doubling)
        return solve_lyapunov_equation(A, C, lyap_ws; lyapunov_algorithm = lyapunov_algorithm)
    end
else
    function solve_lyapunov_for_bench(A, C, ::Nothing; lyapunov_algorithm::Symbol = :doubling)
        return solve_lyapunov_equation(A, C; lyapunov_algorithm = lyapunov_algorithm)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark function
# ──────────────────────────────────────────────────────────────────────────────
function run_benchmarks!(𝓂::ℳ, SUITE::BenchmarkGroup)
    SUITE[𝓂.model_name] = BenchmarkGroup()

    # --- IRF (high-level, works on all versions) ---
    get_irf(𝓂)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["irf"] = @benchmarkable get_irf($𝓂) setup = clear_solution_caches!($𝓂, :first_order)

    # --- NSSS ---
    reference_steady_state, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["NSSS"] = @benchmarkable get_NSSS_and_parameters($𝓂, $𝓂.parameter_values) setup = clear_solution_caches!($𝓂, :first_order)

    # --- Jacobian ---
    ∇₁ = calculate_jacobian_for_bench(𝓂.parameter_values, reference_steady_state, 𝓂)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["jacobian"] = @benchmarkable calculate_jacobian_for_bench($𝓂.parameter_values, $reference_steady_state, $𝓂) setup = clear_solution_caches!($𝓂, :first_order)

    # --- QME (first-order solution) ---
    SUITE[𝓂.model_name]["qme"] = BenchmarkGroup()

    qme_schur_opts    = merge_calculation_options(quadratic_matrix_equation_algorithm = :schur)
    qme_doubling_opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :doubling)

    sol, qme_sol, solved = first_order_solution_for_bench(∇₁, 𝓂; opts = qme_schur_opts)
    clear_solution_caches!(𝓂, :first_order)

    SUITE[𝓂.model_name]["qme"]["schur"]    = @benchmarkable first_order_solution_for_bench($∇₁, $𝓂; opts = $qme_schur_opts) setup = clear_solution_caches!($𝓂, :first_order)
    SUITE[𝓂.model_name]["qme"]["doubling"] = @benchmarkable first_order_solution_for_bench($∇₁, $𝓂; opts = $qme_doubling_opts) setup = clear_solution_caches!($𝓂, :first_order)

    # --- Lyapunov equation ---
    T = get_timings(𝓂)

    A = @views sol[:, 1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]
    C = @views sol[:, T.nPast_not_future_and_mixed+1:end]
    CC = C * C'

    lyap_ws = HAS_WORKSPACE_API ? Lyapunov_workspace(size(A, 1)) : nothing

    # Warm up call
    solve_lyapunov_for_bench(A, CC, lyap_ws)

    SUITE[𝓂.model_name]["lyapunov"] = BenchmarkGroup()
    SUITE[𝓂.model_name]["lyapunov"]["doubling"]         = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :doubling)
    SUITE[𝓂.model_name]["lyapunov"]["bartels_stewart"]   = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :bartels_stewart)
    SUITE[𝓂.model_name]["lyapunov"]["bicgstab"]          = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :bicgstab)
    SUITE[𝓂.model_name]["lyapunov"]["gmres"]             = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :gmres)

    # --- Covariance (high-level, works on all versions) ---
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["covariance"] = @benchmarkable get_covariance($𝓂) setup = clear_solution_caches!($𝓂, :first_order)
end


include("../models/FS2000.jl")
run_benchmarks!(FS2000, SUITE)

include("../models/NAWM_EAUS_2008.jl")
run_benchmarks!(NAWM_EAUS_2008, SUITE)

include("../models/Smets_Wouters_2007.jl")
run_benchmarks!(Smets_Wouters_2007, SUITE)
