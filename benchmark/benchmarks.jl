using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.evals = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

const SUITE = BenchmarkGroup()

import LinearAlgebra as ℒ
using MacroModelling
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters, solve_lyapunov_equation, ℳ, merge_calculation_options

# Workspace-enabled Lyapunov API exists in v0.1.46+.
const HAS_WORKSPACE_API = isdefined(MacroModelling, :Lyapunov_workspace)
if HAS_WORKSPACE_API
    import MacroModelling: Lyapunov_workspace
    import MatrixEquations
end

# Timings live in different places across versions.
function get_timings(𝓂::ℳ)
    if hasproperty(𝓂, :constants) && hasproperty(𝓂.constants, :post_model_macro)
        return 𝓂.constants.post_model_macro
    end
    return 𝓂.timings
end

has_model_field(𝓂::ℳ, field::Symbol) = hasfield(typeof(𝓂), field)
has_nested_field(obj, field::Symbol) = hasfield(typeof(obj), field)

# Dispatch to the matching jacobian API by model layout first.
function calculate_jacobian_for_bench(parameters, SS_and_pars, 𝓂::ℳ)
    if has_model_field(𝓂, :caches) && has_model_field(𝓂, :functions) &&
       has_nested_field(getfield(𝓂, :functions), :jacobian)
        caches_obj = getfield(𝓂, :caches)
        jacobian_funcs = getfield(getfield(𝓂, :functions), :jacobian)

        if has_model_field(𝓂, :workspaces)
            workspaces_obj = getfield(𝓂, :workspaces)
            clear_solution_caches!(𝓂, :first_order)
            return calculate_jacobian(parameters,
                                      SS_and_pars,
                                      caches_obj,
                                      jacobian_funcs,
                                      workspaces_obj;
                                      caching = false)
        end

        if hasmethod(calculate_jacobian,
                     Tuple{typeof(parameters), typeof(SS_and_pars), typeof(caches_obj), typeof(jacobian_funcs)})
            clear_solution_caches!(𝓂, :first_order)
            return calculate_jacobian(parameters, SS_and_pars, caches_obj, jacobian_funcs)
        end
    end

    if hasmethod(calculate_jacobian, Tuple{typeof(parameters), typeof(SS_and_pars), typeof(𝓂)})
        clear_solution_caches!(𝓂, :first_order)
        return calculate_jacobian(parameters, SS_and_pars, 𝓂)
    end

    error("No supported calculate_jacobian benchmark API found for $(typeof(𝓂)).")
end

# Dispatch to the matching first-order API by model layout first, then helper availability.
function first_order_solution_for_bench(∇₁::AbstractMatrix, 𝓂::ℳ; opts = merge_calculation_options())
    if has_model_field(𝓂, :constants) && has_model_field(𝓂, :workspaces) && has_model_field(𝓂, :caches)
        constants_obj = getfield(𝓂, :constants)
        workspaces_obj = getfield(𝓂, :workspaces)
        caches_obj = getfield(𝓂, :caches)

        if hasmethod(calculate_first_order_solution,
                     Tuple{typeof(∇₁), typeof(constants_obj), typeof(workspaces_obj), typeof(caches_obj)})
            return calculate_first_order_solution(∇₁,
                                                  constants_obj,
                                                  workspaces_obj,
                                                  caches_obj;
                                                  opts = opts,
                                                  caching = false)
        end
    end

    if has_model_field(𝓂, :constants) && isdefined(MacroModelling, :ensure_qme_workspace!) && isdefined(MacroModelling, :ensure_sylvester_1st_order_workspace!)
        constants_obj = getfield(𝓂, :constants)
        qme_ws_fn = getfield(MacroModelling, :ensure_qme_workspace!)
        sylv_ws_fn = getfield(MacroModelling, :ensure_sylvester_1st_order_workspace!)
        qme_ws = qme_ws_fn(𝓂)
        sylv_ws = sylv_ws_fn(𝓂)
        if hasmethod(calculate_first_order_solution,
                     Tuple{typeof(∇₁), typeof(constants_obj), typeof(qme_ws), typeof(sylv_ws)})
            return calculate_first_order_solution(∇₁, constants_obj, qme_ws, sylv_ws; opts = opts)
        end
    end

    T = get_timings(𝓂)
    return calculate_first_order_solution(∇₁; T = T, opts = opts)
end

if HAS_WORKSPACE_API
    function solve_lyapunov_for_bench(A, C, lyap_ws; lyapunov_algorithm::Symbol = :doubling)
        return solve_lyapunov_equation(A, C, lyap_ws; lyapunov_algorithm = lyapunov_algorithm)
    end
else
    function solve_lyapunov_for_bench(A, C, ::Nothing; lyapunov_algorithm::Symbol = :doubling)
        return solve_lyapunov_equation(A, C; lyapunov_algorithm = lyapunov_algorithm)
    end
end

function run_benchmarks!(𝓂::ℳ, SUITE::BenchmarkGroup)
    SUITE[𝓂.model_name] = BenchmarkGroup()

    get_irf(𝓂)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["irf"] = @benchmarkable get_irf($𝓂) setup = clear_solution_caches!($𝓂, :first_order)

    reference_steady_state, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["NSSS"] = @benchmarkable get_NSSS_and_parameters($𝓂, $𝓂.parameter_values) setup = clear_solution_caches!($𝓂, :first_order)

    ∇₁ = calculate_jacobian_for_bench(𝓂.parameter_values, reference_steady_state, 𝓂)
    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["jacobian"] = @benchmarkable calculate_jacobian_for_bench($𝓂.parameter_values, $reference_steady_state, $𝓂) setup = clear_solution_caches!($𝓂, :first_order)

    SUITE[𝓂.model_name]["qme"] = BenchmarkGroup()

    qme_schur_opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :schur)
    qme_doubling_opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :doubling)

    sol, qme_sol, solved = first_order_solution_for_bench(∇₁, 𝓂; opts = qme_schur_opts)
    clear_solution_caches!(𝓂, :first_order)

    SUITE[𝓂.model_name]["qme"]["schur"] = @benchmarkable first_order_solution_for_bench($∇₁, $𝓂; opts = $qme_schur_opts) setup = clear_solution_caches!($𝓂, :first_order)
    SUITE[𝓂.model_name]["qme"]["doubling"] = @benchmarkable first_order_solution_for_bench($∇₁, $𝓂; opts = $qme_doubling_opts) setup = clear_solution_caches!($𝓂, :first_order)

    T = get_timings(𝓂)
    A = @views sol[:, 1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]
    C = @views sol[:, T.nPast_not_future_and_mixed+1:end]
    CC = C * C'

    lyap_ws = HAS_WORKSPACE_API ? Lyapunov_workspace(size(A, 1)) : nothing
    solve_lyapunov_for_bench(A, CC, lyap_ws)

    SUITE[𝓂.model_name]["lyapunov"] = BenchmarkGroup()
    SUITE[𝓂.model_name]["lyapunov"]["doubling"] = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :doubling)
    SUITE[𝓂.model_name]["lyapunov"]["bartels_stewart"] = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :bartels_stewart)
    SUITE[𝓂.model_name]["lyapunov"]["bicgstab"] = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :bicgstab)
    SUITE[𝓂.model_name]["lyapunov"]["gmres"] = @benchmarkable solve_lyapunov_for_bench($A, $CC, $lyap_ws, lyapunov_algorithm = :gmres)

    clear_solution_caches!(𝓂, :first_order)
    SUITE[𝓂.model_name]["covariance"] = @benchmarkable get_covariance($𝓂) setup = clear_solution_caches!($𝓂, :first_order)
end

include("../models/FS2000.jl")
run_benchmarks!(FS2000, SUITE)

include("../models/NAWM_EAUS_2008.jl")
run_benchmarks!(NAWM_EAUS_2008, SUITE)

include("../models/Smets_Wouters_2007.jl")
run_benchmarks!(Smets_Wouters_2007, SUITE)
