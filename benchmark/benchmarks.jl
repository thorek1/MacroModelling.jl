
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.evals = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Define a parent BenchmarkGroup to contain our SUITE
const SUITE = BenchmarkGroup()

# Add some child groups to our benchmark SUITE.


# SUITE["FS2000"]["load_time"] = @elapsed using MacroModelling
import LinearAlgebra as ℒ
using MacroModelling
import MatrixEquations
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters, calculate_jacobian, merge_calculation_options, solve_lyapunov_equation, ℳ

# Check if new workspace API is available (not present in old package versions)
const HAS_WORKSPACE_API = isdefined(MacroModelling, :Lyapunov_workspace)

# Conditionally import workspace types only if they exist
if HAS_WORKSPACE_API
    import MacroModelling: Lyapunov_workspace, lyapunov_workspace, ensure_lyapunov_workspace!, ensure_qme_workspace!
end

# Version-aware wrapper for solve_lyapunov_equation benchmarking
# For new API: uses pre-allocated workspace for true benchmark of workspace reuse
# For old API: calls without workspace argument
function solve_lyapunov_for_bench(A, C, lyap_ws; lyapunov_algorithm::Symbol = :doubling)
    if HAS_WORKSPACE_API
        # New API - reuse pre-allocated workspace (shows benefit of workspace caching)
        return solve_lyapunov_equation(A, C, lyap_ws; lyapunov_algorithm = lyapunov_algorithm)
    else
        # Old API - no workspace argument
        return solve_lyapunov_equation(A, C; lyapunov_algorithm = lyapunov_algorithm)
    end
end

function timings_for_bench(𝓂::ℳ)
    if hasproperty(𝓂, :timings)
        out = 𝓂.timings
    else
        out = 𝓂.constants.post_model_macro
    end
    return out
end

function first_order_solution_for_bench(∇₁::AbstractMatrix, 𝓂::ℳ; opts = merge_calculation_options())
    if HAS_WORKSPACE_API
        qme_ws = ensure_qme_workspace!(𝓂)
        sylv_ws = 𝓂.workspaces.sylvester_1st_order
        out = calculate_first_order_solution(∇₁, 𝓂.constants, qme_ws, sylv_ws; opts = opts)
    else
        out = calculate_first_order_solution(∇₁; T = timings_for_bench(𝓂), opts = opts)
    end
    return out
end

function calculate_jacobian_for_bench(parameters, SS_and_pars, 𝓂::ℳ)
    return calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces)
end


function run_benchmarks!(𝓂::ℳ, SUITE::BenchmarkGroup)
    SUITE[𝓂.model_name] = BenchmarkGroup()

    get_irf(𝓂)
    # SUITE[𝓂.model_name]["ttfx_irf"] = BenchmarkTools.Trial(BenchmarkTools.Parameters(seconds=0,samples=1,evals=1,overhead=0,gctrial=false,gcsample=false),[@elapsed get_irf(𝓂)],zeros(1),0,0)

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

    T = timings_for_bench(𝓂)

    A = @views sol[:, 1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]

    C = @views sol[:, T.nPast_not_future_and_mixed+1:end]

    CC = C * C'
    
    # Create workspace once before benchmarks (new API) or use nothing (old API)
    # For new API: pre-allocated workspace shows benefit of workspace caching
    # For old API: workspace is not used
    lyap_ws = HAS_WORKSPACE_API ? Lyapunov_workspace(size(A, 1)) : nothing
    
    # Warm up call
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


# SUITE["trig"] = BenchmarkGroup(["math", "triangles"])
# SUITE["dot"] = BenchmarkGroup(["broadcast", "elementwise"])

# This string will be the same every time because we're seeding the RNG
# teststr = join(rand(MersenneTwister(1), 'a':'d', 10^4))

# Add some benchmarks to the "string" group
# SUITE["string"]["replace"] = @benchmarkable replace($teststr, "a", "b") seconds = Float64(π)
# SUITE["string"]["join"] = @benchmarkable join($teststr, $teststr) samples = 42

# Add some benchmarks to the "trig"/"dot" group
# for f in (sin, cos, tan)
#     for x in (0.0, pi)
#         SUITE["trig"][string(f), x] = @benchmarkable $(f)($x)
#         SUITE["dot"][string(f), x] = @benchmarkable $(f).([$x, $x, $x])
#     end
# end

# If a caches of tuned parameters already exists, use it, otherwise, tune and caches
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `SUITE` every time the file is included.
# paramspath = joinpath(dirname(@__FILE__), "params.json")

# if isfile(paramspath)
#     loadparams!(SUITE, BenchmarkTools.load(paramspath)[1], :evals)
# else
#     tune!(SUITE)
#     BenchmarkTools.save(paramspath, params(SUITE))
# end
