
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
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters, calculate_jacobian, merge_calculation_options, solve_lyapunov_equation, ℳ


function run_benchmarks!(𝓂::ℳ, SUITE::BenchmarkGroup)
    SUITE[𝓂.model_name] = BenchmarkGroup()

    get_irf(𝓂)
    # SUITE[𝓂.model_name]["ttfx_irf"] = BenchmarkTools.Trial(BenchmarkTools.Parameters(seconds=0,samples=1,evals=1,overhead=0,gctrial=false,gcsample=false),[@elapsed get_irf(𝓂)],zeros(1),0,0)

    clear_solution_caches!(𝓂, :first_order)
    
    SUITE[𝓂.model_name]["irf"] = @benchmarkable get_irf($𝓂) setup = clear_solution_caches!($𝓂, :first_order)
    
    reference_steady_state, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)
    
    clear_solution_caches!(𝓂, :first_order)
    
    SUITE[𝓂.model_name]["NSSS"] = @benchmarkable get_NSSS_and_parameters($𝓂, $𝓂.parameter_values) setup = clear_solution_caches!($𝓂, :first_order)
    
    
    ∇₁ = calculate_jacobian(𝓂.parameter_values, reference_steady_state, 𝓂)
    
    clear_solution_caches!(𝓂, :first_order)
    
    SUITE[𝓂.model_name]["jacobian"] = @benchmarkable calculate_jacobian($𝓂.parameter_values, $reference_steady_state, $𝓂) setup = clear_solution_caches!($𝓂, :first_order)
    
    
    SUITE[𝓂.model_name]["qme"] = BenchmarkGroup()
    
    sol, qme_sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings, opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :schur), 𝒬ℂ = 𝓂.caches.qme_caches)
    
    clear_solution_caches!(𝓂, :first_order)
    
    SUITE[𝓂.model_name]["qme"]["schur"] = @benchmarkable calculate_first_order_solution($∇₁; T = $𝓂.timings, opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :schur), 𝒬ℂ = $𝓂.caches.qme_caches) setup = clear_solution_caches!($𝓂, :first_order)
    
    SUITE[𝓂.model_name]["qme"]["doubling"] = @benchmarkable calculate_first_order_solution($∇₁; T = $𝓂.timings, opts = merge_calculation_options(quadratic_matrix_equation_algorithm = :doubling), 𝒬ℂ = $𝓂.caches.qme_caches) setup = clear_solution_caches!($𝓂, :first_order)
    
    
    A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
    
    C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'
    
    solve_lyapunov_equation(A, CC)
    
    SUITE[𝓂.model_name]["lyapunov"] = BenchmarkGroup()
    SUITE[𝓂.model_name]["lyapunov"]["doubling"] = @benchmarkable solve_lyapunov_equation($A, $CC, lyapunov_algorithm = :doubling) # setup = clear_solution_caches!($𝓂, :first_order)
    SUITE[𝓂.model_name]["lyapunov"]["bartels_stewart"] = @benchmarkable solve_lyapunov_equation($A, $CC, lyapunov_algorithm = :bartels_stewart) # setup = clear_solution_caches!($𝓂, :first_order)
    SUITE[𝓂.model_name]["lyapunov"]["bicgstab"] = @benchmarkable solve_lyapunov_equation($A, $CC, lyapunov_algorithm = :bicgstab) # setup = clear_solution_caches!($𝓂, :first_order)
    SUITE[𝓂.model_name]["lyapunov"]["gmres"] = @benchmarkable solve_lyapunov_equation($A, $CC, lyapunov_algorithm = :gmres) # setup = clear_solution_caches!($𝓂, :first_order)
    
    
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

# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `SUITE` every time the file is included.
# paramspath = joinpath(dirname(@__FILE__), "params.json")

# if isfile(paramspath)
#     loadparams!(SUITE, BenchmarkTools.load(paramspath)[1], :evals)
# else
#     tune!(SUITE)
#     BenchmarkTools.save(paramspath, params(SUITE))
# end