# Benchmark script for NSSS calculation with SW07 model

# Add BenchmarkTools if needed
import Pkg
if !haskey(Pkg.project().dependencies, "BenchmarkTools")
    Pkg.add("BenchmarkTools")
end

using MacroModelling
using BenchmarkTools
import MacroModelling: clear_solution_caches!

include("models/Smets_Wouters_2007.jl")

model = Smets_Wouters_2007

println("=== Benchmarking NSSS calculation for Smets_Wouters_2007 ===")
println()

# Warm-up to ensure NSSS solver infrastructure and initial cache are available
println("Warming up...")
get_steady_state(model, derivatives = false)
println("✓ Warm-up complete")
println()

# Benchmark
println("Running benchmark...")
trial = @benchmark begin
    get_steady_state($model, parameters = $model.parameter_values, derivatives = false)
end setup = clear_solution_caches!($model,:first_order) samples=100 seconds=60

println()
println("=== Results ===")
println(trial)
println()
println("Minimum time:   ", minimum(trial).time / 1e6, " ms")
println("Median time:    ", median(trial).time / 1e6, " ms")
println("Mean time:      ", mean(trial).time / 1e6, " ms")
println()
println("Minimum memory: ", minimum(trial).memory / 1024, " KB")
println("Median memory:  ", median(trial).memory / 1024, " KB")
println("Mean memory:    ", mean(trial).memory / 1024, " KB")
println()
println("Allocations:    ", minimum(trial).allocs)
