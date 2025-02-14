
using BenchmarkTools

# Define a parent BenchmarkGroup to contain our SUITE
const SUITE = BenchmarkGroup()

# Add some child groups to our benchmark SUITE.

SUITE["FS2000"] = BenchmarkGroup()

# SUITE["FS2000"]["load_time"] = @elapsed using MacroModelling
using MacroModelling
import MacroModelling: clear_solution_caches!, get_NSSS_and_parameters

# SUITE["FS2000"]["ttfx_excl_load_time"] = @elapsed include("../models/FS2000.jl")
include("../models/FS2000.jl")
model = FS2000

# SUITE["FS2000"]["ttfx_irf"] = @elapsed get_irf(model)
get_irf(model)

SUITE["FS2000"]["irf"] = @benchmarkable get_irf($model) setup = clear_solution_caches!($model, :first_order)

SUITE["FS2000"]["NSSS"] = @benchmarkable get_NSSS_and_parameters($model, $model.parameter_values) setup = clear_solution_caches!($model, :first_order)


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