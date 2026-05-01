using MacroModelling, Zygote, FiniteDifferences, AxisKeys, DelimitedFiles
import LinearAlgebra as ℒ

include("../models/FS2000.jl")

cd("test") do
    global dat, header
    dat, header = readdlm("data/FS2000_data.csv", ',', header = true)
end
dat = Float64.(dat)
names = vec(header)
data = KeyedArray(dat', Variable = Symbol.("log_".*names), Time = axes(dat, 1))
data = log.(data)
observables = sort(Symbol.("log_".*names))
data = data(observables,:)

dat_nan = Matrix{Float64}(collect(data))
# Full pattern: partial (wide m=1) AND fully-missing (m=0)
dat_nan[1, 25] = NaN
dat_nan[2, 47] = NaN
dat_nan[:, 30] .= NaN
dat_nan[:, 31] .= NaN
dat_nan[:, 60] .= NaN

data_nan = KeyedArray(dat_nan, Variable = collect(axiskeys(data, 1)), Time = axes(dat_nan, 2))

println("--- Forward primal (missing pruned 2nd) ---")
ll_nan = get_loglikelihood(FS2000, data_nan, FS2000.parameter_values; algorithm = :pruned_second_order, filter = :inversion)
println("ll_nan = ", ll_nan)
@assert isfinite(ll_nan)

g_zy = nothing
println("--- Zygote pullback (pruned 2nd missing) ---")
f = x -> get_loglikelihood(FS2000, data_nan, x; algorithm = :pruned_second_order, filter = :inversion)
g_zy = nothing
g_zy = nothing
try
    global g_zy = Zygote.gradient(f, FS2000.parameter_values)[1]
    println("Zygote OK, norm = ", ℒ.norm(g_zy))
catch e
    println("Zygote FAILED: ", e)
    rethrow(e)
end

println("--- FD reference ---")
g_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), f, FS2000.parameter_values)[1]
println("FD norm = ", ℒ.norm(g_fd))
println("g_zy types: ", typeof(g_zy))
g_zy_clean = [v === nothing ? 0.0 : v for v in g_zy]
diff = g_zy_clean .- g_fd
relerr = ℒ.norm(diff) / max(ℒ.norm(g_fd), eps())
println("max abs diff = ", maximum(abs, diff))
println("max rel diff per-component = ", maximum(abs.(diff) ./ (abs.(g_fd) .+ 1e-8)))
println("rel norm err = ", relerr)
for (i, (a, b)) in enumerate(zip(g_zy_clean, g_fd))
    if abs(a - b) > 1e-3 * max(abs(b), 1.0)
        println("  param[$i]: zy=$a   fd=$b   diff=$(a-b)")
    end
end

if relerr < 1e-4
    println("PASS")
else
    println("FAIL")
    exit(1)
end
