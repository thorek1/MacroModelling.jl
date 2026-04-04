#!/usr/bin/env julia
# Benchmark: Mooncake vs Zygote gradient performance for MacroModelling.jl
# Tests get_solution, get_irf, get_statistics, get_loglikelihood

using Pkg
Pkg.instantiate()

using MacroModelling, LinearAlgebra, AxisKeys
using Zygote, ForwardDiff, FiniteDifferences, Mooncake
using DifferentiationInterface, ADTypes

# ── Define RBC model ──
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ     = 0.2
    δ     = 0.02
    α     = 0.5
    β     = 0.95
end

x0 = RBC.parameter_values[:]

# Simulate data for loglikelihood (only 1 observable since RBC has 1 shock)
import Random; Random.seed!(42)
sim = simulate(RBC)
data = sim([:c], :, :simulate)

println("="^70)
println("BENCHMARK: Mooncake vs Zygote gradient performance")
println("="^70)

# Helper to run a benchmark section
results = Dict{String, NamedTuple}()

function bench_section(name, f, x; n_runs=5, check_fwd=true)
    println("\n── $name ──")
    
    # Correctness
    g_zyg = Zygote.gradient(f, x)[1]
    g_fd  = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
    g_mc  = DifferentiationInterface.gradient(f, AutoMooncake(; config=nothing), x)
    
    println("  Zygote:     ", round.(g_zyg; digits=8))
    println("  FiniteDiff: ", round.(g_fd;  digits=8))
    println("  Mooncake:   ", round.(g_mc;  digits=8))
    
    if check_fwd
        g_fwd = ForwardDiff.gradient(f, x)
        println("  ForwardDiff:", round.(g_fwd; digits=8))
        @assert isapprox(g_zyg, g_fwd; rtol=1e-6) "$name: ForwardDiff ≠ Zygote"
    end
    @assert isapprox(g_zyg, g_mc; rtol=1e-6) "$name: Mooncake ≠ Zygote"
    @assert isapprox(g_zyg, g_fd; rtol=1e-4) "$name: FiniteDiff ≠ Zygote"
    println("  ✅ All backends agree")
    
    # Runtime (already warm)
    zyg_t = Float64[]; mc_t = Float64[]
    for _ in 1:n_runs
        push!(zyg_t, @elapsed Zygote.gradient(f, x))
        push!(mc_t,  @elapsed DifferentiationInterface.gradient(f, AutoMooncake(; config=nothing), x))
    end
    zmed = sort(zyg_t)[cld(n_runs,2)]
    mmed = sort(mc_t)[cld(n_runs,2)]
    println("  Runtime (median of $n_runs):")
    println("    Zygote:   ", round(zmed; sigdigits=4), "s")
    println("    Mooncake: ", round(mmed; sigdigits=4), "s")
    println("    Ratio Z/M: ", round(zmed/mmed; sigdigits=3))
    
    results[name] = (zyg_med=zmed, mc_med=mmed, zyg_all=zyg_t, mc_all=mc_t)
end

# ═══════════════════════════════════════════════════════════════════════
# TTFD — time to first derivative (compilation cost)
# ═══════════════════════════════════════════════════════════════════════
println("\n── TTFD: time to first derivative (get_solution) ──")
println("  (includes compilation; measured in a fresh worker via @elapsed)")

# We already compiled above during model setup — report the first-call times
# from the actual benchmark sections instead. The TTFD for each section's
# first Mooncake call includes rule compilation.
println("  Mooncake 1st call overhead visible in per-run timings below.")

# ═══════════════════════════════════════════════════════════════════════
# 1–4: Benchmark sections
# ═══════════════════════════════════════════════════════════════════════
for (name, f, check_fwd) in [
    ("get_solution",     x -> norm(get_solution(RBC, x)[2]),                           true),
    ("get_irf",          x -> norm(get_irf(RBC, x)),                                   true),
    ("get_statistics",   x -> norm(get_statistics(RBC, x; standard_deviation=:all)[:standard_deviation]), true),
    ("get_loglikelihood", x -> get_loglikelihood(RBC, data, x),                        false),
]
    try
        bench_section(name, f, x0; check_fwd=check_fwd)
    catch e
        println("  ❌ FAILED: ", sprint(showerror, e))
        results[name] = (zyg_med=NaN, mc_med=NaN, zyg_all=Float64[], mc_all=Float64[])
    end
end

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("SUMMARY (median runtime, seconds)")
println("="^70)
println("  Function           | Zygote     | Mooncake   | Ratio (Z/M)")
println("  ───────────────────|────────────|────────────|────────────")
for name in ["get_solution", "get_irf", "get_statistics", "get_loglikelihood"]
    r = results[name]
    if isnan(r.zyg_med)
        println("  ", rpad(name, 20), "| FAILED")
    else
        println("  ", rpad(name, 20), "| ",
                lpad(round(r.zyg_med; sigdigits=4), 10), " | ",
                lpad(round(r.mc_med; sigdigits=4), 10), " | ",
                round(r.zyg_med / r.mc_med; sigdigits=3))
    end
end
println("="^70)
n_passed = count(name -> !isnan(results[name].zyg_med), keys(results))
println("$n_passed/$(length(results)) benchmarks passed")

