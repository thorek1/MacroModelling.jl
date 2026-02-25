#!/usr/bin/env julia
# Comprehensive test for get_loglikelihood rrule (Zygote reverse-mode AD)
# Collects all get_loglikelihood call patterns from the estimation test scripts
# and verifies that Zygote.gradient produces finite, correct gradients.
#
# Usage:  julia --project=test tasks/test_get_loglikelihood_rrule.jl
#
# For benchmarking mode (uses BenchmarkTools for reliable timing/allocation data):
#   BENCHMARK=1 julia --project=test tasks/test_get_loglikelihood_rrule.jl
#
# Output: prints primal values, gradient norms, and optionally BenchmarkTools
# median time/allocation data for cross-branch comparison.
#
# IMPORTANT: Each @benchmark setup block:
#   1. Calls clear_solution_caches! to wipe QME warm-start, solution
#      matrices, and stochastic-steady-state vectors.
#   2. Runs one get_loglikelihood with slightly perturbed params so that
#      workspaces are allocated at the right size (no first-call penalty).
#   3. Calls clear_solution_caches! again so the timed call doesn't hit
#      cached results.  evals is left at the default (auto-tuned).

using MacroModelling, Random, CSV, DataFrames, AxisKeys, Zygote, ForwardDiff, LinearAlgebra
using BenchmarkTools

const BENCHMARK_MODE = get(ENV, "BENCHMARK", "0") == "1"

# Print git metadata for traceability
println("Julia:  ", VERSION)
println("Branch: ", strip(read(`git branch --show-current`, String)))
println("Commit: ", strip(read(`git log --oneline -1`, String)))
println("Benchmark mode: ", BENCHMARK_MODE)
println()

# ─── helpers ─────────────────────────────────────────────────────────────

struct CaseResult
    name::String
    passed::Bool
    primal::Float64
    grad_norm::Float64
    grad_len::Int
    # BenchmarkTools median results (nanoseconds / bytes)
    median_time_primal_ns::Float64
    median_alloc_primal::Int64
    median_time_grad_ns::Float64
    median_alloc_grad::Int64
    error_msg::String
end

const RESULTS = CaseResult[]

"""
Run a single get_loglikelihood case: verify correctness, then optionally benchmark.

Each @benchmark sample runs with `evals=1` and a `setup` block that
calls `clear_solution_caches!` to prevent warm-start bias.
"""
function run_case(name::String; model, data, params, kwargs...)
    kw = Dict{Symbol,Any}(kwargs)

    # Determine the algorithm so we can clear the right caches
    algo = get(kw, :algorithm, :first_order)

    println("─── CASE: $name ───")

    # --- primal correctness ---
    MacroModelling.clear_solution_caches!(model, algo)
    local llh::Float64
    try
        llh = get_loglikelihood(model, data, params; kw...)
        println("  primal  = $llh")
        if !isfinite(llh)
            push!(RESULTS, CaseResult(name, false, llh, NaN, 0, NaN, 0, NaN, 0, "primal not finite"))
            println("  FAIL: primal not finite")
            return
        end
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        push!(RESULTS, CaseResult(name, false, NaN, NaN, 0, NaN, 0, NaN, 0, "primal error: $msg"))
        println("  FAIL (primal): ", first(split(msg, '\n')))
        return
    end

    # --- Zygote gradient correctness ---
    MacroModelling.clear_solution_caches!(model, algo)
    local grad
    try
        grad = Zygote.gradient(x -> get_loglikelihood(model, data, x; kw...), params)[1]
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        push!(RESULTS, CaseResult(name, false, llh, NaN, 0, NaN, 0, NaN, 0, "Zygote error: $msg"))
        println("  FAIL (Zygote): ", first(split(msg, '\n')))
        return
    end

    gn = norm(grad)
    gl = length(grad)
    passed = isfinite(gn) && gn > 0
    println("  grad    = norm=$gn, len=$gl, finite=$(isfinite(gn))")

    # --- Benchmark with BenchmarkTools ---
    local med_t_p::Float64, med_a_p::Int64, med_t_g::Float64, med_a_g::Int64
    med_t_p = NaN; med_a_p = 0; med_t_g = NaN; med_a_g = 0
    if BENCHMARK_MODE
        println("  benchmarking primal (with workspace warm-up per sample)...")
        warmup_params = params .* 1.0001  # slightly perturbed to avoid cache hit
        b_primal = @benchmark(
            get_loglikelihood($(Ref(model))[], $(Ref(data))[], $(Ref(params))[]; $(kw)...),
            setup = begin
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
                get_loglikelihood($(Ref(model))[], $(Ref(data))[], $(Ref(warmup_params))[]; $(kw)...)
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
            end
        )
        med_p = median(b_primal)
        med_t_p = med_p.time        # nanoseconds
        med_a_p = med_p.memory      # bytes
        println("    primal: $(round(med_t_p/1e6, digits=3)) ms, $(med_a_p) bytes ($(round(med_a_p/1024, digits=1)) KB)")

        println("  benchmarking gradient (with workspace warm-up per sample)...")
        b_grad = @benchmark(
            Zygote.gradient(x -> get_loglikelihood($(Ref(model))[], $(Ref(data))[], x; $(kw)...), $(Ref(params))[]),
            setup = begin
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
                get_loglikelihood($(Ref(model))[], $(Ref(data))[], $(Ref(warmup_params))[]; $(kw)...)
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
            end
        )
        med_g = median(b_grad)
        med_t_g = med_g.time
        med_a_g = med_g.memory
        println("    grad:   $(round(med_t_g/1e6, digits=3)) ms, $(med_a_g) bytes ($(round(med_a_g/1024, digits=1)) KB)")
    end

    println("  => ", passed ? "PASS" : "FAIL")
    push!(RESULTS, CaseResult(name, passed, llh, gn, gl, med_t_p, med_a_p, med_t_g, med_a_g, ""))
end

# helper for SW07 parameter combination
function sw07_combined_params(all_params, fixed)
    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew,
    crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw,
    csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap,
    cfc, crpi, crr, cry, crdy,
    constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params
    ctou, clandaw, cg, curvp, curvw = fixed
    [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost,
     chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy,
     crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw,
     constelab, constepinf, constebeta, ctrend,
     z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]
end

# ─── SW07 Zygote gradient wrapper (differentiates w.r.t. estimated params) ──
function sw07_grad_case(name, model, data_sw, obs_sw, p_est, fixed; kwargs...)
    kw = Dict{Symbol,Any}(kwargs)
    algo = get(kw, :algorithm, :first_order)

    println("─── CASE: $name ───")

    combo = sw07_combined_params(p_est, fixed)

    # primal correctness
    MacroModelling.clear_solution_caches!(model, algo)
    local llh
    try
        llh = get_loglikelihood(model, data_sw(obs_sw), combo; kw...)
        println("  primal  = $llh")
        if !isfinite(llh)
            push!(RESULTS, CaseResult(name, false, llh, NaN, 0, NaN, 0, NaN, 0, "primal not finite"))
            return
        end
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        push!(RESULTS, CaseResult(name, false, NaN, NaN, 0, NaN, 0, NaN, 0, "primal error: $msg"))
        println("  FAIL (primal): ", first(split(msg, '\n')))
        return
    end

    # Zygote gradient correctness
    MacroModelling.clear_solution_caches!(model, algo)
    local grad
    try
        grad = Zygote.gradient(x -> get_loglikelihood(model, data_sw(obs_sw), sw07_combined_params(x, fixed); kw...), p_est)[1]
    catch err
        msg = sprint(showerror, err, catch_backtrace())
        push!(RESULTS, CaseResult(name, false, llh, NaN, 0, NaN, 0, NaN, 0, "Zygote error: $msg"))
        println("  FAIL (Zygote): ", first(split(msg, '\n')))
        return
    end

    gn = norm(grad)
    gl = length(grad)
    passed = isfinite(gn) && gn > 0
    println("  grad    = norm=$gn, len=$gl, finite=$(isfinite(gn))")

    # Benchmark
    local med_t_p::Float64, med_a_p::Int64, med_t_g::Float64, med_a_g::Int64
    med_t_p = NaN; med_a_p = 0; med_t_g = NaN; med_a_g = 0
    if BENCHMARK_MODE
        data_obs = data_sw(obs_sw)
        warmup_combo = combo .* 1.0001  # slightly perturbed to avoid cache hit
        println("  benchmarking primal (with workspace warm-up per sample)...")
        b_primal = @benchmark(
            get_loglikelihood($(Ref(model))[], $(Ref(data_obs))[], $(Ref(combo))[]; $(kw)...),
            setup = begin
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
                get_loglikelihood($(Ref(model))[], $(Ref(data_obs))[], $(Ref(warmup_combo))[]; $(kw)...)
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
            end
        )
        med_p = median(b_primal)
        med_t_p = med_p.time
        med_a_p = med_p.memory
        println("    primal: $(round(med_t_p/1e6, digits=3)) ms, $(med_a_p) bytes ($(round(med_a_p/1024, digits=1)) KB)")

        println("  benchmarking gradient (with workspace warm-up per sample)...")
        b_grad = @benchmark(
            Zygote.gradient(x -> get_loglikelihood($(Ref(model))[], $(Ref(data_obs))[], sw07_combined_params(x, $(Ref(fixed))[]); $(kw)...), $(Ref(p_est))[]),
            setup = begin
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
                get_loglikelihood($(Ref(model))[], $(Ref(data_obs))[], $(Ref(warmup_combo))[]; $(kw)...)
                MacroModelling.clear_solution_caches!($(Ref(model))[], $(Ref(algo))[])
            end
        )
        med_g = median(b_grad)
        med_t_g = med_g.time
        med_a_g = med_g.memory
        println("    grad:   $(round(med_t_g/1e6, digits=3)) ms, $(med_a_g) bytes ($(round(med_a_g/1024, digits=1)) KB)")
    end

    println("  => ", passed ? "PASS" : "FAIL")
    push!(RESULTS, CaseResult(name, passed, llh, gn, gl, med_t_p, med_a_p, med_t_g, med_a_g, ""))
end


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. FS2000 model — Kalman, Inversion, 2nd, pruned-2nd order            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

println("\n", "="^70)
println("  Loading FS2000 model + data")
println("="^70, "\n")

include(joinpath(@__DIR__, "..", "models", "FS2000.jl"))
dat_fs = CSV.read(joinpath(@__DIR__, "..", "test", "data", "FS2000_data.csv"), DataFrame)
data_fs = KeyedArray(permutedims(Matrix(dat_fs)),
                     Variable = Symbol.("log_" .* names(dat_fs)),
                     Time = 1:size(dat_fs,1))
data_fs = log.(data_fs)
obs_fs  = sort(Symbol.("log_" .* names(dat_fs)))
data_fs = data_fs(obs_fs, :)
p_fs    = copy(FS2000.parameter_values)

# Case 1: default (kalman, first_order)
run_case("fs2000_kalman_1st",
         model = FS2000, data = data_fs, params = p_fs)

# Case 2: explicit kalman filter
run_case("fs2000_kalman_explicit",
         model = FS2000, data = data_fs, params = p_fs,
         filter = :kalman)

# Case 3: inversion filter
run_case("fs2000_inversion_1st",
         model = FS2000, data = data_fs, params = p_fs,
         filter = :inversion)

# Case 4: second_order
run_case("fs2000_second_order",
         model = FS2000, data = data_fs, params = p_fs,
         algorithm = :second_order)

# Case 5: pruned_second_order
run_case("fs2000_pruned_second_order",
         model = FS2000, data = data_fs, params = p_fs,
         algorithm = :pruned_second_order)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. Caldara et al 2012 — 3rd order, pruned 3rd order                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

println("\n", "="^70)
println("  Loading Caldara et al 2012 model + data")
println("="^70, "\n")

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))
dat_us = CSV.read(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), DataFrame)
data_us = KeyedArray(permutedims(Matrix(dat_us)),
                     Variable = Symbol.(strip.(names(dat_us))),
                     Time = 1:size(dat_us,1))
data_cal = data_us([:dy], 75:230)
p_cal    = copy(Caldara_et_al_2012_estim.parameter_values)

# Case 6: third_order
run_case("caldara_third_order",
         model = Caldara_et_al_2012_estim, data = data_cal, params = p_cal,
         algorithm = :third_order, on_failure_loglikelihood = -Inf)

# Case 7: pruned_third_order
run_case("caldara_pruned_third_order",
         model = Caldara_et_al_2012_estim, data = data_cal, params = p_cal,
         algorithm = :pruned_third_order, on_failure_loglikelihood = -Inf)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. Smets & Wouters 2007 — linear, kalman with presample & diagonal    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

println("\n", "="^70)
println("  Loading Smets & Wouters 2007 linear model + data")
println("="^70, "\n")

dat_sw  = CSV.read(joinpath(@__DIR__, "..", "test", "data", "usmodel.csv"), DataFrame)
data_sw = KeyedArray(permutedims(Matrix(dat_sw)),
                     Variable = Symbol.(strip.(names(dat_sw))),
                     Time = 1:size(dat_sw,1))
obs_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
obs_sw  = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data_sw = rekey(data_sw(obs_old, 47:230), :Variable => obs_sw)

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_linear.jl"))
fixed_lin   = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw],
                Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]
par_names   = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew,
               :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw,
               :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap,
               :cfc, :crpi, :crr, :cry, :crdy,
               :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]
idx_est_lin = indexin(par_names, Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)
p_est_lin   = copy(Smets_Wouters_2007_linear.parameter_values[idx_est_lin])

# Case 8: SW07 linear, kalman, presample, diagonal
sw07_grad_case("sw07_linear_kalman",
               Smets_Wouters_2007_linear, data_sw, obs_sw, p_est_lin, fixed_lin,
               presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. Smets & Wouters 2007 — nonlinear, kalman with presample & diagonal ║
# ╚══════════════════════════════════════════════════════════════════════════╝

println("\n", "="^70)
println("  Loading Smets & Wouters 2007 nonlinear model")
println("="^70, "\n")

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))
fixed_nl    = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw],
                Smets_Wouters_2007.constants.post_complete_parameters.parameters)]
idx_est_nl  = indexin(par_names, Smets_Wouters_2007.constants.post_complete_parameters.parameters)
p_est_nl    = copy(Smets_Wouters_2007.parameter_values[idx_est_nl])

# Case 9: SW07 nonlinear, kalman, presample, diagonal
sw07_grad_case("sw07_nonlinear_kalman",
               Smets_Wouters_2007, data_sw, obs_sw, p_est_nl, fixed_nl,
               presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Summary                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

println("\n", "="^70)
println("  SUMMARY")
println("="^70)

npass = count(r -> r.passed, RESULTS)
ntot  = length(RESULTS)
println("$npass / $ntot cases passed\n")

# Print table
if BENCHMARK_MODE
    println(rpad("Case", 30), " ", rpad("Pass", 5), " ", rpad("Primal", 22), " ",
            rpad("GradNorm", 22), " ", rpad("GLen", 5), " ",
            rpad("Primal_ms", 12), " ", rpad("Primal_KB", 12), " ",
            rpad("Grad_ms", 12), " ", rpad("Grad_KB", 12))
    println("-"^160)
    for r in RESULTS
        println(rpad(r.name, 30), " ",
                rpad(r.passed ? "✓" : "✗", 5), " ",
                rpad(string(round(r.primal, sigdigits=12)), 22), " ",
                rpad(string(round(r.grad_norm, sigdigits=8)), 22), " ",
                rpad(string(r.grad_len), 5), " ",
                rpad(string(round(r.median_time_primal_ns / 1e6, digits=3)), 12), " ",
                rpad(string(round(r.median_alloc_primal / 1024, digits=1)), 12), " ",
                rpad(string(round(r.median_time_grad_ns / 1e6, digits=3)), 12), " ",
                rpad(string(round(r.median_alloc_grad / 1024, digits=1)), 12))
    end
else
    println(rpad("Case", 30), " ", rpad("Pass", 5), " ", rpad("Primal", 22), " ",
            rpad("GradNorm", 22), " ", rpad("GLen", 5))
    println("-"^90)
    for r in RESULTS
        println(rpad(r.name, 30), " ",
                rpad(r.passed ? "✓" : "✗", 5), " ",
                rpad(string(round(r.primal, sigdigits=12)), 22), " ",
                rpad(string(round(r.grad_norm, sigdigits=8)), 22), " ",
                rpad(string(r.grad_len), 5))
    end
end

println()
for r in RESULTS
    if !r.passed && r.error_msg != ""
        println("FAIL detail [$( r.name)]: $(r.error_msg)")
    end
end

# Machine-readable output for cross-branch comparison (CSV-like)
println("\n\n### MACHINE_READABLE_OUTPUT ###")
println("name,passed,primal,grad_norm,grad_len,primal_median_ms,primal_alloc_kb,grad_median_ms,grad_alloc_kb")
for r in RESULTS
    println(r.name, ",",
            r.passed, ",",
            r.primal, ",",
            r.grad_norm, ",",
            r.grad_len, ",",
            round(r.median_time_primal_ns / 1e6, digits=3), ",",
            round(r.median_alloc_primal / 1024, digits=1), ",",
            round(r.median_time_grad_ns / 1e6, digits=3), ",",
            round(r.median_alloc_grad / 1024, digits=1))
end

npass == ntot || exit(1)
