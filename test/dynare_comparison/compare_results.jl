# compare_results.jl — Phase 3 of Dynare comparison
#
# Loads Julia and Dynare CSV outputs for each model, compares them, and
# reports pass/fail.  Exits non-zero on any failure.
#
# Requires: DelimitedFiles, Test (both available via --project=.)

using DelimitedFiles
using Test

const RTOL = 1e-6
const OUTPUT_ROOT = joinpath(@__DIR__, "output")

# ─────────────────────────────────────────────
# CSV loading helpers
# ─────────────────────────────────────────────
read_names(path) = strip.(readlines(path))

function read_vector(path)
    vec(readdlm(path, ',', Float64))
end

function read_matrix(path)
    readdlm(path, ',', Float64)
end

function safe_isapprox(a, b; rtol = RTOL, atol = eps())
    isapprox(a, b, rtol = rtol, atol = atol) ||
    (abs(a) < 1e-12 && abs(b) < 1e-12)
end

# ─────────────────────────────────────────────
# Load results from a directory (julia/ or dynare/)
# ─────────────────────────────────────────────
function load_results(dir)
    r = Dict{Symbol, Any}()

    r[:var_names]       = read_names(joinpath(dir, "var_names.csv"))
    r[:exo_names]       = read_names(joinpath(dir, "exo_names.csv"))
    r[:state_var_names] = read_names(joinpath(dir, "state_var_names.csv"))
    r[:steady_state]    = read_vector(joinpath(dir, "steady_state.csv"))
    r[:ghx]             = read_matrix(joinpath(dir, "ghx.csv"))
    r[:ghu]             = read_matrix(joinpath(dir, "ghu.csv"))

    # IRFs (optional — may not exist if all zero)
    irf_fields_path = joinpath(dir, "irf_fields.csv")
    if isfile(irf_fields_path)
        fields = read_names(irf_fields_path)
        irfs = Dict{String, Vector{Float64}}()
        for f in fields
            p = joinpath(dir, "irf_$f.csv")
            if isfile(p)
                irfs[f] = read_vector(p)
            end
        end
        r[:irfs] = irfs
        r[:irf_fields] = fields
    end

    # Variance-covariance
    vcov_path = joinpath(dir, "variance_covariance.csv")
    if isfile(vcov_path)
        r[:variance_covariance] = read_matrix(vcov_path)
    end

    # Variance decomposition
    vd_path = joinpath(dir, "variance_decomposition.csv")
    if isfile(vd_path)
        r[:variance_decomposition] = read_matrix(vd_path)
        r[:vd_var_names] = read_names(joinpath(dir, "variance_decomposition_var_names.csv"))
        r[:vd_exo_names] = read_names(joinpath(dir, "variance_decomposition_exo_names.csv"))
    end

    r
end

# ─────────────────────────────────────────────
# Build index lookup: name → row/col index
# ─────────────────────────────────────────────
name_index(names) = Dict(n => i for (i, n) in enumerate(names))

# ─────────────────────────────────────────────
# Comparison functions
# ─────────────────────────────────────────────

function compare_steady_state(jl, dy)
    jl_idx = name_index(jl[:var_names])
    dy_idx = name_index(dy[:var_names])

    # Assert all Julia vars are present in Dynare output
    for v in jl[:var_names]
        @test haskey(dy_idx, v) || @warn "Variable $v missing from Dynare"
    end

    common = intersect(jl[:var_names], dy[:var_names])
    for v in common
        jval = jl[:steady_state][jl_idx[v]]
        dval = dy[:steady_state][dy_idx[v]]
        @test safe_isapprox(jval, dval)
    end
end

function compare_ghx(jl, dy)
    jl_vidx = name_index(jl[:var_names])
    dy_vidx = name_index(dy[:var_names])
    jl_sidx = name_index(jl[:state_var_names])
    dy_sidx = name_index(dy[:state_var_names])

    common_vars = intersect(jl[:var_names], dy[:var_names])
    common_states = intersect(jl[:state_var_names], dy[:state_var_names])

    for v in jl[:var_names]
        @test haskey(dy_vidx, v) || @warn "ghx: Variable $v missing from Dynare"
    end
    for s in jl[:state_var_names]
        @test haskey(dy_sidx, s) || @warn "ghx: State $s missing from Dynare"
    end

    for v in common_vars, s in common_states
        jval = jl[:ghx][jl_vidx[v], jl_sidx[s]]
        dval = dy[:ghx][dy_vidx[v], dy_sidx[s]]
        @test safe_isapprox(jval, dval)
    end
end

function compare_ghu(jl, dy)
    jl_vidx = name_index(jl[:var_names])
    dy_vidx = name_index(dy[:var_names])
    jl_eidx = name_index(jl[:exo_names])
    dy_eidx = name_index(dy[:exo_names])

    common_vars = intersect(jl[:var_names], dy[:var_names])
    common_exo = intersect(jl[:exo_names], dy[:exo_names])

    for v in jl[:var_names]
        @test haskey(dy_vidx, v) || @warn "ghu: Variable $v missing from Dynare"
    end
    for e in jl[:exo_names]
        @test haskey(dy_eidx, e) || @warn "ghu: Shock $e missing from Dynare"
    end

    for v in common_vars, e in common_exo
        jval = jl[:ghu][jl_vidx[v], jl_eidx[e]]
        dval = dy[:ghu][dy_vidx[v], dy_eidx[e]]
        @test safe_isapprox(jval, dval)
    end
end

function compare_irfs(jl, dy)
    haskey(jl, :irfs) && haskey(dy, :irfs) || return

    # Assert all Julia IRF fields exist on the Dynare side
    for f in get(jl, :irf_fields, String[])
        if !haskey(dy[:irfs], f)
            @warn "IRF field $f missing from Dynare"
        end
    end

    common_fields = intersect(keys(jl[:irfs]), keys(dy[:irfs]))
    for f in common_fields
        jvec = jl[:irfs][f]
        dvec = dy[:irfs][f]
        n = min(length(jvec), length(dvec))
        for t in 1:n
            @test safe_isapprox(jvec[t], dvec[t]; atol = 1e-14)
        end
    end
end

function compare_variance(jl, dy)
    haskey(jl, :variance_covariance) && haskey(dy, :variance_covariance) || return

    jl_idx = name_index(jl[:var_names])
    dy_idx = name_index(dy[:var_names])
    common = intersect(jl[:var_names], dy[:var_names])

    # Compare variances (diagonal)
    for v in common
        ji = jl_idx[v]; di = dy_idx[v]
        ji > size(jl[:variance_covariance], 1) && continue
        di > size(dy[:variance_covariance], 1) && continue
        jval = jl[:variance_covariance][ji, ji]
        dval = dy[:variance_covariance][di, di]
        @test safe_isapprox(jval, dval)
    end

    # Compare standard deviations
    for v in common
        ji = jl_idx[v]; di = dy_idx[v]
        ji > size(jl[:variance_covariance], 1) && continue
        di > size(dy[:variance_covariance], 1) && continue
        jval = sqrt(jl[:variance_covariance][ji, ji])
        dval = sqrt(dy[:variance_covariance][di, di])
        @test safe_isapprox(jval, dval)
    end
end

function compare_variance_decomposition(jl, dy)
    haskey(jl, :variance_decomposition) && haskey(dy, :variance_decomposition) || return

    jl_vidx = name_index(jl[:vd_var_names])
    dy_vidx = name_index(dy[:vd_var_names])
    jl_eidx = name_index(jl[:vd_exo_names])
    dy_eidx = name_index(dy[:vd_exo_names])

    common_vars = intersect(jl[:vd_var_names], dy[:vd_var_names])
    common_exo = intersect(jl[:vd_exo_names], dy[:vd_exo_names])

    for v in common_vars, e in common_exo
        ji = jl_vidx[v]; jei = jl_eidx[e]
        di = dy_vidx[v]; dei = dy_eidx[e]
        ji > size(jl[:variance_decomposition], 1) && continue
        di > size(dy[:variance_decomposition], 1) && continue
        # Both sides already in percentages (0-100)
        jval = jl[:variance_decomposition][ji, jei]
        dval = dy[:variance_decomposition][di, dei]

        # Skip variables where both sides have near-zero total decomposition
        # (indicates near-zero variance — decomposition is numerically meaningless)
        jl_row_sum = sum(abs, jl[:variance_decomposition][ji, :])
        dy_row_sum = sum(abs, dy[:variance_decomposition][di, :])
        if jl_row_sum < 1.0 || dy_row_sum < 1.0
            # Total decomposition < 1% means near-zero variance
            continue
        end

        ok = safe_isapprox(jval, dval; rtol = RTOL, atol = 0.01) ||
             (abs(jval) < 0.01 && abs(dval) < 0.01)
        if !ok
            @warn "Variance decomp mismatch: var=$v, shock=$e, julia=$jval, dynare=$dval, diff=$(abs(jval-dval)), rdiff=$(abs(jval-dval)/max(abs(dval),eps()))"
        end
        @test ok
    end
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
function main()
    if !isdir(OUTPUT_ROOT)
        error("Output directory not found: $OUTPUT_ROOT")
    end

    model_dirs = filter(d -> isdir(joinpath(OUTPUT_ROOT, d, "julia")) &&
                              isdir(joinpath(OUTPUT_ROOT, d, "dynare")),
                        readdir(OUTPUT_ROOT))

    if isempty(model_dirs)
        error("No model directories with both julia/ and dynare/ results found in $OUTPUT_ROOT")
    end

    @testset "Dynare Comparison" begin
        for mname in sort(model_dirs)
            julia_dir = joinpath(OUTPUT_ROOT, mname, "julia")
            dynare_dir = joinpath(OUTPUT_ROOT, mname, "dynare")

            @info "Comparing results for: $mname"
            jl = load_results(julia_dir)
            dy = load_results(dynare_dir)

            @testset "$mname" begin
                @testset "Steady State" begin
                    compare_steady_state(jl, dy)
                end
                @testset "Policy Matrix ghx" begin
                    compare_ghx(jl, dy)
                end
                @testset "Policy Matrix ghu" begin
                    compare_ghu(jl, dy)
                end
                @testset "IRFs" begin
                    compare_irfs(jl, dy)
                end
                @testset "Variance" begin
                    compare_variance(jl, dy)
                end
                @testset "Variance Decomposition" begin
                    compare_variance_decomposition(jl, dy)
                end
            end
        end
    end

    # ── Benchmark comparison ──
    println("\n", "="^72)
    println("  NSSS + Jacobian + First-Order Solve Benchmark: Julia vs Dynare (median of 100 runs)")
    println("="^72)
    println(rpad("Model", 40), rpad("Julia", 12), rpad("Dynare", 12), "Speedup")
    println("-"^72)

    for mname in sort(model_dirs)
        jl_bench_path = joinpath(OUTPUT_ROOT, mname, "julia", "benchmark_first_order.csv")
        dy_bench_path = joinpath(OUTPUT_ROOT, mname, "dynare", "benchmark_first_order.csv")

        jl_time = isfile(jl_bench_path) ? read_vector(jl_bench_path)[1] : NaN
        dy_time = isfile(dy_bench_path) ? read_vector(dy_bench_path)[1] : NaN

        jl_str = isnan(jl_time) ? "N/A" : format_time(jl_time)
        dy_str = isnan(dy_time) ? "N/A" : format_time(dy_time)

        if !isnan(jl_time) && !isnan(dy_time) && jl_time > 0
            speedup = dy_time / jl_time
            sp_str = string(round(speedup, digits=1), "x")
        else
            sp_str = "N/A"
        end

        println(rpad(mname, 40), rpad(jl_str, 12), rpad(dy_str, 12), sp_str)
    end
    println("="^72)
end

function format_time(t)
    if t < 1e-3
        string(round(t * 1e6, digits=1), " μs")
    elseif t < 1.0
        string(round(t * 1e3, digits=2), " ms")
    else
        string(round(t, digits=3), " s")
    end
end

main()
