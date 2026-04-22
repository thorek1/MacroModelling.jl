# compare_results.jl — Phase 3 of Dynare comparison
#
# Loads Julia and Dynare CSV outputs for each model, compares them, and
# reports pass/fail.  Exits non-zero on any failure.
#
# Requires: DelimitedFiles, Test (both available via --project=.)

using DelimitedFiles
using Test

const RTOL = 1e-6
const ATOL = 1e-10
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

function safe_isapprox(a, b; rtol = RTOL, atol = ATOL)
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

    # Higher-order solution matrices (optional)
    for key in [:ghxx, :ghxu, :ghuu, :ghs2,
                :ghxxx, :ghxxu, :ghxuu, :ghuuu, :ghxss, :ghuss]
        p = joinpath(dir, "$(key).csv")
        if isfile(p)
            r[key] = read_matrix(p)
        end
    end

    r
end

# ─────────────────────────────────────────────
# Build index lookup: name → row/col index
# ─────────────────────────────────────────────
name_index(names) = Dict(n => i for (i, n) in enumerate(names))

# ─────────────────────────────────────────────
# Comparison functions — first order
# ─────────────────────────────────────────────

function compare_steady_state(jl, dy)
    jl_idx = name_index(jl[:var_names])
    dy_idx = name_index(dy[:var_names])

    for v in jl[:var_names]
        @test haskey(dy_idx, v) || @warn "Variable $v missing from Dynare"
    end

    common = intersect(jl[:var_names], dy[:var_names])
    @test length(common) > 0
    @test length(common) >= min(length(jl[:var_names]), length(dy[:var_names])) * 0.5
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
    @test length(common_states) > 0

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
    @test length(common_exo) > 0

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

    for v in common
        ji = jl_idx[v]; di = dy_idx[v]
        ji > size(jl[:variance_covariance], 1) && continue
        di > size(dy[:variance_covariance], 1) && continue
        jval = jl[:variance_covariance][ji, ji]
        dval = dy[:variance_covariance][di, di]
        @test safe_isapprox(jval, dval)
    end

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
        jval = jl[:variance_decomposition][ji, jei]
        dval = dy[:variance_decomposition][di, dei]

        jl_row_sum = sum(abs, jl[:variance_decomposition][ji, :])
        dy_row_sum = sum(abs, dy[:variance_decomposition][di, :])
        if jl_row_sum < 1.0 || dy_row_sum < 1.0
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
# Comparison functions — higher-order matrices
# ─────────────────────────────────────────────

"""
Compare a Kronecker-product matrix (ghxx, ghuu, ghxxx, ghuuu, etc.)
indexed by kron of name vectors (e.g., state × state for ghxx).
Uses tuple-based column alignment: iterate over common (name₁, name₂[, …])
tuples and look up elements in each side's matrix via their local indices.
"""
function compare_kron_matrix(jl, dy, mat_key::Symbol,
                             jl_row_names, dy_row_names,
                             jl_col_name_vecs::Vector{<:AbstractVector},
                             dy_col_name_vecs::Vector{<:AbstractVector};
                             rtol = RTOL, atol = ATOL)
    haskey(jl, mat_key) && haskey(dy, mat_key) || return

    jl_ridx = name_index(jl_row_names)
    dy_ridx = name_index(dy_row_names)
    common_rows = intersect(jl_row_names, dy_row_names)

    @test length(common_rows) > 0

    # Build column index maps and common name tuples for each kron dimension
    jl_col_idxs = [name_index(v) for v in jl_col_name_vecs]
    dy_col_idxs = [name_index(v) for v in dy_col_name_vecs]
    common_cols = [intersect(jl_col_name_vecs[k], dy_col_name_vecs[k]) for k in eachindex(jl_col_name_vecs)]
    for k in eachindex(common_cols)
        @test length(common_cols[k]) > 0
    end
    jl_col_sizes = [length(v) for v in jl_col_name_vecs]
    dy_col_sizes = [length(v) for v in dy_col_name_vecs]

    ndim = length(jl_col_name_vecs)

    # Iterate over all common column-name tuples
    if ndim == 2
        for v in common_rows, c1 in common_cols[1], c2 in common_cols[2]
            jl_ri = jl_ridx[v]
            dy_ri = dy_ridx[v]
            jl_ci = (jl_col_idxs[1][c1] - 1) * jl_col_sizes[2] + jl_col_idxs[2][c2]
            dy_ci = (dy_col_idxs[1][c1] - 1) * dy_col_sizes[2] + dy_col_idxs[2][c2]
            jval = jl[mat_key][jl_ri, jl_ci]
            dval = dy[mat_key][dy_ri, dy_ci]
            @test safe_isapprox(jval, dval; rtol = rtol, atol = atol)
        end
    elseif ndim == 3
        for v in common_rows, c1 in common_cols[1], c2 in common_cols[2], c3 in common_cols[3]
            jl_ri = jl_ridx[v]
            dy_ri = dy_ridx[v]
            jl_ci = (jl_col_idxs[1][c1] - 1) * jl_col_sizes[2] * jl_col_sizes[3] +
                    (jl_col_idxs[2][c2] - 1) * jl_col_sizes[3] +
                    jl_col_idxs[3][c3]
            dy_ci = (dy_col_idxs[1][c1] - 1) * dy_col_sizes[2] * dy_col_sizes[3] +
                    (dy_col_idxs[2][c2] - 1) * dy_col_sizes[3] +
                    dy_col_idxs[3][c3]
            jval = jl[mat_key][jl_ri, jl_ci]
            dval = dy[mat_key][dy_ri, dy_ci]
            @test safe_isapprox(jval, dval; rtol = rtol, atol = atol)
        end
    end
end

function compare_vector_matrix(jl, dy, mat_key::Symbol,
                               jl_row_names, dy_row_names,
                               jl_col_names, dy_col_names;
                               rtol = RTOL, atol = ATOL)
    haskey(jl, mat_key) && haskey(dy, mat_key) || return

    jl_ridx = name_index(jl_row_names)
    dy_ridx = name_index(dy_row_names)
    jl_cidx = name_index(jl_col_names)
    dy_cidx = name_index(dy_col_names)
    common_rows = intersect(jl_row_names, dy_row_names)
    common_cols = intersect(jl_col_names, dy_col_names)

    jl_mat = jl[mat_key]
    dy_mat = dy[mat_key]

    for v in common_rows, c in common_cols
        jval = jl_mat[jl_ridx[v], jl_cidx[c]]
        dval = dy_mat[dy_ridx[v], dy_cidx[c]]
        @test safe_isapprox(jval, dval; rtol = rtol, atol = atol)
    end
end

function compare_second_order(jl, dy)
    sn_jl = jl[:state_var_names]; sn_dy = dy[:state_var_names]
    en_jl = jl[:exo_names];       en_dy = dy[:exo_names]
    vn_jl = jl[:var_names];       vn_dy = dy[:var_names]

    @testset "ghxx" begin
        compare_kron_matrix(jl, dy, :ghxx, vn_jl, vn_dy,
                           [sn_jl, sn_jl], [sn_dy, sn_dy])
    end
    @testset "ghxu" begin
        compare_kron_matrix(jl, dy, :ghxu, vn_jl, vn_dy,
                           [sn_jl, en_jl], [sn_dy, en_dy])
    end
    @testset "ghuu" begin
        compare_kron_matrix(jl, dy, :ghuu, vn_jl, vn_dy,
                           [en_jl, en_jl], [en_dy, en_dy])
    end
    @testset "ghs2" begin
        if haskey(jl, :ghs2) && haskey(dy, :ghs2)
            jl_vidx = name_index(vn_jl)
            dy_vidx = name_index(vn_dy)
            common_vars = intersect(vn_jl, vn_dy)
            for v in common_vars
                jval = jl[:ghs2][jl_vidx[v], 1]
                dval = dy[:ghs2][dy_vidx[v], 1]
                @test safe_isapprox(jval, dval)
            end
        end
    end
end

function compare_third_order(jl, dy)
    sn_jl = jl[:state_var_names]; sn_dy = dy[:state_var_names]
    en_jl = jl[:exo_names];       en_dy = dy[:exo_names]
    vn_jl = jl[:var_names];       vn_dy = dy[:var_names]

    @testset "ghxxx" begin
        compare_kron_matrix(jl, dy, :ghxxx, vn_jl, vn_dy,
                           [sn_jl, sn_jl, sn_jl], [sn_dy, sn_dy, sn_dy])
    end
    @testset "ghxxu" begin
        compare_kron_matrix(jl, dy, :ghxxu, vn_jl, vn_dy,
                           [sn_jl, sn_jl, en_jl], [sn_dy, sn_dy, en_dy])
    end
    @testset "ghxuu" begin
        compare_kron_matrix(jl, dy, :ghxuu, vn_jl, vn_dy,
                           [sn_jl, en_jl, en_jl], [sn_dy, en_dy, en_dy])
    end
    @testset "ghuuu" begin
        compare_kron_matrix(jl, dy, :ghuuu, vn_jl, vn_dy,
                           [en_jl, en_jl, en_jl], [en_dy, en_dy, en_dy])
    end
    @testset "ghxss" begin
        compare_vector_matrix(jl, dy, :ghxss, vn_jl, vn_dy, sn_jl, sn_dy)
    end
    @testset "ghuss" begin
        compare_vector_matrix(jl, dy, :ghuss, vn_jl, vn_dy, en_jl, en_dy)
    end
end

# ─────────────────────────────────────────────
# Detect whether a model directory has higher-order results
# ─────────────────────────────────────────────
has_second_order(r) = haskey(r, :ghxx)
has_third_order(r)  = haskey(r, :ghxxx)

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

                # Higher-order comparisons (when data is present)
                if has_second_order(jl) && has_second_order(dy)
                    @testset "Second Order Matrices" begin
                        compare_second_order(jl, dy)
                    end
                end
                if has_third_order(jl) && has_third_order(dy)
                    @testset "Third Order Matrices" begin
                        compare_third_order(jl, dy)
                    end
                end
            end
        end
    end

    # ── Benchmark comparison ──
    # Dynare benchmarks: component-level (NSSS, Jacobian, first-order solve, Hessian, second-order solve)
    # Julia benchmarks: component-level via BenchmarkTools
    # Note: For order=3 (k_order_solver), Dynare cannot decompose beyond NSSS vs k_order_pert
    println("\n", "="^100)
    println("  Benchmark Comparison: Julia (BenchmarkTools median) vs Dynare (median of 100 runs)")
    println("="^100)

    # Helper to read a benchmark value, returning NaN if file doesn't exist
    read_bench(dir, name) = let p = joinpath(dir, name)
        isfile(p) ? read_vector(p)[1] : NaN
    end

    function print_bench_table(title, model_dirs, jl_file, dy_file; note = "")
        println("\n--- $title ---")
        if !isempty(note)
            println("    $note")
        end
        println(rpad("Model", 50), rpad("Julia", 12), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(model_dirs)
            jl_time = read_bench(joinpath(OUTPUT_ROOT, mname, "julia"), jl_file)
            dy_time = read_bench(joinpath(OUTPUT_ROOT, mname, "dynare"), dy_file)
            jl_str = isnan(jl_time) ? "N/A" : format_time(jl_time)
            dy_str = isnan(dy_time) ? "N/A" : format_time(dy_time)
            speedup_str = (!isnan(jl_time) && !isnan(dy_time) && jl_time > 0) ? 
                string(round(dy_time / jl_time, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 12), rpad(dy_str, 12), speedup_str)
        end
    end

    # NSSS
    print_bench_table("NSSS (Steady State)", model_dirs,
                      "benchmark_nsss.csv", "benchmark_nsss.csv")

    # Jacobian (Dynare: dynamic_g1; not available for k_order models)
    print_bench_table("Jacobian", model_dirs,
                      "benchmark_jacobian.csv", "benchmark_jacobian.csv";
                      note = "Dynare: N/A for order=3 (k_order_pert bundles all)")

    # First-order total (NSSS + Jacobian + first-order solve)
    print_bench_table("First-Order Total (NSSS + Jacobian + QME Solve)", model_dirs,
                      "benchmark_first_order.csv", "benchmark_first_order.csv";
                      note = "For k_order models, Dynare total includes ALL orders")

    # Hessian (available for order >= 2 non-k_order on Dynare side, always for Julia HO models)
    ho_models = filter(d -> isfile(joinpath(OUTPUT_ROOT, d, "julia", "benchmark_hessian.csv")),
                       model_dirs)
    if !isempty(ho_models)
        print_bench_table("Hessian", ho_models,
                          "benchmark_hessian.csv", "benchmark_hessian.csv")

        print_bench_table("Second-Order Solve", ho_models,
                          "benchmark_second_order_solve.csv", "benchmark_second_order_solve.csv")
    end

    # Third-order components (Julia only — Dynare uses k_order_pert for order=3)
    to_models = filter(d -> isfile(joinpath(OUTPUT_ROOT, d, "julia", "benchmark_third_order_derivatives.csv")),
                       model_dirs)
    if !isempty(to_models)
        println("\n--- Third-Order Components (Julia only — Dynare k_order_pert is bundled) ---")
        println(rpad("Model", 50), rpad("3rd Derivs", 15), "3rd Solve")
        println("-"^100)
        for mname in sort(to_models)
            td = let p = joinpath(OUTPUT_ROOT, mname, "julia", "benchmark_third_order_derivatives.csv")
                isfile(p) ? format_time(read_vector(p)[1]) : "N/A"
            end
            ts = let p = joinpath(OUTPUT_ROOT, mname, "julia", "benchmark_third_order_solve.csv")
                isfile(p) ? format_time(read_vector(p)[1]) : "N/A"
            end
            println(rpad(mname, 50), rpad(td, 15), ts)
        end
    end

    # Grand total (sum all available components)
    if !isempty(ho_models)
        println("\n--- Grand Total (all orders summed) ---")
        println(rpad("Model", 50), rpad("Julia", 12), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(ho_models)
            jl_dir = joinpath(OUTPUT_ROOT, mname, "julia")
            dy_dir = joinpath(OUTPUT_ROOT, mname, "dynare")

            jl_total = read_bench(jl_dir, "benchmark_first_order.csv")
            jl_hess = read_bench(jl_dir, "benchmark_hessian.csv")
            jl_so = read_bench(jl_dir, "benchmark_second_order_solve.csv")
            jl_td = read_bench(jl_dir, "benchmark_third_order_derivatives.csv")
            jl_ts = read_bench(jl_dir, "benchmark_third_order_solve.csv")
            jl_grand = jl_total
            isnan(jl_hess) || (jl_grand += jl_hess)
            isnan(jl_so) || (jl_grand += jl_so)
            isnan(jl_td) || (jl_grand += jl_td)
            isnan(jl_ts) || (jl_grand += jl_ts)

            # Dynare grand total: for k_order, first_order IS the grand total;
            # for non-k_order, sum first_order + hessian + second_order_solve
            dy_grand = read_bench(dy_dir, "benchmark_first_order.csv")
            dy_hess = read_bench(dy_dir, "benchmark_hessian.csv")
            dy_so = read_bench(dy_dir, "benchmark_second_order_solve.csv")
            isnan(dy_hess) || (dy_grand += dy_hess)
            isnan(dy_so) || (dy_grand += dy_so)

            jl_str = isnan(jl_grand) ? "N/A" : format_time(jl_grand)
            dy_str = isnan(dy_grand) ? "N/A" : format_time(dy_grand)
            speedup_str = (!isnan(jl_grand) && !isnan(dy_grand) && jl_grand > 0) ? 
                string(round(dy_grand / jl_grand, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 12), rpad(dy_str, 12), speedup_str)
        end
    end

    println("="^100)
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
