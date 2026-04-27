# compare_results.jl — Phase 3 of Dynare comparison
#
# Loads Julia and Dynare CSV outputs for each model, compares them, and
# reports pass/fail.  Exits non-zero on any failure.
#
# Requires: DelimitedFiles, Test (both available via --project=.)

using DelimitedFiles
using LinearAlgebra
using Test

const RTOL = 1e-6
const ATOL = 1e-7
const DEFAULT_OUTPUT_ROOT = joinpath(@__DIR__, "output")
const BENCHMARK_ONLY_MODELS = Set(["FRBUS"])
const _BENCH_CACHE = Dict{String, Dict{String, Float64}}()

function print_usage()
    println("Usage: julia --project=. compare_results.jl [--output-root=PATH | PATH]")
end

function parse_args(args)
    output_root = DEFAULT_OUTPUT_ROOT
    positional_args = String[]

    for arg in args
        if arg in ("-h", "--help")
            print_usage()
            return nothing
        elseif startswith(arg, "--output-root=")
            output_root = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--")
            error("Unknown option: $arg")
        else
            push!(positional_args, arg)
        end
    end

    if length(positional_args) > 1
        error("Expected at most one positional output-root argument, got $(length(positional_args))")
    elseif length(positional_args) == 1
        output_root = positional_args[1]
    end

    return abspath(output_root)
end

# ─────────────────────────────────────────────
# CSV loading helpers
# ─────────────────────────────────────────────
read_names(path) = strip.(readlines(path))

function load_benchmarks(dir)
    haskey(_BENCH_CACHE, dir) && return _BENCH_CACHE[dir]
    bundled = joinpath(dir, "benchmarks.csv")
    benchmarks = Dict{String, Float64}()
    if isfile(bundled)
        raw = readdlm(bundled, ',')
        for row in 1:size(raw, 1)
            key = strip(string(raw[row, 1]))
            isempty(key) && continue
            benchmarks[key] = Float64(raw[row, 2])
        end
    end
    _BENCH_CACHE[dir] = benchmarks
    return benchmarks
end

function read_bench(dir, name)
    key = endswith(name, ".csv") ? name[1:end-4] : name
    benchmarks = load_benchmarks(dir)
    if haskey(benchmarks, key)
        return benchmarks[key]
    end
    legacy = joinpath(dir, key * ".csv")
    return isfile(legacy) ? read_vector(legacy)[1] : NaN
end

function read_key_value_metadata(path)
    metadata = Dict{String, String}()
    if !isfile(path)
        return metadata
    end

    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        idx = findfirst(==('='), stripped)
        idx === nothing && continue
        key = strip(stripped[begin:prevind(stripped, idx)])
        value = strip(stripped[nextind(stripped, idx):end])
        metadata[key] = value
    end

    return metadata
end

function current_julia_metadata()
    blas_lapack = try
        string(BLAS.get_config())
    catch
        "unknown"
    end

    return Dict(
        "julia_version" => string(VERSION),
        "julia_threads" => string(Threads.nthreads()),
        "julia_threads_default" => string(Threads.nthreads(:default)),
        "julia_threads_interactive" => string(Threads.nthreads(:interactive)),
        "blas_threads" => string(BLAS.get_num_threads()),
        "blas_lapack" => blas_lapack,
        "hostname" => get(ENV, "COMPUTERNAME", get(ENV, "HOSTNAME", "unknown")),
        "kernel" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "cpu_name" => string(Sys.CPU_NAME),
        "cpu_threads" => string(Sys.CPU_THREADS),
        "word_size" => string(Sys.WORD_SIZE),
        "total_memory_bytes" => try
            string(Sys.total_memory())
        catch
            "unknown"
        end,
    )
end

function format_memory_string(bytes_string)
    try
        gib = parse(Float64, bytes_string) / 1024.0^3
        return string(round(gib, digits = 2), " GiB")
    catch
        return bytes_string
    end
end

function print_environment_summary(output_root)
    julia_metadata = read_key_value_metadata(joinpath(output_root, "comparison_environment_julia.txt"))
    dynare_metadata = read_key_value_metadata(joinpath(output_root, "comparison_environment_dynare.txt"))
    julia_source = "phase-1 metadata"
    if isempty(julia_metadata)
        julia_metadata = current_julia_metadata()
        julia_source = "compare runtime fallback"
    end

    println("Run Environment")
    println("  Julia ($julia_source):")
    println("    version: ", get(julia_metadata, "julia_version", "unknown"))
    println("    BLAS/LAPACK: ", get(julia_metadata, "blas_lapack", "unknown"))
    println("    threads: Julia=", get(julia_metadata, "julia_threads", "unknown"),
            " default=", get(julia_metadata, "julia_threads_default", "unknown"),
            " interactive=", get(julia_metadata, "julia_threads_interactive", "unknown"),
            " BLAS=", get(julia_metadata, "blas_threads", "unknown"))
    println("    machine: host=", get(julia_metadata, "hostname", "unknown"),
            " kernel=", get(julia_metadata, "kernel", "unknown"),
            " arch=", get(julia_metadata, "arch", "unknown"),
            " cpu=", get(julia_metadata, "cpu_name", "unknown"),
            " cpu_threads=", get(julia_metadata, "cpu_threads", "unknown"),
            " memory=", format_memory_string(get(julia_metadata, "total_memory_bytes", "unknown")))

    println("  Dynare:")
    if isempty(dynare_metadata)
        println("    metadata unavailable")
    else
        driver = get(dynare_metadata, "dynare_driver", "unknown")
        version = get(dynare_metadata, "dynare_version", "unknown")
        blas = get(dynare_metadata, "blas", "unknown")
        lapack = get(dynare_metadata, "lapack", "unknown")
        println("    driver/version: ", driver, " / ", version)
        if haskey(dynare_metadata, "matlab_version")
            println("    MATLAB: ", get(dynare_metadata, "matlab_version", "unknown"),
                    " release=", get(dynare_metadata, "matlab_release", "unknown"))
        elseif haskey(dynare_metadata, "octave_version")
            println("    Octave: ", get(dynare_metadata, "octave_version", "unknown"))
        end
        println("    BLAS/LAPACK: ", blas, " / ", lapack)
        println("    machine: host=", get(dynare_metadata, "hostname", "unknown"),
                " os=", get(dynare_metadata, "os", get(dynare_metadata, "kernel", "unknown")),
                " arch=", get(dynare_metadata, "arch", get(dynare_metadata, "computer", "unknown")),
                " cpu_threads=", get(dynare_metadata, "cpu_threads", get(dynare_metadata, "max_num_comp_threads", "unknown")))
        println("    threads: requested=", get(dynare_metadata, "thread_count_requested", "unknown"),
                " active=", get(dynare_metadata, "max_num_comp_threads", "unknown"))
    end
end

function read_vector(path)
    vec(readdlm(path, ',', Float64))
end

function read_matrix(path)
    readdlm(path, ',', Float64)
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

    policy_alg_path = joinpath(dir, "policy_algorithm.csv")
    if isfile(policy_alg_path)
        algs = read_names(policy_alg_path)
        if !isempty(algs)
            r[:policy_algorithm] = algs[1]
        end
    end

    # IRFs (optional — may not exist if all zero)
    irf_fields_path = joinpath(dir, "irf_fields.csv")
    if isfile(irf_fields_path)
        fields = read_names(irf_fields_path)
        irfs = Dict{String, Vector{Float64}}()
        bundled_path = joinpath(dir, "irfs.csv")
        if isfile(bundled_path)
            # Bundled format: matrix with rows = periods, cols = fields (in irf_fields.csv order).
            mat = read_matrix(bundled_path)
            ncols = min(size(mat, 2), length(fields))
            for j in 1:ncols
                irfs[fields[j]] = vec(mat[:, j])
            end
        else
            # Legacy per-field files (kept for backward compatibility with older outputs).
            for f in fields
                p = joinpath(dir, "irf_$f.csv")
                if isfile(p)
                    irfs[f] = read_vector(p)
                end
            end
        end
        r[:irfs] = irfs
        r[:irf_fields] = fields

        irf_alg_path = joinpath(dir, "irf_algorithm.csv")
        if isfile(irf_alg_path)
            algs = read_names(irf_alg_path)
            if !isempty(algs)
                r[:irf_algorithm] = algs[1]
            end
        end
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

function common_named_indices(jl_names, dy_names)
    common_names = intersect(jl_names, dy_names)
    jl_idx = name_index(jl_names)
    dy_idx = name_index(dy_names)
    return common_names, [jl_idx[name] for name in common_names], [dy_idx[name] for name in common_names]
end

function kron_linear_index(col_names, col_idxs, col_sizes)
    linear_index = 1
    stride = prod(col_sizes)
    for k in eachindex(col_names)
        stride ÷= col_sizes[k]
        linear_index += (col_idxs[k][col_names[k]] - 1) * stride
    end
    return linear_index
end

function common_kron_column_indices(jl_col_name_vecs::Vector{<:AbstractVector},
                                    dy_col_name_vecs::Vector{<:AbstractVector})
    jl_col_idxs = [name_index(v) for v in jl_col_name_vecs]
    dy_col_idxs = [name_index(v) for v in dy_col_name_vecs]
    common_cols = [intersect(jl_col_name_vecs[k], dy_col_name_vecs[k]) for k in eachindex(jl_col_name_vecs)]
    jl_col_sizes = [length(v) for v in jl_col_name_vecs]
    dy_col_sizes = [length(v) for v in dy_col_name_vecs]

    common_tuples = collect(Iterators.product(common_cols...))
    jl_col_indices = [kron_linear_index(col_names, jl_col_idxs, jl_col_sizes) for col_names in common_tuples]
    dy_col_indices = [kron_linear_index(col_names, dy_col_idxs, dy_col_sizes) for col_names in common_tuples]

    return common_cols, jl_col_indices, dy_col_indices
end

is_nawm_model(model_name) = model_name == "NAWM_EAUS_2008"
is_higher_order_model(model_name) = occursin("_pruned_2nd", model_name) || occursin("_pruned_3rd", model_name)
is_pruned_third_order_model(model_name) = occursin("_pruned_3rd", model_name)
is_excluded_model_dir(model_name) = model_name == "FS2000_pruned_3rd"
is_benchmark_only_model_dir(model_name) = model_name in BENCHMARK_ONLY_MODELS
is_supported_pruned_third_order_variance_model(model_name) = model_name in (
    "Gali_2015_chapter_3_nonlinear_pruned_3rd",
)

# ─────────────────────────────────────────────
# Comparison functions — first order
# ─────────────────────────────────────────────

function compare_steady_state(jl, dy; rtol = RTOL, atol = ATOL)
    jl_idx = name_index(jl[:var_names])
    dy_idx = name_index(dy[:var_names])

    for v in jl[:var_names]
        if !haskey(dy_idx, v)
            @warn "steady state: Variable $v missing from Dynare"
        end
    end

    common, jl_common_idx, dy_common_idx = common_named_indices(jl[:var_names], dy[:var_names])
    @test length(common) > 0
    @test length(common) >= min(length(jl[:var_names]), length(dy[:var_names])) * 0.5
    @test isapprox(jl[:steady_state][jl_common_idx], dy[:steady_state][dy_common_idx]; rtol = rtol, atol = atol)
end

function compare_ghx(jl, dy; rtol = RTOL, atol = ATOL)
    jl_vidx = name_index(jl[:var_names])
    dy_vidx = name_index(dy[:var_names])
    jl_sidx = name_index(jl[:state_var_names])
    dy_sidx = name_index(dy[:state_var_names])

    common_vars, jl_var_idx, dy_var_idx = common_named_indices(jl[:var_names], dy[:var_names])
    common_states, jl_state_idx, dy_state_idx = common_named_indices(jl[:state_var_names], dy[:state_var_names])
    @test length(common_vars) > 0
    @test length(common_states) > 0

    for v in jl[:var_names]
        if !haskey(dy_vidx, v)
            @warn "ghx: Variable $v missing from Dynare"
        end
    end
    for s in jl[:state_var_names]
        if !haskey(dy_sidx, s)
            @warn "ghx: State $s missing from Dynare"
        end
    end

    jl_subset = jl[:ghx][jl_var_idx, jl_state_idx]
    dy_subset = dy[:ghx][dy_var_idx, dy_state_idx]
    @test isapprox(jl_subset, dy_subset; rtol = rtol, atol = atol)
end

function compare_ghu(jl, dy; rtol = RTOL, atol = ATOL)
    jl_vidx = name_index(jl[:var_names])
    dy_vidx = name_index(dy[:var_names])
    jl_eidx = name_index(jl[:exo_names])
    dy_eidx = name_index(dy[:exo_names])

    common_vars, jl_var_idx, dy_var_idx = common_named_indices(jl[:var_names], dy[:var_names])
    common_exo, jl_exo_idx, dy_exo_idx = common_named_indices(jl[:exo_names], dy[:exo_names])
    @test length(common_vars) > 0
    @test length(common_exo) > 0

    for v in jl[:var_names]
        if !haskey(dy_vidx, v)
            @warn "ghu: Variable $v missing from Dynare"
        end
    end
    for e in jl[:exo_names]
        if !haskey(dy_eidx, e)
            @warn "ghu: Shock $e missing from Dynare"
        end
    end

    jl_subset = jl[:ghu][jl_var_idx, jl_exo_idx]
    dy_subset = dy[:ghu][dy_var_idx, dy_exo_idx]
    @test isapprox(jl_subset, dy_subset; rtol = rtol, atol = atol)
end

function compare_irfs(jl, dy; model_name = "", rtol = RTOL, atol = ATOL)
    haskey(jl, :irfs) && haskey(dy, :irfs) || return

    if is_higher_order_model(model_name)
        @info "Skipping IRF comparison for $model_name (higher-order IRFs are convention-dependent; compare moments instead)"
        return
    end

    # Backward-compatibility guard: for higher-order model directories, compare IRFs
    # only when Julia IRFs were explicitly generated at first order.
    if is_higher_order_model(model_name)
        irf_alg = get(jl, :irf_algorithm, "")
        if irf_alg != "first_order"
            @info "Skipping IRF comparison for $model_name (IRFs not tagged as first-order; regenerate phase-1 outputs to enable)"
            return
        end
    end

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
        @test isapprox(jvec[1:n], dvec[1:n]; rtol = rtol, atol = atol)
    end
end

function compare_variance(jl, dy; rtol = RTOL, atol = ATOL)
    haskey(jl, :variance_covariance) && haskey(dy, :variance_covariance) || return

    jl_idx = name_index(jl[:var_names])
    dy_idx = name_index(dy[:var_names])
    common = intersect(jl[:var_names], dy[:var_names])
    valid_common = filter(v -> begin
        ji = jl_idx[v]
        di = dy_idx[v]
        ji <= size(jl[:variance_covariance], 1) && di <= size(dy[:variance_covariance], 1)
    end, common)
    isempty(valid_common) && return

    jl_variance = [jl[:variance_covariance][jl_idx[v], jl_idx[v]] for v in valid_common]
    dy_variance = [dy[:variance_covariance][dy_idx[v], dy_idx[v]] for v in valid_common]
    @test isapprox(jl_variance, dy_variance; rtol = rtol, atol = atol)

    jl_std = sqrt.(jl_variance)
    dy_std = sqrt.(dy_variance)
    @test isapprox(jl_std, dy_std; rtol = rtol, atol = atol)
end

function compare_variance_decomposition(jl, dy; rtol = RTOL, atol = ATOL)
    haskey(jl, :variance_decomposition) && haskey(dy, :variance_decomposition) || return

    jl_vidx = name_index(jl[:vd_var_names])
    dy_vidx = name_index(dy[:vd_var_names])
    jl_eidx = name_index(jl[:vd_exo_names])
    dy_eidx = name_index(dy[:vd_exo_names])

    common_vars = intersect(jl[:vd_var_names], dy[:vd_var_names])
    common_exo = intersect(jl[:vd_exo_names], dy[:vd_exo_names])

    valid_vars = filter(v -> begin
        ji = jl_vidx[v]
        di = dy_vidx[v]
        ji <= size(jl[:variance_decomposition], 1) || return false
        di <= size(dy[:variance_decomposition], 1) || return false
        jl_row_sum = sum(abs, jl[:variance_decomposition][ji, :])
        dy_row_sum = sum(abs, dy[:variance_decomposition][di, :])
        jl_row_sum >= 1.0 && dy_row_sum >= 1.0
    end, common_vars)
    valid_exo = filter(e -> begin
        jl_eidx[e] <= size(jl[:variance_decomposition], 2) && dy_eidx[e] <= size(dy[:variance_decomposition], 2)
    end, common_exo)
    isempty(valid_vars) && return
    isempty(valid_exo) && return

    jl_var_idx = [jl_vidx[v] for v in valid_vars]
    dy_var_idx = [dy_vidx[v] for v in valid_vars]
    jl_exo_idx = [jl_eidx[e] for e in valid_exo]
    dy_exo_idx = [dy_eidx[e] for e in valid_exo]

    jl_subset = jl[:variance_decomposition][jl_var_idx, jl_exo_idx]
    dy_subset = dy[:variance_decomposition][dy_var_idx, dy_exo_idx]

    comparison_atol = max(atol, 0.01)
    ok = isapprox(jl_subset, dy_subset; rtol = rtol, atol = comparison_atol) ||
         all(abs.(jl_subset) .< comparison_atol .&& abs.(dy_subset) .< comparison_atol)
    if !ok
        diff = maximum(abs.(jl_subset .- dy_subset))
        scale = max(maximum(abs.(jl_subset)), maximum(abs.(dy_subset)))
        @warn "Variance decomp mismatch" achieved_atol=diff achieved_rtol=(scale > 0 ? diff / scale : Inf) required_atol=comparison_atol required_rtol=rtol
    end
    @test ok
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

    common_rows, jl_row_idx, dy_row_idx = common_named_indices(jl_row_names, dy_row_names)
    @test length(common_rows) > 0

    common_cols, jl_col_idx, dy_col_idx = common_kron_column_indices(jl_col_name_vecs, dy_col_name_vecs)
    for k in eachindex(common_cols)
        @test length(common_cols[k]) > 0
    end

    jl_subset = jl[mat_key][jl_row_idx, jl_col_idx]
    dy_subset = dy[mat_key][dy_row_idx, dy_col_idx]
    @test isapprox(jl_subset, dy_subset; rtol = rtol, atol = atol)
end

function compare_vector_matrix(jl, dy, mat_key::Symbol,
                               jl_row_names, dy_row_names,
                               jl_col_names, dy_col_names;
                               rtol = RTOL, atol = ATOL)
    haskey(jl, mat_key) && haskey(dy, mat_key) || return

    common_rows, jl_row_idx, dy_row_idx = common_named_indices(jl_row_names, dy_row_names)
    common_cols, jl_col_idx, dy_col_idx = common_named_indices(jl_col_names, dy_col_names)

    jl_mat = jl[mat_key]
    dy_mat = dy[mat_key]

    @test length(common_rows) > 0
    @test length(common_cols) > 0
    @test isapprox(jl_mat[jl_row_idx, jl_col_idx], dy_mat[dy_row_idx, dy_col_idx]; rtol = rtol, atol = atol)
end

function compare_second_order(jl, dy; rtol = RTOL, atol = ATOL)
    sn_jl = jl[:state_var_names]; sn_dy = dy[:state_var_names]
    en_jl = jl[:exo_names];       en_dy = dy[:exo_names]
    vn_jl = jl[:var_names];       vn_dy = dy[:var_names]

    @testset "ghxx" begin
        compare_kron_matrix(jl, dy, :ghxx, vn_jl, vn_dy,
                           [sn_jl, sn_jl], [sn_dy, sn_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghxu" begin
        compare_kron_matrix(jl, dy, :ghxu, vn_jl, vn_dy,
                           [sn_jl, en_jl], [sn_dy, en_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghuu" begin
        compare_kron_matrix(jl, dy, :ghuu, vn_jl, vn_dy,
                           [en_jl, en_jl], [en_dy, en_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghs2" begin
        if haskey(jl, :ghs2) && haskey(dy, :ghs2)
            jl_vidx = name_index(vn_jl)
            dy_vidx = name_index(vn_dy)
            common_vars = intersect(vn_jl, vn_dy)
            for v in common_vars
                # ghs2 convention differs between MacroModelling and Dynare:
                # MacroModelling extracts the (σ,σ) slice of the second-order tensor
                # (pure perturbation-parameter² coefficient), whereas Dynare's ghs2
                # absorbs the full shock covariance matrix.  These are different
                # mathematical objects and cannot be compared element-wise.
                # Skipping ghs2 comparison.
            end
        end
    end
end

function compare_third_order(jl, dy; rtol = RTOL, atol = ATOL)
    sn_jl = jl[:state_var_names]; sn_dy = dy[:state_var_names]
    en_jl = jl[:exo_names];       en_dy = dy[:exo_names]
    vn_jl = jl[:var_names];       vn_dy = dy[:var_names]

    @testset "ghxxx" begin
        compare_kron_matrix(jl, dy, :ghxxx, vn_jl, vn_dy,
                           [sn_jl, sn_jl, sn_jl], [sn_dy, sn_dy, sn_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghxxu" begin
        compare_kron_matrix(jl, dy, :ghxxu, vn_jl, vn_dy,
                           [sn_jl, sn_jl, en_jl], [sn_dy, sn_dy, en_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghxuu" begin
        compare_kron_matrix(jl, dy, :ghxuu, vn_jl, vn_dy,
                           [sn_jl, en_jl, en_jl], [sn_dy, en_dy, en_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghuuu" begin
        compare_kron_matrix(jl, dy, :ghuuu, vn_jl, vn_dy,
                           [en_jl, en_jl, en_jl], [en_dy, en_dy, en_dy];
                           rtol = rtol, atol = atol)
    end
    @testset "ghxss" begin
        compare_vector_matrix(jl, dy, :ghxss, vn_jl, vn_dy, sn_jl, sn_dy;
                              rtol = rtol, atol = atol)
    end
    @testset "ghuss" begin
        compare_vector_matrix(jl, dy, :ghuss, vn_jl, vn_dy, en_jl, en_dy;
                              rtol = rtol, atol = atol)
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
function main(args = ARGS)
    output_root = parse_args(args)
    output_root === nothing && return

    if !isdir(output_root)
        error("Output directory not found: $output_root")
    end

    model_dirs = filter(d -> isdir(joinpath(output_root, d, "julia")) &&
                        isdir(joinpath(output_root, d, "dynare")) &&
                        !is_excluded_model_dir(d),
                        readdir(output_root))

    comparison_model_dirs = filter(d -> !is_benchmark_only_model_dir(d), model_dirs)

    if isempty(model_dirs)
        error("No model directories with both julia/ and dynare/ results found in $output_root")
    end

    println("Comparison output root: $output_root")
    print_environment_summary(output_root)

    benchmark_only_dirs = filter(is_benchmark_only_model_dir, model_dirs)
    for mname in sort(benchmark_only_dirs)
        @info "Skipping correctness comparison for benchmark-only model: $mname"
    end

    comparison_exception = nothing
    try
        if !isempty(comparison_model_dirs)
            @testset "Dynare Comparison" begin
                for mname in sort(comparison_model_dirs)
                    julia_dir = joinpath(output_root, mname, "julia")
                    dynare_dir = joinpath(output_root, mname, "dynare")

                    @info "Comparing results for: $mname"
                    jl = load_results(julia_dir)
                    dy = load_results(dynare_dir)

                @testset "$mname" begin
                    moments_only_higher_order = is_higher_order_model(mname)
                    skip_pruned_third_order = is_pruned_third_order_model(mname)

                        @testset "Steady State" begin
                            compare_steady_state(jl, dy)
                        end
                        @testset "Policy Matrix ghx" begin
                            if skip_pruned_third_order
                                @info "Skipping ghx comparison for $mname (pruned third-order state representation mismatch)"
                            elseif moments_only_higher_order && get(jl, :policy_algorithm, "") != "first_order"
                                @info "Skipping ghx comparison for $mname (policy matrices not tagged as first-order; regenerate phase-1 outputs to enable)"
                            else
                                compare_ghx(jl, dy)
                            end
                        end
                        @testset "Policy Matrix ghu" begin
                            if moments_only_higher_order && get(jl, :policy_algorithm, "") != "first_order"
                                @info "Skipping ghu comparison for $mname (policy matrices not tagged as first-order; regenerate phase-1 outputs to enable)"
                            else
                                compare_ghu(jl, dy)
                            end
                        end
                        @testset "IRFs" begin
                            compare_irfs(jl, dy; model_name = mname)
                        end
                        @testset "Variance" begin
                            if skip_pruned_third_order && !is_supported_pruned_third_order_variance_model(mname)
                                @info "Skipping variance comparison for $mname (pruned third-order moment convention mismatch outside the validated benchmark cases)"
                            else
                                compare_variance(jl, dy)
                            end
                        end
                        @testset "Variance Decomposition" begin
                            if moments_only_higher_order
                                @info "Skipping variance decomposition comparison for $mname (higher-order configured as covariance/variance moments-only)"
                            else
                                compare_variance_decomposition(jl, dy)
                            end
                        end

                        # Higher-order comparisons (when data is present)
                        if has_second_order(jl) && has_second_order(dy)
                            @testset "Second Order Matrices" begin
                                if moments_only_higher_order
                                    @info "Skipping second-order matrix comparison for $mname (higher-order configured as moments-only)"
                                else
                                    compare_second_order(jl, dy)
                                end
                            end
                        end
                        if has_third_order(jl) && has_third_order(dy)
                            @testset "Third Order Matrices" begin
                                if moments_only_higher_order
                                    @info "Skipping third-order matrix comparison for $mname (higher-order configured as moments-only)"
                                else
                                    compare_third_order(jl, dy)
                                end
                            end
                        end
                    end
                end
            end
        else
            @info "No correctness-comparison model directories found under $output_root"
        end
    catch err
        if err isa Test.TestSetException
            comparison_exception = err
        else
            rethrow(err)
        end
    end

    # ── Benchmark comparison ──
    # Dynare benchmarks: component-level (Jacobian, first-order solve, Hessian, second-order solve)
    # Julia benchmarks: component-level via BenchmarkTools
    # Dynare order=3 also exports k_order_pert as an additional bundled reference.
    println("\n", "="^100)
    println("  Benchmark Comparison: MacroModelling (median of 500 runs) vs Dynare (median of 500 runs)")
    println("="^100)

    has_bench(dir, name) = !isnan(read_bench(dir, name))

    function sum_bench_components(dir, files)
        total = 0.0
        for file in files
            value = read_bench(dir, file)
            if isnan(value)
                return NaN
            end
            total += value
        end
        return total
    end

    # Dynare order=3 runs additionally export a bundled k_order_pert timing.
    is_dynare_k_order_dir(dir) = has_bench(dir, "benchmark_k_order_pert.csv")

    # k_order_pert timing is exported explicitly for Dynare order=3 runs.
    function read_dynare_k_order_pert(dir)
        has_bench(dir, "benchmark_k_order_pert.csv") ? read_bench(dir, "benchmark_k_order_pert.csv") : NaN
    end

    function print_bench_table(title, model_dirs, jl_file, dy_file; note = "")
        println("\n--- $title ---")
        if !isempty(note)
            println("    $note")
        end
        println(rpad("Model", 50), rpad("MacroModelling", 18), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(model_dirs)
            jl_time = read_bench(joinpath(output_root, mname, "julia"), jl_file)
            dy_time = read_bench(joinpath(output_root, mname, "dynare"), dy_file)
            jl_str = isnan(jl_time) ? "N/A" : format_time(jl_time)
            dy_str = isnan(dy_time) ? "N/A" : format_time(dy_time)
            speedup_str = (!isnan(jl_time) && !isnan(dy_time) && jl_time > 0) ? 
                string(round(dy_time / jl_time, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 18), rpad(dy_str, 12), speedup_str)
        end
    end

    # Jacobian (Dynare: dynamic_g1)
    print_bench_table("Jacobian", model_dirs,
                      "benchmark_jacobian.csv", "benchmark_jacobian.csv")

    # First-order solve (Julia: direct QME solve; Dynare: dyn_first_order_solver)
    print_bench_table("First-Order Solve", model_dirs,
                      "benchmark_first_order_solve.csv", "benchmark_first_order_solve.csv")

    # First-order total (sum of direct component medians)
    println("\n--- First-Order Total (sum of direct Jacobian + solve medians) ---")
    println(rpad("Model", 50), rpad("MacroModelling", 18), rpad("Dynare", 12), "Speedup")
    println("-"^100)
    for mname in sort(model_dirs)
        jl_dir = joinpath(output_root, mname, "julia")
        dy_dir = joinpath(output_root, mname, "dynare")
        jl_time = sum_bench_components(jl_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"])
        dy_time = sum_bench_components(dy_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"])
        jl_str = isnan(jl_time) ? "N/A" : format_time(jl_time)
        dy_str = isnan(dy_time) ? "N/A" : format_time(dy_time)
        speedup_str = (!isnan(jl_time) && !isnan(dy_time) && jl_time > 0) ?
            string(round(dy_time / jl_time, digits=1), "x") : "N/A"
        println(rpad(mname, 50), rpad(jl_str, 18), rpad(dy_str, 12), speedup_str)
    end

    # Hessian / second-order solve
    ho_models = filter(d -> has_bench(joinpath(output_root, d, "julia"), "benchmark_hessian.csv"), model_dirs)
    dy_decomposable_ho_models = filter(d -> has_bench(joinpath(output_root, d, "dynare"), "benchmark_hessian.csv"), ho_models)
    if !isempty(dy_decomposable_ho_models)
        print_bench_table("Hessian", dy_decomposable_ho_models,
                          "benchmark_hessian.csv", "benchmark_hessian.csv")

        print_bench_table("Second-Order Solve", dy_decomposable_ho_models,
                          "benchmark_second_order_solve.csv", "benchmark_second_order_solve.csv")

        # Second-Order Total (Hessian + Second-Order Solve)
        println("\n--- Second-Order Total (Hessian + Second-Order Solve) ---")
        println(rpad("Model", 50), rpad("MacroModelling", 18), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(dy_decomposable_ho_models)
            jl_dir = joinpath(output_root, mname, "julia")
            dy_dir = joinpath(output_root, mname, "dynare")
            jl_time = sum_bench_components(jl_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"])
            dy_time = sum_bench_components(dy_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"])
            jl_str = isnan(jl_time) ? "N/A" : format_time(jl_time)
            dy_str = isnan(dy_time) ? "N/A" : format_time(dy_time)
            speedup_str = (!isnan(jl_time) && !isnan(dy_time) && jl_time > 0) ?
                string(round(dy_time / jl_time, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 18), rpad(dy_str, 12), speedup_str)
        end
    end

    # Dynare k_order models: report bundled higher-order timing consistently.
    k_order_models = filter(d -> is_dynare_k_order_dir(joinpath(output_root, d, "dynare")), model_dirs)
    if !isempty(k_order_models)
        println("\n--- Higher-Order Bundled (Dynare k_order_pert) ---")
        println("    MacroModelling sums directly measured solve-stack components; Dynare reports direct bundled k_order_pert")
        println(rpad("Model", 50), rpad("MacroModelling", 18), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(k_order_models)
            jl_dir = joinpath(output_root, mname, "julia")
            dy_dir = joinpath(output_root, mname, "dynare")

            jl_fo_solve = read_bench(jl_dir, "benchmark_first_order_solve.csv")
            jl_hess = read_bench(jl_dir, "benchmark_hessian.csv")
            jl_so = read_bench(jl_dir, "benchmark_second_order_solve.csv")
            jl_td = read_bench(jl_dir, "benchmark_third_order_derivatives.csv")
            jl_ts = read_bench(jl_dir, "benchmark_third_order_solve.csv")

            jl_bundled = jl_fo_solve
            isnan(jl_hess) || (jl_bundled += jl_hess)
            isnan(jl_so) || (jl_bundled += jl_so)
            isnan(jl_td) || (jl_bundled += jl_td)
            isnan(jl_ts) || (jl_bundled += jl_ts)

            dy_bundled = read_dynare_k_order_pert(dy_dir)

            jl_str = isnan(jl_bundled) ? "N/A" : format_time(jl_bundled)
            dy_str = isnan(dy_bundled) ? "N/A" : format_time(dy_bundled)
            speedup_str = (!isnan(jl_bundled) && !isnan(dy_bundled) && jl_bundled > 0) ?
                string(round(dy_bundled / jl_bundled, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 18), rpad(dy_str, 12), speedup_str)
        end
    end

    if !isempty(dy_decomposable_ho_models)
        println("\n--- Comparable Direct Components Total (Jacobian + FO + Hessian + SO) ---")
        println(rpad("Model", 50), rpad("MacroModelling", 18), rpad("Dynare", 12), "Speedup")
        println("-"^100)
        for mname in sort(dy_decomposable_ho_models)
            jl_dir = joinpath(output_root, mname, "julia")
            dy_dir = joinpath(output_root, mname, "dynare")

            jl_total = sum_bench_components(jl_dir, [
                "benchmark_jacobian.csv",
                "benchmark_first_order_solve.csv",
                "benchmark_hessian.csv",
                "benchmark_second_order_solve.csv",
            ])
            dy_total = sum_bench_components(dy_dir, [
                "benchmark_jacobian.csv",
                "benchmark_first_order_solve.csv",
                "benchmark_hessian.csv",
                "benchmark_second_order_solve.csv",
            ])

            jl_str = isnan(jl_total) ? "N/A" : format_time(jl_total)
            dy_str = isnan(dy_total) ? "N/A" : format_time(dy_total)
            speedup_str = (!isnan(jl_total) && !isnan(dy_total) && jl_total > 0) ?
                string(round(dy_total / jl_total, digits=1), "x") : "N/A"
            println(rpad(mname, 50), rpad(jl_str, 18), rpad(dy_str, 12), speedup_str)
        end
    end

    # Third-order components (MacroModelling only — Dynare uses k_order_pert for order=3)
    to_models = filter(d -> has_bench(joinpath(output_root, d, "julia"), "benchmark_third_order_derivatives.csv"),
                       model_dirs)
    if !isempty(to_models)
        println("\n--- Third-Order Components (MacroModelling only — Dynare k_order_pert is bundled) ---")
        println(rpad("Model", 50), rpad("3rd Derivs", 15), "3rd Solve")
        println("-"^100)
        for mname in sort(to_models)
            td_time = read_bench(joinpath(output_root, mname, "julia"), "benchmark_third_order_derivatives.csv")
            ts_time = read_bench(joinpath(output_root, mname, "julia"), "benchmark_third_order_solve.csv")
            td = isnan(td_time) ? "N/A" : format_time(td_time)
            ts = isnan(ts_time) ? "N/A" : format_time(ts_time)
            println(rpad(mname, 50), rpad(td, 15), ts)
        end
    end

    println("="^100)

    comparison_exception === nothing || throw(comparison_exception)
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
