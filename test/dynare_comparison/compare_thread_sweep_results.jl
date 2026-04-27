using DelimitedFiles

const DEFAULT_SWEEP_ROOT = joinpath(@__DIR__, "output_thread_sweep")

read_vector(path) = vec(readdlm(path, ',', Float64))

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

function format_memory_string(bytes_string)
    try
        gib = parse(Float64, bytes_string) / 1024.0^3
        return string(round(gib, digits = 2), " GiB")
    catch
        return bytes_string
    end
end

function print_environment_summary(thread_counts, thread_dirs)
    first_thread = first(thread_counts)
    metadata_root = thread_dirs[first_thread]
    julia_metadata = read_key_value_metadata(joinpath(metadata_root, "comparison_environment_julia.txt"))
    dynare_metadata = read_key_value_metadata(joinpath(metadata_root, "comparison_environment_dynare.txt"))

    println("Run Environment (metadata source: threads_$(first_thread))")
    println("  Julia:")
    if isempty(julia_metadata)
        println("    metadata unavailable")
    else
        println("    version: ", get(julia_metadata, "julia_version", "unknown"))
        println("    BLAS/LAPACK: ", get(julia_metadata, "blas_lapack", "unknown"))
        println("    machine: host=", get(julia_metadata, "hostname", "unknown"),
                " kernel=", get(julia_metadata, "kernel", "unknown"),
                " arch=", get(julia_metadata, "arch", "unknown"),
                " cpu=", get(julia_metadata, "cpu_name", "unknown"),
                " cpu_threads=", get(julia_metadata, "cpu_threads", "unknown"),
                " memory=", format_memory_string(get(julia_metadata, "total_memory_bytes", "unknown")))
    end

    println("  Dynare:")
    if isempty(dynare_metadata)
        println("    metadata unavailable")
    else
        println("    driver/version: ", get(dynare_metadata, "dynare_driver", "unknown"),
                " / ", get(dynare_metadata, "dynare_version", "unknown"))
        if haskey(dynare_metadata, "matlab_version")
            println("    MATLAB: ", get(dynare_metadata, "matlab_version", "unknown"),
                    " release=", get(dynare_metadata, "matlab_release", "unknown"))
        elseif haskey(dynare_metadata, "octave_version")
            println("    Octave: ", get(dynare_metadata, "octave_version", "unknown"))
        end
        println("    BLAS/LAPACK: ", get(dynare_metadata, "blas", "unknown"),
                " / ", get(dynare_metadata, "lapack", "unknown"))
        println("    machine: host=", get(dynare_metadata, "hostname", "unknown"),
                " os=", get(dynare_metadata, "os", get(dynare_metadata, "kernel", "unknown")),
                " arch=", get(dynare_metadata, "arch", get(dynare_metadata, "computer", "unknown")),
                " cpu_threads=", get(dynare_metadata, "cpu_threads", get(dynare_metadata, "max_num_comp_threads", "unknown")))
        println("    threads: requested=", get(dynare_metadata, "thread_count_requested", "unknown"),
                " active=", get(dynare_metadata, "max_num_comp_threads", "unknown"))
    end
end

# Cache of parsed benchmarks.csv files: dir => Dict{String,Float64}.
const _BENCH_CACHE = Dict{String, Dict{String, Float64}}()

function load_benchmarks(dir)
    haskey(_BENCH_CACHE, dir) && return _BENCH_CACHE[dir]
    bundled = joinpath(dir, "benchmarks.csv")
    d = Dict{String, Float64}()
    if isfile(bundled)
        raw = readdlm(bundled, ',')
        for r in 1:size(raw, 1)
            key = strip(string(raw[r, 1]))
            isempty(key) && continue
            d[key] = Float64(raw[r, 2])
        end
    end
    _BENCH_CACHE[dir] = d
    return d
end

function read_bench(dir, name)
    # `name` may be either a bare metric name (e.g. "benchmark_jacobian") or a
    # legacy filename ("benchmark_jacobian.csv"). Strip the .csv if present and
    # consult the bundled benchmarks.csv first; fall back to the per-file CSV.
    key = endswith(name, ".csv") ? name[1:end-4] : name
    bench = load_benchmarks(dir)
    if haskey(bench, key)
        return bench[key]
    end
    legacy = joinpath(dir, key * ".csv")
    return isfile(legacy) ? read_vector(legacy)[1] : NaN
end

function print_usage()
    println("Usage: julia --project=. compare_thread_sweep_results.jl [--output-root=PATH | PATH]")
end

function parse_args(args)
    output_root = DEFAULT_SWEEP_ROOT
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

function sum_components(dir, files)
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

function sum_optional_components(dir, files)
    total = 0.0
    found_value = false
    for file in files
        value = read_bench(dir, file)
        if !isnan(value)
            total += value
            found_value = true
        end
    end
    return found_value ? total : NaN
end

function collect_thread_dirs(output_root)
    thread_dirs = Dict{Int, String}()
    for entry in readdir(output_root)
        full_path = joinpath(output_root, entry)
        isdir(full_path) || continue
        match_result = match(r"^threads_(\d+)$", entry)
        match_result === nothing && continue
        thread_dirs[parse(Int, match_result.captures[1])] = full_path
    end

    isempty(thread_dirs) && error("No thread-sweep directories found under $output_root")

    thread_counts = sort(collect(keys(thread_dirs)))
    return thread_counts, thread_dirs
end

function model_names(thread_dir)
    filter(model_name -> isdir(joinpath(thread_dir, model_name, "julia")) &&
                         isdir(joinpath(thread_dir, model_name, "dynare")),
           readdir(thread_dir))
end

function collect_model_names(thread_counts, thread_dirs)
    model_set = Set{String}()
    for thread_count in thread_counts
        for model_name in model_names(thread_dirs[thread_count])
            push!(model_set, model_name)
        end
    end
    return sort!(collect(model_set))
end

first_order_total(julia_dir, dynare_dir) = (
    sum_components(julia_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"]),
    sum_components(dynare_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"]),
)

second_order_total(julia_dir, dynare_dir) = (
    sum_components(julia_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"]),
    sum_components(dynare_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"]),
)

comparable_direct_total(julia_dir, dynare_dir) = (
    sum_components(julia_dir, [
        "benchmark_jacobian.csv",
        "benchmark_first_order_solve.csv",
        "benchmark_hessian.csv",
        "benchmark_second_order_solve.csv",
    ]),
    sum_components(dynare_dir, [
        "benchmark_jacobian.csv",
        "benchmark_first_order_solve.csv",
        "benchmark_hessian.csv",
        "benchmark_second_order_solve.csv",
    ]),
)

higher_order_bundled(julia_dir, dynare_dir) = (
    sum_optional_components(julia_dir, [
        "benchmark_first_order_solve.csv",
        "benchmark_hessian.csv",
        "benchmark_second_order_solve.csv",
        "benchmark_third_order_derivatives.csv",
        "benchmark_third_order_solve.csv",
    ]),
    read_bench(dynare_dir, "benchmark_k_order_pert.csv"),
)

function collect_rows(thread_counts, thread_dirs, metric_fn)
    rows = Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}()

    for model_name in collect_model_names(thread_counts, thread_dirs)
        julia_values = Float64[]
        dynare_values = Float64[]
        has_any_value = false

        for thread_count in thread_counts
            model_dir = joinpath(thread_dirs[thread_count], model_name)
            if isdir(joinpath(model_dir, "julia")) && isdir(joinpath(model_dir, "dynare"))
                julia_value, dynare_value = metric_fn(joinpath(model_dir, "julia"), joinpath(model_dir, "dynare"))
            else
                julia_value, dynare_value = NaN, NaN
            end

            push!(julia_values, julia_value)
            push!(dynare_values, dynare_value)
            has_any_value |= !isnan(julia_value) || !isnan(dynare_value)
        end

        has_any_value && push!(rows, (model_name, julia_values, dynare_values))
    end

    return rows
end

function write_summary_csv(path, thread_counts, rows)
    ncols = 1 + 2 * length(thread_counts)
    table = Matrix{Any}(undef, length(rows) + 1, ncols)
    table[1, 1] = "Model"

    column_index = 2
    for thread_count in thread_counts
        table[1, column_index] = "Julia_$(thread_count)"
        table[1, column_index + 1] = "Dynare_$(thread_count)"
        column_index += 2
    end

    for (row_index, (model_name, julia_values, dynare_values)) in enumerate(rows)
        table[row_index + 1, 1] = model_name
        column_index = 2
        for value_index in eachindex(thread_counts)
            table[row_index + 1, column_index] = isnan(julia_values[value_index]) ? "" : julia_values[value_index]
            table[row_index + 1, column_index + 1] = isnan(dynare_values[value_index]) ? "" : dynare_values[value_index]
            column_index += 2
        end
    end

    writedlm(path, table, ',')
end

function format_time(value)
    if isnan(value)
        return "N/A"
    elseif value < 1e-3
        return string(round(value * 1e6, digits = 1), " us")
    elseif value < 1.0
        return string(round(value * 1e3, digits = 2), " ms")
    else
        return string(round(value, digits = 3), " s")
    end
end

function print_summary_table(title, thread_counts, rows)
    println("\n--- $title ---")
    if isempty(rows)
        println("No benchmark rows found.")
        return
    end

    header = rpad("Model", 50)
    for thread_count in thread_counts
        header *= rpad("Julia_$(thread_count)", 12)
        header *= rpad("Dynare_$(thread_count)", 12)
    end
    println(header)
    println("-"^length(header))

    for (model_name, julia_values, dynare_values) in rows
        row_text = rpad(model_name, 50)
        for value_index in eachindex(thread_counts)
            row_text *= rpad(format_time(julia_values[value_index]), 12)
            row_text *= rpad(format_time(dynare_values[value_index]), 12)
        end
        println(row_text)
    end
end

function main(args = ARGS)
    output_root = parse_args(args)
    output_root === nothing && return
    isdir(output_root) || error("Output directory not found: $output_root")

    thread_counts, thread_dirs = collect_thread_dirs(output_root)

    println("Thread sweep output root: $output_root")
    println("Detected thread counts: $(join(string.(thread_counts), ", "))")
    print_environment_summary(thread_counts, thread_dirs)

    summaries = [
        ("First-Order Total", "benchmark_first_order_total_by_thread.csv", first_order_total),
        ("Second-Order Total", "benchmark_second_order_total_by_thread.csv", second_order_total),
        ("Comparable Direct Components Total", "benchmark_comparable_direct_total_by_thread.csv", comparable_direct_total),
        ("Higher-Order Bundled", "benchmark_higher_order_bundled_by_thread.csv", higher_order_bundled),
    ]

    for (title, file_name, metric_fn) in summaries
        rows = collect_rows(thread_counts, thread_dirs, metric_fn)
        print_summary_table(title, thread_counts, rows)
        if !isempty(rows)
            summary_path = joinpath(output_root, file_name)
            write_summary_csv(summary_path, thread_counts, rows)
            println("Wrote $(summary_path)")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end