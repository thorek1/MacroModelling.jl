include(joinpath(@__DIR__, "compare_results.jl"))

const DEFAULT_SWEEP_ROOT = joinpath(@__DIR__, "output_thread_sweep")

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

function print_environment_summary(thread_counts, thread_dirs)
    first_thread = first(thread_counts)
    metadata_root = thread_dirs[first_thread]
    julia_metadata = read_key_value_metadata(joinpath(metadata_root, "comparison_environment_julia.txt"))
    dynare_metadata = read_key_value_metadata(joinpath(metadata_root, "comparison_environment_dynare.txt"))
    julia_source = "threads_$(first_thread) metadata"

    if isempty(julia_metadata)
        julia_metadata = current_julia_metadata()
        julia_source = "compare runtime fallback"
    end

    println("Run Environment (metadata source: threads_$(first_thread))")
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
    return sort!(collect(keys(thread_dirs))), thread_dirs
end

function model_names(thread_dir)
    filter(model_name -> isdir(joinpath(thread_dir, model_name, "julia")) &&
                         isdir(joinpath(thread_dir, model_name, "dynare")) &&
                         !is_excluded_model_dir(model_name),
           readdir(thread_dir))
end

function collect_model_names(thread_counts, thread_dirs; include_benchmark_only = true)
    model_set = Set{String}()
    for thread_count in thread_counts
        for model_name in model_names(thread_dirs[thread_count])
            push!(model_set, model_name)
        end
    end

    model_dir_names = sort!(collect(model_set))
    return include_benchmark_only ? model_dir_names : filter(mname -> !is_benchmark_only_model_dir(mname), model_dir_names)
end

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

is_dynare_k_order_dir(dir) = has_bench(dir, "benchmark_k_order_pert.csv")
read_dynare_k_order_pert(dir) = has_bench(dir, "benchmark_k_order_pert.csv") ? read_bench(dir, "benchmark_k_order_pert.csv") : NaN

jacobian_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_jacobian.csv"),
    read_bench(dynare_dir, "benchmark_jacobian.csv"),
)

first_order_solve_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_first_order_solve.csv"),
    read_bench(dynare_dir, "benchmark_first_order_solve.csv"),
)

first_order_total(julia_dir, dynare_dir) = (
    sum_bench_components(julia_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"]),
    sum_bench_components(dynare_dir, ["benchmark_jacobian.csv", "benchmark_first_order_solve.csv"]),
)

hessian_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_hessian.csv"),
    read_bench(dynare_dir, "benchmark_hessian.csv"),
)

second_order_solve_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_second_order_solve.csv"),
    read_bench(dynare_dir, "benchmark_second_order_solve.csv"),
)

second_order_total(julia_dir, dynare_dir) = (
    sum_bench_components(julia_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"]),
    sum_bench_components(dynare_dir, ["benchmark_hessian.csv", "benchmark_second_order_solve.csv"]),
)

comparable_direct_total(julia_dir, dynare_dir) = (
    sum_bench_components(julia_dir, [
        "benchmark_jacobian.csv",
        "benchmark_first_order_solve.csv",
        "benchmark_hessian.csv",
        "benchmark_second_order_solve.csv",
    ]),
    sum_bench_components(dynare_dir, [
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
    read_dynare_k_order_pert(dynare_dir),
)

third_order_derivatives_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_third_order_derivatives.csv"),
    NaN,
)

third_order_solve_metric(julia_dir, dynare_dir) = (
    read_bench(julia_dir, "benchmark_third_order_solve.csv"),
    NaN,
)

function collect_rows(thread_counts, thread_dirs, metric_fn; model_dir_names = collect_model_names(thread_counts, thread_dirs))
    rows = Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}()

    for model_name in model_dir_names
        macro_values = Float64[]
        dynare_values = Float64[]
        has_any_value = false

        for thread_count in thread_counts
            model_dir = joinpath(thread_dirs[thread_count], model_name)
            if isdir(joinpath(model_dir, "julia")) && isdir(joinpath(model_dir, "dynare"))
                macro_value, dynare_value = metric_fn(joinpath(model_dir, "julia"), joinpath(model_dir, "dynare"))
            else
                macro_value, dynare_value = NaN, NaN
            end

            push!(macro_values, macro_value)
            push!(dynare_values, dynare_value)
            has_any_value |= !isnan(macro_value) || !isnan(dynare_value)
        end

        has_any_value && push!(rows, (model_name, macro_values, dynare_values))
    end

    return rows
end

function write_summary_csv(path, thread_counts, rows)
    ncols = 1 + 2 * length(thread_counts)
    table = Matrix{Any}(undef, length(rows) + 1, ncols)
    table[1, 1] = "Model"

    column_index = 2
    for thread_count in thread_counts
        table[1, column_index] = "MacroModelling_$(thread_count)"
        table[1, column_index + 1] = "Dynare_$(thread_count)"
        column_index += 2
    end

    for (row_index, (model_name, macro_values, dynare_values)) in enumerate(rows)
        table[row_index + 1, 1] = model_name
        column_index = 2
        for value_index in eachindex(thread_counts)
            table[row_index + 1, column_index] = isnan(macro_values[value_index]) ? "" : macro_values[value_index]
            table[row_index + 1, column_index + 1] = isnan(dynare_values[value_index]) ? "" : dynare_values[value_index]
            column_index += 2
        end
    end

    writedlm(path, table, ',')
end

function print_summary_table(title, thread_counts, rows; note = "")
    println("\n--- $title ---")
    if !isempty(note)
        println("    $note")
    end
    if isempty(rows)
        println("No benchmark rows found.")
        return
    end

    header = rpad("Model", 50)
    for thread_count in thread_counts
        header *= rpad("MacroModelling_$(thread_count)", 18)
        header *= rpad("Dynare_$(thread_count)", 18)
    end
    println(header)
    println("-"^length(header))

    for (model_name, macro_values, dynare_values) in rows
        row_text = rpad(model_name, 50)
        for value_index in eachindex(thread_counts)
            row_text *= rpad(format_time(macro_values[value_index]), 18)
            row_text *= rpad(format_time(dynare_values[value_index]), 18)
        end
        println(row_text)
    end
end

function compare_result_sets(reference_results, candidate_results; model_name)
    moments_only_higher_order = is_higher_order_model(model_name)
    skip_pruned_third_order = is_pruned_third_order_model(model_name)

    @testset "Steady State" begin
        compare_steady_state(reference_results, candidate_results)
    end
    @testset "Policy Matrix ghx" begin
        if skip_pruned_third_order
            @info "Skipping ghx comparison for $model_name (pruned third-order state representation mismatch)"
        elseif moments_only_higher_order && get(reference_results, :policy_algorithm, "") != "first_order"
            @info "Skipping ghx comparison for $model_name (policy matrices not tagged as first-order; regenerate phase-1 outputs to enable)"
        else
            compare_ghx(reference_results, candidate_results)
        end
    end
    @testset "Policy Matrix ghu" begin
        if moments_only_higher_order && get(reference_results, :policy_algorithm, "") != "first_order"
            @info "Skipping ghu comparison for $model_name (policy matrices not tagged as first-order; regenerate phase-1 outputs to enable)"
        else
            compare_ghu(reference_results, candidate_results)
        end
    end
    @testset "IRFs" begin
        compare_irfs(reference_results, candidate_results; model_name = model_name)
    end
    @testset "Variance" begin
        if skip_pruned_third_order && !is_supported_pruned_third_order_variance_model(model_name)
            @info "Skipping variance comparison for $model_name (pruned third-order moment convention mismatch outside the validated benchmark cases)"
        else
            compare_variance(reference_results, candidate_results)
        end
    end
    @testset "Variance Decomposition" begin
        if moments_only_higher_order
            @info "Skipping variance decomposition comparison for $model_name (higher-order configured as covariance/variance moments-only)"
        else
            compare_variance_decomposition(reference_results, candidate_results)
        end
    end

    if has_second_order(reference_results) && has_second_order(candidate_results)
        @testset "Second Order Matrices" begin
            if moments_only_higher_order
                @info "Skipping second-order matrix comparison for $model_name (higher-order configured as moments-only)"
            else
                compare_second_order(reference_results, candidate_results)
            end
        end
    end
    if has_third_order(reference_results) && has_third_order(candidate_results)
        @testset "Third Order Matrices" begin
            if moments_only_higher_order
                @info "Skipping third-order matrix comparison for $model_name (higher-order configured as moments-only)"
            else
                compare_third_order(reference_results, candidate_results)
            end
        end
    end
end

function compare_thread_consistency(thread_counts, thread_dirs)
    if length(thread_counts) <= 1
        @info "Only one thread count detected; skipping cross-thread correctness comparison"
        return
    end

    reference_thread = first(thread_counts)
    all_model_dirs = collect_model_names(thread_counts, thread_dirs)
    comparison_model_dirs = collect_model_names(thread_counts, thread_dirs; include_benchmark_only = false)

    for mname in sort(filter(is_benchmark_only_model_dir, all_model_dirs))
        @info "Skipping cross-thread correctness comparison for benchmark-only model: $mname"
    end

    isempty(comparison_model_dirs) && return

    comparison_exception = nothing
    try
        @testset "Cross-Thread Consistency (reference: threads_$reference_thread)" begin
            for mname in sort(comparison_model_dirs)
                reference_model_dir = joinpath(thread_dirs[reference_thread], mname)
                reference_julia_dir = joinpath(reference_model_dir, "julia")
                reference_dynare_dir = joinpath(reference_model_dir, "dynare")

                @info "Comparing cross-thread results for: $mname"
                @testset "$mname" begin
                    if !isdir(reference_julia_dir) || !isdir(reference_dynare_dir)
                        @test isdir(reference_julia_dir)
                        @test isdir(reference_dynare_dir)
                        continue
                    end

                    reference_julia_results = load_results(reference_julia_dir)
                    reference_dynare_results = load_results(reference_dynare_dir)

                    for thread_count in thread_counts[2:end]
                        candidate_model_dir = joinpath(thread_dirs[thread_count], mname)
                        candidate_julia_dir = joinpath(candidate_model_dir, "julia")
                        candidate_dynare_dir = joinpath(candidate_model_dir, "dynare")

                        @testset "threads_$thread_count" begin
                            if !isdir(candidate_julia_dir) || !isdir(candidate_dynare_dir)
                                @test isdir(candidate_julia_dir)
                                @test isdir(candidate_dynare_dir)
                                continue
                            end

                            candidate_julia_results = load_results(candidate_julia_dir)
                            candidate_dynare_results = load_results(candidate_dynare_dir)

                            @testset "MacroModelling" begin
                                compare_result_sets(reference_julia_results, candidate_julia_results; model_name = mname)
                            end
                            @testset "Dynare" begin
                                compare_result_sets(reference_dynare_results, candidate_dynare_results; model_name = mname)
                            end
                        end
                    end
                end
            end
        end
    catch err
        if err isa Test.TestSetException
            comparison_exception = err
        else
            rethrow(err)
        end
    end

    comparison_exception === nothing || throw(comparison_exception)
end

function main(args = ARGS)
    output_root = parse_args(args)
    output_root === nothing && return
    isdir(output_root) || error("Output directory not found: $output_root")

    thread_counts, thread_dirs = collect_thread_dirs(output_root)

    println("Thread sweep output root: $output_root")
    println("Detected thread counts: $(join(string.(thread_counts), ", "))")
    print_environment_summary(thread_counts, thread_dirs)
    compare_thread_consistency(thread_counts, thread_dirs)

    println("\n", "="^100)
    println("  Benchmark Summary by Thread: MacroModelling vs Dynare")
    println("="^100)

    summaries = [
        ("Jacobian", "benchmark_jacobian_by_thread.csv", jacobian_metric, ""),
        ("First-Order Solve", "benchmark_first_order_solve_by_thread.csv", first_order_solve_metric, ""),
        ("First-Order Total (sum of direct Jacobian + solve medians)", "benchmark_first_order_total_by_thread.csv", first_order_total, ""),
        ("Hessian", "benchmark_hessian_by_thread.csv", hessian_metric, ""),
        ("Second-Order Solve", "benchmark_second_order_solve_by_thread.csv", second_order_solve_metric, ""),
        ("Second-Order Total (Hessian + Second-Order Solve)", "benchmark_second_order_total_by_thread.csv", second_order_total, ""),
        ("Comparable Direct Components Total (Jacobian + FO + Hessian + SO)", "benchmark_comparable_direct_total_by_thread.csv", comparable_direct_total, ""),
        ("Higher-Order Bundled (Dynare k_order_pert)", "benchmark_higher_order_bundled_by_thread.csv", higher_order_bundled,
         "MacroModelling sums directly measured solve-stack components; Dynare reports direct bundled k_order_pert"),
        ("Third-Order Derivatives (MacroModelling only)", "benchmark_third_order_derivatives_by_thread.csv", third_order_derivatives_metric,
         "Dynare does not export a directly comparable component-level metric."),
        ("Third-Order Solve (MacroModelling only)", "benchmark_third_order_solve_by_thread.csv", third_order_solve_metric,
         "Dynare does not export a directly comparable component-level metric."),
    ]

    model_dir_names = collect_model_names(thread_counts, thread_dirs)
    for (title, file_name, metric_fn, note) in summaries
        rows = collect_rows(thread_counts, thread_dirs, metric_fn; model_dir_names = model_dir_names)
        print_summary_table(title, thread_counts, rows; note = note)
        if !isempty(rows)
            summary_path = joinpath(output_root, file_name)
            write_summary_csv(summary_path, thread_counts, rows)
            println("Wrote $(summary_path)")
        end
    end

    println("="^100)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end