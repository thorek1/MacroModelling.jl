# generate_julia_results.jl — Phase 1 of Dynare comparison
#
# Loads models, exports .mod files, and saves Julia-computed results as CSV.
# Output structure:
#   output/{model_name}/
#     {model_name}.mod
#     julia/
#       var_names.csv, exo_names.csv, state_var_names.csv
#       steady_state.csv, ghx.csv, ghu.csv
#       irf_fields.csv, irf_{var}_{shock}.csv
#       variance_covariance.csv
#
#   For higher-order models (pruned 2nd/3rd order):
#   output/{model_name}_pruned_2nd/  and  output/{model_name}_pruned_3rd/
#     Includes first-order comparable outputs plus higher-order moments:
#       steady_state.csv, ghx.csv, ghu.csv, irf_*.csv
#       variance_covariance.csv
#     Excludes higher-order solution-matrix CSVs (ghxx/ghxu/..., ghxxx/...)

# On Windows we deliberately switch the LinearAlgebra BLAS backend to MKL so the
# Julia side mirrors what Dynare/MATLAB use. MKL.jl must be loaded BEFORE any
# BLAS calls (including BLAS.set_num_threads) for it to take effect. On non-
# Windows platforms (e.g. Linux CI runners where MKL.jl may not be installed)
# we silently fall back to the default OpenBLAS backend.

# @static if Sys.iswindows()
#     using MKL
#     MKL.set_num_threads(Threads.nthreads())
#     @info "Using MKL.jl for BLAS on Windows with $(MKL.get_num_threads()) threads"
# end

using MacroModelling
using DelimitedFiles
using LinearAlgebra
using Sockets

const IRF_PERIODS = 40
const DEFAULT_OUTPUT_ROOT = joinpath(@__DIR__, "output")
const MODELS_DIR = joinpath(@__DIR__, "..", "..", "models")

# Models to test (first order)
const MODEL_FILES = [
    "FS2000",
    "Gali_2015_chapter_3_nonlinear",
    "Smets_Wouters_2007",
    "Smets_Wouters_2003",
    "NAWM_EAUS_2008",
    "GNSS_2010",
    "QUEST3_2009",
    "FRBUS",
]

# Models to also test at pruned 2nd order
const SECOND_ORDER_MODELS = [
    "FS2000",
    "Gali_2015_chapter_3_nonlinear",
    "Smets_Wouters_2007",
]

# Models to also test at pruned 3rd order
const THIRD_ORDER_MODELS = [
    "Gali_2015_chapter_3_nonlinear",
    "Caldara_et_al_2012",
]

# Models that skip variance/covariance
const SKIP_MOMENTS_MODELS = Set(["FRBUS", "NAWM"])

# Models for which only the benchmark timings are exported (no names, steady state,
# policy matrices, IRFs, or moments). The .mod file is still written so the Dynare
# phase can run and produce its own benchmark CSVs.
const BENCHMARK_ONLY_MODELS = Set(["FRBUS", "NAWM"])

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

function ascii_name(sym::Symbol)
    MacroModelling.translate_symbol_to_ascii(sym)
end

function original_vars(model)
    setdiff(model.constants.post_model_macro.vars_in_ss_equations,
            model.constants.post_model_macro.➕_vars)
end

function write_names(path, names)
    open(path, "w") do io
        for n in names
            println(io, n)
        end
    end
end

function print_usage()
    println("Usage: julia --project=. --threads=N generate_julia_results.jl [--output-root=PATH | PATH]")
end

function parse_args(args)
    output_root = DEFAULT_OUTPUT_ROOT
    positional_args = String[]
    only_models = String[]

    for arg in args
        if arg in ("-h", "--help")
            print_usage()
            return nothing
        elseif startswith(arg, "--output-root=")
            output_root = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--only-models=")
            value = split(arg, "=", limit = 2)[2]
            for token in split(value, ',')
                trimmed = strip(token)
                if !isempty(trimmed)
                    push!(only_models, String(trimmed))
                end
            end
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

    env_only = get(ENV, "DYNARE_COMPARE_ONLY_MODELS", "")
    if isempty(only_models) && !isempty(env_only)
        for token in split(env_only, ',')
            trimmed = strip(token)
            if !isempty(trimmed)
                push!(only_models, String(trimmed))
            end
        end
    end

    return (abspath(output_root), only_models)
end

function configure_julia_threads!()
    julia_threads = Threads.nthreads()
    BLAS.set_num_threads(julia_threads)
    blas_threads = BLAS.get_num_threads()
    println("Julia thread configuration: julia_threads=$julia_threads blas_threads=$blas_threads")
    println("  Threads.nthreads()       = ", Threads.nthreads())
    println("  Threads.nthreads(:default) = ", Threads.nthreads(:default))
    println("  Threads.nthreads(:interactive) = ", Threads.nthreads(:interactive))
    println("  BLAS.get_num_threads()   = ", BLAS.get_num_threads())
    blas_vendor = try
        string(BLAS.get_config())
    catch
        "unknown"
    end
    println("  BLAS vendor: $blas_vendor")
    for var in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        println("  ENV $var=", get(ENV, var, "<unset>"))
    end
    @info "Julia thread configuration" julia_threads blas_threads
    return julia_threads, blas_threads
end

function write_thread_configuration(output_root, julia_threads, blas_threads)
    open(joinpath(output_root, "julia_thread_configuration.txt"), "w") do io
        println(io, "julia_threads=$julia_threads")
        println(io, "blas_threads=$blas_threads")
    end
end

sanitize_metadata_value(value) = replace(string(value), r"[\r\n]+" => " | ")

function machine_hostname()
    try
        return Sockets.gethostname()
    catch
        return get(ENV, "COMPUTERNAME", get(ENV, "HOSTNAME", "unknown"))
    end
end

function write_key_value_metadata(path, entries)
    open(path, "w") do io
        for (key, value) in entries
            println(io, key, "=", sanitize_metadata_value(value))
        end
    end
end

function write_julia_environment_metadata(output_root, julia_threads, blas_threads)
    blas_lapack = try
        string(BLAS.get_config())
    catch
        "unknown"
    end
    total_memory_bytes = try
        Sys.total_memory()
    catch
        "unknown"
    end

    entries = [
        "julia_version" => VERSION,
        "julia_threads" => julia_threads,
        "julia_threads_default" => Threads.nthreads(:default),
        "julia_threads_interactive" => Threads.nthreads(:interactive),
        "blas_threads" => blas_threads,
        "blas_lapack" => blas_lapack,
        "hostname" => machine_hostname(),
        "kernel" => Sys.KERNEL,
        "arch" => Sys.ARCH,
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS,
        "word_size" => Sys.WORD_SIZE,
        "total_memory_bytes" => total_memory_bytes,
    ]

    write_key_value_metadata(joinpath(output_root, "comparison_environment_julia.txt"), entries)
end

function export_names_and_steady_state(model, julia_dir, orig, state_vars, exo_vars)
    var_names_ascii  = [ascii_name(v) for v in orig]
    exo_names_ascii  = [ascii_name(e) for e in exo_vars]
    state_names_ascii = [ascii_name(s) for s in state_vars]

    write_names(joinpath(julia_dir, "var_names.csv"), var_names_ascii)
    write_names(joinpath(julia_dir, "exo_names.csv"), exo_names_ascii)
    write_names(joinpath(julia_dir, "state_var_names.csv"), state_names_ascii)

    ss = get_SS(model, derivatives = false)
    ss_vals = [Float64(ss(v)) for v in orig]
    writedlm(joinpath(julia_dir, "steady_state.csv"), ss_vals, ',')

    return var_names_ascii, exo_names_ascii
end

function export_first_order_matrices(model, julia_dir, orig, state_vars, exo_vars)
    sol = get_solution(model, algorithm = :first_order)

    ghx = zeros(length(orig), length(state_vars))
    for (si, s) in enumerate(state_vars)
        s_key = Symbol(string(s) * "₍₋₁₎")
        for (vi, v) in enumerate(orig)
            ghx[vi, si] = Float64(sol(s_key, v))
        end
    end
    writedlm(joinpath(julia_dir, "ghx.csv"), ghx, ',')

    ghu = zeros(length(orig), length(exo_vars))
    for (ei, e) in enumerate(exo_vars)
        e_key = Symbol(string(e) * "₍ₓ₎")
        for (vi, v) in enumerate(orig)
            ghu[vi, ei] = Float64(sol(e_key, v))
        end
    end
    writedlm(joinpath(julia_dir, "ghu.csv"), ghu, ',')
    write_names(joinpath(julia_dir, "policy_algorithm.csv"), ["first_order"])
end

function export_irfs(model, julia_dir, orig, exo_vars; algorithm = :first_order)
    irfs = get_irf(model, periods = IRF_PERIODS, algorithm = algorithm)
    irf_fields = String[]
    n_cols = length(orig) * length(exo_vars)
    irf_matrix = Matrix{Float64}(undef, IRF_PERIODS, n_cols)
    col = 0
    for v in orig
        v_ascii = ascii_name(v)
        for e in exo_vars
            e_ascii = ascii_name(e)
            push!(irf_fields, "$(v_ascii)_$(e_ascii)")
            col += 1
            for t in 1:IRF_PERIODS
                irf_matrix[t, col] = Float64(irfs(v, t, e))
            end
        end
    end
    # Single bundled file: rows = periods, cols = irf_fields (in same order).
    writedlm(joinpath(julia_dir, "irfs.csv"), irf_matrix, ',')
    write_names(joinpath(julia_dir, "irf_fields.csv"), irf_fields)
    write_names(joinpath(julia_dir, "irf_algorithm.csv"), [String(algorithm)])
end

function write_benchmarks(julia_dir, bench::AbstractDict)
    keys_sorted = sort(collect(keys(bench)))
    table = Matrix{Any}(undef, length(keys_sorted), 2)
    for (i, k) in enumerate(keys_sorted)
        table[i, 1] = k
        table[i, 2] = bench[k]
    end
    writedlm(joinpath(julia_dir, "benchmarks.csv"), table, ',')
end

function export_moments(model, julia_dir, orig, exo_vars;
                        algorithm = :first_order,
                        var_names_ascii = nothing,
                        exo_names_ascii = nothing)
    moments = get_moments(model, algorithm = algorithm,
                          derivatives = false,
                          non_stochastic_steady_state = false,
                          mean = false,
                          variance = true,
                          standard_deviation = false,
                          covariance = true)

    vcov = zeros(length(orig), length(orig))
    covar_ka = moments[:covariance]
    for (ri, rv) in enumerate(orig)
        for (ci, cv) in enumerate(orig)
            vcov[ri, ci] = Float64(covar_ka(rv, cv))
        end
    end
    writedlm(joinpath(julia_dir, "variance_covariance.csv"), vcov, ',')
end

# ─────────────────────────────────────────────
# Benchmark helpers — manual median of N_BENCH runs
# ─────────────────────────────────────────────
const N_BENCH = 500

function median_elapsed(f, n = N_BENCH)
    times = Vector{Float64}(undef, n)
    for i in 1:n
        times[i] = @elapsed f()
    end
    sort!(times)
    m = length(times) ÷ 2
    return isodd(length(times)) ? times[m + 1] : (times[m] + times[m + 1]) / 2
end

function benchmark_first_order(model, bench::AbstractDict)
    params = copy(model.parameter_values)
    opts = MacroModelling.merge_calculation_options(verbose = true)

    # Warm up to ensure functions are compiled and reusable inputs are available
    MacroModelling.invalidate_cache_validity!(model)
    SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(model, params, opts = opts, caching = false)
    ∇₁ = MacroModelling.calculate_jacobian(params, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces, caching = false)
    MacroModelling.calculate_first_order_solution(∇₁, model.constants, model.workspaces, model.caches;
                                                  opts = opts, initial_guess = model.caches.qme_solution,
                                                  parameter_values = params, caching = false)

    opts = MacroModelling.merge_calculation_options()

    # Benchmark Jacobian (given precomputed steady-state inputs)
    median_jac = median_elapsed() do
        MacroModelling.calculate_jacobian(params, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces, caching = false)
    end
    bench["benchmark_jacobian"] = median_jac

    # Benchmark first-order solve (given Jacobian)
    median_fo = median_elapsed() do
        MacroModelling.calculate_first_order_solution(∇₁, model.constants, model.workspaces, model.caches;
                                                      opts = opts, initial_guess = model.caches.qme_solution,
                                                      parameter_values = params, caching = false)
    end
    median_fo_total = median_jac + median_fo
    bench["benchmark_first_order_solve"] = median_fo
    bench["benchmark_first_order_total"] = median_fo_total
    bench["benchmark_first_order"] = median_fo_total

    @info "Benchmark $(model.model_name) [first order]:"
    @info "  Jacobian:  $(round(median_jac*1e6, digits=1)) μs"
    @info "  QME solve: $(round(median_fo*1e6, digits=1)) μs"
    @info "  Total:     $(round(median_fo_total*1e6, digits=1)) μs"
end

function benchmark_second_order(model, bench::AbstractDict)
    params = copy(model.parameter_values)
    opts = MacroModelling.merge_calculation_options()

    # Warm up: run full pipeline once
    MacroModelling.invalidate_cache_validity!(model)
    SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(model, params, opts = opts, caching = false)
    ∇₁ = MacroModelling.calculate_jacobian(params, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces, caching = false)
    𝐒₁, _, _ = MacroModelling.calculate_first_order_solution(∇₁, model.constants, model.workspaces, model.caches;
                                                              opts = opts, initial_guess = model.caches.qme_solution,
                                                              parameter_values = params, caching = false)
    ∇₂ = MacroModelling.calculate_hessian(params, SS_and_pars, model.caches, model.functions.hessian, model.workspaces, caching = false)
    MacroModelling.calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, model.constants, model.workspaces, model.caches;
                                                   initial_guess = model.caches.second_order_solution,
                                                   opts = opts, parameter_values = params, caching = false)

    # Benchmark Hessian
    median_hess = median_elapsed() do
        MacroModelling.calculate_hessian(params, SS_and_pars, model.caches, model.functions.hessian, model.workspaces, caching = false)
    end
    bench["benchmark_hessian"] = median_hess

    # Benchmark second-order solve (given first-order solution + Hessian)
    median_so = median_elapsed() do
        MacroModelling.calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, model.constants, model.workspaces, model.caches;
                                                       initial_guess = model.caches.second_order_solution,
                                                       opts = opts, parameter_values = params, caching = false)
    end
    bench["benchmark_second_order_solve"] = median_so

    @info "Benchmark $(model.model_name) [second order]:"
    @info "  Hessian:          $(round(median_hess*1e6, digits=1)) μs"
    @info "  2nd order solve:  $(round(median_so*1e6, digits=1)) μs"
end

function benchmark_third_order(model, bench::AbstractDict)
    params = copy(model.parameter_values)
    opts = MacroModelling.merge_calculation_options()

    # Warm up: run full pipeline once
    MacroModelling.invalidate_cache_validity!(model)
    SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(model, params, opts = opts, caching = false)
    ∇₁ = MacroModelling.calculate_jacobian(params, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces, caching = false)
    𝐒₁, _, _ = MacroModelling.calculate_first_order_solution(∇₁, model.constants, model.workspaces, model.caches;
                                                              opts = opts, initial_guess = model.caches.qme_solution,
                                                              parameter_values = params, caching = false)
    ∇₂ = MacroModelling.calculate_hessian(params, SS_and_pars, model.caches, model.functions.hessian, model.workspaces, caching = false)
    𝐒₂, _ = MacroModelling.calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, model.constants, model.workspaces, model.caches;
                                                            initial_guess = model.caches.second_order_solution,
                                                            opts = opts, parameter_values = params, caching = false)
    ∇₃ = MacroModelling.calculate_third_order_derivatives(params, SS_and_pars, model.caches, model.functions.third_order_derivatives, model.workspaces, caching = false)
    MacroModelling.calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, model.constants, model.workspaces, model.caches;
                                                  initial_guess = model.caches.third_order_solution,
                                                  opts = opts, parameter_values = params, caching = false)

    # Benchmark third-order derivatives
    median_d3 = median_elapsed() do
        MacroModelling.calculate_third_order_derivatives(params, SS_and_pars, model.caches, model.functions.third_order_derivatives, model.workspaces, caching = false)
    end
    bench["benchmark_third_order_derivatives"] = median_d3

    # Benchmark third-order solve
    median_to = median_elapsed() do
        MacroModelling.calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, model.constants, model.workspaces, model.caches;
                                                      initial_guess = model.caches.third_order_solution,
                                                      opts = opts, parameter_values = params, caching = false)
    end
    bench["benchmark_third_order_solve"] = median_to

    @info "Benchmark $(model.model_name) [third order]:"
    @info "  3rd order derivs: $(round(median_d3*1e6, digits=1)) μs"
    @info "  3rd order solve:  $(round(median_to*1e6, digits=1)) μs"
end

# ─────────────────────────────────────────────
# Higher-order solution matrix extraction
# ─────────────────────────────────────────────

function orig_var_indices(model, orig)
    all_vars = model.constants.post_model_macro.var
    [findfirst(==(v), all_vars) for v in orig]
end

function export_second_order_matrices(model, julia_dir, orig)
    nPast = model.constants.post_model_macro.nPast_not_future_and_mixed
    nExo  = model.constants.post_model_macro.nExo
    nVars = model.constants.post_model_macro.nVars
    n_aug = nPast + 1 + nExo  # [states, σ, shocks]

    # Expand compressed solution to full tensor (nVars × n_aug²)
    sol2_full = model.caches.second_order_solution * model.constants.second_order.𝐔₂
    sol2_raw = reshape(Matrix(sol2_full), nVars, n_aug, n_aug)

    oi = orig_var_indices(model, orig)
    sol2 = sol2_raw[oi, :, :]
    nOrig = length(orig)
    state_range = 1:nPast
    σ_idx = nPast + 1
    exo_range = (nPast + 2):(nPast + 1 + nExo)

    # ghxx: nOrig × nState² — symmetric, column-major reshape matches Dynare kron(x,x)
    ghxx = reshape(sol2[:, state_range, state_range], nOrig, nPast * nPast)
    writedlm(joinpath(julia_dir, "ghxx.csv"), ghxx, ',')

    # ghxu: nOrig × (nState × nExo) — permutedims to match Dynare kron(x,u) convention
    ghxu_block = sol2[:, state_range, exo_range]  # (nOrig, nPast, nExo)
    ghxu = reshape(permutedims(ghxu_block, (1, 3, 2)), nOrig, nPast * nExo)
    writedlm(joinpath(julia_dir, "ghxu.csv"), ghxu, ',')

    # ghuu: nOrig × nExo² — symmetric
    ghuu = reshape(sol2[:, exo_range, exo_range], nOrig, nExo * nExo)
    writedlm(joinpath(julia_dir, "ghuu.csv"), ghuu, ',')

    # ghs2: nOrig × 1 — volatility correction
    ghs2 = sol2[:, σ_idx, σ_idx]
    writedlm(joinpath(julia_dir, "ghs2.csv"), ghs2, ',')
end

function export_third_order_matrices(model, julia_dir, orig)
    nPast = model.constants.post_model_macro.nPast_not_future_and_mixed
    nExo  = model.constants.post_model_macro.nExo
    nVars = model.constants.post_model_macro.nVars
    n_aug = nPast + 1 + nExo

    sol3_full = model.caches.third_order_solution * model.constants.third_order.𝐔₃
    sol3_raw = reshape(Matrix(sol3_full), nVars, n_aug, n_aug, n_aug)

    oi = orig_var_indices(model, orig)
    sol3 = sol3_raw[oi, :, :, :]
    nOrig = length(orig)
    sr = 1:nPast
    σ = nPast + 1
    er = (nPast + 2):(nPast + 1 + nExo)

    # ghxxx: nOrig × nState³ — symmetric, direct reshape
    ghxxx = reshape(sol3[:, sr, sr, sr], nOrig, nPast^3)
    writedlm(joinpath(julia_dir, "ghxxx.csv"), ghxxx, ',')

    # ghxxu: nOrig × (nState² × nExo) — permutedims [1,4,3,2] to match kron(x,kron(x,u))
    ghxxu = reshape(permutedims(sol3[:, sr, sr, er], (1, 4, 3, 2)), nOrig, nPast^2 * nExo)
    writedlm(joinpath(julia_dir, "ghxxu.csv"), ghxxu, ',')

    # ghxuu: nOrig × (nState × nExo²) — same permutation
    ghxuu = reshape(permutedims(sol3[:, sr, er, er], (1, 4, 3, 2)), nOrig, nPast * nExo^2)
    writedlm(joinpath(julia_dir, "ghxuu.csv"), ghxuu, ',')

    # ghuuu: nOrig × nExo³ — symmetric, direct reshape
    ghuuu = reshape(sol3[:, er, er, er], nOrig, nExo^3)
    writedlm(joinpath(julia_dir, "ghuuu.csv"), ghuuu, ',')

    # ghxss: nOrig × nState — coefficient for x_i * σ²
    ghxss = sol3[:, sr, σ, σ]
    writedlm(joinpath(julia_dir, "ghxss.csv"), ghxss, ',')

    # ghuss: nOrig × nExo — coefficient for u_j * σ²
    ghuss = sol3[:, er, σ, σ]
    writedlm(joinpath(julia_dir, "ghuss.csv"), ghuss, ',')
end

# ─────────────────────────────────────────────
# Export one model's first-order results
# ─────────────────────────────────────────────
function export_model(model, outdir; include_moments = true, benchmark_only = false)
    julia_dir = joinpath(outdir, "julia")
    mkpath(julia_dir)

    if benchmark_only
        # ── Export .mod file only (needed for the Dynare phase) ──
        cd(outdir) do
            write_mod_file(model)
        end

        # ── Benchmarks ──
        bench = Dict{String, Float64}()
        benchmark_first_order(model, bench)
        write_benchmarks(julia_dir, bench)

        @info "Exported Julia benchmark-only results for $(model.model_name) → $outdir"
        return
    end

    orig = original_vars(model)
    state_vars = model.constants.post_model_macro.past_not_future_and_mixed
    exo_vars = model.constants.post_model_macro.exo

    var_names_ascii, exo_names_ascii = export_names_and_steady_state(model, julia_dir, orig, state_vars, exo_vars)
    export_first_order_matrices(model, julia_dir, orig, state_vars, exo_vars)
    export_irfs(model, julia_dir, orig, exo_vars, algorithm = :first_order)

    if include_moments
        export_moments(model, julia_dir, orig, exo_vars;
                       algorithm = :first_order,
                       var_names_ascii = var_names_ascii,
                       exo_names_ascii = exo_names_ascii)
    end

    # ── Export .mod file ──
    cd(outdir) do
        write_mod_file(model)
    end

    # ── Benchmarks ──
    bench = Dict{String, Float64}()
    benchmark_first_order(model, bench)
    write_benchmarks(julia_dir, bench)

    @info "Exported Julia results for $(model.model_name) → $outdir"
end

# ─────────────────────────────────────────────
# Export one model's higher-order results
# ─────────────────────────────────────────────
function export_higher_order_model(model, outdir, dir_name, order)
    julia_dir = joinpath(outdir, "julia")
    mkpath(julia_dir)

    algorithm = order == 2 ? :pruned_second_order : :pruned_third_order

    # Trigger solve at the requested order (populates all caches up to that order)
    get_solution(model, algorithm = algorithm)

    orig = original_vars(model)
    state_vars = model.constants.post_model_macro.past_not_future_and_mixed
    exo_vars = model.constants.post_model_macro.exo

    export_names_and_steady_state(model, julia_dir, orig, state_vars, exo_vars)
    # Export first-order comparable objects even for higher-order model directories.
    export_first_order_matrices(model, julia_dir, orig, state_vars, exo_vars)
    export_irfs(model, julia_dir, orig, exo_vars, algorithm = :first_order)

    # Higher-order-specific outputs are moments-only (no higher-order solution matrices).
    export_moments(model, julia_dir, orig, exo_vars;
                   algorithm = algorithm)

    # ── Export .mod file with correct order and pruning, renamed to match directory ──
    cd(outdir) do
        write_mod_file(model, order = order, pruning = true)
        mv("$(model.model_name).mod", "$(dir_name).mod", force = true)
    end

    # ── Benchmarks ──
    bench = Dict{String, Float64}()
    benchmark_first_order(model, bench)
    benchmark_second_order(model, bench)
    if order >= 3
        benchmark_third_order(model, bench)
    end
    write_benchmarks(julia_dir, bench)

    @info "Exported Julia higher-order results (order=$order) for $(model.model_name) → $outdir"
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
function main(args = ARGS)
    parsed = parse_args(args)
    parsed === nothing && return
    output_root, only_models = parsed

    julia_threads, blas_threads = configure_julia_threads!()

    if isdir(output_root)
        rm(output_root, recursive = true)
    end
    mkpath(output_root)
    write_thread_configuration(output_root, julia_threads, blas_threads)
    write_julia_environment_metadata(output_root, julia_threads, blas_threads)

    only_set = Set(only_models)
    keep(name) = isempty(only_set) || (name in only_set)

    if !isempty(only_set)
        @info "Restricting Phase 1 to selected models" only_models
    end

    # Runtime tracking (wall-clock time post-startup for each model)
    model_runtimes = Vector{Pair{String, Float64}}()
    total_start = time()

    # Phase 1a: First-order exports for all models
    for mname in MODEL_FILES
        keep(mname) || continue
        @info "Processing model (first order): $mname"
        model_start = time()
        include(joinpath(MODELS_DIR, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(output_root, mname)
        mkpath(outdir)
        Base.invokelatest(export_model, model, outdir;
                         include_moments = !(mname in SKIP_MOMENTS_MODELS),
                         benchmark_only = mname in BENCHMARK_ONLY_MODELS)
        elapsed = time() - model_start
        push!(model_runtimes, mname => elapsed)
        @info "  Wall-clock time for $mname: $(round(elapsed, digits=3)) s"
    end

    # Phase 1b: Second-order exports for selected models
    for mname in SECOND_ORDER_MODELS
        keep(mname) || continue
        dir_name = "$(mname)_pruned_2nd"
        @info "Processing model (pruned order 2): $mname → $dir_name"

        model_start = time()
        include(joinpath(MODELS_DIR, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(output_root, dir_name)
        mkpath(outdir)
        Base.invokelatest(export_higher_order_model, model, outdir, dir_name, 2)
        elapsed = time() - model_start
        push!(model_runtimes, dir_name => elapsed)
        @info "  Wall-clock time for $dir_name: $(round(elapsed, digits=3)) s"
    end

    # Phase 1c: Third-order exports for selected models
    for mname in THIRD_ORDER_MODELS
        keep(mname) || continue
        dir_name = "$(mname)_pruned_3rd"
        @info "Processing model (pruned order 3): $mname → $dir_name"

        model_start = time()
        include(joinpath(MODELS_DIR, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(output_root, dir_name)
        mkpath(outdir)
        Base.invokelatest(export_higher_order_model, model, outdir, dir_name, 3)
        elapsed = time() - model_start
        push!(model_runtimes, dir_name => elapsed)
        @info "  Wall-clock time for $dir_name: $(round(elapsed, digits=3)) s"
    end

    total_elapsed = time() - total_start

    # Write runtime summary
    open(joinpath(output_root, "runtime_julia.csv"), "w") do io
        println(io, "model,elapsed_seconds")
        for (name, t) in model_runtimes
            println(io, "$name,$t")
        end
        println(io, "TOTAL,$total_elapsed")
    end
    @info "Total wall-clock time (post-startup): $(round(total_elapsed, digits=3)) s"
    @info "Phase 1 complete. Results in $output_root"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
