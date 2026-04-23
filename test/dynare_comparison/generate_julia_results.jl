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
#       variance_decomposition.csv, variance_decomposition_var_names.csv,
#       variance_decomposition_exo_names.csv
#
#   For higher-order models (pruned 2nd/3rd order):
#   output/{model_name}_pruned_2nd/  and  output/{model_name}_pruned_3rd/
#     Includes first-order comparable outputs plus higher-order moments:
#       steady_state.csv, ghx.csv, ghu.csv, irf_*.csv
#       variance_covariance.csv
#     Excludes higher-order solution-matrix CSVs (ghxx/ghxu/..., ghxxx/...)

using MacroModelling
using DelimitedFiles
using BenchmarkTools

const IRF_PERIODS = 40
const OUTPUT_ROOT = joinpath(@__DIR__, "output")

# Models to test (first order)
const MODEL_FILES = [
    "FS2000",
    "Gali_2015_chapter_3_nonlinear",
    "Smets_Wouters_2007",
    "NAWM_EAUS_2008",
    "GNSS_2010",
    "QUEST3_2009",
]

# Models to also test at pruned 2nd order
const SECOND_ORDER_MODELS = [
    "FS2000",
    "Gali_2015_chapter_3_nonlinear",
]

# Models to also test at pruned 3rd order
const THIRD_ORDER_MODELS = [
    "Gali_2015_chapter_3_nonlinear",
]

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
    for v in orig
        v_ascii = ascii_name(v)
        for e in exo_vars
            e_ascii = ascii_name(e)
            field = "$(v_ascii)_$(e_ascii)"
            push!(irf_fields, field)
            irf_vec = [Float64(irfs(v, t, e)) for t in 1:IRF_PERIODS]
            writedlm(joinpath(julia_dir, "irf_$(field).csv"), irf_vec, ',')
        end
    end
    write_names(joinpath(julia_dir, "irf_fields.csv"), irf_fields)
    write_names(joinpath(julia_dir, "irf_algorithm.csv"), [String(algorithm)])
end

function export_moments(model, julia_dir, orig, exo_vars;
                        algorithm = :first_order,
                        include_variance_decomposition = true,
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

    if include_variance_decomposition
        vd = get_variance_decomposition(model)
        vd_mat = zeros(length(orig), length(exo_vars))
        for (vi, v) in enumerate(orig)
            for (ei, e) in enumerate(exo_vars)
                vd_mat[vi, ei] = Float64(vd(v, e)) * 100.0
            end
        end
        writedlm(joinpath(julia_dir, "variance_decomposition.csv"), vd_mat, ',')
        write_names(joinpath(julia_dir, "variance_decomposition_var_names.csv"), something(var_names_ascii, [ascii_name(v) for v in orig]))
        write_names(joinpath(julia_dir, "variance_decomposition_exo_names.csv"), something(exo_names_ascii, [ascii_name(e) for e in exo_vars]))
    end
end

# ─────────────────────────────────────────────
# Benchmark helpers using BenchmarkTools
# ─────────────────────────────────────────────

function benchmark_first_order(model, julia_dir)
    params = copy(model.parameter_values)
    opts = MacroModelling.merge_calculation_options()

    # Warm up to ensure functions are compiled and reusable inputs are available
    MacroModelling.invalidate_cache_validity!(model)
    SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(model, params, opts = opts, caching = false)
    ∇₁ = MacroModelling.calculate_jacobian(params, SS_and_pars, model.caches, model.functions.jacobian, model.workspaces, caching = false)
    MacroModelling.calculate_first_order_solution(∇₁, model.constants, model.workspaces, model.caches;
                                                  opts = opts, initial_guess = model.caches.qme_solution,
                                                  parameter_values = params, caching = false)

    # Benchmark Jacobian (given precomputed steady-state inputs)
    b_jac = @benchmark begin
        MacroModelling.calculate_jacobian($params, $SS_and_pars, $model.caches, $model.functions.jacobian, $model.workspaces, caching = false)
    end
    median_jac = median(b_jac).time / 1e9
    writedlm(joinpath(julia_dir, "benchmark_jacobian.csv"), [median_jac], ',')

    # Benchmark first-order solve (given Jacobian)
    b_fo = @benchmark begin
        MacroModelling.calculate_first_order_solution($∇₁, $model.constants, $model.workspaces, $model.caches;
                                                      opts = $opts, initial_guess = $model.caches.qme_solution,
                                                      parameter_values = $params, caching = false)
    end
    median_fo = median(b_fo).time / 1e9
    median_fo_total = median_jac + median_fo
    writedlm(joinpath(julia_dir, "benchmark_first_order_solve.csv"), [median_fo], ',')
    writedlm(joinpath(julia_dir, "benchmark_first_order_total.csv"), [median_fo_total], ',')
    writedlm(joinpath(julia_dir, "benchmark_first_order.csv"), [median_fo_total], ',')

    @info "Benchmark $(model.model_name) [first order]:"
    @info "  Jacobian:  $(round(median_jac*1e6, digits=1)) μs"
    @info "  QME solve: $(round(median_fo*1e6, digits=1)) μs"
    @info "  Total:     $(round(median_fo_total*1e6, digits=1)) μs"
end

function benchmark_second_order(model, julia_dir)
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
    b_hess = @benchmark begin
        MacroModelling.calculate_hessian($params, $SS_and_pars, $model.caches, $model.functions.hessian, $model.workspaces, caching = false)
    end
    median_hess = median(b_hess).time / 1e9
    writedlm(joinpath(julia_dir, "benchmark_hessian.csv"), [median_hess], ',')

    # Benchmark second-order solve (given first-order solution + Hessian)
    b_so = @benchmark begin
        MacroModelling.calculate_second_order_solution($∇₁, $∇₂, $𝐒₁, $model.constants, $model.workspaces, $model.caches;
                                                       initial_guess = $model.caches.second_order_solution,
                                                       opts = $opts, parameter_values = $params, caching = false)
    end
    median_so = median(b_so).time / 1e9
    writedlm(joinpath(julia_dir, "benchmark_second_order_solve.csv"), [median_so], ',')

    @info "Benchmark $(model.model_name) [second order]:"
    @info "  Hessian:          $(round(median_hess*1e6, digits=1)) μs"
    @info "  2nd order solve:  $(round(median_so*1e6, digits=1)) μs"
end

function benchmark_third_order(model, julia_dir)
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
    b_d3 = @benchmark begin
        MacroModelling.calculate_third_order_derivatives($params, $SS_and_pars, $model.caches, $model.functions.third_order_derivatives, $model.workspaces, caching = false)
    end
    median_d3 = median(b_d3).time / 1e9
    writedlm(joinpath(julia_dir, "benchmark_third_order_derivatives.csv"), [median_d3], ',')

    # Benchmark third-order solve
    b_to = @benchmark begin
        MacroModelling.calculate_third_order_solution($∇₁, $∇₂, $∇₃, $𝐒₁, $𝐒₂, $model.constants, $model.workspaces, $model.caches;
                                                      initial_guess = $model.caches.third_order_solution,
                                                      opts = $opts, parameter_values = $params, caching = false)
    end
    median_to = median(b_to).time / 1e9
    writedlm(joinpath(julia_dir, "benchmark_third_order_solve.csv"), [median_to], ',')

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
function export_model(model, outdir)
    julia_dir = joinpath(outdir, "julia")
    mkpath(julia_dir)

    orig = original_vars(model)
    state_vars = model.constants.post_model_macro.past_not_future_and_mixed
    exo_vars = model.constants.post_model_macro.exo

    var_names_ascii, exo_names_ascii = export_names_and_steady_state(model, julia_dir, orig, state_vars, exo_vars)
    export_first_order_matrices(model, julia_dir, orig, state_vars, exo_vars)
    export_irfs(model, julia_dir, orig, exo_vars, algorithm = :first_order)

    export_moments(model, julia_dir, orig, exo_vars;
                   algorithm = :first_order,
                   include_variance_decomposition = true,
                   var_names_ascii = var_names_ascii,
                   exo_names_ascii = exo_names_ascii)

    # ── Export .mod file ──
    cd(outdir) do
        write_mod_file(model)
    end

    # ── Benchmarks ──
    benchmark_first_order(model, julia_dir)

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
                   algorithm = algorithm,
                   include_variance_decomposition = false)

    # ── Export .mod file with correct order and pruning, renamed to match directory ──
    cd(outdir) do
        write_mod_file(model, order = order, pruning = true)
        mv("$(model.model_name).mod", "$(dir_name).mod", force = true)
    end

    # ── Benchmarks ──
    benchmark_first_order(model, julia_dir)
    benchmark_second_order(model, julia_dir)
    if order >= 3
        benchmark_third_order(model, julia_dir)
    end

    @info "Exported Julia higher-order results (order=$order) for $(model.model_name) → $outdir"
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
function main()
    if isdir(OUTPUT_ROOT)
        rm(OUTPUT_ROOT, recursive = true)
    end
    mkpath(OUTPUT_ROOT)

    models_dir = joinpath(@__DIR__, "..", "..", "models")

    # Phase 1a: First-order exports for all models
    for mname in MODEL_FILES
        @info "Processing model (first order): $mname"
        include(joinpath(models_dir, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(OUTPUT_ROOT, mname)
        mkpath(outdir)
        Base.invokelatest(export_model, model, outdir)
    end

    # Phase 1b: Second-order exports for selected models
    for mname in SECOND_ORDER_MODELS
        dir_name = "$(mname)_pruned_2nd"
        @info "Processing model (pruned order 2): $mname → $dir_name"

        include(joinpath(models_dir, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(OUTPUT_ROOT, dir_name)
        mkpath(outdir)
        Base.invokelatest(export_higher_order_model, model, outdir, dir_name, 2)
    end

    # Phase 1c: Third-order exports for selected models
    for mname in THIRD_ORDER_MODELS
        dir_name = "$(mname)_pruned_3rd"
        @info "Processing model (pruned order 3): $mname → $dir_name"

        include(joinpath(models_dir, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(OUTPUT_ROOT, dir_name)
        mkpath(outdir)
        Base.invokelatest(export_higher_order_model, model, outdir, dir_name, 3)
    end

    @info "Phase 1 complete. Results in $OUTPUT_ROOT"
end

main()
