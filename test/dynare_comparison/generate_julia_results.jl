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

using MacroModelling
using DelimitedFiles

const IRF_PERIODS = 40
const OUTPUT_ROOT = joinpath(@__DIR__, "output")

# Models to test
const MODEL_FILES = [
    "RBC_baseline",
    "FS2000",
    "Ireland_2004",
    "Gali_2015_chapter_3_nonlinear",
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

function ascii_name(sym::Symbol)
    MacroModelling.translate_symbol_to_ascii(sym)
end

function original_vars(model)
    # Same variable set that write_mod_file exports to Dynare
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

# ─────────────────────────────────────────────
# Export one model's Julia results
# ─────────────────────────────────────────────
function export_model(model, outdir)
    julia_dir = joinpath(outdir, "julia")
    mkpath(julia_dir)

    orig = original_vars(model)
    state_vars = model.constants.post_model_macro.past_not_future_and_mixed
    exo_vars = model.constants.post_model_macro.exo

    # ASCII name lists (only original, non-auxiliary variables)
    var_names_ascii = [ascii_name(v) for v in orig]
    exo_names_ascii = [ascii_name(e) for e in exo_vars]
    state_names_ascii = [ascii_name(s) for s in state_vars]

    write_names(joinpath(julia_dir, "var_names.csv"), var_names_ascii)
    write_names(joinpath(julia_dir, "exo_names.csv"), exo_names_ascii)
    write_names(joinpath(julia_dir, "state_var_names.csv"), state_names_ascii)

    # ── Steady state ──
    ss = get_SS(model, derivatives = false)
    ss_vals = [Float64(ss(v)) for v in orig]
    writedlm(joinpath(julia_dir, "steady_state.csv"), ss_vals, ',')

    # ── First-order solution (ghx, ghu) ──
    sol = get_solution(model, algorithm = :first_order)

    # ghx: nVars × nStates matrix (rows = orig vars, cols = state vars)
    # In MM: sol(Symbol("k₍₋₁₎"), :c) = coefficient of var c w.r.t. lagged state k
    # Dynare convention: ghx[var_row, state_col]
    ghx = zeros(length(orig), length(state_vars))
    for (si, s) in enumerate(state_vars)
        s_key = Symbol(string(s) * "₍₋₁₎")
        for (vi, v) in enumerate(orig)
            ghx[vi, si] = Float64(sol(s_key, v))
        end
    end
    writedlm(joinpath(julia_dir, "ghx.csv"), ghx, ',')

    # ghu: nVars × nExo matrix (rows = orig vars, cols = shocks)
    ghu = zeros(length(orig), length(exo_vars))
    for (ei, e) in enumerate(exo_vars)
        e_key = Symbol(string(e) * "₍ₓ₎")
        for (vi, v) in enumerate(orig)
            ghu[vi, ei] = Float64(sol(e_key, v))
        end
    end
    writedlm(joinpath(julia_dir, "ghu.csv"), ghu, ',')

    # ── IRFs ──
    # IRF axes: (Variables, Periods 1:N, Shocks)
    irfs = get_irf(model, periods = IRF_PERIODS, algorithm = :first_order)
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

    # ── Variance-covariance ──
    moments = get_moments(model, algorithm = :first_order,
                          derivatives = false,
                          non_stochastic_steady_state = false,
                          mean = false,
                          variance = true,
                          standard_deviation = false,
                          covariance = true)

    # Build nVars × nVars covariance matrix for original vars
    vcov = zeros(length(orig), length(orig))
    covar_ka = moments[:covariance]
    for (ri, rv) in enumerate(orig)
        for (ci, cv) in enumerate(orig)
            vcov[ri, ci] = Float64(covar_ka(rv, cv))
        end
    end
    writedlm(joinpath(julia_dir, "variance_covariance.csv"), vcov, ',')

    # ── Variance decomposition (as percentages 0-100) ──
    vd = get_variance_decomposition(model)
    vd_mat = zeros(length(orig), length(exo_vars))
    for (vi, v) in enumerate(orig)
        for (ei, e) in enumerate(exo_vars)
            vd_mat[vi, ei] = Float64(vd(v, e)) * 100.0
        end
    end
    writedlm(joinpath(julia_dir, "variance_decomposition.csv"), vd_mat, ',')
    write_names(joinpath(julia_dir, "variance_decomposition_var_names.csv"), var_names_ascii)
    write_names(joinpath(julia_dir, "variance_decomposition_exo_names.csv"), exo_names_ascii)

    # ── Export .mod file ──
    cd(outdir) do
        write_mod_file(model)
    end

    # ── Benchmark: NSSS + Jacobian + first-order solve ──
    # Match Dynare's resol benchmark path by timing the full first-order
    # pipeline from a cold solution cache on each iteration.
    N_BENCH = 100
    times = Vector{Float64}(undef, N_BENCH)
    params = copy(model.parameter_values)

    # Warm-up on a cold cache (mirrors Dynare's warm-up resol call)
    MacroModelling.clear_solution_caches!(model, :first_order)
    _, _, solved_warmup = get_solution(model, params; algorithm = :first_order, caching = false)
    @assert solved_warmup "Warm-up first-order solve failed for $(model.model_name)"

    for i in 1:N_BENCH
        MacroModelling.clear_solution_caches!(model, :first_order)
        times[i] = @elapsed begin
            _, _, solved = get_solution(model, params; algorithm = :first_order, caching = false)
            @assert solved "First-order solve failed for $(model.model_name) in benchmark iteration $i"
        end
    end
    median_time = sort(times)[div(N_BENCH, 2) + 1]
    writedlm(joinpath(julia_dir, "benchmark_first_order.csv"), [median_time], ',')
    @info "Benchmark $(model.model_name) (NSSS + Jacobian + first-order solve): median=$(round(median_time*1e6, digits=1))μs over $N_BENCH runs"

    @info "Exported Julia results for $(model.model_name) → $outdir"
end

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
function main()
    # Clean output directory
    if isdir(OUTPUT_ROOT)
        rm(OUTPUT_ROOT, recursive = true)
    end
    mkpath(OUTPUT_ROOT)

    models_dir = joinpath(@__DIR__, "..", "..", "models")

    for mname in MODEL_FILES
        @info "Processing model: $mname"
        include(joinpath(models_dir, "$mname.jl"))
        model = Base.invokelatest(getfield, Main, Symbol(mname))
        outdir = joinpath(OUTPUT_ROOT, mname)
        mkpath(outdir)
        Base.invokelatest(export_model, model, outdir)
    end

    @info "Phase 1 complete. Results in $OUTPUT_ROOT"
end

main()
