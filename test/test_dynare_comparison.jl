using MacroModelling
using Test
using DelimitedFiles

const RTOL = 1e-6
const IRF_PERIODS = 40

# ─────────────────────────────────────────────
# Helper: check Octave + Dynare availability
# ─────────────────────────────────────────────
function check_octave_dynare()
    octave_cmd = coalesce(Sys.which("octave"), Sys.which("octave-cli"))
    octave_cmd === nothing && return false

    dynare_probe = """
        if exist('dynare', 'file') ~= 2
            dynare_paths = {'/usr/lib/dynare/matlab', '/usr/share/dynare/matlab', '/usr/local/lib/dynare/matlab'};
            for i = 1:length(dynare_paths)
                if exist(dynare_paths{i}, 'dir')
                    addpath(dynare_paths{i});
                end
            end
        end
        if exist('dynare', 'file') == 2
            disp('dynare_ok');
        else
            disp('no_dynare');
        end
    """

    try
        out = read(Cmd([octave_cmd, "--no-gui", "--quiet", "--eval", dynare_probe]), String)
        return contains(out, "dynare_ok")
    catch
        return false
    end
end

# ─────────────────────────────────────────────
# Helper: get original (non-auxiliary) variable names
# ─────────────────────────────────────────────
function original_var_names(model)
    filter(v -> !contains(string(v), "➕"), model.constants.post_model_macro.var)
end

ascii_name(sym::Symbol) = MacroModelling.translate_symbol_to_ascii(sym)

function write_name_file(path, names)
    open(path, "w") do io
        for name in names
            println(io, name)
        end
    end
end

function key_index(keys, key)
    idx = findfirst(==(key), keys)
    idx === nothing && error("Key $(repr(key)) not found in saved Julia output axes")
    return idx
end

function write_julia_results(model, output_dir)
    mkpath(output_dir)

    steady_state = get_SS(model, derivatives = false)
    solution = get_solution(model, algorithm = :first_order, silent = true)
    steady_state_keys = Set(collect(axiskeys(steady_state, 1)))
    solution_rows = collect(axiskeys(solution, 1))
    solution_cols = collect(axiskeys(solution, 2))
    solution_row_keys = Set(solution_rows)
    solution_col_keys = Set(solution_cols)
    solution_array = Array(solution)

    var_syms = filter(v -> v in steady_state_keys && v in solution_col_keys, original_var_names(model))
    state_syms = filter(s -> Symbol(string(s) * "₍₋₁₎") in solution_row_keys,
                        collect(model.constants.post_model_macro.past_not_future_and_mixed))
    exo_syms = filter(e -> Symbol(string(e) * "₍ₓ₎") in solution_row_keys,
                      collect(model.constants.post_model_macro.exo))

    var_names = ascii_name.(var_syms)
    state_var_names = ascii_name.(state_syms)
    exo_names = ascii_name.(exo_syms)

    write_name_file(joinpath(output_dir, "var_names.csv"), var_names)
    write_name_file(joinpath(output_dir, "state_var_names.csv"), state_var_names)
    write_name_file(joinpath(output_dir, "exo_names.csv"), exo_names)
    writedlm(joinpath(output_dir, "steady_state.csv"),
             Float64[steady_state(v) for v in var_syms], ',')

    ghx = Matrix{Float64}(undef, length(var_syms), length(state_syms))
    for (v_idx, v) in enumerate(var_syms)
        col_idx = key_index(solution_cols, v)
        for (s_idx, s) in enumerate(state_syms)
            row_idx = key_index(solution_rows, Symbol(string(s) * "₍₋₁₎"))
            ghx[v_idx, s_idx] = solution_array[row_idx, col_idx]
        end
    end
    writedlm(joinpath(output_dir, "ghx.csv"), ghx, ',')

    ghu = Matrix{Float64}(undef, length(var_syms), length(exo_syms))
    for (v_idx, v) in enumerate(var_syms)
        col_idx = key_index(solution_cols, v)
        for (e_idx, e) in enumerate(exo_syms)
            row_idx = key_index(solution_rows, Symbol(string(e) * "₍ₓ₎"))
            ghu[v_idx, e_idx] = solution_array[row_idx, col_idx]
        end
    end
    writedlm(joinpath(output_dir, "ghu.csv"), ghu, ',')

    irfs = get_irf(model, periods = IRF_PERIODS, algorithm = :first_order)
    irf_vars = collect(axiskeys(irfs, 1))
    irf_shocks = collect(axiskeys(irfs, 3))
    irf_array = Array(irfs)
    irf_fields = String[]

    for v in var_syms
        v_idx = findfirst(==(v), irf_vars)
        v_idx === nothing && continue
        for e in exo_syms
            e_idx = findfirst(==(e), irf_shocks)
            e_idx === nothing && continue

            field = "$(ascii_name(v))_$(ascii_name(e))"
            push!(irf_fields, field)
            writedlm(joinpath(output_dir, "irf_$field.csv"),
                     vec(irf_array[v_idx, :, e_idx]), ',')
        end
    end

    write_name_file(joinpath(output_dir, "irf_fields.csv"), irf_fields)

    moments = get_moments(model,
                          algorithm = :first_order,
                          derivatives = false,
                          non_stochastic_steady_state = false,
                          mean = false,
                          variance = true,
                          standard_deviation = true,
                          covariance = true)
    variance_covariance = [Float64(moments[:covariance](v1, v2)) for v1 in var_syms, v2 in var_syms]
    writedlm(joinpath(output_dir, "variance_covariance.csv"), variance_covariance, ',')

    variance_decomposition = get_variance_decomposition(model)
    vd = [Float64(variance_decomposition(v, e)) * 100.0 for v in var_syms, e in exo_syms]
    writedlm(joinpath(output_dir, "variance_decomposition.csv"), vd, ',')
    write_name_file(joinpath(output_dir, "variance_decomposition_var_names.csv"), var_names)
    write_name_file(joinpath(output_dir, "variance_decomposition_exo_names.csv"), exo_names)

    return output_dir
end

function prepare_comparison_artifacts(model, workdir)
    cd(workdir) do
        write_mod_file(model)
    end

    julia_output_dir = joinpath(workdir, "$(model.model_name)_julia_results")
    write_julia_results(model, julia_output_dir)
    return julia_output_dir
end

# ─────────────────────────────────────────────
# Run Dynare via Octave in a working directory
# ─────────────────────────────────────────────
function run_dynare(model, workdir)
    mod_name = string(model.model_name)

    script_dir = joinpath(@__DIR__, "dynare_comparison")
    cp(joinpath(script_dir, "extract_dynare_results.m"),
       joinpath(workdir, "extract_dynare_results.m"), force = true)
    cp(joinpath(script_dir, "run_model.m"),
       joinpath(workdir, "run_model.m"), force = true)

    cmd = `octave --no-gui --eval "model_name='$mod_name'; output_dir='$(mod_name)_results'; run('run_model.m')"`
    cd(() -> run(cmd), workdir)

    return joinpath(workdir, "$(mod_name)_results")
end

# ─────────────────────────────────────────────
# Parse CSV files written by Octave extraction
# ─────────────────────────────────────────────
function parse_dynare_results(output_dir)
    results = Dict{Symbol, Any}()

    results[:var_names] = strip.(readlines(joinpath(output_dir, "var_names.csv")))
    results[:exo_names] = strip.(readlines(joinpath(output_dir, "exo_names.csv")))
    results[:state_var_names] = strip.(readlines(joinpath(output_dir, "state_var_names.csv")))

    results[:steady_state] = vec(readdlm(joinpath(output_dir, "steady_state.csv"), ',', Float64))
    results[:ghx] = readdlm(joinpath(output_dir, "ghx.csv"), ',', Float64)
    results[:ghu] = readdlm(joinpath(output_dir, "ghu.csv"), ',', Float64)

    if isfile(joinpath(output_dir, "irf_fields.csv"))
        irf_fields = strip.(readlines(joinpath(output_dir, "irf_fields.csv")))
        irfs = Dict{String, Vector{Float64}}()
        for f in irf_fields
            path = joinpath(output_dir, "irf_$f.csv")
            if isfile(path)
                irfs[f] = vec(readdlm(path, ',', Float64))
            end
        end
        results[:irfs] = irfs
    end

    vcov_path = joinpath(output_dir, "variance_covariance.csv")
    if isfile(vcov_path)
        results[:variance_covariance] = readdlm(vcov_path, ',', Float64)
    end

    vd_path = joinpath(output_dir, "variance_decomposition.csv")
    if isfile(vd_path)
        results[:variance_decomposition] = readdlm(vd_path, ',', Float64)
        results[:vd_var_names] = strip.(readlines(joinpath(output_dir, "variance_decomposition_var_names.csv")))
        results[:vd_exo_names] = strip.(readlines(joinpath(output_dir, "variance_decomposition_exo_names.csv")))
    end

    results
end

# ─────────────────────────────────────────────
# Compare: Steady State
# ─────────────────────────────────────────────
function compare_steady_state(julia_results, dynare; rtol = RTOL)
    for (j_idx, var_name) in enumerate(julia_results[:var_names])
        d_idx = findfirst(==(var_name), dynare[:var_names])
        if d_idx === nothing
            @warn "Variable $var_name not found in Dynare output, skipping"
            continue
        end
        mm_val = julia_results[:steady_state][j_idx]
        dy_val = dynare[:steady_state][d_idx]
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end
end

# ─────────────────────────────────────────────
# Compare: Policy / transition matrices
# ─────────────────────────────────────────────
function compare_policy_matrices(julia_results, dynare; rtol = RTOL)
    for (j_v_idx, var_name) in enumerate(julia_results[:var_names])
        dy_v_idx = findfirst(==(var_name), dynare[:var_names])
        dy_v_idx === nothing && continue

        for (j_s_idx, state_name) in enumerate(julia_results[:state_var_names])
            dy_s_idx = findfirst(==(state_name), dynare[:state_var_names])
            dy_s_idx === nothing && continue

            mm_val = julia_results[:ghx][j_v_idx, j_s_idx]
            dy_val = dynare[:ghx][dy_v_idx, dy_s_idx]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
                  (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
        end
    end

    for (j_v_idx, var_name) in enumerate(julia_results[:var_names])
        dy_v_idx = findfirst(==(var_name), dynare[:var_names])
        dy_v_idx === nothing && continue

        for (j_e_idx, exo_name) in enumerate(julia_results[:exo_names])
            dy_e_idx = findfirst(==(exo_name), dynare[:exo_names])
            dy_e_idx === nothing && continue

            mm_val = julia_results[:ghu][j_v_idx, j_e_idx]
            dy_val = dynare[:ghu][dy_v_idx, dy_e_idx]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
                  (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
        end
    end
end

# ─────────────────────────────────────────────
# Compare: Impulse Response Functions
# ─────────────────────────────────────────────
function compare_irfs(julia_results, dynare; rtol = RTOL)
    haskey(julia_results, :irfs) || return
    haskey(dynare, :irfs) || return

    for (irf_name, julia_irf) in julia_results[:irfs]
        haskey(dynare[:irfs], irf_name) || continue
        dynare_irf = dynare[:irfs][irf_name]
        n_periods = min(length(julia_irf), length(dynare_irf), IRF_PERIODS)

        for t in 1:n_periods
            mm_val = julia_irf[t]
            dy_val = dynare_irf[t]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = 1e-14) ||
                  (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
        end
    end
end

# ─────────────────────────────────────────────
# Compare: Variance (diagonal of covariance)
# ─────────────────────────────────────────────
function compare_variance(julia_results, dynare; rtol = RTOL)
    haskey(julia_results, :variance_covariance) || return
    haskey(dynare, :variance_covariance) || return

    julia_vcov = julia_results[:variance_covariance]
    dy_vcov = dynare[:variance_covariance]

    for (j_idx, var_name) in enumerate(julia_results[:var_names])
        d_idx = findfirst(==(var_name), dynare[:var_names])
        d_idx === nothing && continue
        d_idx > size(dy_vcov, 1) && continue

        mm_val = julia_vcov[j_idx, j_idx]
        dy_val = dy_vcov[d_idx, d_idx]
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end

    for (j_idx, var_name) in enumerate(julia_results[:var_names])
        d_idx = findfirst(==(var_name), dynare[:var_names])
        d_idx === nothing && continue
        d_idx > size(dy_vcov, 1) && continue

        mm_val = sqrt(julia_vcov[j_idx, j_idx])
        dy_val = sqrt(dy_vcov[d_idx, d_idx])
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end
end

# ─────────────────────────────────────────────
# Compare: Variance Decomposition
# ─────────────────────────────────────────────
function compare_variance_decomposition(julia_results, dynare; rtol = RTOL)
    haskey(julia_results, :variance_decomposition) || return
    haskey(dynare, :variance_decomposition) || return

    dy_vd = dynare[:variance_decomposition]

    for (j_v_idx, var_name) in enumerate(julia_results[:vd_var_names])
        dy_v_idx = findfirst(==(var_name), dynare[:vd_var_names])
        dy_v_idx === nothing && continue
        dy_v_idx > size(dy_vd, 1) && continue

        if haskey(julia_results, :variance_covariance) && haskey(dynare, :variance_covariance)
            julia_var_idx = findfirst(==(var_name), julia_results[:var_names])
            dynare_var_idx = findfirst(==(var_name), dynare[:var_names])
            if julia_var_idx !== nothing && dynare_var_idx !== nothing &&
               julia_var_idx <= size(julia_results[:variance_covariance], 1) &&
               dynare_var_idx <= size(dynare[:variance_covariance], 1)
                julia_var = julia_results[:variance_covariance][julia_var_idx, julia_var_idx]
                dynare_var = dynare[:variance_covariance][dynare_var_idx, dynare_var_idx]
                abs(julia_var) < 1e-12 && abs(dynare_var) < 1e-12 && continue
            end
        end

        for (j_e_idx, exo_name) in enumerate(julia_results[:vd_exo_names])
            dy_e_idx = findfirst(==(exo_name), dynare[:vd_exo_names])
            dy_e_idx === nothing && continue

            mm_val = julia_results[:variance_decomposition][j_v_idx, j_e_idx]
            dy_val = dy_vd[dy_v_idx, dy_e_idx]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = 0.01) ||
                  (abs(mm_val) < 0.01 && abs(dy_val) < 0.01)
        end
    end
end

# ─────────────────────────────────────────────
# Main test runner for a single model
# ─────────────────────────────────────────────
function run_model_comparison(model; rtol = RTOL)
    workdir = joinpath(dirname(@__DIR__), "tasks", "dynare_comparison", string(model.model_name))
    rm(workdir, recursive = true, force = true)
    mkpath(workdir)
    @info "Running Dynare comparison for $(model.model_name) in $workdir"

    julia_output_dir = prepare_comparison_artifacts(model, workdir)
    dynare_output_dir = run_dynare(model, workdir)

    julia_results = parse_dynare_results(julia_output_dir)
    dynare = parse_dynare_results(dynare_output_dir)

    @testset "Steady State" begin
        compare_steady_state(julia_results, dynare; rtol = rtol)
    end

    @testset "Policy Matrices (ghx, ghu)" begin
        compare_policy_matrices(julia_results, dynare; rtol = rtol)
    end

    @testset "IRFs ($IRF_PERIODS periods)" begin
        compare_irfs(julia_results, dynare; rtol = rtol)
    end

    @testset "Variance & Std Dev" begin
        compare_variance(julia_results, dynare; rtol = rtol)
    end

    @testset "Variance Decomposition" begin
        compare_variance_decomposition(julia_results, dynare; rtol = rtol)
    end
end

# ═══════════════════════════════════════════════
# Test Suite Entry Point
# ═══════════════════════════════════════════════
@testset "Dynare Comparison" begin
    if !check_octave_dynare()
        @warn "Octave not available — skipping Dynare comparison tests"
        @test_broken false  # register as broken so CI is aware
        return
    end

    models_dir = joinpath(@__DIR__, "..", "models")

    @testset "RBC_baseline" begin
        include(joinpath(models_dir, "RBC_baseline.jl"))
        run_model_comparison(RBC_baseline)
        global RBC_baseline = nothing
    end

    @testset "FS2000" begin
        include(joinpath(models_dir, "FS2000.jl"))
        run_model_comparison(FS2000)
        global FS2000 = nothing
    end

    @testset "Ireland_2004" begin
        include(joinpath(models_dir, "Ireland_2004.jl"))
        run_model_comparison(Ireland_2004)
        global Ireland_2004 = nothing
    end

    @testset "Gali_2015_chapter_3_nonlinear" begin
        include(joinpath(models_dir, "Gali_2015_chapter_3_nonlinear.jl"))
        run_model_comparison(Gali_2015_chapter_3_nonlinear)
        global Gali_2015_chapter_3_nonlinear = nothing
    end
end
