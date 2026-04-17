using MacroModelling
using Test
using DelimitedFiles

const RTOL = 1e-6
const IRF_PERIODS = 40

# ─────────────────────────────────────────────
# Helper: check Octave + Dynare availability
# ─────────────────────────────────────────────
function check_octave_dynare()
    try
        out = read(`octave --no-gui --eval "try; dynare_version(); disp('dynare_ok'); catch; disp('no_dynare'); end"`, String)
        return contains(out, "dynare_ok")
    catch
        return false
    end
end

# ─────────────────────────────────────────────
# Helper: build Unicode → ASCII name mapping
# ─────────────────────────────────────────────
function build_name_mapping(model)
    mapping = Dict{String, String}()
    for v in model.constants.post_model_macro.var
        ascii = MacroModelling.translate_symbol_to_ascii(v)
        mapping[string(v)] = ascii
    end
    for e in model.constants.post_model_macro.exo
        ascii = MacroModelling.translate_symbol_to_ascii(e)
        mapping[string(e)] = ascii
    end
    for s in model.constants.post_model_macro.past_not_future_and_mixed
        ascii = MacroModelling.translate_symbol_to_ascii(s)
        mapping[string(s)] = ascii
    end
    mapping
end

# ─────────────────────────────────────────────
# Helper: get original (non-auxiliary) variable names
# ─────────────────────────────────────────────
function original_var_names(model)
    filter(v -> !contains(string(v), "➕"), model.constants.post_model_macro.var)
end

# ─────────────────────────────────────────────
# Run Dynare via Octave in a working directory
# ─────────────────────────────────────────────
function run_dynare(model, workdir)
    mod_name = string(model.model_name)

    cd(workdir) do
        write_mod_file(model)
    end

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
function compare_steady_state(model, dynare; rtol = RTOL)
    name_map = build_name_mapping(model)
    mm_ss = get_SS(model, derivatives = false)
    orig_vars = original_var_names(model)

    for v in orig_vars
        v_ascii = name_map[string(v)]
        idx = findfirst(==(v_ascii), dynare[:var_names])
        if idx === nothing
            @warn "Variable $v ($v_ascii) not found in Dynare output, skipping"
            continue
        end
        mm_val = Float64(mm_ss(v))
        dy_val = dynare[:steady_state][idx]
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end
end

# ─────────────────────────────────────────────
# Compare: Policy / transition matrices
# ─────────────────────────────────────────────
function compare_policy_matrices(model, dynare; rtol = RTOL)
    name_map = build_name_mapping(model)
    mm_sol = get_solution(model, algorithm = :first_order)

    state_vars = model.constants.post_model_macro.past_not_future_and_mixed
    exo_vars = model.constants.post_model_macro.exo
    orig_vars = original_var_names(model)

    n_states = length(state_vars)
    n_exo = length(exo_vars)

    # --- ghx comparison ---
    # MM solution: rows = [Steady_state; states₍₋₁₎; shocks₍ₓ₎], columns = variables
    # ghx in MM: rows 2:(1+n_states), columns = all vars
    # Dynare ghx (declaration order): rows = all endo vars, columns = state vars
    # Relationship: MM_ghx[s, v] == Dynare_ghx[v, s]  (transposed, after name alignment)

    for v in orig_vars
        v_ascii = name_map[string(v)]
        dy_v_idx = findfirst(==(v_ascii), dynare[:var_names])
        dy_v_idx === nothing && continue

        for (s_i, s) in enumerate(state_vars)
            s_ascii = name_map[string(s)]
            dy_s_idx = findfirst(==(s_ascii), dynare[:state_var_names])
            dy_s_idx === nothing && continue

            mm_val = Float64(mm_sol[Symbol(string(s) * "₍₋₁₎"), v])
            dy_val = dynare[:ghx][dy_v_idx, dy_s_idx]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
                  (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
        end
    end

    # --- ghu comparison ---
    # MM ghu: rows (2+n_states):end of solution, columns = all vars
    # Dynare ghu (declaration order): rows = all endo vars, columns = shocks
    # Relationship: MM_ghu[e, v] == Dynare_ghu[v, e]  (transposed, after name alignment)

    for v in orig_vars
        v_ascii = name_map[string(v)]
        dy_v_idx = findfirst(==(v_ascii), dynare[:var_names])
        dy_v_idx === nothing && continue

        for (e_i, e) in enumerate(exo_vars)
            e_ascii = name_map[string(e)]
            dy_e_idx = findfirst(==(e_ascii), dynare[:exo_names])
            dy_e_idx === nothing && continue

            mm_val = Float64(mm_sol[Symbol(string(e) * "₍ₓ₎"), v])
            dy_val = dynare[:ghu][dy_v_idx, dy_e_idx]
            @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
                  (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
        end
    end
end

# ─────────────────────────────────────────────
# Compare: Impulse Response Functions
# ─────────────────────────────────────────────
function compare_irfs(model, dynare; rtol = RTOL)
    name_map = build_name_mapping(model)
    mm_irfs = get_irf(model, periods = IRF_PERIODS, algorithm = :first_order)

    haskey(dynare, :irfs) || return

    orig_vars = original_var_names(model)
    exo_vars = model.constants.post_model_macro.exo

    for v in orig_vars
        v_ascii = name_map[string(v)]
        for e in exo_vars
            e_ascii = name_map[string(e)]
            # Dynare IRF field naming convention: varname_shockname
            dy_key = "$(v_ascii)_$(e_ascii)"
            haskey(dynare[:irfs], dy_key) || continue

            dy_irf = dynare[:irfs][dy_key]
            n_periods = min(length(dy_irf), IRF_PERIODS)

            for t in 1:n_periods
                mm_val = Float64(mm_irfs[v, e, t])
                dy_val = dy_irf[t]
                @test isapprox(mm_val, dy_val, rtol = rtol, atol = 1e-14) ||
                      (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
            end
        end
    end
end

# ─────────────────────────────────────────────
# Compare: Variance (diagonal of covariance)
# ─────────────────────────────────────────────
function compare_variance(model, dynare; rtol = RTOL)
    haskey(dynare, :variance_covariance) || return

    name_map = build_name_mapping(model)
    mm_moments = get_moments(model, algorithm = :first_order,
                             derivatives = false,
                             non_stochastic_steady_state = false,
                             mean = false,
                             variance = true,
                             standard_deviation = true,
                             covariance = false)

    orig_vars = original_var_names(model)
    dy_vcov = dynare[:variance_covariance]

    # Compare variances
    mm_var = mm_moments[:variance]
    for v in orig_vars
        v_ascii = name_map[string(v)]
        idx = findfirst(==(v_ascii), dynare[:var_names])
        idx === nothing && continue
        # Dynare's oo_.var may not have rows for all variables; check bounds
        idx > size(dy_vcov, 1) && continue

        mm_val = Float64(mm_var(v))
        dy_val = dy_vcov[idx, idx]
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end

    # Compare standard deviations
    mm_std = mm_moments[:standard_deviation]
    for v in orig_vars
        v_ascii = name_map[string(v)]
        idx = findfirst(==(v_ascii), dynare[:var_names])
        idx === nothing && continue
        idx > size(dy_vcov, 1) && continue

        mm_val = Float64(mm_std(v))
        dy_val = sqrt(dy_vcov[idx, idx])
        @test isapprox(mm_val, dy_val, rtol = rtol, atol = eps()) ||
              (abs(mm_val) < 1e-12 && abs(dy_val) < 1e-12)
    end
end

# ─────────────────────────────────────────────
# Compare: Variance Decomposition
# ─────────────────────────────────────────────
function compare_variance_decomposition(model, dynare; rtol = RTOL)
    haskey(dynare, :variance_decomposition) || return

    name_map = build_name_mapping(model)
    mm_vd = get_variance_decomposition(model)

    orig_vars = original_var_names(model)
    dy_vd = dynare[:variance_decomposition]
    dy_vd_vars = dynare[:vd_var_names]
    dy_vd_exos = dynare[:vd_exo_names]

    exo_vars = model.constants.post_model_macro.exo

    for v in orig_vars
        v_ascii = name_map[string(v)]
        dy_v_idx = findfirst(==(v_ascii), dy_vd_vars)
        dy_v_idx === nothing && continue
        dy_v_idx > size(dy_vd, 1) && continue

        for e in exo_vars
            e_ascii = name_map[string(e)]
            dy_e_idx = findfirst(==(e_ascii), dy_vd_exos)
            dy_e_idx === nothing && continue

            # MM returns fractions (0-1), Dynare returns percentages (0-100)
            mm_val = Float64(mm_vd(v, e)) * 100.0
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
    workdir = mktempdir()
    @info "Running Dynare comparison for $(model.model_name) in $workdir"

    output_dir = run_dynare(model, workdir)
    dynare = parse_dynare_results(output_dir)

    @testset "Steady State" begin
        compare_steady_state(model, dynare; rtol = rtol)
    end

    @testset "Policy Matrices (ghx, ghu)" begin
        compare_policy_matrices(model, dynare; rtol = rtol)
    end

    @testset "IRFs ($IRF_PERIODS periods)" begin
        compare_irfs(model, dynare; rtol = rtol)
    end

    @testset "Variance & Std Dev" begin
        compare_variance(model, dynare; rtol = rtol)
    end

    @testset "Variance Decomposition" begin
        compare_variance_decomposition(model, dynare; rtol = rtol)
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
