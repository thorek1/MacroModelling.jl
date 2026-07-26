using MacroModelling

symbolic_mode = get(ENV, "MACROMODELLING_SW07_SS_SYMBOLIC_MODE", "auto")
model_source = read(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"), String)
model_source = replace(
    model_source,
    "@parameters Smets_Wouters_2007 begin" =>
        "@parameters Smets_Wouters_2007 ss_symbolic_mode=$(symbolic_mode) begin",
)

model_started = time_ns()
include_string(Main, model_source, "Smets_Wouters_2007_$(symbolic_mode).jl")
resolved_mode = MacroModelling.resolve_steady_state_symbolic_mode(Smets_Wouters_2007, Symbol(symbolic_mode))
println("SW07 definition complete: $((time_ns() - model_started) / 1e9) s mode=$(symbolic_mode), resolved=$(resolved_mode)")
flush(stdout)

function run_sw07_workflow(model)
    nsss_started = time_ns()
    nsss = get_steady_state(
        model;
        derivatives = false,
        stochastic = false,
        return_variables_only = true,
        verbose = false,
        silent = true,
    )
    println("NSSS complete: $((time_ns() - nsss_started) / 1e9) s")
    flush(stdout)

    irf_started = time_ns()
    irf = get_irf(
        model;
        algorithm = :first_order,
        periods = 40,
        variables = :all,
        shocks = :all,
        verbose = false,
    )
    println("IRF complete: $((time_ns() - irf_started) / 1e9) s")
    flush(stdout)

    moments_started = time_ns()
    moments = get_moments(
        model;
        algorithm = :first_order,
        variables = :all,
        non_stochastic_steady_state = true,
        mean = true,
        standard_deviation = true,
        variance = true,
        covariance = true,
        correlation = true,
        derivatives = false,
        silent = true,
        verbose = false,
    )
    println("Moments complete: $((time_ns() - moments_started) / 1e9) s")
    flush(stdout)

    return nsss, irf, moments
end

nsss, irf, moments = run_sw07_workflow(Smets_Wouters_2007)

@assert !isempty(nsss) "NSSS output is empty"
@assert all(isfinite, nsss) "NSSS output contains non-finite values"
@assert ndims(irf) == 3 && size(irf, 2) == 40 "IRF output has an unexpected shape"
@assert !isempty(moments) "Moments output is empty"
for (key, value) in moments
    # Correlations involving a numerically degenerate variable are intentionally
    # NaN; all other moment outputs must be finite and no output may contain Inf.
    @assert all(x -> isfinite(x) || (key == :correlation && isnan(x)), value) "$(key) output contains an unsupported non-finite value"
end

moment_keys = sort!(string.(collect(keys(moments))))
moment_checksum = sum(sum(x -> isfinite(x) ? abs2(x) : 0.0, value) for value in values(moments))
println("NSSS output: size=$(size(nsss)), checksum=$(sum(abs2, nsss))")
println("IRF output: size=$(size(irf)), checksum=$(sum(abs2, irf))")
println("Moments output: keys=$(join(moment_keys, ',')), checksum=$(moment_checksum), undefined_correlations=$(sum(isnan, moments[:correlation]))")
