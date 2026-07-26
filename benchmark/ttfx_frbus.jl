using MacroModelling

model_started = time_ns()
include(joinpath(@__DIR__, "..", "models", "FRBUS.jl"))
println("FRBUS definition complete: $((time_ns() - model_started) / 1e9) s")
flush(stdout)

function run_frbus_workflow(model)
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

    # FRBUS has unit-root variables, so its full covariance-based moments are
    # not defined. Mean and NSSS moments remain a valid supported path.
    moments_started = time_ns()
    moments = get_moments(
        model;
        algorithm = :first_order,
        variables = :all,
        non_stochastic_steady_state = true,
        mean = true,
        standard_deviation = false,
        variance = false,
        covariance = false,
        correlation = false,
        derivatives = false,
        parameter_derivatives = :all,
        silent = true,
        verbose = false,
    )
    println("Supported moments complete: $((time_ns() - moments_started) / 1e9) s")
    flush(stdout)

    return nsss, irf, moments
end

nsss, irf, moments = run_frbus_workflow(FRBUS)

@assert !isempty(nsss) "NSSS output is empty"
@assert all(isfinite, nsss) "NSSS output contains non-finite values"
@assert ndims(irf) == 3 && size(irf, 2) == 40 "IRF output has an unexpected shape"
@assert !isempty(moments) "Moments output is empty"
for value in values(moments)
    @assert all(isfinite, value) "Moments output contains non-finite values"
end

moment_keys = sort!(string.(collect(keys(moments))))
moment_checksum = sum(sum(abs2, value) for value in values(moments))
println("NSSS output: size=$(size(nsss)), checksum=$(sum(abs2, nsss))")
println("IRF output: size=$(size(irf)), checksum=$(sum(abs2, irf))")
println("Moments output: keys=$(join(moment_keys, ',')), checksum=$(moment_checksum)")
