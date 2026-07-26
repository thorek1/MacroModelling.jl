using MacroModelling

model_started = time_ns()
@model BasicRBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
println("Model definition complete: $((time_ns() - model_started) / 1e9) s")
flush(stdout)

parameters_started = time_ns()
@parameters BasicRBC silent = true begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
println("Parameters definition complete: $((time_ns() - parameters_started) / 1e9) s")
flush(stdout)

function run_basic_workflow(model)
    nsss_started = time_ns()
    nsss = get_steady_state(
        model;
        derivatives = false,
        stochastic = false,
        return_variables_only = true,
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

nsss, irf, moments = run_basic_workflow(BasicRBC)

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
