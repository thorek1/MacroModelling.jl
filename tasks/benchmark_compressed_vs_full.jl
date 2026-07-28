using MacroModelling
using AxisKeys
using Random
using LinearAlgebra

@model BenchRBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α * exp(g[0])
    z[0] = ρz * z[-1] + std_z * eps_z[x]
    g[0] = ρg * g[-1] + std_g * eps_g[x]
end

@parameters BenchRBC begin
    std_z = 0.01
    std_g = 0.01
    ρz = 0.4
    ρg = 0.6
    δ = 0.02
    α = 0.5
    β = 0.95
end

function timed_repeated(f, repeats)
    f() # compile and populate the model cache
    times = Float64[]
    bytes = Int[]
    values = Any[]
    for _ in 1:repeats
        GC.gc()
        t0 = time_ns()
        result = @timed f()
        push!(times, result.time)
        push!(bytes, result.bytes)
        push!(values, result.value)
    end
    order = sortperm(times)
    middle = times[order[cld(length(order), 2)]]
    return middle, bytes[order[cld(length(order), 2)]], values[end]
end

function print_case(label, f, repeats)
    seconds, bytes, value = timed_repeated(f, repeats)
    println(label, " median_seconds=", round(seconds; digits = 6),
            " median_bytes=", bytes, " result=", value)
    return seconds, bytes
end

direct_only = get(ENV, "BENCH_DIRECT_ONLY", "0") == "1"
if !direct_only
    Random.seed!(1234)
    periods = 32
    simulated = simulate(BenchRBC, periods = periods)
    data = simulated([:c, :q], :, :simulate)
    p = BenchRBC.parameter_values
    shock_path = 1e-3 .* randn(2, periods + 1)
    measurement_error_std = 0.002

    println("model_vars=", BenchRBC.constants.post_model_macro.nVars,
            " model_past=", BenchRBC.constants.post_model_macro.nPast_not_future_and_mixed,
            " model_exo=", BenchRBC.constants.post_model_macro.nExo)

    filter_free = print_case("filter_free_pruned_third_order", () ->
        get_loglikelihood(BenchRBC, data, p, shock_path, measurement_error_std;
                          algorithm = :pruned_third_order, warmup_iterations = 2,
                          verbose = false), 3)

    particle = print_case("bootstrap_particle_pruned_third_order", () ->
        get_loglikelihood(BenchRBC, data, p;
                          algorithm = :pruned_third_order,
                          filter = :bootstrap_particle,
                          n_particles = 2048,
                          presample_periods = 2,
                          measurement_error = measurement_error_std^2,
                          particle_rng = MersenneTwister(42),
                          verbose = false), 3)

    println("filter_free_seconds=", filter_free[1], " particle_seconds=", particle[1])
    println("filter_free_bytes=", filter_free[2], " particle_bytes=", particle[2])
end

# Direct transition sweep with the same dimensions as user-facing state updates.
# It isolates the duplicated Kronecker work from model solving and particle
# resampling, and is useful for checking scaling beyond the small RBC fixture.
full_reference = get(ENV, "BENCH_FULL", "0") == "1"

pair_index(i, j) = begin
    hi = max(i, j)
    lo = min(i, j)
    (hi - 1) * hi ÷ 2 + lo
end

triple_index(i, j, k) = begin
    a, b, c = sort((i, j, k); rev = true)
    (a - 1) * a * (a + 1) ÷ 6 + (b - 1) * b ÷ 2 + c
end

function make_direct_case(n_past, n_exo, n_vars, order, full_reference)
    n_aug = n_past + 1 + n_exo
    n_pair = n_aug * (n_aug + 1) ÷ 2
    n_triple = n_aug * (n_aug + 1) * (n_aug + 2) ÷ 6
    rng = MersenneTwister(99 + n_aug)
    first = 0.01 .* randn(rng, n_vars, n_aug)
    second = 0.01 .* randn(rng, n_vars, n_pair)
    third = 0.01 .* randn(rng, n_vars, n_triple)
    second_full = if full_reference
        matrix = zeros(n_vars, n_aug^2)
        for i in 1:n_aug, j in 1:n_aug
            matrix[:, (i - 1) * n_aug + j] .= second[:, pair_index(i, j)]
        end
        matrix
    else
        nothing
    end
    third_full = if full_reference && order == 3
        matrix = zeros(n_vars, n_aug^3)
        for i in 1:n_aug, j in 1:n_aug, k in 1:n_aug
            matrix[:, (i - 1) * n_aug^2 + (j - 1) * n_aug + k] .= third[:, triple_index(i, j, k)]
        end
        matrix
    else
        nothing
    end
    state = randn(rng, n_vars)
    shock = randn(rng, n_exo)
    augmented = Vector{Float64}(undef, n_aug)
    pair = Vector{Float64}(undef, n_pair)
    triple = Vector{Float64}(undef, n_triple)
    output = similar(state)
    pair_full = full_reference ? Vector{Float64}(undef, n_aug^2) : nothing
    triple_full = full_reference && order == 3 ? Vector{Float64}(undef, n_aug^3) : nothing

    return (; n_past, n_exo, order, first, second, third, second_full, third_full,
            state0 = copy(state), state, shock, augmented, pair, triple, output,
            pair_full, triple_full, full_reference)
end

function direct_transition_sweep(case, steps)
    (; n_past, n_exo, order, first, second, third, second_full, third_full,
       state0, state, shock, augmented, pair, triple, output,
       pair_full, triple_full, full_reference) = case
    copyto!(state, state0)

    for _ in 1:steps
        copyto!(augmented, 1, state, 1, n_past)
        augmented[n_past + 1] = 1.0
        copyto!(augmented, n_past + 2, shock, 1, n_exo)
        if full_reference && order >= 2
            copyto!(pair_full, kron(augmented, augmented))
        elseif order >= 2
            MacroModelling.compressed_kron²_power!(pair, augmented)
        end
        if full_reference && order == 3
            copyto!(triple_full, kron(pair_full, augmented))
        elseif order == 3
            MacroModelling.compressed_kron³_power!(triple, augmented)
        end
        mul!(output, first, augmented)
        if order >= 2
            mul!(output, full_reference ? second_full : second, full_reference ? pair_full : pair, 0.5, 1.0)
        end
        if order == 3
            mul!(output, full_reference ? third_full : third, full_reference ? triple_full : triple, 1 / 6, 1.0)
        end
        copyto!(state, output)
    end
    return sum(state)
end

println("transition_representation=", full_reference ? "full" : "compressed")
for (n_past, n_exo, n_vars, order) in ((3, 2, 5, 2), (8, 4, 12, 2), (14, 6, 20, 3), (24, 8, 32, 3))
    label = full_reference ? "direct_full" : "direct_compressed"
    case = make_direct_case(n_past, n_exo, n_vars, order, full_reference)
    print_case("$(label)_naug_$(n_past + 1 + n_exo)_order_$(order)",
               () -> direct_transition_sweep(case, 4000), 5)
end
