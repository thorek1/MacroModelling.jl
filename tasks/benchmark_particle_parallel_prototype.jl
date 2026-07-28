using MacroModelling
using Random
using LinearAlgebra
using Base.Threads

const n_past = 14
const n_exo = 6
const n_vars = 20
const n_particles = 32768
const n_steps = 12
const n_aug = n_past + 1 + n_exo
const observables = collect(1:7)
const data_col = zeros(Float64, length(observables))
const measurement_variance = ones(Float64, length(observables))
const past_idx = collect(1:n_past)

function make_particles(rng)
    return [[0.01 .* randn(rng, n_vars), zeros(n_vars), zeros(n_vars)]
            for _ in 1:n_particles]
end

function make_outputs()
    return [[zeros(n_vars), zeros(n_vars), zeros(n_vars)] for _ in 1:n_particles]
end

function serial_steps!(out, particles, shocks, policies, scratch, full_buf, shock_buf)
    for _ in 1:n_steps
        @inbounds for p in 1:n_particles
            for e in 1:n_exo
                shock_buf[e] = shocks[e, p]
            end
            MacroModelling.higher_propagate!(Val(:pruned_third_order), out[p], particles[p],
                                             shock_buf, past_idx, policies, scratch)
            full = MacroModelling.measurement_full(out[p], full_buf)
            MacroModelling.particle_log_measurement_density(full, data_col, observables,
                                                             measurement_variance, eachindex(observables), log(2π))
        end
        out, particles = particles, out
    end
    return sum(particles[1][1])
end

function threaded_chunk!(out, particles, shocks, policies, scratch, full_buf, shock_buf,
                         first_particle, last_particle)
    @inbounds for p in first_particle:last_particle
        for e in 1:n_exo
            shock_buf[e] = shocks[e, p]
        end
        MacroModelling.higher_propagate!(Val(:pruned_third_order), out[p], particles[p],
                                         shock_buf, past_idx, policies, scratch)
        full = MacroModelling.measurement_full(out[p], full_buf)
        MacroModelling.particle_log_measurement_density(full, data_col, observables,
                                                         measurement_variance, eachindex(observables), log(2π))
    end
    return nothing
end

function threaded_steps!(out, particles, shocks, policies, scratch_pool, full_pool, shock_pool)
    for _ in 1:n_steps
        @sync begin
            for chunk in 1:nthreads()
                first_particle = fld((chunk - 1) * n_particles, nthreads()) + 1
                last_particle = fld(chunk * n_particles, nthreads())
                @spawn begin
                    threaded_chunk!(out, particles, shocks, policies, scratch_pool[chunk],
                                    full_pool[chunk], shock_pool[chunk], first_particle, last_particle)
                end
            end
        end
        out, particles = particles, out
    end
    return sum(particles[1][1])
end

rng = MersenneTwister(123)
policies = [0.01 .* randn(rng, n_vars, n_aug),
            0.01 .* randn(rng, n_vars, n_aug * (n_aug + 1) ÷ 2),
            0.01 .* randn(rng, n_vars, n_aug * (n_aug + 1) * (n_aug + 2) ÷ 6)]
shocks = randn(rng, n_exo, n_particles)
serial_particles = make_particles(rng)
serial_out = make_outputs()
threaded_particles = deepcopy(serial_particles)
threaded_out = make_outputs()
serial_scratch = MacroModelling.build_higher_scratch(Val(:pruned_third_order), n_past, n_exo)
threaded_scratch = [MacroModelling.build_higher_scratch(Val(:pruned_third_order), n_past, n_exo)
                    for _ in 1:(nthreads() + 1)]
serial_full = zeros(n_vars)
threaded_full = [zeros(n_vars) for _ in 1:(nthreads() + 1)]
serial_shock = zeros(n_exo)
threaded_shock = [zeros(n_exo) for _ in 1:(nthreads() + 1)]

serial_steps!(serial_out, serial_particles, shocks, policies, serial_scratch, serial_full, serial_shock)
threaded_steps!(threaded_out, threaded_particles, shocks, policies, threaded_scratch, threaded_full, threaded_shock)

serial_result = @timed serial_steps!(serial_out, serial_particles, shocks, policies, serial_scratch, serial_full, serial_shock)
threaded_result = @timed threaded_steps!(threaded_out, threaded_particles, shocks, policies, threaded_scratch, threaded_full, threaded_shock)

println("threads=", nthreads(), " particles=", n_particles, " steps=", n_steps,
        " n_aug=", n_aug)
println("serial_seconds=", serial_result.time, " threaded_seconds=", threaded_result.time,
        " speedup=", serial_result.time / threaded_result.time,
        " serial_bytes=", serial_result.bytes, " threaded_bytes=", threaded_result.bytes)
println("serial_result=", serial_result.value, " threaded_result=", threaded_result.value)
