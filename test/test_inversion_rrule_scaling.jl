using Test
using MacroModelling
import Zygote
using AxisKeys, Random, Statistics

# ──────────────────────────────────────────────────────────────────────────────
# Regression test: dense first-order inversion-filter rrule is O(T·t) per call,
# not O(T·t²).  A doubling of the sample length T should at most double the
# Zygote-gradient runtime (allowing a generous 3× slack for noise and constant
# overhead).
#
# This guards against a regression of the closure-capture-as-Any boxing bug
# (see PR #295) and any future quadratic blow-up in the per-period work.
# ──────────────────────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "models", "FS2000.jl"))

import DelimitedFiles

dat_csv, header = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "FS2000_data.csv"),
                                          ',', header = true)
names = vec(strip.(header))
data = KeyedArray(Float64.(dat_csv)',
                  Variable = Symbol.("log_" .* names),
                  Time = axes(dat_csv, 1))
data = log.(data)
observables = sort(Symbol.("log_" .* names))
data = data(observables, :)

# Tile cyclically so we can address up to 600 periods without using simulate
# (which returns a 3-D KeyedArray that complicates slicing).
n_have = size(data, 2)
n_need = 600
mat = collect(data)
tiled = reduce(hcat, [mat for _ in 1:cld(n_need, n_have)])[:, 1:n_need]
data_full = KeyedArray(tiled, Variable = observables, Time = 1:n_need)

const p0 = FS2000.parameter_values

function loglik_at(p, T_short)
    sub = data_full(:, 1:T_short)
    get_loglikelihood(FS2000, sub, p;
                      filter = :inversion, algorithm = :first_order,
                      presample_periods = 0)
end

# Warm up (compile + cache).
loglik_at(p0, 64)
Zygote.gradient(p -> loglik_at(p, 64), p0)
Zygote.gradient(p -> loglik_at(p, 128), p0)

function med_time_ns(T_short; n = 5)
    samples = Float64[]
    for _ in 1:n
        t = @elapsed Zygote.gradient(p -> loglik_at(p, T_short), p0)
        push!(samples, t)
    end
    median(samples) * 1e9
end

@testset "Inversion first_order rrule scales ≤ linearly in T" begin
    t_small = med_time_ns(100)
    t_big   = med_time_ns(400)  # 4× the sample
    ratio   = t_big / t_small

    @info "first_order inversion-rrule scaling" t_small_ns=t_small t_big_ns=t_big ratio=ratio
    # Linear scaling => ratio ≈ 4.  Allow up to 12× to absorb constant overhead
    # and noise; a quadratic regression would push ratio ≥ 16.
    @test ratio < 12
end
