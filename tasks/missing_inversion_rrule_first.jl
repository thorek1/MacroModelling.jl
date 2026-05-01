using MacroModelling
using FiniteDifferences
using DelimitedFiles, AxisKeys
import LinearAlgebra as LA
import ChainRulesCore

cd(@__DIR__)
include("../models/FS2000.jl")

dat, header = readdlm("../test/data/FS2000_data.csv", ',', header = true)
dat = Float64.(dat)
names_ = vec(header)
data = KeyedArray(dat', Variable = Symbol.("log_".*names_), Time = axes(dat, 1))
data = log.(data)
observables = sort(Symbol.("log_".*names_))
data = data(observables, :)

dA_full = Matrix{Float64}(copy(collect(data)))
dA = copy(dA_full)
dA[1, 25] = NaN
dA[2, 47] = NaN
dA[:, 30] .= NaN
dA[:, 60] .= NaN
dataA_ka = KeyedArray(dA, Variable = observables, Time = axes(dat,1))

import MacroModelling as MM

θ₀ = FS2000.parameter_values
algo = :first_order
filt = :inversion

ll_check = get_loglikelihood(FS2000, dataA_ka, θ₀, filter = filt, algorithm = algo)
println("ll forward = ", ll_check); flush(stdout)

opts = MM.merge_calculation_options()
constants_obj, SS_and_pars, 𝐒, state, solved =
    MM.get_relevant_steady_state_and_state_update(Val(algo), θ₀, FS2000, opts = opts, estimation = true)
@assert solved
SS_and_pars_names = FS2000.constants.post_complete_parameters.SS_and_pars_names
obs_indices = convert(Vector{Int}, indexin(observables, SS_and_pars_names))
dev = MM.missing_data_to_nan(dA) .- SS_and_pars[obs_indices]

y, pb = ChainRulesCore.rrule(MM.calculate_loglikelihood, Val(filt), Val(algo),
                              obs_indices, 𝐒, dev, constants_obj, state, FS2000.workspaces)
println("rrule primal = ", y); flush(stdout)

println("calling pullback..."); flush(stdout)
out = pb(1.0)
println("pullback returned, len=", length(out)); flush(stdout)
∂𝐒  = out[5]
∂dev = out[6]
println("typeof(∂𝐒)=", typeof(∂𝐒)); flush(stdout)
println("typeof(∂dev)=", typeof(∂dev)); flush(stdout)

# FD
ε = 1e-6
function ll_dev(d)
    yy, _ = ChainRulesCore.rrule(MM.calculate_loglikelihood, Val(filt), Val(algo),
            obs_indices, 𝐒, d, constants_obj, state, FS2000.workspaces)
    yy
end

println("\n--- ∂dev FD ---"); flush(stdout)
global fail = 0
for (i, j) in [(1, 1), (2, 1), (1, 26), (2, 47), (2, 25), (1, 31), (1, 100)]
    if !isfinite(dev[i, j]); println("skip [$i,$j] missing"); continue; end
    dp = copy(dev); dp[i,j] += ε
    dm = copy(dev); dm[i,j] -= ε
    fd = (ll_dev(dp) - ll_dev(dm)) / (2ε)
    ad = ∂dev[i,j]
    ok = isapprox(ad, fd; rtol = 1e-3, atol = 1e-5)
    println("∂dev[$i,$j]: ad=", ad, "  fd=", fd, "  ", ok ? "OK" : "FAIL"); flush(stdout)
    global fail += !ok
end

function ll_S(S)
    yy, _ = ChainRulesCore.rrule(MM.calculate_loglikelihood, Val(filt), Val(algo),
            obs_indices, S, dev, constants_obj, state, FS2000.workspaces)
    yy
end

println("\n--- ∂𝐒 FD ---"); flush(stdout)
nS, mS = size(𝐒)
for (i, j) in [(1, 1), (1, mS), (5, 4), (10, mS-1), (12, mS), (1, mS-1), (3, 1)]
    Sp = copy(𝐒); Sp[i,j] += ε
    Sm = copy(𝐒); Sm[i,j] -= ε
    fd = (ll_S(Sp) - ll_S(Sm)) / (2ε)
    ad = ∂𝐒[i,j]
    ok = isapprox(ad, fd; rtol = 1e-3, atol = 1e-5)
    println("∂𝐒[$i,$j]: ad=", ad, "  fd=", fd, "  ", ok ? "OK" : "FAIL"); flush(stdout)
    global fail += !ok
end

println(fail == 0 ? "\nALL PASS" : "\n$fail FAILURES")
