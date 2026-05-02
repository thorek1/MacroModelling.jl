#=
Verify aumann_shapley_shock_decomposition_pruned_{2,3}_order! against the
polynomial-coefficient driver on RBC_CME, then benchmark.
=#

using MacroModelling
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "test", "models", "RBC_CME.jl"))

const M = MacroModelling
const SHOCK_REF = M.SHOCK_DECOMP_MC_METHOD

println("=== AS shock decomposition verification + benchmark ===")
println("Model: RBC_CME, nExo = ", m.constants.post_model_macro.nExo)

Random.seed!(42)

function gen_data(algo, nT)
    sim = simulate(m, algorithm = algo, periods = nT)
    return sim
end

# Pick exactly nExo observables to keep inversion well-posed.
function pick_obs(sim)
    nE = m.constants.post_model_macro.nExo
    full = sim[:, 2:end, 1]
    return full[1:nE, :]
end

function bench(f, n)
    f()  # warmup
    t = Inf
    for _ in 1:n
        t = min(t, @elapsed f())
    end
    return t
end

function compare_orders(algorithm::Symbol, nT::Int)
    println("\n--- Algorithm: $algorithm, nT = $nT ---")
    sim = gen_data(algorithm, nT)
    data_arr = pick_obs(sim)

    SHOCK_REF[] = :polynomial
    sd_poly = get_shock_decomposition(m, data_arr;
                                      algorithm = algorithm,
                                      filter = :inversion,
                                      marginal_contribution = true)

    SHOCK_REF[] = :aumann_shapley
    sd_as = get_shock_decomposition(m, data_arr;
                                    algorithm = algorithm,
                                    filter = :inversion,
                                    marginal_contribution = true)
    SHOCK_REF[] = :polynomial

    diff = abs.(Array(sd_poly) .- Array(sd_as))
    println("  shape: ", size(sd_poly))
    println("  max abs diff: ", maximum(diff))
    println("  max abs (poly): ", maximum(abs.(Array(sd_poly))))
    rel = maximum(diff) / max(maximum(abs.(Array(sd_poly))), eps())
    println("  max rel diff:  ", rel)
    # per-period max diff
    nT_eff = size(diff, 3)
    per_t = [maximum(diff[:,:,t]) for t in 1:nT_eff]
    println("  per-period max diff: t=1: ", per_t[1], "  t=mid: ", per_t[nT_eff÷2+1], "  t=end: ", per_t[end])
    # column-wise
    println("  per-shock max diff: ", [maximum(diff[:, c, :]) for c in 1:size(diff,2)])

    SHOCK_REF[] = :polynomial
    t_poly = bench(() -> get_shock_decomposition(m, data_arr;
                                                 algorithm = algorithm,
                                                 filter = :inversion,
                                                 marginal_contribution = true), 3)
    SHOCK_REF[] = :aumann_shapley
    t_as = bench(() -> get_shock_decomposition(m, data_arr;
                                               algorithm = algorithm,
                                               filter = :inversion,
                                               marginal_contribution = true), 3)
    SHOCK_REF[] = :polynomial
    println("  poly time: ", round(t_poly*1000, digits=2), " ms")
    println("  AS   time: ", round(t_as*1000, digits=2), " ms")
    println("  AS / poly ratio: ", round(t_as/t_poly, digits=2))

    return rel
end

rel2 = compare_orders(:pruned_second_order, 30)
rel3 = compare_orders(:pruned_third_order, 30)

println("\n=== summary ===")
println("pruned_second_order rel diff: ", rel2)
println("pruned_third_order  rel diff: ", rel3)
