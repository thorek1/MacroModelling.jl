#=
SW07 verification + benchmark of AS shock decomposition vs polynomial.
nE for SW07 ≈ 7, much larger than RBC_CME (nE=2).
=#

using MacroModelling
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "SW07_nonlinear_zscaled.jl"))
const m = SW07_nonlinear_zscaled
const SHOCK_REF = MacroModelling.SHOCK_DECOMP_MC_METHOD

println("=== AS shock decomposition on SW07 ===")
nE = m.constants.post_model_macro.nExo
println("nExo = ", nE, "; nVars = ", m.constants.post_model_macro.nVars)

Random.seed!(123)

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
    sim = simulate(m, algorithm = algorithm, periods = nT)
    obs_names = [:dy, :dc, :dinve, :dwobs, :labobs, :pinfobs, :robs]
    data_arr = sim(obs_names, 2:nT+1, :simulate)

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
    sc = max(maximum(abs.(Array(sd_poly))), eps())
    println("  shape: ", size(sd_poly))
    println("  max abs diff: ", maximum(diff))
    println("  max abs (poly): ", sc)
    println("  max rel diff:  ", maximum(diff)/sc)
    println("  per-shock max diff: ",
            [round(maximum(diff[:, c, :]), sigdigits=3) for c in 1:size(diff,2)])

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
    println("  AS / poly ratio: ", round(t_as/t_poly, digits=3))
end

compare_orders(:pruned_second_order, 30)
compare_orders(:pruned_third_order, 30)

println("\nDONE")
