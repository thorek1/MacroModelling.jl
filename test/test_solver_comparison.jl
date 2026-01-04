"""
Test comparison between LBFGS and Lagrange-Newton solvers for conditional forecasts.
Checks total output (y) across perturbation algorithms.
"""

include("../models/Smets_Wouters_2007.jl")

sw_vars = Smets_Wouters_2007.var
sw_shocks = Smets_Wouters_2007.exo

@test length(sw_shocks) == 7

idx_y = findfirst(==(:y), sw_vars)
idx_c = findfirst(==(:c), sw_vars)
idx_inve = findfirst(==(:inve), sw_vars)
idx_pinf = findfirst(==(:pinf), sw_vars)
idx_r = findfirst(==(:r), sw_vars)
idx_w = findfirst(==(:w), sw_vars)
idx_lab = findfirst(==(:lab), sw_vars)

periods = 6
conditions = Matrix{Union{Nothing,Float64}}(nothing, length(sw_vars), periods)

conditions[idx_y, 1] = 0.001
conditions[idx_c, 2] = 0.0008
conditions[idx_y, 3] = 0.0009
conditions[idx_c, 3] = 0.0007
conditions[idx_y, 4] = 0.0011
conditions[idx_pinf, 4] = 0.04
conditions[idx_y, 5] = 0.0006
conditions[idx_c, 5] = 0.0005
conditions[idx_inve, 5] = 0.0007
conditions[idx_pinf, 5] = 0.003
conditions[idx_y, 6] = 0.0005
conditions[idx_c, 6] = 0.0004
conditions[idx_inve, 6] = 0.0006
conditions[idx_pinf, 6] = 0.0002
conditions[idx_r, 6] = 0.0003
conditions[idx_w, 6] = 0.0004
conditions[idx_lab, 6] = 0.02

shocks = Matrix{Union{Nothing,Float64}}(nothing, length(sw_shocks), periods)

free_shocks_by_period = [
    [1],
    [1, 2],
    [1, 2],
    [1, 2, 3],
    collect(1:7),
    collect(1:7),
]

for (period, free_idx) in enumerate(free_shocks_by_period)
    for i in axes(shocks, 1)
        shocks[i, period] = (i in free_idx) ? nothing : 0.0
    end
end

expected_counts = Dict(
    1 => (1, 1),
    2 => (2, 1),
    3 => (2, 2),
    4 => (3, 2),
    5 => (7, 4),
    6 => (7, 7),
)

tol = 1e-6
@testset "Output comparison acros algorithms" begin
for algorithm in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
        forecast_ln = get_conditional_forecast(
            Smets_Wouters_2007,
            conditions,
            shocks = shocks,
            conditions_in_levels = false,
            algorithm = algorithm,
            periods = periods,
            conditional_forecast_solver = :LagrangeNewton,
        )

        forecast_lbfgs = get_conditional_forecast(
            Smets_Wouters_2007,
            conditions,
            shocks = shocks,
            conditions_in_levels = false,
            algorithm = algorithm,
            periods = periods,
            conditional_forecast_solver = :LBFGS,
        )

        for p in 1:periods
            ln_shocks = forecast_ln[length(sw_vars)+1:end, p]
            lbfgs_shocks = forecast_lbfgs[length(sw_vars)+1:end, p]
            ln_conds = forecast_ln[1:length(sw_vars), p]
            lbfgs_conds = forecast_lbfgs[1:length(sw_vars), p]

            println("Algorithm: $algorithm, Period: $p")
            println("  Norm of shocks (LagrangeNewton): $(ℒ.norm(ln_shocks))")
            println("  Norm of shocks (LBFGS): $(ℒ.norm(lbfgs_shocks))")
            println("  Norm of distance to conditions (LagrangeNewton): $(ℒ.norm(ln_conds[findall(.!isnothing.(conditions[:,p]))] - conditions[findall(.!isnothing.(conditions[:,p])),p]))")
            println("  Norm of distance to conditions (LBFGS): $(ℒ.norm(lbfgs_conds[findall(.!isnothing.(conditions[:,p]))] - conditions[findall(.!isnothing.(conditions[:,p])),p]))")
           
            # free_shocks_ln = ln_shocks[free_shocks_by_period[p]]
            # free_shocks_lbfgs = lbfgs_shocks[free_shocks_by_period[p]]

            # @test ℒ.norm(free_shocks_ln - free_shocks_lbfgs) / max(ℒ.norm(free_shocks_ln), ℒ.norm(free_shocks_lbfgs), 1e-12) < 0.01

            @test ℒ.norm(ln_conds[findall(.!isnothing.(conditions[:,p]))] - lbfgs_conds[findall(.!isnothing.(conditions[:,p]))]) / max(ℒ.norm(ln_conds[findall(.!isnothing.(conditions[:,p]))]), ℒ.norm(lbfgs_conds[findall(.!isnothing.(conditions[:,p]))])) < tol
        end
    end
end
