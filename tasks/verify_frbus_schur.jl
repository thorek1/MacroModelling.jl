using MacroModelling

include("../models/FRBUS.jl")

println("=" ^ 70)
println("Default solve (verbose):")
println("=" ^ 70)
sol = get_solution(FRBUS, algorithm = :first_order, verbose = true)
println("has_unit_roots: ", FRBUS.caches.has_unit_roots)

# Validate first-order solution coefficients (same as in test/test_models.jl)
@assert size(sol) == (433, 428)
@assert isapprox(sol(:rff₍₋₁₎, :rff), 0.84575710864915, rtol = 1e-5)
@assert isapprox(sol(:eco_l₍₋₁₎, :eco_l), 1.1848016760901816, rtol = 1e-5)
@assert isapprox(sol(:ebfi_l₍₋₁₎, :ebfi_l), 1.27660626172, rtol = 1e-5)
@assert isapprox(sol(:ex_l₍₋₁₎, :ex_l), 0.892272127137, rtol = 1e-5)

irf = get_irf(FRBUS, algorithm = :first_order, shocks = [:fiscal_aerr], periods = 5)
@assert isapprox(irf(:rff, 1, :fiscal_aerr), 0.0144267064, rtol = 1e-4)
@assert isapprox(irf(:xgap2, 1, :fiscal_aerr), 0.0961780423, rtol = 1e-4)
@assert isapprox(irf(:eco_l, 1, :fiscal_aerr), 0.0010262957, rtol = 1e-4)
@assert isapprox(irf(:debt_to_gdp, 1, :fiscal_aerr), 0.0065268971, rtol = 1e-4)
@assert isapprox(irf(:rff, 5, :fiscal_aerr), 0.1445122073, rtol = 1e-4)
@assert isapprox(irf(:xgap2, 5, :fiscal_aerr), 0.3546553714, rtol = 1e-4)
@assert isapprox(irf(:debt_to_gdp, 5, :fiscal_aerr), 0.0685009926, rtol = 1e-4)

println("\nFRBUS solution + IRF assertions PASSED")
