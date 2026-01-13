using LinearAlgebra
using MacroModelling

A = 0.1 .* Matrix{Float64}(I, 2, 2)
B = Matrix{Float64}(I, 2, 2)
C = 0.01 .* Matrix{Float64}(I, 2, 2)

caches = MacroModelling.Caches()
MacroModelling.set_timings!(caches, MacroModelling.Empty_timings())
ws = MacroModelling.Workspaces()

sol1, solved1 = MacroModelling.solve_quadratic_matrix_equation(A, B, C, caches;
                                                                quadratic_matrix_equation_algorithm = :doubling,
                                                                workspace = ws.quadratic_matrix_equation)
sol2, solved2 = MacroModelling.solve_quadratic_matrix_equation(A, B, C, caches;
                                                                quadratic_matrix_equation_algorithm = :doubling,
                                                                workspace = ws.quadratic_matrix_equation)

println(solved1)
println(solved2)
println(norm(sol1 - sol2))
