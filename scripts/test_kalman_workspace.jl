using LinearAlgebra
using MacroModelling

A = [0.8 0.0; 0.1 0.7]
B = [0.1 0.0; 0.0 0.2]
C = [1.0 0.0]
P = Matrix{Float64}(I, 2, 2)
data = zeros(1, 5)

ws = MacroModelling.Kalman_workspaces()

llh1 = MacroModelling.run_kalman_iterations(A, B * B', C, copy(P), data; workspace = ws)
llh2 = MacroModelling.run_kalman_iterations(A, B * B', C, copy(P), data; workspace = ws)

println(llh1)
println(llh2)
