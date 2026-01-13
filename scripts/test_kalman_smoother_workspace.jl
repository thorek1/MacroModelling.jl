using LinearAlgebra
using MacroModelling

include("../models/RBC_baseline.jl")

data = zeros(1, 5)
ws = MacroModelling.Workspaces()

out1 = MacroModelling.filter_and_smooth(RBC_baseline, data, [:y];
                                        workspace = ws.kalman.smoother)
out2 = MacroModelling.filter_and_smooth(RBC_baseline, data, [:y];
                                        workspace = ws.kalman.smoother)

println(size(out1[1]))
println(norm(out1[1] - out2[1]))
