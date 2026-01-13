using LinearAlgebra
using MacroModelling

include("../models/RBC_baseline.jl")

solve!(RBC_baseline)
ss1 = copy(RBC_baseline.solution.non_stochastic_steady_state)

solve!(RBC_baseline)
ss2 = copy(RBC_baseline.solution.non_stochastic_steady_state)

println(norm(ss1 - ss2))
