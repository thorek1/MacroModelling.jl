using Revise
using MacroModelling

# This will trigger the SS solve during @parameters
include(joinpath(@__DIR__, "models/FS2000.jl"))
println("FS2000 model defined.")
