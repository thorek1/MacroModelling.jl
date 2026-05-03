using Pkg

Pkg.activate(temp = true)
Pkg.add("DataStructures")
Pkg.add(PackageSpec(url = "https://github.com/penelopeysm/FlexiChains.jl"))

using Statistics
using FlexiChains

include(joinpath(@__DIR__, "..", "test", "test_helpers.jl"))

# Validate the exact `sample_array`/`sample_names` layout consumed from Pigeons.
sample_array = Array{Float64}(undef, 3, 3, 2)
sample_array[:, 1, 1] = [1.0, 2.0, 3.0]
sample_array[:, 1, 2] = [4.0, 5.0, 6.0]
sample_array[:, 2, 1] = [10.0, 11.0, 12.0]
sample_array[:, 2, 2] = [13.0, 14.0, 15.0]
sample_array[:, 3, 1] = [-1.0, -2.0, -3.0]
sample_array[:, 3, 2] = [-4.0, -5.0, -6.0]
sample_names = [:θ, :ϕ, :log_density]

chain = pigeons_flexichain(sample_array, sample_names)
means = parameter_means(chain)

@assert chain isa FlexiChain
@assert size(sample_array, 3) == 2
@assert :θ in Symbol.(sample_names)
@assert :ϕ in Symbol.(sample_names)
@assert means == [3.5, 12.5]

println("pigeons_sample_names=$(sample_names)")
println("converted_chain_type=$(typeof(chain))")
println("parameter_means=$(means)")