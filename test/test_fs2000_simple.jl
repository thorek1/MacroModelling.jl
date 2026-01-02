using MacroModelling, AxisKeys

println("Loading FS2000 model...")
include("../models/FS2000.jl")
println("Model loaded successfully")

periods = 1

# Very simple test: just 1 period, small condition
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                       Variables = [:y], 
                       Periods = 1:periods)
conditions[1,1] = 0.001

shocks = Matrix{Union{Nothing,Float64}}(undef, 2, periods)
shocks[1,1] = nothing  # e_a free
shocks[2,1] = nothing  # e_m free

println("\nAttempting conditional forecast...")
println("Algorithm: second_order")
println("Conditions: y = 0.001")
println("Free shocks: 2 (e_a, e_m)")

try
    @time result = get_conditional_forecast(FS2000, 
                                            conditions, 
                                            shocks = shocks,
                                            conditions_in_levels = false, 
                                            algorithm = :second_order,
                                            periods = periods)
    
    y_idx = findfirst(x -> x == :y, axiskeys(result, 1))
    println("\n✓ SUCCESS!")
    println("Result: y = $(result[y_idx, 1])")
    println("Target: y = $(conditions[1, 1])")
    println("Error: $(abs(result[y_idx, 1] - conditions[1, 1]))")
catch e
    println("\n✗ FAILED")
    println("Error: $e")
end
