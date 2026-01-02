using MacroModelling, AxisKeys

# Load FS2000 model
include("../models/FS2000.jl")

periods = 1  # Start with single period for debugging

# Condition only on y (output)  
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                       Variables = [:y], 
                       Periods = 1:periods)
conditions[1,1] = 0.001  # Small condition

# Both shocks are free
shocks = Matrix{Union{Nothing,Float64}}(undef, 2, periods)
shocks[1,1] = nothing  # e_a is free
shocks[2,1] = nothing  # e_m is free

println("Testing second_order algorithm on FS2000...")
println("Conditions: y = $(conditions[1,1])")
println("Free shocks: e_a, e_m")

try
    result = get_conditional_forecast(FS2000, 
                                     conditions, 
                                     shocks = shocks,
                                     conditions_in_levels = false, 
                                     algorithm = :second_order,
                                     periods = periods)
    
    y_idx = findfirst(x -> x == :y, axiskeys(result, 1))
    println("SUCCESS! y = $(result[y_idx, 1]), target = $(conditions[1, 1])")
catch e
    println("FAILED with error: $e")
    if isa(e, AssertionError)
        println("AssertionError - algorithm didn't converge")
    else
        rethrow(e)
    end
end
