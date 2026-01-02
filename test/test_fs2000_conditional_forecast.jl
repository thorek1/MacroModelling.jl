using MacroModelling, Test, AxisKeys

@testset "FS2000 Conditional Forecast - Multi-shock tests" begin
    # Load FS2000 model (2 shocks: e_a and e_m)
    include("../models/FS2000.jl")
    
    periods = 3
    
    # Condition only on y (output) - 1 condition, but have 2 shocks free
    conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef, 1, periods), 
                           Variables = [:y], 
                           Periods = 1:periods)
    conditions[1,1] = 0.001
    conditions[1,2] = 0.0005
    conditions[1,3] = 0.0002

    # Both shocks are free (2 shocks > 1 condition per period)
    shocks = Matrix{Union{Nothing,Float64}}(undef, 2, periods)
    for p in 1:periods
        shocks[1,p] = nothing  # e_a is free
        shocks[2,p] = nothing  # e_m is free
    end

    # Test each algorithm
    for algorithm in [:first_order, :second_order, :third_order, :pruned_second_order, :pruned_third_order]
        @testset "Algorithm: $algorithm" begin
            println("\n=== Testing $algorithm ===")
            
            result = get_conditional_forecast(FS2000, 
                                             conditions, 
                                             shocks = shocks,
                                             conditions_in_levels = false, 
                                             algorithm = algorithm,
                                             periods = periods)
            
            # Find the index of variable y
            y_idx = findfirst(x -> x == :y, axiskeys(result, 1))
            @test !isnothing(y_idx)
            
            # Check that conditions are met for all periods
            for p in 1:periods
                deviation = abs(result[y_idx, p] - conditions[1, p])
                println("  Period $p: y = $(result[y_idx, p]), target = $(conditions[1, p]), deviation = $deviation")
                @test deviation < 1e-8  # Should match conditions very closely
            end
            
            println("  ✓ $algorithm test passed!")
        end
    end
end
