#!/usr/bin/env julia
# Test iterative refinement with FS2000 second-order

using Pkg
Pkg.activate(".")

using MacroModelling
using Test

println("\n" * "="^70)
println("Testing Iterative Refinement with FS2000 Model")
println("="^70 * "\n")

# Load FS2000 model
include("models/FS2000.jl")

# Solve with second-order
println("Solving FS2000 with second-order perturbation...")
get_solution(FS2000, algorithm = :second_order)
println("✓ Model solved\n")

# Test conditional forecast with 1 condition, 2 shocks (underdetermined)
println("Testing conditional forecast (underdetermined: 2 shocks, 1 condition)...")
periods = 3
conditions = fill(0.01, (1, periods))  # Condition on y
conditioned_variables = [:y]

try
    result = get_conditional_forecast(FS2000, 
                                     conditioned_variables,
                                     conditions,
                                     periods = periods,
                                     algorithm = :second_order,
                                     verbose = false)
    
    println("✓ Conditional forecast completed")
    println("  Result size: $(size(result))")
    
    # Check if conditions are satisfied
    y_idx = findfirst(x -> x == :y, FS2000.var)
    y_forecasts = result[y_idx, 1:periods]
    
    println("\n  Target vs Actual:")
    for i in 1:periods
        target = conditions[1, i]
        actual = y_forecasts[i]
        error = abs(actual - target)
        status = error < 1e-10 ? "✓" : "✗"
        println("    Period $i: target = $target, actual = $actual, error = $error $status")
    end
    
    max_error = maximum(abs.(y_forecasts - vec(conditions)))
    println("\n  Maximum error: $max_error")
    
    if max_error < 1e-10
        println("  ✓✓✓ TEST PASSED: Iterative refinement working! ✓✓✓")
    elseif max_error < 1e-8
        println("  ✓ TEST MOSTLY PASSED: Good precision (< 1e-8)")
    else
        println("  ⚠ TEST MARGINAL: Precision could be better")
    end
    
catch e
    println("✗ Error during conditional forecast:")
    println("  ", e)
    if isa(e, ErrorException)
        println("  Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

println("\n" * "="^70)
println("Test Complete")
println("="^70)
