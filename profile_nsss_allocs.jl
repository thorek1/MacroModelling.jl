# Profile allocations for NSSS calculation with SW07 model
# This requires ProfileCanvas.jl or @profview_allocs macro

using MacroModelling
import MacroModelling: clear_solution_caches!

include("models/Smets_Wouters_2007.jl")

model = Smets_Wouters_2007

# Warm-up
println("Warming up...")
get_steady_state(model, derivatives = false)
println("✓ Warm-up complete")
println()

# Profile allocations
println("Profiling allocations...")
println("Run this with @profview_allocs if you have ProfileCanvas installed:")
println()
println("@profview_allocs for i in 1:1000")
println("    clear_solution_caches!(model, :first_order)")
println("    get_steady_state(model, parameters = model.parameter_values, derivatives = false)")
println("end")
println()

# Simple allocation test without profiling
println("Running simple allocation test (100 iterations)...")
for i in 1:100
    clear_solution_caches!(model, :first_order)
    get_steady_state(model, parameters = model.parameter_values, derivatives = false)
end
println("✓ Complete")
