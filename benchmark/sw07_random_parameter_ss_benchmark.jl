using Revise
using MacroModelling
using BenchmarkTools
using Random
import MacroModelling: clear_solution_caches!

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))

model = Smets_Wouters_2007

# Warm-up to ensure NSSS solver infrastructure and initial cache are available.
get_steady_state(model, derivatives = false)

trial = @benchmark begin
    get_steady_state($model, parameters = $model.parameter_values, derivatives = false)
end setup = clear_solution_caches!($model,:first_order)

@profview_allocs for i in 1:1000 
    clear_solution_caches!(model,:first_order)
    get_steady_state(model, parameters = model.parameter_values, derivatives = false)
end


@profview for i in 1:100000
    clear_solution_caches!(model,:first_order)
    get_steady_state(model, parameters = model.parameter_values, derivatives = false)
end

import MacroModelling: update_post_complete_parameters
model.constants.post_complete_parameters = update_post_complete_parameters(
            model.constants.post_complete_parameters;
            nsss_fastest_solver_parameter_idx = 13,
        );

clear_solution_caches!(model,:first_order)
get_steady_state(model, parameters = model.parameter_values, derivatives = false, verbose = true)

model.caches.solver_cache

println(trial)
println("Minimum time: ", minimum(trial).time, " ns")
println("Minimum memory: ", minimum(trial).memory, " bytes")
