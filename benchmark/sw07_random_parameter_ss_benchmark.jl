using Revise
using MacroModelling
using BenchmarkTools
using Random
import MacroModelling: clear_solution_caches!

include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl"))

model = Smets_Wouters_2007

init_pars = deepcopy(model.parameter_values)
# Warm-up to ensure NSSS solver infrastructure and initial cache are available.
get_steady_state(model, derivatives = false)

while length(model.caches.solver_cache) > 1
    pop!(model.caches.solver_cache)
end

get_steady_state(model, parameters = init_pars, derivatives = false, verbose = true)


while length(model.caches.solver_cache) > 2
    pop!(model.caches.solver_cache)
end

get_steady_state(model, parameters = init_pars .+ .001, derivatives = false, verbose = true)

trial = @benchmark begin
    # get_steady_state($model, parameters = $init_pars .+ .001, derivatives = false)
    get_solution($model,  $init_pars .+ .001)
end setup = while length(model.caches.solver_cache) > 2
    pop!(model.caches.solver_cache)
end


@profview_allocs for i in 1:10000 
    while length(model.caches.solver_cache) > 2
        pop!(model.caches.solver_cache)
    end

    get_solution(model, init_pars .+ .001)
    # get_steady_state(model, parameters = init_pars .+ .001, derivatives = false)
end


@profview for i in 1:100000
    while length(model.caches.solver_cache) > 2
        pop!(model.caches.solver_cache)
    end

    get_solution(model, init_pars .+ .001)
    # get_steady_state(model, parameters = init_pars .+ .001, derivatives = false)
end

# import MacroModelling: update_post_complete_parameters
# model.constants.post_complete_parameters = update_post_complete_parameters(
#             model.constants.post_complete_parameters;
#             nsss_fastest_solver_parameter_idx = 13,
#         );


model.caches.solver_cache

println(trial)
println("Minimum time: ", minimum(trial).time, " ns")
println("Minimum memory: ", minimum(trial).memory, " bytes")
