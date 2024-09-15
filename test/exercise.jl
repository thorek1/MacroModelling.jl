using MacroModelling
using Zygote
import ForwardDiff
import Turing
# import Pigeons
import Turing: NUTS, sample, logpdf, AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/Gali_2015_chapter_3_nonlinear.jl")

SS(Gali_2015_chapter_3_nonlinear)

get_std(Gali_2015_chapter_3_nonlinear, derivatives = false)
get_std(Gali_2015_chapter_3_nonlinear, derivatives = false, algorithm = :pruned_second_order)


# Optimal simple rule
parameter_value_inputs = [1.5, 0.125]

model_statistics = get_statistics(Gali_2015_chapter_3_nonlinear, 
                                parameter_value_inputs, 
                                algorithm = :pruned_second_order, 
                                parameters = [:ϕᵖⁱ, :ϕʸ], 
                                variance = [:y_gap, :pi_ann])

                                
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015_chapter_3_nonlinear,
                                        parameter_value_inputs, 
                                        algorithm = :pruned_second_order, 
                                        parameters = [:ϕᵖⁱ, :ϕʸ], 
                                        standard_deviation = [:y_gap, :pi_ann])

    weights = [3,1]

    return model_statistics[1]' * weights ./ sum(weights)
end

distance_to_target(parameter_value_inputs)

sol = Optim.optimize(distance_to_target,
                               [1,0.001],
                               [3,.5],
                               parameter_value_inputs,
                            #    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                               Optim.Fminbox(Optim.NelderMead()),
                               Optim.Options(outer_iterations = 1,
                                                time_limit = 600,
                                                show_trace = true))
                                                
sol.minimizer
                 
sol = Optim.optimize(distance_to_target,
[0.5,1,0.001,0.001],
[.975,3,.5,.5],
                               sol.minimizer,
                                Optim.Fminbox(Optim.NelderMead()),
                            #    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                               Optim.Options(outer_iterations = 1,
                                            show_trace = true,
                                            time_limit = 600))


model_statistics = get_statistics(Smets_Wouters_2007,
                                # sol.minimizer, 
                                parameter_value_inputs,
                                algorithm = :pruned_second_order, 
                                parameters = [:crr, :crpi, :cry, :crdy], 
                                variance = [:drobs, :dpinfobs, :ygap])
                                # 4-element Vector{Float64}:
                                # 0.9803094444557999
                                # 8.639241288770886
                                # 0.9171416346407335
                                # 0.9972515638017101

distance_to_target(sol.minimizer)

producivity_shock_size = []
OSR_coefficients = []

for i in range(.4, .6, 10)
    push!(producivity_shock_size, i)
    SS(Smets_Wouters_2007, parameters = :z_ea => i, derivatives = false)
    sol = Optim.optimize(distance_to_target,
    [0.5,1,0.001,0.001],
    [.975,3,.5,.5],
                        sol.minimizer,
                        Optim.Fminbox(Optim.NelderMead()),
                        # Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                        Optim.Options(outer_iterations = 3,
                                    # show_trace = true,
                                    time_limit = 600))
    push!(OSR_coefficients, sol.minimizer)
    println("z_ea: $i => OSR coeffs: $(sol.minimizer)")
end



function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Smets_Wouters_2007, 
                                        parameter_value_inputs, 
                                        algorithm = :first_order, 
                                        parameters = [:crr, :crpi, :cry, :crdy], 
                                        standard_deviation = [:drobs, :pinfobs, :ygap])

    weights = [4,10,3]

    return model_statistics[1]' * weights ./ sum(weights)
end

distance_to_target(parameter_value_inputs)

sol = Optim.optimize(distance_to_target,
                    [0.5,1,0.001,0.001],
                    [.975,3,.5,.5],
                    parameter_value_inputs,
                    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))),
                    # Optim.Fminbox(Optim.NelderMead()),
                    Optim.Options(outer_iterations = 10,
                                        time_limit = 600,
                                        show_trace = true))

sol.minimizer

distance_to_target(sol.minimizer)
