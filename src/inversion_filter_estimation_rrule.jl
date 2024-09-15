using Revise
using MacroModelling
import MacroModelling: find_shocks, expand_steady_state, get_and_check_observables, calculate_inversion_filter_loglikelihood, check_bounds, get_NSSS_and_parameters, get_relevant_steady_state_and_state_update, ℳ, calculate_second_order_stochastic_steady_state, timings, second_order_auxilliary_matrices, calculate_third_order_stochastic_steady_state
using Random
using BenchmarkTools
import LinearAlgebra as ℒ
# import Optim, LineSearches
import FiniteDifferences
import Zygote
import Zygote: @ignore_derivatives
import Accessors
import ThreadedSparseArrays
import Polyester
using SparseArrays

import ForwardDiff
# import CSV
# using DataFrames
# using Test

using TimerOutputs
TimerOutputs.enable_debug_timings(MacroModelling)

include("../models/Gali_2015_chapter_3_nonlinear.jl")

include("../models/SGU_2003_debt_premium.jl")

include("../models/Smets_Wouters_2007.jl")

include("../models/Ghironi_Melitz_2005.jl")



# @model RBC_baseline begin
# 	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

# 	ψ * c[0] ^ σ / (1 - l[0]) = z[0] * k[-1] ^ α * l[0] ^ (1 - α) * (1 - α) / l[0]

# 	z[0] * k[-1] ^ α * l[0] ^ (1 - α) = c[0] + k[0] - (1 - δ) * k[-1] + g[0]

# 	# y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

# 	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

# 	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

# end


# @parameters RBC_baseline begin
# 	σᶻ = 0.066

# 	σᵍ = .104

# 	σ = 1

# 	α = 1/3

# 	i_y = 0.25

# 	k_y = 10.4

# 	ρᶻ = 0.97

# 	ρᵍ = 0.989

# 	g_y = 0.2038

# 	# ḡ | ḡ = g_y * y[ss]
#     # z[0] * k[-1] ^ α * l[0] ^ (1 - α)
# 	ḡ | ḡ = g_y * k[ss] ^ α * l[ss] ^ (1 - α)

#     δ = i_y / k_y

#     β = 1 / (α / k_y + (1 - δ))

# 	ψ | l[ss] = 1/3
# end
𝓂 = Ghironi_Melitz_2005
oobbss = [:C, :Q]

𝓂 = Smets_Wouters_2003
get_variables(𝓂)
oobbss = [:L, :W, :R, :pi, :I, :C, :Y]

𝓂 = Smets_Wouters_2007
oobbss = [:labobs, :dwobs, :robs, :pinfobs, :dinve, :dc, :dy]

𝓂 = SGU_2003_debt_premium
get_variables(𝓂)
oobbss = [:r]

𝓂 = Gali_2015_chapter_3_nonlinear
oobbss = [:Y, :R, :Pi]
# 𝓂 = RBC_baseline


T = 𝓂.timings
tol = 1e-12
parameter_values = 𝓂.parameter_values
parameters = 𝓂.parameter_values
verbose = false
presample_periods = 0
sylvester_algorithm = :doubling


periods = 10
# speed up solution and filtering
# algorithm = :second_order
algorithm = :pruned_second_order
# algorithm = :third_order
# algorithm = :pruned_third_order
timer = TimerOutput()
rr = rand()
# Random.seed!(9)
data = simulate(𝓂, 
                algorithm = algorithm, 
                periods = periods, 
                # parameters = :constebeta => .99 + rr * 1e-5, 
                # parameters = :β  => .992, 
                timer = timer)(oobbss,:,:simulate)
timer


timer = TimerOutput()
rr = rand()
# Random.seed!(9)
data = simulate(𝓂, 
                # algorithm = algorithm, 
                periods = periods, 
                parameters = :constebeta => .99 + rr * 1e-5, 
                # parameters = :β  => .992, 
                timer = timer)(oobbss,:,:simulate)
timer

timer = TimerOutput()
get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm, timer = timer)
timer


Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

timer = TimerOutput()
# for i in 1:10
zygdiff = Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm, timer = timer), 𝓂.parameter_values)[1]
# end
timer

@profview for i in 1:3 Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm, timer = timer), 𝓂.parameter_values)[1] end

using BenchmarkTools
@benchmark get_loglikelihood(𝓂, data[:,1:10], 𝓂.parameter_values, algorithm = algorithm)

@benchmark Zygote.gradient(x-> get_loglikelihood(𝓂, data[:,1:10], x, algorithm = algorithm), 𝓂.parameter_values)[1]

get_parameters(𝓂)
# get_parameters(𝓂, values = true)


findiff = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1, max_range = 1e-5), x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

# findiff = FiniteDifferences.grad(FiniteDifferences.forward_fdm(5,1, max_range = 1e-3), x-> get_loglikelihood(𝓂, data, vcat(x,𝓂.parameter_values[2:end]), algorithm = algorithm), 𝓂.parameter_values[1])[1]

zygdiff = Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

isapprox(findiff, zygdiff)
findiff - zygdiff

@benchmark get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm)

@benchmark Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

@benchmark FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1, max_range = 1e-4), x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]


@profview Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

@profview for i in 1:5 Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1] end

@profview for i in 1:50 get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm) end

# fordiff = ForwardDiff.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)

# @benchmark ForwardDiff.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)



findiff = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1, max_range = 1e-4), x-> get_loglikelihood(𝓂, data(:,[1,2]), x, algorithm = algorithm), 𝓂.parameter_values)[1]

zygdiff = Zygote.gradient(x-> get_loglikelihood(𝓂, data(:,[1]), x, algorithm = algorithm), 𝓂.parameter_values)[1]

# fordiff = ForwardDiff.gradient(x-> get_loglikelihood(𝓂, data([:c],[1,2]), x, algorithm = algorithm), 𝓂.parameter_values)

isapprox(findiff, zygdiff)

# isapprox(findiff, fordiff)


get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm)



to = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)

to[7]
to[8]
to[9]
to[10]

# third order
# all_SS + state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃
for1 = ForwardDiff.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)
zyg1 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)[1]
fin1 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)[1]
isapprox(zyg1,fin1)
isapprox(for1,fin1)
zyg1-fin1


zyg2 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[3], 𝓂.parameter_values)[1]
fin2 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[3], 𝓂.parameter_values)[1]
isapprox(zyg2,fin2)


zyg3 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[5], 𝓂.parameter_values)[1]
fin3 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[5], 𝓂.parameter_values)[1]
isapprox(zyg3,fin3)


zyg4 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[6], 𝓂.parameter_values)[1]
fin4 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[6], 𝓂.parameter_values)[1]
isapprox(zyg4,fin4)


zyg5 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[7] |> ℒ.norm, 𝓂.parameter_values)[1]
fin5 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[7] |> ℒ.norm, 𝓂.parameter_values)[1]
isapprox(zyg5,fin5)


zyg6 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[8] |> ℒ.norm, 𝓂.parameter_values)[1]
fin6 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[8] |> ℒ.norm, 𝓂.parameter_values)[1]
isapprox(zyg6,fin6)


zyg7 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[9], 𝓂.parameter_values)[1]
fin7 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[9], 𝓂.parameter_values)[1]
isapprox(zyg7,fin7)

ℒ.norm(zyg7 - fin7) / max(ℒ.norm(fin7), ℒ.norm(zyg7))

zyg8 = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[10], 𝓂.parameter_values)[1]
fin8 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[10], 𝓂.parameter_values)[1]
isapprox(zyg8,fin8)

ℒ.norm(zyg8 - fin8) / max(ℒ.norm(fin8), ℒ.norm(zyg8))


full_partial = sparse([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 17, 17, 18, 18, 22, 22, 25, 25, 26, 26, 27, 27, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 35, 35, 36, 36], [0.26614015732727614, 0.10704230805268979, -0.009897729836585153, -0.010833679333138708, 0.05236745293389423, -0.2274088222544429, 0.027986427059693342, 0.011256218440323292, 0.003563146282100018, -0.015473177596694056, -0.009897729836585153, -0.010833679333138708, -0.0009129912889280832, 0.0007293406100448242, 0.009305520744558133, 0.005673498279434401, -0.0010408128442920687, -0.001139234227145021, 0.0006331591434441617, 0.0003860318416934748, 0.05236745293389422, -0.2274088222544429, 0.009305520744558133, 0.005673498279434401, 0.0024794853525378253, -0.1882138065031564, 0.005506789792846308, -0.02391356674869773, 0.00016870725079123482, -0.012806300236297242, -0.1289019726996042, 0.09192578414111252, 0.027986427059693342, 0.011256218440323292, -0.0010408128442920687, -0.001139234227145021, 0.005506789792846308, -0.023913566748697733, 0.002942960985043586, 0.001183667055403056, 0.0003746887900287177, -0.00162710866537531, 0.003563146282100018, -0.015473177596694055, 0.0006331591434441617, 0.0003860318416934748, 0.00016870725079123482, -0.012806300236297242, 0.0003746887900287177, -0.0016271086653753101, 1.1479050053836699e-5, -0.0008713565109233175], 3, 36)
full_partial_dense = [0.26614015732727614 -0.009897729836585153 0.05236745293389423 0.0 0.027986427059693342 0.003563146282100018 -0.009897729836585153 -0.0009129912889280832 0.009305520744558133 0.0 -0.0010408128442920687 0.0006331591434441617 0.05236745293389422 0.009305520744558133 0.0024794853525378253 0.0 0.005506789792846308 0.00016870725079123482 0.0 0.0 0.0 -0.1289019726996042 0.0 0.0 0.027986427059693342 -0.0010408128442920687 0.005506789792846308 0.0 0.002942960985043586 0.0003746887900287177 0.003563146282100018 0.0006331591434441617 0.00016870725079123482 0.0 0.0003746887900287177 1.1479050053836699e-5; 0.10704230805268979 -0.010833679333138708 -0.2274088222544429 0.0 0.011256218440323292 -0.015473177596694056 -0.010833679333138708 0.0007293406100448242 0.005673498279434401 0.0 -0.001139234227145021 0.0003860318416934748 -0.2274088222544429 0.005673498279434401 -0.1882138065031564 0.0 -0.02391356674869773 -0.012806300236297242 0.0 0.0 0.0 0.09192578414111252 0.0 0.0 0.011256218440323292 -0.001139234227145021 -0.023913566748697733 0.0 0.001183667055403056 -0.00162710866537531 -0.015473177596694055 0.0003860318416934748 -0.012806300236297242 0.0 -0.0016271086653753101 -0.0008713565109233175; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

isapprox(full_partial, full_partial_dense)

full = sparse([1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7, 1, 2, 4, 6, 7], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 22, 22, 22, 22, 22, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36], [0.26614015732727614, 0.10704230805268979, 0.2857473207894207, -0.28120376627292276, 0.11143643362717742, -0.009897729836585153, -0.010833679333138708, -0.010741968939938392, 0.014112495481316222, -0.011485288026574848, 0.05236745293389423, -0.2274088222544429, 0.06268210053147728, 0.03534229348582014, -0.2260732549633295, 0.027986427059693342, 0.011256218440323292, 0.03004825213559124, -0.029570466827486267, 0.011718290290421082, 0.003563146282100018, -0.015473177596694056, 0.004264967665028352, 0.0024047333712001323, -0.0153823039459585, -0.009897729836585153, -0.010833679333138708, -0.010741968939938392, 0.014112495481316222, -0.011485288026574848, -0.0009129912889280832, 0.0007293406100448242, -0.0009765787985923954, -0.0006721062196381086, 0.0007916590338529345, 0.009305520744558133, 0.005673498279434401, 0.010191518097445198, 0.025295581283200603, 0.0055604810571578, -0.0010408128442920687, -0.001139234227145021, -0.0011295902626426617, 0.0014840237917663152, -0.0012077552626529665, 0.0006331591434441617, 0.0003860318416934748, 0.0006934434994137965, 0.0017211426440115888, 0.00037834201007465405, 0.05236745293389422, -0.2274088222544429, 0.06268210053147728, 0.03534229348582014, -0.2260732549633295, 0.009305520744558133, 0.005673498279434401, 0.010191518097445198, 0.025295581283200603, 0.0055604810571578, 0.0024794853525378253, -0.1882138065031564, -0.02224326054102648, 0.1400010996455336, -0.211035127697189, 0.005506789792846308, -0.02391356674869773, 0.00659144434304714, 0.0037164798003289537, -0.02377312286773128, 0.00016870725079123482, -0.012806300236297242, -0.0015134589646471769, 0.009525848017118803, -0.014359091162901527, -0.1289019726996042, 0.09192578414111252, -0.13982245556467268, 0.3446155065565036, 0.09525544397057248, 0.027986427059693342, 0.011256218440323292, 0.03004825213559124, -0.029570466827486267, 0.011718290290421082, -0.0010408128442920687, -0.001139234227145021, -0.0011295902626426617, 0.0014840237917663152, -0.0012077552626529665, 0.005506789792846308, -0.023913566748697733, 0.00659144434304714, 0.0037164798003289537, -0.02377312286773128, 0.002942960985043586, 0.001183667055403056, 0.0031597757554110094, -0.0031095334176527534, 0.001232257017395137, 0.0003746887900287177, -0.00162710866537531, 0.0004484900274650646, 0.00025287388332134817, -0.0016175526899693453, 0.003563146282100018, -0.015473177596694055, 0.004264967665028352, 0.0024047333712001323, -0.0153823039459585, 0.0006331591434441617, 0.0003860318416934748, 0.0006934434994137965, 0.0017211426440115888, 0.00037834201007465405, 0.00016870725079123482, -0.012806300236297242, -0.0015134589646471769, 0.009525848017118803, -0.014359091162901527, 0.0003746887900287177, -0.0016271086653753101, 0.0004484900274650646, 0.00025287388332134817, -0.0016175526899693453, 1.1479050053836699e-5, -0.0008713565109233175, -0.00010297762027496294, 0.000648150483639013, -0.0009770103265479397], 13, 36)

deriv = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

s3 = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)[10]





𝓂
data
parameter_values
# algorithm = :first_order
# filter = :kalman
# warmup_iterations = 0
# presample_periods = 0
# initial_covariance = :theoretical
filter_algorithm = :LagrangeNewton
# tol = 1e-12
# verbose = false



observables = get_and_check_observables(𝓂, data)

solve!(𝓂, verbose = verbose, algorithm = algorithm)

NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

obs_indices = convert(Vector{Int}, indexin(observables, NSSS_labels))

TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)

if collect(axiskeys(data,1)) isa Vector{String}
    data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
end

dt = collect(data(observables))

# prepare data
data_in_deviations = dt .- SS_and_pars[obs_indices]




precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

shocks² = 0.0
logabsdets = 0.0

s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
shock_idxs = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
shock²_idxs = tmp.nzind

shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
var_vol²_idxs = tmp.nzind

tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
var²_idxs = tmp.nzind

𝐒⁻¹  = 𝐒[1][T.past_not_future_and_mixed_idx,:]
𝐒⁻¹ᵉ = 𝐒[1][T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
𝐒¹⁻  = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
𝐒¹ᵉ  = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
𝐒²⁻  = 𝐒[2][cond_var_idx,var²_idxs]
𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
𝐒²ᵉ  = 𝐒[2][cond_var_idx,shock²_idxs]
𝐒⁻²  = 𝐒[2][T.past_not_future_and_mixed_idx,:]

𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
var_vol³_idxs = tmp.nzind

tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
shock_idxs2 = tmp.nzind

tmp = ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), zero(e_in_s⁺) .+ 1) |> sparse
shock_idxs3 = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
shock³_idxs = tmp.nzind

tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
shockvar1_idxs = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
shockvar2_idxs = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
shockvar3_idxs = tmp.nzind

shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)#, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)

𝐒³⁻ᵛ  = 𝐒[3][cond_var_idx,var_vol³_idxs]
𝐒³⁻ᵉ² = 𝐒[3][cond_var_idx,shockvar³2_idxs]
𝐒³⁻ᵉ  = 𝐒[3][cond_var_idx,shockvar³_idxs]
𝐒³ᵉ   = 𝐒[3][cond_var_idx,shock³_idxs]
𝐒⁻³   = 𝐒[3][T.past_not_future_and_mixed_idx,:]

𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³


stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]

kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)

kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

II = sparse(ℒ.I(T.nExo^2))

lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

    𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

    init_guess = zeros(size(𝐒ⁱ, 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kronxxx[i],
                            kron_buffer2,
                            kron_buffer3,
                            kron_buffer4,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ[i],
                            𝐒ⁱ³ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)
    fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])

    ℒ.kron!(kronxxx[i], x[i], kronxx[i])

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
end

# See: https://pcubaborda.net/documents/CGIZ-final.pdf
llh = -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2



∂llh = 1

∂state = similar(state)

∂𝐒 = copy(𝐒)

∂data_in_deviations = similar(data_in_deviations)

∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒²ᵉ = zero(𝐒²ᵉ)
∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)

∂𝐒⁻¹ = zero(𝐒⁻¹)
∂𝐒⁻² = zero(𝐒⁻²)
∂𝐒⁻³ = zero(𝐒⁻³)

∂state¹⁻_vol = zero(state¹⁻_vol)
∂x = zero(x[1])
∂kronxx = zero(kronxx[1])
∂state = zeros(T.nPast_not_future_and_mixed)
∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)

n_end = 3 # size(data_in_deviations, 2)

for i in 3:-1:1 # reverse(axes(data_in_deviations,2))
    ∂kronstate¹⁻_vol *= 0

    # stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
    ∂aug_state = 𝐒⁻¹' * ∂state
    ∂kronaug_state = 𝐒⁻²' * ∂state / 2
    ∂kronkronaug_state = 𝐒⁻³' * ∂state / 6

    re∂kronkronaug_state = reshape(∂kronkronaug_state, 
                                    length(aug_state[i]), 
                                    length(aug_state[i])^2)

    ei = 1
    for e in eachslice(re∂kronkronaug_state; dims = (1))
        ∂aug_state[ei] += ℒ.dot(ℒ.kron(aug_state[i], aug_state[i]),e)
        ei += 1
    end
    
    ei = 1
    for e in eachslice(re∂kronkronaug_state; dims = (2))
        ∂kronaug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    re∂kronaug_state = reshape(∂kronaug_state, 
                            length(aug_state[i]), 
                            length(aug_state[i]))

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (1))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (2))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    if i > 1 && i < n_end # size(data_in_deviations,2)
        ∂state *= 0
    end

    # aug_state[i] = [stt; 1; x[i]]
    ∂state += ∂aug_state[1:length(∂state)]

    # aug_state[i] = [stt; 1; x[i]]
    ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

    # shocks² += sum(abs2,x[i])
    if i < n_end # size(data_in_deviations,2)
        ∂x -= copy(x[i])
    else
        ∂x += copy(x[i])
    end

    # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    ∂jacc = inv(jacc[i])'

    # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
    ∂𝐒ⁱ = -∂jacc / 2 # fine

    ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

    re∂kronIx = reshape(∂kronIx, 
                            T.nExo, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIx; dims = (1,3))
        if i < n_end # size(data_in_deviations,2)
            ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

    ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2
    
    re∂kronIxx = reshape(∂kronIxx, 
                            T.nExo^2, 
                            T.nExo, 
                            1,
                            T.nExo)
          
    ∂kronxx *= 0

    ei = 1
    for e in eachslice(re∂kronIxx; dims = (1,3))
        if i < n_end # size(data_in_deviations,2)
            ∂kronxx[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂kronxx[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    re∂kronxx = reshape(∂kronxx, 
                            T.nExo, 
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronxx; dims = (2))
        ∂x[ei] += ℒ.dot(x[i],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronxx; dims = (1))
        ∂x[ei] += ℒ.dot(x[i],e)
        ei += 1
    end

    # find_shocks
    ∂xλ = vcat(∂x, zero(λ[i]))

    S = fXλp[i]' \ ∂xλ

    if i < n_end # size(data_in_deviations,2)
        S *= -1
    end

    ∂shock_independent = S[T.nExo+1:end] # fine

    ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine

    ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

    # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    state¹⁻_vol = [aug_state[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later

    ∂state¹⁻_vol *= 0

    ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

    re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
        ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        ei += 1
    end

    ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

    re∂kronIstate¹⁻_volstate¹⁻_vol = reshape(∂kronIstate¹⁻_volstate¹⁻_vol, 
                            length(state¹⁻_vol)^2, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIstate¹⁻_volstate¹⁻_vol; dims = (1,3))
        ∂kronstate¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e) # ∂kronstate¹⁻_vol is dealt with later
        ei += 1
    end

    # 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
    ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

    re∂kronIIstate¹⁻_vol = reshape(∂kronIIstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            T.nExo^2, 
                            1,
                            T.nExo^2)

    ei = 1
    for e in eachslice(re∂kronIIstate¹⁻_vol; dims = (1,3))
        ∂state¹⁻_vol[ei] += ℒ.dot(II,e)
        ei += 1
    end


    # shock_independent = copy(data_in_deviations[:,i])
    ∂data_in_deviations[:,i] = ∂shock_independent

    # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent # fine

    # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

    # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
    ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

    re∂kronstate¹⁻_volstate¹⁻_vol = reshape(∂kronstate¹⁻_volstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            length(state¹⁻_vol)^2)
                    
    ei = 1
    for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (2))
        ∂kronstate¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (1))
        ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.kron(state¹⁻_vol, state¹⁻_vol),e) # fine
        ei += 1
    end        

    re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, # fine
                            length(state¹⁻_vol), 
                            length(state¹⁻_vol))

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
        ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
        ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e) # fine
        ei += 1
    end

    # state¹⁻_vol = vcat(state¹⁻, 1)
    ∂state += ∂state¹⁻_vol[1:end-1]
end

∂𝐒 = [copy(𝐒[1]) * 0, copy(𝐒[2]) * 0, copy(𝐒[3]) * 0]

∂𝐒[1][cond_var_idx,end-T.nExo+1:end] += ∂𝐒¹ᵉ
∂𝐒[2][cond_var_idx,shockvar²_idxs] += ∂𝐒²⁻ᵉ
∂𝐒[2][cond_var_idx,shock²_idxs] += ∂𝐒²ᵉ
∂𝐒[3][cond_var_idx,shockvar³2_idxs] += ∂𝐒³⁻ᵉ²
∂𝐒[3][cond_var_idx,shockvar³_idxs] += ∂𝐒³⁻ᵉ
∂𝐒[3][cond_var_idx,shock³_idxs] += ∂𝐒ⁱ³ᵉ / 6 # 𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

∂𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1] += ∂𝐒¹⁻ᵛ
∂𝐒[2][cond_var_idx,var_vol²_idxs] += ∂𝐒²⁻ᵛ
∂𝐒[3][cond_var_idx,var_vol³_idxs] += ∂𝐒³⁻ᵛ

∂𝐒[1][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻¹
∂𝐒[2][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻²
∂𝐒[3][T.past_not_future_and_mixed_idx,:] += ∂𝐒⁻³

∂𝐒[1] *= ∂llh
∂𝐒[2] *= ∂llh
∂𝐒[3] *= ∂llh




findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    # stt = copy(state[T.past_not_future_and_mixed_idx])
                    stt = X
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    # dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    # dtt = X

                    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

                    for i in 1:2# axes(data_in_deviations,2)
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(data_in_deviations[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                    end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                copy(state[T.past_not_future_and_mixed_idx]))[1]'


# check where it breaks




findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                    # stt = X
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    # dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    dtt = X

                    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

                    for i in 1:3# axes(data_in_deviations,2)
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                    end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                # copy(state[T.past_not_future_and_mixed_idx]))[1]'
                copy(data_in_deviations[:,[1,2,3]]))[1]'

findiff




# sequential instead of loop

findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                    # stt = X
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    # dtt = X

                    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

                    # for i in 1:2# axes(data_in_deviations,2)
                    i = 1
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(dtt[:,i])
                        shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6

                        
                    # for i in 1:2# axes(data_in_deviations,2)
                    i = 2
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])
                        # shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6

                    # for i in 1:2# axes(data_in_deviations,2)
                        i = 3
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])
                        # shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                    # end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                # copy(state[T.past_not_future_and_mixed_idx]))[1]'
                copy(data_in_deviations[:,[1]]))[1]'


# check where it breaks across iterations

findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                    # stt = X
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    # dtt = X

                    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

                    # for i in 1:2# axes(data_in_deviations,2)
                    i = 1
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])
                        # shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))
                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(X, X))
                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), X) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))
                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), X) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(X, X))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                            # shocks² += sum(abs2,X)
                        end

                        aug_state[i] = [stt; 1; x[i]]
                        # aug_state[i] = [stt; 1; X]
                        # aug_state[i] = [X; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                        # stt = 𝐒⁻¹ * X + 𝐒⁻² * ℒ.kron(X, X) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(X,X),X) / 6

                        
                    # for i in 1:2# axes(data_in_deviations,2)
                    i = 2
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])
                        # shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state[i] = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                    # end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                # copy(state[T.past_not_future_and_mixed_idx]))[1]'
                copy(x[1]))[1]'








stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kronxxx = [zeros(T.nExo^3) for _ in 1:size(data_in_deviations,2)]

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

kron_buffer3 = ℒ.kron(J, zeros(T.nExo^2))

kron_buffer4 = ℒ.kron(ℒ.kron(J, J), zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = [zero(𝐒²ᵉ) for _ in 1:size(data_in_deviations,2)]

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[1] * ℒ.kron(ℒ.I(T.nExo), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ[1]' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]

kronxxλ_tmp = ℒ.kron(x[1], kronxλ_tmp)

kronxxλ = [kronxxλ_tmp for _ in 1:size(data_in_deviations,2)]

II = sparse(ℒ.I(T.nExo^2))

lI = 2 * ℒ.I(size(𝐒ⁱ, 2))

𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

    𝐒ⁱ²ᵉ[i] = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

    init_guess = zeros(size(𝐒ⁱ, 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kronxxx[i],
                            kron_buffer2,
                            kron_buffer3,
                            kron_buffer4,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ[i],
                            𝐒ⁱ³ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    if !matched println("failed to find shocks") end
        
    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxxx[i], x[i], kronxx[i])

    jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), kronxx[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)
    fXλp[i] = [reshape((2 * 𝐒ⁱ²ᵉ[i] + 6 * 𝐒ⁱ³ᵉ * ℒ.kron(II, x[i]))' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - lI  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    ℒ.kron!(kronxxλ[i], x[i], kronxλ[i])
            
    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
end







∂llh = 1

∂state = similar(state)

∂𝐒 = copy(𝐒)

∂data_in_deviations = similar(data_in_deviations)

∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒²ᵉ = zero(𝐒²ᵉ)
∂𝐒ⁱ³ᵉ = zero(𝐒ⁱ³ᵉ)

∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)
∂𝐒³⁻ᵉ = zero(𝐒³⁻ᵉ)
∂𝐒³⁻ᵉ² = zero(𝐒³⁻ᵉ²)

∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)
∂𝐒³⁻ᵛ = zero(𝐒³⁻ᵛ)

∂𝐒⁻¹ = zero(𝐒⁻¹)
∂𝐒⁻² = zero(𝐒⁻²)
∂𝐒⁻³ = zero(𝐒⁻³)

∂state¹⁻_vol = zero(state¹⁻_vol)
∂x = zero(x[1])
∂kronxx = zero(kronxx[1])
∂state = zeros(T.nPast_not_future_and_mixed)
∂kronstate¹⁻_vol = zeros(length(state¹⁻_vol)^2)

n_end = 1

i = 1
    ∂kronstate¹⁻_vol *= 0

    # stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
    ∂aug_state = 𝐒⁻¹' * ∂state
    ∂kronaug_state = 𝐒⁻²' * ∂state / 2
    ∂kronkronaug_state = 𝐒⁻³' * ∂state / 6

    re∂kronkronaug_state = reshape(∂kronkronaug_state, 
                                    length(aug_state[i]), 
                                    length(aug_state[i])^2)

    ei = 1
    for e in eachslice(re∂kronkronaug_state; dims = (1))
        ∂aug_state[ei] += ℒ.dot(ℒ.kron(aug_state[i], aug_state[i]),e)
        ei += 1
    end
    
    ei = 1
    for e in eachslice(re∂kronkronaug_state; dims = (2))
        ∂kronaug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    re∂kronaug_state = reshape(∂kronaug_state, 
                            length(aug_state[i]), 
                            length(aug_state[i]))

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (1))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (2))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    if i > 1 && i < n_end # size(data_in_deviations,2)
        ∂state *= 0
    end

    # aug_state[i] = [stt; 1; x[i]]
    ∂state += ∂aug_state[1:length(∂state)]

    # aug_state[i] = [stt; 1; x[i]]
    ∂x = ∂aug_state[T.nPast_not_future_and_mixed+2:end]

    # shocks² += sum(abs2,x[i])
    if i < n_end # size(data_in_deviations,2)
        ∂x -= copy(x[i])
    else
        ∂x += copy(x[i])
    end

    # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    ∂jacc = inv(ℒ.svd(jacc[i]))'

    # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ[i] * ℒ.kron(ℒ.I(T.nExo), x) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x))
    # ∂𝐒ⁱ = -∂jacc / 2 # fine

    ∂kronIx = 𝐒ⁱ²ᵉ[i]' * ∂jacc

    re∂kronIx = reshape(∂kronIx, 
                            T.nExo, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIx; dims = (1,3))
        if i < n_end # size(data_in_deviations,2)
            ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    ∂𝐒ⁱ²ᵉ = -∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

    ∂kronIxx = 𝐒ⁱ³ᵉ' * ∂jacc * 3 / 2
    
    re∂kronIxx = reshape(∂kronIxx, 
                            T.nExo^2, 
                            T.nExo, 
                            1,
                            T.nExo)
          
    ∂kronxx *= 0

    ei = 1
    for e in eachslice(re∂kronIxx; dims = (1,3))
        if i < n_end # size(data_in_deviations,2)
            ∂kronxx[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂kronxx[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    re∂kronxx = reshape(∂kronxx, 
                            T.nExo, 
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronxx; dims = (2))
        ∂x[ei] += ℒ.dot(x[i],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronxx; dims = (1))
        ∂x[ei] += ℒ.dot(x[i],e)
        ei += 1
    end

    # find_shocks
    # λ = tmp' \ x * 2

    ∂xλ = vcat(∂x, zero(λ[i]))

    S = fXλp[i]' \ ∂xλ

    if i < n_end # size(data_in_deviations,2)
        S *= -1
    end

    ∂shock_independent = S[T.nExo+1:end] # fine

    copyto!(∂𝐒ⁱ, ℒ.kron(S[1:T.nExo], λ[i]) - ℒ.kron(x[i], S[T.nExo+1:end]))
    ∂𝐒ⁱ -= ∂jacc / 2 # fine

    # ∂𝐒ⁱ += S[1:T.nExo] * λ[i]' - S[T.nExo + 1:end] * x[i]' # fine
    ℒ.kron(x[i],λ[i])
    ∂𝐒ⁱ²ᵉ += reshape(2 * ℒ.kron(S[1:T.nExo], ℒ.kron(x[i],λ[i])) - ℒ.kron(kronxx[i], S[T.nExo+1:end]), size(∂𝐒ⁱ²ᵉ))
    # ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] * kronxλ[i]' - S[T.nExo + 1:end] * kronxx[i]'

    # 𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2
    state¹⁻_vol = [aug_state[i][1:T.nPast_not_future_and_mixed];1] # define here as it is used multiple times later

    ∂state¹⁻_vol *= 0

    ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

    re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
        ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        ei += 1
    end

    ∂kronIstate¹⁻_volstate¹⁻_vol = 𝐒³⁻ᵉ²' * ∂𝐒ⁱ / 2

    re∂kronIstate¹⁻_volstate¹⁻_vol = reshape(∂kronIstate¹⁻_volstate¹⁻_vol, 
                            length(state¹⁻_vol)^2, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIstate¹⁻_volstate¹⁻_vol; dims = (1,3))
        ∂kronstate¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e) # ∂kronstate¹⁻_vol is dealt with later
        ei += 1
    end

    # 𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2
    ∂kronIIstate¹⁻_vol = 𝐒³⁻ᵉ' * ∂𝐒ⁱ²ᵉ / 2

    re∂kronIIstate¹⁻_vol = reshape(∂kronIIstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            T.nExo^2, 
                            1,
                            T.nExo^2)

    ei = 1
    for e in eachslice(re∂kronIIstate¹⁻_vol; dims = (1,3))
        ∂state¹⁻_vol[ei] += ℒ.dot(II,e)
        ei += 1
    end


    # shock_independent = copy(data_in_deviations[:,i])
    ∂data_in_deviations[:,i] = ∂shock_independent

    # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent # fine

    # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    ∂kronstate¹⁻_vol -= 𝐒²⁻ᵛ' * ∂shock_independent / 2

    # ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
    ∂kronstate¹⁻_volstate¹⁻_vol = -𝐒³⁻ᵛ' * ∂shock_independent / 6

    re∂kronstate¹⁻_volstate¹⁻_vol = reshape(∂kronstate¹⁻_volstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            length(state¹⁻_vol)^2)
                    
    ei = 1
    for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (2))
        ∂kronstate¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_volstate¹⁻_vol; dims = (1))
        ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.kron(state¹⁻_vol, state¹⁻_vol),e) # fine
        ei += 1
    end        

    re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, # fine
                            length(state¹⁻_vol), 
                            length(state¹⁻_vol))

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
        ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
        ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e) # fine
        ei += 1
    end

    # state¹⁻_vol = vcat(state¹⁻, 1)
    ∂state += ∂state¹⁻_vol[1:end-1]
# end


stt = copy(state[T.past_not_future_and_mixed_idx])

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

dtt = copy(data_in_deviations)

shock_independent = copy(dtt[:,i])
# shock_independent = copy(X[:,i])

ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)

ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                    # stt = X
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    # dtt = X

                    𝐒ⁱ³ᵉ = 𝐒³ᵉ / 6

                    # for i in 1:2# axes(data_in_deviations,2)
                    i = 1
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(dtt[:,i])
                        # shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ² * ℒ.kron(ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol), state¹⁻_vol) / 2

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 + 𝐒³⁻ᵉ * ℒ.kron(II, state¹⁻_vol) / 2

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kronxxx[i],
                                                kron_buffer2,
                                                kron_buffer3,
                                                kron_buffer4,
                                                J,
                                                𝐒ⁱ,
                                                X,
                                                𝐒ⁱ³ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )
# return x[i]
                        jacc[i] =  𝐒ⁱ + 2 * X * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))
                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i]) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(X, X))
                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), X) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x[i], x[i]))
                        # jacc[i] =  𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), X) + 3 * 𝐒ⁱ³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(X, X))

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                                # logabsdets += ℒ.logabsdet(X)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                                # logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(X))
                            end

                            shocks² += sum(abs2,x[i])
                            # shocks² += sum(abs2,X)
                        end

                        aug_state[i] = [stt; 1; x[i]]
                        # aug_state[i] = [stt; 1; X]
                        # aug_state[i] = [X; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state[i],aug_state[i]),aug_state[i]) / 6
                        # stt = 𝐒⁻¹ * X + 𝐒⁻² * ℒ.kron(X, X) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(X,X),X) / 6

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                # copy(state[T.past_not_future_and_mixed_idx]))[1]'
                # copy(state[T.past_not_future_and_mixed_idx]))[1]'
                copy(𝐒ⁱ²ᵉ))[1]'

                jacc[i]



# -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])

# jacc' \ x[1]

∂state = similar(state)

∂𝐒 = copy(𝐒)

∂data_in_deviations = similar(data_in_deviations)

∂llh = 1



# shocks² += sum(abs2,x[i])
∂x = copy(x[1])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc')

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[1]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end


fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
-jacc  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

init_guess = zeros(size(𝐒ⁱ, 2))

∂shock_independent = similar(data_in_deviations[:,1])

∂xλ = vcat(∂x, zero(λ[1]))

S = -fXλp' \ ∂xλ

copyto!(∂shock_independent, S[length(init_guess)+1:end])

# ∂𝐒ⁱ = similar(𝐒ⁱ)

# ∂𝐒ⁱ²ᵉ = similar(𝐒ⁱ²ᵉ)


# copyto!(∂𝐒ⁱ, ℒ.kron(S[1:length(init_guess)], λ[1]) - ℒ.kron(x[1], S[length(init_guess)+1:end]))

# copyto!(∂𝐒ⁱ²ᵉ, 2 * ℒ.kron(S[1:length(init_guess)], kronxλ[1]) - ℒ.kron(kronxx[1], S[length(init_guess)+1:end]))


# shock_independent = data_in_deviations[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
∂data_in_deviations[:,1] = -∂shock_independent'



∂x = copy(x[1])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[2])), x[1])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc')

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[1]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end


fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
-jacc  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

∂shock_independent = similar(data_in_deviations[:,1])

∂xλ = vcat(∂x, zero(λ[1]))

S = -fXλp' \ ∂xλ

copyto!(∂shock_independent, S[length(init_guess)+1:end])

∂x = ∂shock_independent


𝐒⁻¹ᵉ * ∂x

𝐒⁻¹[:,end-T.nExo+1:end] * ∂x
aug_state = [stt; 1; x[1]]

∂aug_state = zero(aug_state)

∂aug_state[end-T.nExo+1:end] = ∂x

∂state = 𝐒⁻¹ * ∂aug_state + 𝐒⁻² * ℒ.kron(∂aug_state, aug_state)

∂state¹⁻_vol = zero(state¹⁻_vol)
∂state¹⁻_vol[1:T.nPast_not_future_and_mixed] = ∂state

∂shock_independent = 𝐒¹⁻ᵛ * ∂state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(∂state¹⁻_vol, state¹⁻_vol)



stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# Zygote.gradient(x->sum(abs2,x), ones(5) .+1.3)

# Zygote.gradient(x->ℒ.logabsdet(x)[1], jacc)[1]
# inv(jacc)'

findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), x -> calculate_inversion_filter_loglikelihood(Val(:second_order),
                                                    state, 
                                                    𝐒, 
                                                    x, 
                                                    observables,
                                                    T), data_in_deviations[:,1:2])[1]



findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), x -> calculate_inversion_filter_loglikelihood(Val(:second_order),
                                                    state, 
                                                    𝐒, 
                                                    x, 
                                                    observables,
                                                    T), data_in_deviations[:,[1]])[1]





findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    dtt[:,1] = X[:,1]

                    # dt = X

                    for i in 1:2#axes(data_in_deviations,2)
                        state¹⁻ = stt

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(data_in_deviations[:,i])
                        shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                        init_guess = zeros(size(𝐒ⁱ, 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
                    end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
data_in_deviations[:,[1]])[1]




∂shock_independent = zero(data_in_deviations[:,1])

∂𝐒ⁱ = zero(𝐒ⁱ)

∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

∂state = zero(state)

∂aug_state = zero(aug_state)


aug_state
state¹⁻_vol

i = 2
∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc')

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
-jacc  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp' \ ∂xλ

∂shock_independent += S[length(init_guess)+1:end]

copyto!(∂𝐒ⁱ, ℒ.kron(S[1:length(init_guess)], λ[i]) - ℒ.kron(x[i], S[length(init_guess)+1:end]))
        
copyto!(∂𝐒ⁱ²ᵉ, 2 * ℒ.kron(S[1:length(init_guess)], kronxλ[i]) - ℒ.kron(kronxx[i], S[length(init_guess)+1:end]))

state¹⁻_vol = [stt
                1]

∂state¹⁻_vol = 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = 𝐒²⁻ᵛ' * ∂shock_independent

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, size(state¹⁻_vol,1), size(state¹⁻_vol,1), size(state¹⁻_vol,2), size(state¹⁻_vol,2))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2,4))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state = ∂state¹⁻_vol[1:length(∂state¹⁻_vol)-1]

∂aug_state = 𝐒⁻¹' * ∂state

∂kronaug_state = 𝐒⁻²' * ∂state

re∂kronaug_state = reshape(∂kronaug_state, size(∂aug_state,1), size(∂aug_state,1), size(∂aug_state,2), size(∂aug_state,2))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1,3))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2,4))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

∂state = ∂aug_state[1:length(∂state)]

∂x += ∂aug_state[length(∂state)+2:end]


i = 1

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
-jacc  zeros(size(𝐒ⁱ, 1),size(𝐒ⁱ, 1))]

∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp' \ ∂xλ

∂shock_independent += S[length(init_guess)+1:end]



















findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    dtt[:,1] = X[:,1]

                    i = 1
                    # for i in 1:2#axes(data_in_deviations,2)
                        state¹⁻_vol = vcat(stt, 1)
                        
                        shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

                        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                        if i > presample_periods
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        aug_state = [stt; 1; x[i]]

                        stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
                    # end
                    
                        i = 2
                        
                        state¹⁻_vol = vcat(stt, 1)
                            
                        shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

                        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

                        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                        if i > presample_periods
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
data_in_deviations[:,[1]])[1]









stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)

    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

    if i > presample_periods
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# end

    i = 2
    
    state¹⁻_vol = vcat(stt, 1)
        
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)


findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), X)

                        if i > presample_periods
                            if T.nExo == length(observables)
                                logabsdets = ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² = sum(abs2,X)
                        end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
x[2])[1]




∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc')

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end



###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)

    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

    if i > presample_periods
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# end

    i = 2
    
    state¹⁻_vol = vcat(stt, 1)
        
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 






findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                X)

                        jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                        if i > presample_periods
                            if T.nExo == length(observables)
                                logabsdets = ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² = sum(abs2,x[i])
                        end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                shock_independent)[1]




x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]

∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp' \ ∂xλ

∂shock_independent = -S[length(init_guess)+1:end]'





###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)

    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

    if i > presample_periods
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# end

    i = 2
    
    state¹⁻_vol = vcat(stt, 1)
        
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 






findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                        yy, matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                zero(kronxx[i]),
                                                zero(kron_buffer2),
                                                J,
                                                𝐒ⁱ,
                                                X,
                                                shock_independent)

                        jacc = 𝐒ⁱ + 2 * X * ℒ.kron(ℒ.I(T.nExo), yy)

                        # if i > presample_periods
                            if T.nExo == length(observables)
                                logabsdets = ℒ.logabsdet(jacc ./ precision_factor)[1]
                            else
                                logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                            end

                            shocks² = sum(abs2,yy)
                        # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                𝐒ⁱ²ᵉ)[1]'





x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]'

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = ℒ.kron(S[1:length(init_guess)], λ[i]) - ℒ.kron(x[i], S[length(init_guess)+1:end])
∂𝐒ⁱ -= vec(∂jacc)/2

∂𝐒ⁱ²ᵉ = 2 * ℒ.kron(S[1:length(init_guess)], kronxλ[i]) - ℒ.kron(kronxx[i], S[length(init_guess)+1:end])
∂𝐒ⁱ²ᵉ -= vec(∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])')





###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)

    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

    if i > presample_periods
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# end

    i = 2
    
    state¹⁻_vol = vcat(stt, 1)
        
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 






findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                    i = 2
                    
                    state¹⁻_vol = vcat(stt, 1)
                        
                    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
                
                    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
                
                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
                    
                    yy, matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            zero(kronxx[i]),
                                            zero(kron_buffer2),
                                            J,
                                            𝐒ⁱ,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent)

                    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)

                    # if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets = ℒ.logabsdet(jacc ./ precision_factor)[1]
                        else
                            logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                        end

                        shocks² = sum(abs2,yy)
                    # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                state¹⁻_vol)[1]'




∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)

x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂𝐒ⁱ²ᵉ = 2 * S[1:length(init_guess)] * kronxλ[i]' - S[length(init_guess)+1:end] * kronxx[i]'
∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2

∂𝐒²⁻ᵉ = ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

∂𝐒¹ᵉ = ∂𝐒ⁱ

∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state¹⁻ = ∂state¹⁻_vol[1:end-1]



###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)

    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

    if i > presample_periods
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state = [stt; 1; x[i]]

# end




findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                    stt = 𝐒⁻¹ * X + 𝐒⁻² * ℒ.kron(X, X) / 2

                    i = 2
                    
                    state¹⁻_vol = vcat(stt, 1)
                        
                    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
                
                    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
                
                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
                    
                    yy, matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            zero(kronxx[i]),
                                            zero(kron_buffer2),
                                            J,
                                            𝐒ⁱ,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent)

                    jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)

                    # if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets = ℒ.logabsdet(jacc ./ precision_factor)[1]
                        else
                            logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                        end

                        shocks² = sum(abs2,yy)
                    # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                aug_state)[1]'




stt = copy(state[T.past_not_future_and_mixed_idx])

shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
state¹⁻_vol = vcat(stt, 1)

shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

x[i], matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        kronxx[i],
                        kron_buffer2,
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

if i > presample_periods
    if T.nExo == length(observables)
        logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    else
        logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
    end

    shocks² += sum(abs2,x[i])
end

aug_state = [stt; 1; x[i]]

stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

i = 2

state¹⁻_vol = vcat(stt, 1)
    
shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

yy, matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        zero(kronxx[i]),
                        zero(kron_buffer2),
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)

∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)

x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂𝐒ⁱ²ᵉ = 2 * S[1:length(init_guess)] * kronxλ[i]' - S[length(init_guess)+1:end] * kronxx[i]'
∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2

∂𝐒²⁻ᵉ = ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

∂𝐒¹ᵉ = ∂𝐒ⁱ

∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

# i = 1
# aug_state = [stt; 1; x[i]]
# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
# i = 2
# aug_state = [stt; 1; x[i]]
re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state), 
                        length(aug_state))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

∂aug_state







###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)
    
    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            zeros(size(𝐒ⁱ, 2)),
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ,
                            𝐒ⁱ²ᵉ,
                            shock_independent)


# end




findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                    i = 1

                    jacc2 = 𝐒ⁱ + 2 * X * ℒ.kron(ℒ.I(T.nExo), x[i])

                    if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets = ℒ.logabsdet(jacc2 ./ precision_factor)[1]
                        else
                            logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc2 ./ precision_factor))
                        end
                
                        shocks² = sum(abs2,x[i])
                    end

                    aug_state = [stt; 1; x[i]]

                    stt2 = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

                    i = 2
                    
                    state¹⁻_vol = vcat(stt2, 1)
                        
                    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)
                
                    𝐒ⁱ2 = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
                
                    # 𝐒ⁱ²ᵉ2 = 𝐒²ᵉ / 2 
                    
                    yy, matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            zero(kronxx[i]),
                                            zero(kron_buffer2),
                                            J,
                                            𝐒ⁱ2,
                                            X,
                                            shock_independent)

                    jacc = 𝐒ⁱ2 + 2 * X * ℒ.kron(ℒ.I(T.nExo), yy)

                    # if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                        else
                            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                        end

                        shocks² += sum(abs2,yy)
                    # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                𝐒ⁱ²ᵉ)[1]'




stt = copy(state[T.past_not_future_and_mixed_idx])

shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
state¹⁻_vol = vcat(stt, 1)

shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

x[i], matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        kronxx[i],
                        kron_buffer2,
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

if i > presample_periods
    if T.nExo == length(observables)
        logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    else
        logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
    end

    shocks² += sum(abs2,x[i])
end

aug_state = [stt; 1; x[i]]

stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

i = 2

state¹⁻_vol = vcat(stt, 1)
    
shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

yy, matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        zero(kronxx[i]),
                        zero(kron_buffer2),
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)



∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)

x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂𝐒ⁱ²ᵉ = 2 * S[1:length(init_guess)] * kronxλ[i]' - S[length(init_guess)+1:end] * kronxx[i]'
∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2

∂𝐒²⁻ᵉ = ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

∂𝐒¹ᵉ = ∂𝐒ⁱ

∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state), 
                        length(aug_state))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

∂state = ∂aug_state[1:length(∂state)]

∂x = ∂aug_state[length(∂state)+2:end]

i = 1

stt = copy(state[T.past_not_future_and_mixed_idx])

state¹⁻_vol = vcat(stt, 1)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)


∂x -= copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = -inv(jacc)'


# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂𝐒ⁱ²ᵉ += ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'




###################



stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
    state¹⁻_vol = vcat(stt, 1)

    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 



findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                    i = 1

                    shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

                    𝐒ⁱ = 𝐒¹ᵉ + X * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                    x[i], matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            kronxx[i],
                                            kron_buffer2,
                                            J,
                                            𝐒ⁱ,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent)

                    jacc2 = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                    if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets = ℒ.logabsdet(jacc2 ./ precision_factor)[1]
                        else
                            logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc2 ./ precision_factor))
                        end
                
                        shocks² = sum(abs2,x[i])
                    end

                    aug_state = [stt; 1; x[i]]

                    stt2 = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

                    i = 2
                    
                    state¹⁻_vol2 = vcat(stt2, 1)
                        
                    shock_independent2 = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol2 + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol2, state¹⁻_vol2) / 2)
                
                    𝐒ⁱ2 = 𝐒¹ᵉ + X * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol2)
                    
                    yy, matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            zero(kronxx[i]),
                                            zero(kron_buffer2),
                                            J,
                                            𝐒ⁱ2,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent2)

                    jacc = 𝐒ⁱ2 + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)

                    # if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                        else
                            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                        end

                        shocks² += sum(abs2,yy)
                    # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                𝐒²⁻ᵉ)[1]'




stt = copy(state[T.past_not_future_and_mixed_idx])

shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
state¹⁻_vol = vcat(stt, 1)

shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

x[i], matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        kronxx[i],
                        kron_buffer2,
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

if i > presample_periods
    if T.nExo == length(observables)
        logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    else
        logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
    end

    shocks² += sum(abs2,x[i])
end

aug_state = [stt; 1; x[i]]

stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

i = 2

state¹⁻_vol = vcat(stt, 1)
    
shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

yy, matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        zero(kronxx[i]),
                        zero(kron_buffer2),
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)



∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)

x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent

ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂𝐒ⁱ²ᵉ = 2 * S[1:length(init_guess)] * kronxλ[i]' - S[length(init_guess)+1:end] * kronxx[i]'
∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2

∂𝐒²⁻ᵉ = ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

∂𝐒¹ᵉ = ∂𝐒ⁱ

∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state), 
                        length(aug_state))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

∂state = ∂aug_state[1:length(∂state)]

∂x = ∂aug_state[length(∂state)+2:end]

i = 1

stt = copy(state[T.past_not_future_and_mixed_idx])

state¹⁻_vol = vcat(stt, 1)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)


∂x -= copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = -inv(jacc)'


# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end


λ[i] = jacc' \ x[i] * 2

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# ∂x *= 0
# ∂x[3] = 1
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = -S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent


ℒ.kron!(kronxx[i], x[i], x[i])

ℒ.kron!(kronxλ[i], x[i], λ[i])

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂𝐒ⁱ²ᵉ -= 2 * S[1:length(init_guess)] * kronxλ[i]' - S[length(init_guess)+1:end] * kronxx[i]'
∂𝐒ⁱ²ᵉ += ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'



∂𝐒²ᵉ -= ∂𝐒ⁱ²ᵉ / 2

∂𝐒²⁻ᵉ -= ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)'

∂𝐒¹ᵉ -= ∂𝐒ⁱ



isapprox(findiff, vec(∂𝐒²⁻ᵉ))



#####
# this seems to work for two rounds
# let's start by getting derivs for data_in_deviations in a round




findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(5,1, max_range = 1e-6), 
                X -> begin
                    i = 1

                    stt = copy(state[T.past_not_future_and_mixed_idx])
                                    
                    shocks² = 0.0
                    logabsdets = 0.0

                    dtt = copy(data_in_deviations)
                    # dtt[:,1] = X[:,1]

                    i = 1
                    # for i in 1:2#axes(data_in_deviations,2)
                    state¹⁻_vol = vcat(stt, 1)

                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                    shock_independent = X[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

                    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                    x[i], matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            kronxx[i],
                                            kron_buffer2,
                                            J,
                                            𝐒ⁱ,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent)

                    jacc2 = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

                    if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets = ℒ.logabsdet(jacc2 ./ precision_factor)[1]
                        else
                            logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc2 ./ precision_factor))
                        end
                
                        shocks² = sum(abs2,x[i])
                    end

                    aug_state = [stt; 1; x[i]]

                    stt2 = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

                    i = 2
                    
                    state¹⁻_vol2 = vcat(stt2, 1)
                        
                    shock_independent2 = X[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol2 + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol2, state¹⁻_vol2) / 2)
                
                    𝐒ⁱ2 = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol2)
                    
                    yy, matched = find_shocks(Val(filter_algorithm), 
                                            zeros(size(𝐒ⁱ, 2)),
                                            zero(kronxx[i]),
                                            zero(kron_buffer2),
                                            J,
                                            𝐒ⁱ2,
                                            𝐒ⁱ²ᵉ,
                                            shock_independent2)

                    jacc = 𝐒ⁱ2 + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)

                    # if i > presample_periods
                        if T.nExo == length(observables)
                            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
                        else
                            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
                        end

                        shocks² += sum(abs2,yy)
                    # end


                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                dtt)[1]'


stt = copy(state[T.past_not_future_and_mixed_idx])

shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt[:,1] = X[:,1]

i = 1
# for i in 1:2#axes(data_in_deviations,2)
state¹⁻_vol = vcat(stt, 1)

shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

x[i], matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        kronxx[i],
                        kron_buffer2,
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[i])

if i > presample_periods
    if T.nExo == length(observables)
        logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    else
        logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
    end

    shocks² += sum(abs2,x[i])
end

aug_state = [stt; 1; x[i]]

stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2

i = 2

state¹⁻_vol = vcat(stt, 1)
    
shock_independent = dtt[:,i] - (𝐒¹⁻ᵛ * state¹⁻_vol + 𝐒²⁻ᵛ * ℒ.kron(state¹⁻_vol, state¹⁻_vol) / 2)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

yy, matched = find_shocks(Val(filter_algorithm), 
                        zeros(size(𝐒ⁱ, 2)),
                        zero(kronxx[i]),
                        zero(kron_buffer2),
                        J,
                        𝐒ⁱ,
                        𝐒ⁱ²ᵉ,
                        shock_independent)

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), yy)



∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)

x[i], matched = find_shocks(Val(filter_algorithm), 
                                                zeros(size(𝐒ⁱ, 2)),
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent)

∂x = copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
# [i]

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

λ[i] = jacc' \ x[i] * 2
# [i]

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# [i]

∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = ∂shock_independent


∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state), 
                        length(aug_state))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state,e)
    ei += 1
end

∂state = ∂aug_state[1:length(∂state)]

∂x = ∂aug_state[length(stt)+2:end]


i = 1

stt = copy(state[T.past_not_future_and_mixed_idx])

state¹⁻_vol = vcat(stt, 1)

𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

∂x -= copy(x[i])

jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
# [i]

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc)'


# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end


λ[i] = jacc' \ x[i] * 2
# [i]

fXλp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ, 2), size(𝐒ⁱ, 2)) - 2 * ℒ.I(size(𝐒ⁱ, 2))  jacc'
        -jacc  zeros(size(𝐒ⁱ, 1), size(𝐒ⁱ, 1))]
# [i]

∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂data_in_deviations[:,i] = -∂shock_independent


isapprox(findiff[1:6], vec(∂data_in_deviations[:,1:2]), rtol = 1e-5)








stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱtmp = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ = [zero(𝐒ⁱtmp) for _ in 1:size(data_in_deviations,2)]

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱtmp + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ[1], 2), size(𝐒ⁱ[1], 2)) - 2 * ℒ.I(size(𝐒ⁱ[1], 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ[1], 1),size(𝐒ⁱ[1], 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]


for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    init_guess = zeros(size(𝐒ⁱ[i], 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ[i],
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)

    fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ[i], 2), size(𝐒ⁱ[i], 2)) - 2 * ℒ.I(size(𝐒ⁱ[i], 2))  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ[i], 1),size(𝐒ⁱ[i], 1))]

    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2
end




∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)
∂data_in_deviations = zero(data_in_deviations)

i = 2
# shocks² += sum(abs2,x[i])
∂x = copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]


i = 1
# shocks² += sum(abs2,x[i])
∂x -= copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = -∂shock_independent


∂data_in_deviations[:,1:3]



##############
# in a loop

∂𝐒ⁱ = zero(𝐒ⁱ[1])
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)
∂x = zero(x[1])
∂state = zeros(T.nPast_not_future_and_mixed)

for i in 3:-1:1#reverse(axes(data_in_deviations,2))

    # shocks² += sum(abs2,x[i])
    if i < 3
        ∂x -= copy(x[i])
    else
        ∂x += copy(x[i])
    end

    # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    ∂jacc = inv(jacc[i])'

    # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
    ∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

    re∂kronIx = reshape(∂kronIx, 
                            T.nExo, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIx; dims = (1,3))
        if i< 3
            ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    # find_shocks
    ∂xλ = vcat(∂x, zero(λ[i]))

    S = fXλp[i]' \ ∂xλ

    if i < 3
        S *= -1
    end

    ∂shock_independent = S[T.nExo+1:end]

    # shock_independent = copy(data_in_deviations[:,i])
    ∂data_in_deviations[:,i] = ∂shock_independent

    # aug_state[i] = [stt; 1; x[i]]
    if i >= 3-2
        ∂state *= 0
    end

    if i > 1
        # stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        ∂aug_state = 𝐒⁻¹' * ∂state
        ∂kronaug_state  = 𝐒⁻²' * ∂state / 2
        # ∂aug_state *= 0
        re∂kronaug_state = reshape(∂kronaug_state, 
                                length(aug_state[i]), 
                                length(aug_state[i]))
    
        ei = 1
        for e in eachslice(re∂kronaug_state; dims = (1))
            ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
            ei += 1
        end
    
        ∂state += ∂aug_state[1:length(∂state)]

        ei = 1
        for e in eachslice(re∂kronaug_state; dims = (2))
            ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
            ei += 1
        end
    
        # aug_state[i] = [stt; 1; x[i]]
        ∂x = ∂aug_state[length(stt)+2:end]

        # find_shocks
        ∂𝐒ⁱ = S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]'
        ∂𝐒ⁱ -= ∂jacc / 2

        # 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
        ∂state¹⁻_vol *= 0
        ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

        re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                                length(state¹⁻_vol), 
                                T.nExo, 
                                1,
                                T.nExo)

        ei = 1
        for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
            ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
            ei += 1
        end

        # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
        ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

        # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
        ∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

        re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                                length(state¹⁻_vol), 
                                length(state¹⁻_vol))

        ei = 1
        for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
            ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)]; 1],e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
            ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)]; 1],e)
            ei += 1
        end

        # state¹⁻_vol = vcat(state¹⁻, 1)
        ∂state += ∂state¹⁻_vol[1:end-1]
    end
end

∂data_in_deviations

∂data_in_deviations[:,1:3]

reshape(findiff,3,3)
# julia> ∂data_in_deviations
# 3×40 Matrix{Float64}:
#   378.78     330.723   0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0
#  -388.83     -24.1221  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0
#    73.7683  -124.506   0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0

### fin diff
reshape(findiff,3,3)
findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    # dtt = copy(data_in_deviations)
                    # dtt[:,1] = X[:,1]
                    dtt = X

                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                    for i in axes(dtt,2)
                        state¹⁻ = stt

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(data_in_deviations[:,i])
                        shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² += sum(abs2,xx)
                        end

                        aug_statee = [stt; 1; xx]

                        stt = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2
                    end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
data_in_deviations[:,1:3])[1]

reshape(findiff,3,3)



#### three periods
findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    # dtt = X

                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                    # for i in axes(dtt,2)
                    i = 1
                        state¹⁻ = stt

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(data_in_deviations[:,i])
                        shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² += sum(abs2,xx)
                        end

                        aug_statee = [stt; 1; xx]

                        stt = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2
                    # end

                    i = 2
                        state¹⁻ = stt

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(data_in_deviations[:,i])
                        # shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² += sum(abs2,xx)
                        end

                        aug_statee = [stt; 1; xx]

                        # return aug_statee

                        stt = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2

                    i = 3
                        state¹⁻ = stt

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(data_in_deviations[:,i])
                        # shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² += sum(abs2,xx)
                        end

                        aug_statee = [stt; 1; xx]

                        stt = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
copy(data_in_deviations[:,1:3]))[1]

reshape(findiff,3,3)



####################

∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)
∂data_in_deviations = zero(data_in_deviations)


i = 3
# shocks² += sum(abs2,x[i])
∂x = copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]


i = 2
# shocks² += sum(abs2,x[i])
∂x -= copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        T.nExo, 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
# ∂state¹⁻_vol *= 0
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]


i = 1
# shocks² += sum(abs2,x[i])
∂x -= copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i]')

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        T.nExo, 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent




∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
# ∂state¹⁻_vol *= 0
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end


# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]



∂data_in_deviations[:,1:3]

reshape(findiff,3,3)

########
# go back step by step


stt = copy(state[T.past_not_future_and_mixed_idx])
                
shocks² = 0.0
logabsdets = 0.0

dtt = copy(data_in_deviations)
# dtt = X

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

# for i in axes(dtt,2)
i = 1
    state¹⁻ = stt

    state¹⁻_vols = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])
    # shock_independent = copy(X[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

    𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

    init_guess = zeros(size(𝐒ⁱs, 2))

    xx, matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱs,
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
        end

        shocks² += sum(abs2,xx)
    end

    aug_statee = [stt; 1; xx]

    stt = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2
# end

i = 2
state¹⁻ = stt

state¹⁻_vols = vcat(state¹⁻, 1)

shock_independent = copy(data_in_deviations[:,i])
# shock_independent = copy(X[:,i])

ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)

ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

init_guess = zeros(size(𝐒ⁱs, 2))

xx, matched = find_shocks(Val(filter_algorithm), 
                        init_guess,
                        kronxx[i],
                        kron_buffer2,
                        J,
                        𝐒ⁱs,
                        𝐒ⁱ²ᵉ,
                        shock_independent,
                        # max_iter = 100
                        )

jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

if i > presample_periods
    # due to change of variables: jacobian determinant adjustment
    if T.nExo == length(observables)
        logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
    else
        logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
    end

    shocks² = sum(abs2,xx)
end

aug_statee = [stt; 1; xx]

stt2 = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2


findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                        i = 3
                        state¹⁻ = stt2

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(data_in_deviations[:,i])
                        shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets = ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² = sum(abs2,xx)
                        end

                        aug_statee = [stt2; 1; xx]

                        stt3 = 𝐒⁻¹ * aug_statee + 𝐒⁻² * ℒ.kron(aug_statee, aug_statee) / 2

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
data_in_deviations[:,1:3])[1]

reshape(findiff,3,3)




∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)
∂data_in_deviations = zero(data_in_deviations)


i = 3
# shocks² += sum(abs2,x[i])
∂x = copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp[i]' \ ∂xλ

∂shock_independent = S[length(init_guess)+1:end]

∂𝐒ⁱ = S[1:length(init_guess)] * λ[i]' - S[length(init_guess)+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot(state¹⁻_vol,e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state = ∂state¹⁻_vol[1:end-1]


# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i-1],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]





######################
# go back step by step





stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱtmp = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ = [zero(𝐒ⁱtmp) for _ in 1:size(data_in_deviations,2)]

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱtmp + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ[1], 2), size(𝐒ⁱ[1], 2)) - 2 * ℒ.I(size(𝐒ⁱ[1], 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ[1], 1),size(𝐒ⁱ[1], 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]


for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    init_guess = zeros(size(𝐒ⁱ[i], 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ[i],
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)

    fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ[i], 2), size(𝐒ⁱ[i], 2)) - 2 * ℒ.I(size(𝐒ⁱ[i], 2))  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ[i], 1),size(𝐒ⁱ[i], 1))]

    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2
end



findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                X -> begin
                        stt = state[T.past_not_future_and_mixed_idx]

                    i = 1

                        state¹⁻_vol = vcat(stt, 1)

                        shock_independent = copy(X[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                        init_guess = zeros(size(𝐒ⁱ[i], 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ[i],
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets = ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² = sum(abs2,x[i])
                        end

                        aug_states = [stt; 1; x[i]]

                        state¹⁻s = 𝐒⁻¹ * aug_states + 𝐒⁻² * ℒ.kron(aug_states, aug_states) / 2

                    i = 2

                        state¹⁻_vol = vcat(state¹⁻s, 1)
                        
                        shock_independent = copy(data_in_deviations[:,i])
                    
                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
                    
                        𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
                    
                        init_guess = zeros(size(𝐒ⁱ[i], 2))
                    
                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ[i],
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )
                    
                        jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])
                    
                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end
                    
                            shocks² += sum(abs2,x[i])
                        end
                    
                        aug_state[i] = [state¹⁻s; 1; x[i]]
                    
                        stt2 = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2

                    i = 3
                        
                        state¹⁻ = stt2

                        state¹⁻_vol = vcat(state¹⁻, 1)
                        
                        shock_independent = copy(data_in_deviations[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

                        𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

                        init_guess = zeros(size(𝐒ⁱ[i], 2))

                        x[i], matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                kronxx[i],
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱ[i],
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
                            end

                            shocks² += sum(abs2,x[i])
                        end

                        # aug_state[i] = [stt; 1; x[i]]

                        # stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                    # shocks²/2
                end, 
                copy(data_in_deviations))[1]'

reshape(findiff,3,3)

∂aug_state

𝐒⁻¹' * [
177.37832468258554
-87.23046293002642
 -4.452094885937977
-12.826327254466602
]


stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱtmp = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ = [zero(𝐒ⁱtmp) for _ in 1:size(data_in_deviations,2)]

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱtmp + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ[1], 2), size(𝐒ⁱ[1], 2)) - 2 * ℒ.I(size(𝐒ⁱ[1], 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ[1], 1),size(𝐒ⁱ[1], 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]


for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    init_guess = zeros(size(𝐒ⁱ[i], 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ[i],
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)

    fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ[i], 2), size(𝐒ⁱ[i], 2)) - 2 * ℒ.I(size(𝐒ⁱ[i], 2))  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ[i], 1),size(𝐒ⁱ[i], 1))]

    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2
end










∂𝐒ⁱ = zero(𝐒ⁱ)
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)
∂state¹⁻_vol = zero(state¹⁻_vol)
∂data_in_deviations = zero(data_in_deviations)
∂state = zeros(T.nPast_not_future_and_mixed)

i = 3
# shocks² += sum(abs2,x[i])
∂x = copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        length(x[i]), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = fXλp[i]' \ ∂xλ

∂shock_independent = S[T.nExo+1:end]

∂𝐒ⁱ = S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]'
∂𝐒ⁱ -= ∂jacc / 2

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state += ∂state¹⁻_vol[1:end-1]
println(∂state)

i = 2

# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2

re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
    ei += 1
end

# aug_state[i] = [stt; 1; x[i]]
∂state = ∂aug_state[1:length(∂state)]

println(∂state)
# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]


# shocks² += sum(abs2,x[i])
∂x -= copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        T.nExo, 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e) # fine
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp[i]' \ ∂xλ

∂shock_independent = S[T.nExo+1:end] # fine

∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]') # fine
∂𝐒ⁱ -= ∂jacc / 2 # fine

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂state¹⁻_vol *= 0
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent


# ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

# ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        length(state¹⁻_vol))

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
    ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e) # fine
    ei += 1
end

# state¹⁻_vol = vcat(state¹⁻, 1)
∂state += ∂state¹⁻_vol[1:end-1]

println(∂state)
i = 1
# this transition doesnt work; as in ∂state is correct but the next ∂aug_state isnt
# stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
∂aug_state = 𝐒⁻¹' * ∂state
∂kronaug_state  = 𝐒⁻²' * ∂state / 2
# ∂aug_state *= 0
re∂kronaug_state = reshape(∂kronaug_state, 
                        length(aug_state[i]), 
                        length(aug_state[i]))

ei = 1
for e in eachslice(re∂kronaug_state; dims = (1))
    ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
    ei += 1
end

ei = 1
for e in eachslice(re∂kronaug_state; dims = (2))
    ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
    ei += 1
end


# aug_state[i] = [stt; 1; x[i]]
∂state += ∂aug_state[1:length(∂state)]

# aug_state[i] = [stt; 1; x[i]]
∂x = ∂aug_state[length(stt)+2:end]

# shocks² += sum(abs2,x[i])
∂x -= copy(x[i])

# logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
∂jacc = inv(jacc[i])'

# jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

re∂kronIx = reshape(∂kronIx, 
                        T.nExo, 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIx; dims = (1,3))
    ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e) # fine
    ei += 1
end

# find_shocks
∂xλ = vcat(∂x, zero(λ[i]))

S = -fXλp[i]' \ ∂xλ

∂shock_independent = S[T.nExo+1:end] # fine

∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]') # fine
∂𝐒ⁱ -= ∂jacc / 2 # fine

# 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
∂state¹⁻_vol *= 0
∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                        length(state¹⁻_vol), 
                        T.nExo, 
                        1,
                        T.nExo)

ei = 1
for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
    ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
    ei += 1
end

# shock_independent = copy(data_in_deviations[:,i])
∂data_in_deviations[:,i] = ∂shock_independent

∂data_in_deviations[:,1:3]
findiff[1:9]





#### try loop again

stt = state[T.past_not_future_and_mixed_idx]

kronxx = [zeros(T.nExo^2) for _ in 1:size(data_in_deviations,2)]

J = ℒ.I(T.nExo)

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))

x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

state¹⁻ = stt

state¹⁻_vol = vcat(state¹⁻, 1)

𝐒ⁱtmp = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

𝐒ⁱ = [zero(𝐒ⁱtmp) for _ in 1:size(data_in_deviations,2)]

𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

aug_state = [zeros(size(𝐒⁻¹,2)) for _ in 1:size(data_in_deviations,2)]

tmp = 𝐒ⁱtmp + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[1])), x[1])

jacc = [zero(tmp) for _ in 1:size(data_in_deviations,2)]

λ = [zeros(size(tmp, 1)) for _ in 1:size(data_in_deviations,2)]

λ[1] = tmp' \ x[1] * 2

fXλp_tmp = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[1], size(𝐒ⁱ[1], 2), size(𝐒ⁱ[1], 2)) - 2 * ℒ.I(size(𝐒ⁱ[1], 2))  tmp'
            -tmp  zeros(size(𝐒ⁱ[1], 1),size(𝐒ⁱ[1], 1))]

fXλp = [zero(fXλp_tmp) for _ in 1:size(data_in_deviations,2)]

kronxλ_tmp = ℒ.kron(x[1], λ[1])

kronxλ = [kronxλ_tmp for _ in 1:size(data_in_deviations,2)]


for i in axes(data_in_deviations,2)
    state¹⁻ = stt

    state¹⁻_vol = vcat(state¹⁻, 1)
    
    shock_independent = copy(data_in_deviations[:,i])

    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)

    𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)

    init_guess = zeros(size(𝐒ⁱ[i], 2))

    x[i], matched = find_shocks(Val(filter_algorithm), 
                            init_guess,
                            kronxx[i],
                            kron_buffer2,
                            J,
                            𝐒ⁱ[i],
                            𝐒ⁱ²ᵉ,
                            shock_independent,
                            # max_iter = 100
                            )

    jacc[i] =  𝐒ⁱ[i] + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(length(x[i])), x[i])

    λ[i] = jacc[i]' \ x[i] * 2
    # ℒ.ldiv!(λ[i], tmp', x[i])
    # ℒ.rmul!(λ[i], 2)

    fXλp[i] = [reshape(2 * 𝐒ⁱ²ᵉ' * λ[i], size(𝐒ⁱ[i], 2), size(𝐒ⁱ[i], 2)) - 2 * ℒ.I(size(𝐒ⁱ[i], 2))  jacc[i]'
                -jacc[i]  zeros(size(𝐒ⁱ[i], 1),size(𝐒ⁱ[i], 1))]

    ℒ.kron!(kronxx[i], x[i], x[i])

    ℒ.kron!(kronxλ[i], x[i], λ[i])

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[i] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[i] ./ precision_factor))
        end

        shocks² += sum(abs2,x[i])
    end

    aug_state[i] = [stt; 1; x[i]]

    stt = 𝐒⁻¹ * aug_state[i] + 𝐒⁻² * ℒ.kron(aug_state[i], aug_state[i]) / 2
end






n_end = size(data_in_deviations,2)

∂𝐒¹ᵉ = zero(𝐒¹ᵉ)
∂𝐒²⁻ᵉ = zero(𝐒²⁻ᵉ)

∂𝐒¹⁻ᵛ = zero(𝐒¹⁻ᵛ)
∂𝐒²⁻ᵛ = zero(𝐒²⁻ᵛ)

∂𝐒⁻¹ = zero(𝐒⁻¹)
∂𝐒⁻² = zero(𝐒⁻²)

∂𝐒ⁱ = zero(𝐒ⁱ[1])
∂𝐒ⁱ²ᵉ = zero(𝐒ⁱ²ᵉ)

∂state¹⁻_vol = zero(state¹⁻_vol)
∂x = zero(x[1])
∂state = zeros(T.nPast_not_future_and_mixed)

for i in reverse(axes(data_in_deviations,2))
    # stt = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
    ∂𝐒⁻¹ += ∂state * aug_state[i]'
    
    ∂𝐒⁻² += ∂state * ℒ.kron(aug_state[i], aug_state[i])' / 2

    ∂aug_state = 𝐒⁻¹' * ∂state
    ∂kronaug_state  = 𝐒⁻²' * ∂state / 2

    re∂kronaug_state = reshape(∂kronaug_state, 
                            length(aug_state[i]), 
                            length(aug_state[i]))

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (1))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronaug_state; dims = (2))
        ∂aug_state[ei] += ℒ.dot(aug_state[i],e)
        ei += 1
    end

    if i > 1 && i < n_end
        ∂state *= 0
    end
    # aug_state[i] = [stt; 1; x[i]]
    ∂state += ∂aug_state[1:length(∂state)]

    # aug_state[i] = [stt; 1; x[i]]
    ∂x = ∂aug_state[length(stt)+2:end]

    # shocks² += sum(abs2,x[i])
    if i < n_end
        ∂x -= copy(x[i])
    else
        ∂x += copy(x[i])
    end

    # logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
    ∂jacc = inv(jacc[i])'

    # jacc = 𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), x[1])
    ∂kronIx = 𝐒ⁱ²ᵉ' * ∂jacc

    re∂kronIx = reshape(∂kronIx, 
                            T.nExo, 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIx; dims = (1,3))
        if i < n_end
            ∂x[ei] -= ℒ.dot(ℒ.I(T.nExo),e)
        else
            ∂x[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        end
        ei += 1
    end

    ∂𝐒ⁱ²ᵉ -= ∂jacc * ℒ.kron(ℒ.I(T.nExo), x[i])'

    # find_shocks
    ∂xλ = vcat(∂x, zero(λ[i]))

    S = fXλp[i]' \ ∂xλ

    if i < n_end
        S *= -1
    end

    ∂shock_independent = S[T.nExo+1:end] # fine

    ∂𝐒ⁱ = (S[1:T.nExo] * λ[i]' - S[T.nExo+1:end] * x[i]') # fine
    ∂𝐒ⁱ -= ∂jacc / 2 # fine

    ∂𝐒ⁱ²ᵉ += 2 * S[1:T.nExo] *  kronxλ[i]' - S[T.nExo+1:end] * kronxx[i]'

    # 𝐒ⁱ[i] = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)
    ∂state¹⁻_vol *= 0
    ∂kronIstate¹⁻_vol = 𝐒²⁻ᵉ' * ∂𝐒ⁱ

    re∂kronIstate¹⁻_vol = reshape(∂kronIstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            T.nExo, 
                            1,
                            T.nExo)

    ei = 1
    for e in eachslice(re∂kronIstate¹⁻_vol; dims = (1,3))
        ∂state¹⁻_vol[ei] += ℒ.dot(ℒ.I(T.nExo),e)
        ei += 1
    end

    ∂𝐒¹ᵉ += ∂𝐒ⁱ

    ∂𝐒²⁻ᵉ += ∂𝐒ⁱ * ℒ.kron(ℒ.I(T.nExo), [aug_state[i][1:length(stt)];1])'

    # shock_independent = copy(data_in_deviations[:,i])
    ∂data_in_deviations[:,i] = ∂shock_independent


    # ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    ∂𝐒¹⁻ᵛ -= ∂shock_independent * [aug_state[i][1:length(stt)];1]'

    ∂state¹⁻_vol -= 𝐒¹⁻ᵛ' * ∂shock_independent

    # ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    ∂𝐒²⁻ᵛ -= ∂shock_independent * ℒ.kron([aug_state[i][1:length(stt)];1], [aug_state[i][1:length(stt)];1])' / 2

    ∂kronstate¹⁻_vol = -𝐒²⁻ᵛ' * ∂shock_independent / 2

    re∂kronstate¹⁻_vol = reshape(∂kronstate¹⁻_vol, 
                            length(state¹⁻_vol), 
                            length(state¹⁻_vol))

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (1))
        ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e)
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronstate¹⁻_vol; dims = (2))
        ∂state¹⁻_vol[ei] += ℒ.dot([aug_state[i][1:length(stt)];1],e) # fine
        ei += 1
    end

    # state¹⁻_vol = vcat(state¹⁻, 1)
    ∂state += ∂state¹⁻_vol[1:end-1]
end

∂𝐒²ᵉ = ∂𝐒ⁱ²ᵉ / 2



∂𝐒¹ᵉ

∂𝐒¹⁻ᵛ

∂data_in_deviations[:,1:n_end]


#### findiff loop

findiff = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1, max_range = 1e-6), 
                X -> begin
                    stt = copy(state[T.past_not_future_and_mixed_idx])
                
                    shocks² = 0.0
                    logabsdets = 0.0
                    
                    dtt = copy(data_in_deviations)
                    # dtt = copy(data_in_deviations[:,[1]])
                    # dtt[:,1] = X[:,1]
                    # dtt = X

                    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 

                    for i in axes(dtt,2)
                        state¹⁻ = stt

                        state¹⁻_vols = vcat(state¹⁻, 1)
                        
                        # shock_independent = copy(data_in_deviations[:,i])
                        shock_independent = copy(dtt[:,i])

                        ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vols, -1, 1)
                        
                        ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vols, state¹⁻_vols), -1/2, 1)

                        𝐒ⁱs = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vols)

                        init_guess = zeros(size(𝐒ⁱs, 2))

                        xx, matched = find_shocks(Val(filter_algorithm), 
                                                init_guess,
                                                copy(kronxx[i]),
                                                kron_buffer2,
                                                J,
                                                𝐒ⁱs,
                                                𝐒ⁱ²ᵉ,
                                                shock_independent,
                                                # max_iter = 100
                                                )

                        jaccc = 𝐒ⁱs + 2 * 𝐒ⁱ²ᵉ * ℒ.kron(ℒ.I(T.nExo), xx)

                        if i > presample_periods
                            # due to change of variables: jacobian determinant adjustment
                            if T.nExo == length(observables)
                                logabsdets += ℒ.logabsdet(jaccc ./ precision_factor)[1]
                            else
                                logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jaccc ./ precision_factor))
                            end

                            shocks² += sum(abs2,xx)
                        end

                        aug_statee = [stt; 1; xx]

                        stt = 𝐒⁻¹ * aug_statee + X * ℒ.kron(aug_statee, aug_statee) / 2
                    end

                    -(logabsdets + shocks² + (length(observables) * (0 + n_obs - 0)) * log(2 * 3.141592653589793)) / 2
                end, 
                𝐒⁻²)[1]'

                ∂𝐒⁻¹
                reshape(findiff,4,8)
                isapprox(∂𝐒⁻¹, reshape(findiff,4,8))

                ∂𝐒⁻²
                reshape(findiff,4,64)
                isapprox(∂𝐒⁻², reshape(findiff,4,64))

                ∂𝐒²⁻ᵛ
                reshape(findiff,3,25)
                isapprox(∂𝐒²⁻ᵛ, reshape(findiff,3,25))

                ∂𝐒¹⁻ᵛ
                reshape(findiff,3,5)
                isapprox(∂𝐒¹⁻ᵛ, reshape(findiff,3,5))


                ∂𝐒¹ᵉ
                reshape(findiff,3,3)
                isapprox(∂𝐒¹ᵉ, reshape(findiff,3,3))

                ∂𝐒²⁻ᵉ
                reshape(findiff,3,15)
                isapprox(∂𝐒²⁻ᵉ, reshape(findiff,3,15))


                ∂𝐒ⁱ²ᵉ
                reshape(findiff,3,9)
                isapprox(∂𝐒ⁱ²ᵉ, reshape(findiff,3,9))

                
∂data_in_deviations[:,1:n_end]
reshape(findiff,3,n_end)

reshape(findiff,3,9)
∂𝐒²⁻ᵉ

isapprox(∂data_in_deviations[:,1:n_end], reshape(findiff,3,n_end))





fin_debug = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[10], 𝓂.parameter_values)[1]
zyg_debug = Zygote.jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[10], 𝓂.parameter_values)[1]
isapprox(zyg_debug, fin_debug)

ℒ.norm(zyg_debug - fin_debug) / max(ℒ.norm(fin_debug), ℒ.norm(zyg_debug))

import DifferentiationInterface as 𝒟
backend = 𝒟.AutoZygote()

xxx = 𝒟.value_and_jacobian(x -> calculate_third_order_stochastic_steady_state(x, 𝓂, pruning = true)[10], backend, 𝓂.parameter_values)
xxx[2]
isapprox(s3,xxx[1])
# second order
for1 = ForwardDiff.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)
zyg1 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)[1]
fin1 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[1], 𝓂.parameter_values)[1]
isapprox(zyg1,fin1)
zyg1-fin1

zyg2 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[3], 𝓂.parameter_values)[1]
fin2 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[3], 𝓂.parameter_values)[1]
isapprox(zyg2,fin2)

zyg3 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[5], 𝓂.parameter_values)[1]
fin3 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[5], 𝓂.parameter_values)[1]
isapprox(zyg3,fin3)

zyg4 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[6], 𝓂.parameter_values)[1]
fin4 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[6], 𝓂.parameter_values)[1]
isapprox(zyg4,fin4)

zyg5 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[7], 𝓂.parameter_values)[1]
fin5 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[7], 𝓂.parameter_values)[1]
isapprox(zyg5,fin5)

zyg6 = Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[8], 𝓂.parameter_values)[1]
fin6 = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[8], 𝓂.parameter_values)[1]
isapprox(zyg6,fin6)


Zygote.jacobian(x -> calculate_second_order_stochastic_steady_state(x, 𝓂, pruning = true)[3], 𝓂.parameter_values)[1]

sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(parameter_values, 𝓂, pruning = true)







TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), 𝓂.parameter_values, 𝓂, tol);


# hessian derivatives
X = [𝓂.parameter_values; SS_and_pars]

vals = zeros(Float64, length(𝓂.model_hessian_SS_and_pars_vars[1]))

for f in 𝓂.model_hessian_SS_and_pars_vars[1]
    out = f(X)
    
    @inbounds vals[out[2]] = out[1]
end

Accessors.@reset 𝓂.model_hessian_SS_and_pars_vars[2].nzval = vals;
        
analytical_hess_SS_and_pars_vars = 𝓂.model_hessian_SS_and_pars_vars[2] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC



par_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_hessian(x, SS_and_pars, 𝓂), parameters)[1]
    
SS_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_hessian(parameters, x, 𝓂), SS_and_pars)[1]
                            
findiff = hcat(par_hess,SS_hess)' |>sparse 


isapprox(analytical_hess_SS_and_pars_vars,findiff)


# third order
vals = zeros(Float64, length(𝓂.model_third_order_derivatives_SS_and_pars_vars[1]))

Polyester.@batch minbatch = 200 for f in 𝓂.model_third_order_derivatives_SS_and_pars_vars[1]
    out = f(X)
    
    @inbounds vals[out[2]] = out[1]
end

Accessors.@reset 𝓂.model_third_order_derivatives_SS_and_pars_vars[2].nzval = vals

analytical_third_order_derivatives_SS_and_pars_vars = 𝓂.model_third_order_derivatives_SS_and_pars_vars[2] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC


par_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_third_order_derivatives(x, SS_and_pars, 𝓂), parameters)[1]
    
SS_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_third_order_derivatives(parameters, x, 𝓂), SS_and_pars)[1]
                            
findiff = hcat(par_hess,SS_hess)' |>sparse 


isapprox(analytical_third_order_derivatives_SS_and_pars_vars,findiff)




# second order solution

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix
    
𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂) * 𝓂.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂

𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, sylvester_algorithm = sylvester_algorithm, tol = tol, verbose = verbose)

# droptol!(𝐒₂,1e-6)


𝐒₁zyg = Zygote.jacobian(x -> calculate_second_order_solution(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1][5], 𝐒₁)[1]




∇₂fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_second_order_solution(∇₁, sparse(x), 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1][5], collect(∇₂))[1]

∇₂zyg = Zygote.jacobian(x -> calculate_second_order_solution(∇₁, x, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1][5], ∇₂)[1]

isapprox(∇₂fin, ∇₂zyg)



∇₁fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_second_order_solution(x, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1], ∇₁)[1]

∇₁zyg = Zygote.jacobian(x -> calculate_second_order_solution(x, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1], ∇₁)[1]

isapprox(∇₁zyg, ∇₁fin)





𝐒₁fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_second_order_solution(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1], 𝐒₁)[1]

𝐒₁zyg = Zygote.jacobian(x -> calculate_second_order_solution(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1], 𝐒₁)[1]

isapprox(𝐒₁fin, 𝐒₁zyg)

Zygote.jacobian(x->kron(x,x),collect(aa))[1]




# go manual

𝐒₁zyg = Zygote.jacobian(x -> calculate_second_order_solution(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)[1], 𝐒₁)[1]



function calculate_second_order_solution_short(∇₁::AbstractMatrix{<: Real}, #first order derivatives
    ∇₂::SparseMatrixCSC{<: Real}, #second order derivatives
    𝑺₁::AbstractMatrix{<: Real},#first order solution
    M₂::second_order_auxilliary_matrices;  # aux matrices
    T::timings,
    sylvester_algorithm::Symbol = :doubling,
    tol::AbstractFloat = eps(),
    verbose::Bool = false)

    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = - ∇₂ * ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔 * M₂.𝐂₂ 
end


𝐒₁fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> MacroModelling.calculate_second_order_solution_short(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings), 𝐒₁)[1]|>sparse

𝐒₁zyg = Zygote.jacobian(x -> MacroModelling.calculate_second_order_solution_short(∇₁, ∇₂, x, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings), 𝐒₁)[1]|>sparse

isapprox(𝐒₁fin,𝐒₁zyg)
MacroModelling.calculate_second_order_solution_short(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)




# fix kron derivative
aa = randn(2,3)

kron(aa,aa)

zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]

@profview zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]

∂kronaa = zero(kron(aa,aa))
# ∂kronaa[1,1] = 1
∂kronaa[1,2] = 1

reshape(∂kronaa,6,6) * vec(aa)

reshape(∂kronaa,18,2) * (aa')

reshape(∂kronaa,6,6) .* vec(aa) + vec(aa) * vec(∂kronaa)'

reshape(∂kronaa,3,3,2,2)

mapslices(x -> ℒ.dot(aa,x), reshape(∂kronaa,3,3,2,2); dims = (1, 3))[1,:,1,:]

2 * mapslices(x -> ℒ.dot(aa,x), reshape(∂kronaa,3,3,2,2); dims = (2, 4))[:,1,:,1]

mapslices(x -> ℒ.dot(aa,x), reshape(∂kronaa,2,2,3,3); dims = (1, 3))[1,:,1,:]


zygkrondiff[:,1]



aa = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8], [0.6909919605821032, -0.27259360909626684, -2.53628717368683e-15, -26.45556000884485, -17.05911351808633, -0.25461959596982364, 0.13392857142856754, 7.04619863731337e-16, 10.195837547363936, 5.228985756735007, 0.24628104913305907, 0.08807182556648097, 0.5000000000000004, 5.791255979395697, 2.410844892168823, -0.49256209826611513, -0.17614365113296282, -3.3768393832643107e-16, -10.168772175270396, -3.2312325278764695, 0.007677688450912701, -0.0030288178788474914, -0.2939506667649496, -0.18954570575651816, -0.0024628104913305524, -0.0008807182556648164, -0.050843860876352225, -0.016156162639382483, -0.02462810491330557, -0.008807182556648158, -0.05, -0.5791255979395746, -0.24108448921688472], 35, 8) |> collect

kron(aa,aa)

zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]

∂kronaa = zero(kron(aa,aa))
# ∂kronaa = zero(mat1_rsh).*(zero(mat2_rsh).+1);
# size(∂kronaa)

∂kronaa[4] = 1
re∂kronaa = reshape(∂kronaa,size(aa,1),size(aa,2),size(aa,1) ,size(aa,2));

using Combinatorics
# perms = (2,3,4,1)
# perms = (2,3,1,4)
        # perms = (1,3,4,2)
        # perms = (1,3,2,4)
        # perms = (3,1,4,2)

        # perms = [(2,3,4,1), (3,4,1,2), (4,1,2,3), (1,2,3,4)]

        for perm in permutations(1:4)
            perm∂kronaa = permutedims(re∂kronaa, perm)
            result = (vec(aa)' * reshape(perm∂kronaa, size(aa,1)*size(aa,2), size(aa,1)*size(aa,2)))[4]
            if result == aa[1] println("Permutation $perm: $result") end
        end
# Permutation [2, 3, 1, 4]: 0.6909919605821032
# Permutation [3, 2, 1, 4]: 0.6909919605821032
# Permutation [3, 4, 1, 2]: 0.6909919605821032
# Permutation [4, 3, 1, 2]: 0.6909919605821032

perms = (2,3,1,4)
perm∂kronaa = permutedims(re∂kronaa, perms); 

(vec(aa)' * reshape(perm∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)))

zygkrondiff


for i in 1:size(zygkrondiff,1)
    nn = i
    ∂kronaa = zero(kron(aa,aa))

    ∂kronaa[nn] = 1

    re∂kronaa = reshape(∂kronaa,size(aa,1),size(aa,2),size(aa,1) ,size(aa,2));

    perm = [2, 3, 1, 4]
    perm = [3, 2, 1, 4]
    perm = [4, 3, 1, 2]
    perm = [3, 2, 1, 4]
    perm∂kronaa = permutedims(re∂kronaa, perm); 

    vec(aa)' * reshape(perm∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2))

    vec(aa)' * (reshape(∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)) + reshape(perm∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)))

    holds = isapprox((vec(aa)' * (reshape(∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)) + reshape(perm∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2))))[1,:], zygkrondiff[nn,:])

    if !holds 
        println(i) 
        break
    end
end

zygkrondiff[36,:]



nn = 360
perm = [2, 1, 4, 3]
perm2 = [4, 3, 2, 1]

∂kronaa = zero(kron(aa,aa))
∂kronaa[nn] = 1
re∂kronaa = reshape(∂kronaa,size(aa,1),size(aa,2),size(aa,1) ,size(aa,2));

perm∂kronaa = permutedims(re∂kronaa, perm);
perm∂kronaa2 = permutedims(re∂kronaa, perm2);
result = (vec(aa)' * (reshape(perm∂kronaa2, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)) + reshape( perm∂kronaa, size(aa,1)*size(aa,2), size(aa,1)*size(aa,2))))

result[1,:] == zygkrondiff[nn,:]


nn = 36


aa = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8], [0.6909919605821032, -0.27259360909626684, -2.53628717368683e-15, -26.45556000884485, -17.05911351808633, -0.25461959596982364, 0.13392857142856754, 7.04619863731337e-16, 10.195837547363936, 5.228985756735007, 0.24628104913305907, 0.08807182556648097, 0.5000000000000004, 5.791255979395697, 2.410844892168823, -0.49256209826611513, -0.17614365113296282, -3.3768393832643107e-16, -10.168772175270396, -3.2312325278764695, 0.007677688450912701, -0.0030288178788474914, -0.2939506667649496, -0.18954570575651816, -0.0024628104913305524, -0.0008807182556648164, -0.050843860876352225, -0.016156162639382483, -0.02462810491330557, -0.008807182556648158, -0.05, -0.5791255979395746, -0.24108448921688472], 35, 8) |> collect

aa = randn(3,3)

kron(aa,aa)

zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]





candidate_perms = Set{Tuple{Vector{Int64}, Vector{Int64}}}()
# length(perms)
for i in 1:1#size(zygkrondiff,1)
    ∂kronaa = zeros(size(aa,1) * size(aa,2), size(aa,1) * size(aa,2))
    # ∂kronaa = zero(mat1_rsh).*(zero(mat2_rsh).+1);
    # size(∂kronaa)
    ∂kronaa[i] = 1
    re∂kronaa = reshape(∂kronaa,size(aa,1),size(aa,2),size(aa,1) ,size(aa,2));

    perms = Set{Tuple{Vector{Int64}, Vector{Int64}}}()

    for perm in permutations(1:4)
        for perm2 in permutations(1:4)
            perm∂kronaa = permutedims(re∂kronaa, perm)
            perm∂kronaa2 = permutedims(re∂kronaa, perm2)
            result = reshape(perm∂kronaa2, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)) * vec(aa) + reshape( perm∂kronaa, size(aa,1)*size(aa,2), size(aa,1)*size(aa,2)) * vec(aa)
            # println(result)
            if result == zygkrondiff[nn,:] 
                push!(perms, (perm, perm2))
                # println("$i Permutation $perm, $perm2") 
            end
        end
    end
    if length(candidate_perms) == 0 
        for p in perms
            push!(candidate_perms, p)
        end
    else
        println(length(perms))
        intersect!(candidate_perms, perms)
    end
end

vec(aa)' * reshape(∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2))

using Combinatorics
perms = collect(permutations(1:4))
vec(aa)' * reshape(perm∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2))




aa = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8], [0.6909919605821032, -0.27259360909626684, -2.53628717368683e-15, -26.45556000884485, -17.05911351808633, -0.25461959596982364, 0.13392857142856754, 7.04619863731337e-16, 10.195837547363936, 5.228985756735007, 0.24628104913305907, 0.08807182556648097, 0.5000000000000004, 5.791255979395697, 2.410844892168823, -0.49256209826611513, -0.17614365113296282, -3.3768393832643107e-16, -10.168772175270396, -3.2312325278764695, 0.007677688450912701, -0.0030288178788474914, -0.2939506667649496, -0.18954570575651816, -0.0024628104913305524, -0.0008807182556648164, -0.050843860876352225, -0.016156162639382483, -0.02462810491330557, -0.008807182556648158, -0.05, -0.5791255979395746, -0.24108448921688472], 35, 8) |> collect

aa = randn(3,3)

kron(aa,aa)

zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]


using LinearAlgebra

for i in 1:size(zygkrondiff,1)
    ∂kronaa = zeros(size(aa,1) * size(aa,2), size(aa,1) * size(aa,2))
    
    ∂kronaa[i] = 1
    re∂kronaa = reshape(∂kronaa,size(aa,1),size(aa,1),size(aa,2) ,size(aa,2));
    result = zero(aa)
    
    ei = 1
    for e in eachslice(re∂kronaa; dims = (2, 4))
        # result[ei] += dot(aa,e)
        result += dot.(aa,e)
        # push!(daa, dot(aa,e))
        ei += 1
    end

    ei = 1
    for e in eachslice(re∂kronaa; dims = (1, 3))
        result[ei] += dot(aa,e)
        # push!(dab, dot(aa,e))
        ei += 1
    end
    # println(daa == dab)

    # result = daa + dab
    if !(vec(result) == zygkrondiff[i,:])
        println("$i failed")
        break
    else
        println("$i passed")
    end
    # println(result == zygkrondiff[i,:])
end


x̄ = @thunk(project_x(_dot_collect.(Ref(y), eachslice(dz; dims = (2, 4)))))
ȳ = @thunk(project_y(_dot_collect.(Ref(x), eachslice(dz; dims = (1, 3)))))





function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end




function rrule(::typeof(_kron), mat1::AbstractMatrix,mat2::AbstractMatrix)

    function _kron_pullback(∂kron) 
        ∂mat1 = zero(mat1)
        ∂mat2 = zero(mat2)

        re∂kron = reshape(∂kron,size(mat1,1),size(mat2,1),size(mat1,2) ,size(mat2,2));

        ei = 1
        for e in eachslice(re∂kron; dims = (2, 4))
            ∂mat1[ei] += dot(mat1,e)
            ei += 1
        end

        ei = 1
        for e in eachslice(re∂kron; dims = (1, 3))
            ∂mat2[ei] += dot(mat2,e)
            ei += 1
        end

        return NoTangent(), ∂mat1, ∂mat2
    end
    return kron(mat1,mat2), _kron_pullback
end


aa = sparse([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 2, 4, 5, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8], [0.6909919605821032, -0.27259360909626684, -2.53628717368683e-15, -26.45556000884485, -17.05911351808633, -0.25461959596982364, 0.13392857142856754, 7.04619863731337e-16, 10.195837547363936, 5.228985756735007, 0.24628104913305907, 0.08807182556648097, 0.5000000000000004, 5.791255979395697, 2.410844892168823, -0.49256209826611513, -0.17614365113296282, -3.3768393832643107e-16, -10.168772175270396, -3.2312325278764695, 0.007677688450912701, -0.0030288178788474914, -0.2939506667649496, -0.18954570575651816, -0.0024628104913305524, -0.0008807182556648164, -0.050843860876352225, -0.016156162639382483, -0.02462810491330557, -0.008807182556648158, -0.05, -0.5791255979395746, -0.24108448921688472], 35, 8) |> collect

aa = randn(3,3)


MacroModelling._kron(aa,aa)
zygkrondifforig = Zygote.jacobian(x->kron(x,x),aa)[1]

zygkrondiff = Zygote.jacobian(x->MacroModelling._kron(x,x),aa)[1]

zygkrondiff1 = Zygote.jacobian(x->MacroModelling._kron(aa,x),aa)[1]

zygkrondiff2 = Zygote.jacobian(x->MacroModelling._kron(x,aa),aa)[1]

zygkrondiff - zygkrondiff1 - zygkrondiff2


isapprox(zygkrondifforig, zygkrondiff)

perm∂kronaa = permutedims(re∂kronaa, perms[1])
perm∂kronaa2 = permutedims(re∂kronaa, perms[4])
result = (vec(aa)' * (reshape(perm∂kronaa2, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2)) + reshape( perm∂kronaa, size(aa,1)*size(aa,2), size(aa,1)*size(aa,2))))
zygkrondiff[nn,:] 

# aa = randn(3,2)


function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end


_kron(aa,aa)


m1, n1 = size(aa)
mat1_rsh = reshape(aa,(1,m1,1,n1))

m2, n2 = size(aa)
mat2_rsh = reshape(aa,(m2,1,n2,1))

kronaa = reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))

∂kronaa = zero(kron(aa,aa))
# ∂kronaa = zero(mat1_rsh).*(zero(mat2_rsh).+1);
# size(∂kronaa)
∂kronaa[1] = 1

vec(aa)' * reshape(∂kronaa, size(aa,1)*size(aa,2),size(aa,1) *size(aa,2))


reshape(reshape(∂kronaa .* mat2_rsh, m1*m2,n1*n2),6,6)


mat1_rsh0 = zero(mat1_rsh)
mat1_rsh0[2] = 1
∂kronaa += mat1_rsh0.*(zero(mat2_rsh).+1)
∂kronaa[1] = 1

zygkrondiff = Zygote.jacobian(x->x .* mat2_rsh,mat1_rsh)[1]

zygkrondiff = Zygote.jacobian(x->x .* mat1_rsh,mat2_rsh)[1]

reshape(reshape(∂kronaa .* mat2_rsh, m1*m2,n1*n2),6,6)

∂kronaa * mat2_rsh'

∂kronaa = zero(mat1_rsh).*(zero(mat2_rsh).+1)
∂kronaa[4] = 1

vec(mat2_rsh)' * reshape(∂kronaa,6,6)


# forward diff
kron(x,x)

# derivative of kron(x,x) wrt x

# reverse mode AD
# derivative of x wrt to kron(x,x)



∂kronaa = zero(kron(aa,aa))
# ∂kronaa[1,1] = 1
# ∂kronaa[1,2] = 1
∂kronaa[2,1] = 1


# ∂kronaa .* mat1_rsh

grad_mat1_rsh = reshape(∂kronaa, (m2, m1, n2, n1)) .* reshape(aa, (m2, 1, n2, 1))

grad_aa_1 = sum(grad_mat1_rsh, dims=(2, 4))
grad_aa_2 = sum(grad_mat1_rsh, dims=(1, 3))



zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]




vec(mat1_rsh .* mat2_rsh)

using SparseArrays
aa = sprand(10,5,.2)

kron(aa,aa)

zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]

@profview zygkrondiff = Zygote.jacobian(x->kron(x,x),aa)[1]


zygkrondiff
reshape(∂kronaa,6,6) * vec(aa)


reshape(kron(vec(aa),vec(aa)),4,9)'


vec(kron(aa,aa)) - vec(kron(vec(aa')',vec(aa')'))

difff = sparse([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 23, 64)

droptol!(difff,eps())




SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, verbose = verbose)
    
all_SS = expand_steady_state(SS_and_pars,𝓂)

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
par_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_hessian(x, SS_and_pars, 𝓂), parameters)[1]
    
SS_hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1),
                            x -> calculate_hessian(parameters, x, 𝓂), SS_and_pars)[1]
                            
hcat(par_hess,SS_hess)' |>sparse |>findnz
analytical_hess_SS_and_pars_vars    |> findnz  

maximum(hcat(par_hess,SS_hess)' - analytical_hess_SS_and_pars_vars)

sparse(hcat(SS_hess,par_hess))    |> findnz                  
sparse(hcat(SS_hess,par_hess)).nzval .|> Float32|> unique |> sort

𝓂.model_hessian_SS_and_pars_vars[2].nzval .|> Float32 |> unique |> sort
𝓂.model_hessian_SS_and_pars_vars[2]
# if !solved return -Inf end



SS_hess_zyg = Zygote.jacobian(x -> calculate_hessian(parameters, x, 𝓂), SS_and_pars)[1]
isapprox(SS_hess, SS_hess_zyg)                      
# if collect(axiskeys(data,1)) isa Vector{String}
#     data = @ignore_derivatives rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
# end

# dt = @ignore_derivatives collect(data(observables))

# # prepare data
# data_in_deviations = dt .- SS_and_pars[obs_indices]
