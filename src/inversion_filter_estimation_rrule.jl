using Revise
using MacroModelling
import MacroModelling: find_shocks, expand_steady_state, get_and_check_observables, check_bounds, get_NSSS_and_parameters, get_relevant_steady_state_and_state_update, ℳ, calculate_second_order_stochastic_steady_state, timings, second_order_auxilliary_matrices, calculate_third_order_stochastic_steady_state
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

# include("../models/Gali_2015_chapter_3_nonlinear.jl")



@model RBC_baseline begin
	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ * c[0] ^ σ / (1 - l[0]) = z[0] * k[-1] ^ α * l[0] ^ (1 - α) * (1 - α) / l[0]

	z[0] * k[-1] ^ α * l[0] ^ (1 - α) = c[0] + k[0] - (1 - δ) * k[-1] + g[0]

	# y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

end


@parameters RBC_baseline begin
	σᶻ = 0.066

	σᵍ = .104

	σ = 1

	α = 1/3

	i_y = 0.25

	k_y = 10.4

	ρᶻ = 0.97

	ρᵍ = 0.989

	g_y = 0.2038

	# ḡ | ḡ = g_y * y[ss]
    # z[0] * k[-1] ^ α * l[0] ^ (1 - α)
	ḡ | ḡ = g_y * k[ss] ^ α * l[ss] ^ (1 - α)

    δ = i_y / k_y

    β = 1 / (α / k_y + (1 - δ))

	ψ | l[ss] = 1/3
end


# 𝓂 = Gali_2015_chapter_3_nonlinear
𝓂 = RBC_baseline


T = 𝓂.timings
tol = 1e-12
parameter_values = 𝓂.parameter_values
parameters = 𝓂.parameter_values
verbose = false
presample_periods = 0
sylvester_algorithm = :doubling



# oobbss = [:Y, :Pi, :R]
oobbss = [:c, :k]
algorithm = :pruned_second_order

Random.seed!(9)
data = simulate(𝓂, algorithm = algorithm)(oobbss,:,:simulate)


get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm)

get_parameters(𝓂, values = true)

findiff = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1, max_range = 1e-4), x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

zygdiff = Zygote.gradient(x-> get_loglikelihood(𝓂, data, x, algorithm = algorithm), 𝓂.parameter_values)[1]

get_loglikelihood(𝓂, data, 𝓂.parameter_values, algorithm = algorithm)

isapprox(findiff, zygdiff)


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
