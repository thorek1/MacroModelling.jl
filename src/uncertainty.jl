using Revise
using MacroModelling
using StatsPlots
using Optim, LineSearches
using Optimization, OptimizationNLopt#, OptimizationOptimJL
using Zygote, ForwardDiff
using BenchmarkTools
using JuMP, MadNLP

# @model Gali_2015_chapter_3_nonlinear begin
# 	# W_real[0] = C[0] ^ σ * N[0] ^ φ

# 	# Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / π[1]

# 	1 / R[0] = β * (Y[1] / Y[0]) ^ (-σ) * Z[1] / Z[0] / π[1]

# 	# R[0] = 1 / Q[0]

# 	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

# 	# R[0] = π[1] * realinterest[0]

# 	# C[0] = Y[0]
    
# 	# MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

# 	# MC[0] = Y[0] ^ σ * N[0] ^ φ / (S[0] * Y[0] * (1 - α) / N[0])

# 	1 = θ * π[0] ^ (ϵ - 1) + (1 - θ) * π_star[0] ^ (1 - ϵ)

# 	S[0] = (1 - θ) * π_star[0] ^ (( - ϵ) / (1 - α)) + θ * π[0] ^ (ϵ / (1 - α)) * S[-1]

# 	π_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

# 	x_aux_1[0] = Y[0] ^ σ * N[0] ^ φ / (S[0] * Y[0] * (1 - α) / N[0]) * Y[0] * Z[0] * Y[0] ^ (-σ) + β * θ * π[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

# 	x_aux_2[0] = Y[0] * Z[0] * Y[0] ^ (-σ) + β * θ * π[1] ^ (ϵ - 1) * x_aux_2[1]

# 	# log_y[0] = log(Y[0])

# 	# log_W_real[0] = log(W_real[0])

# 	# log_N[0] = log(N[0])

# 	# π_ann[0] = 4 * log(π[0])

# 	# i_ann[0] = 4 * log(R[0])

# 	# r_real_ann[0] = 4 * log(realinterest[0])

# 	# M_real[0] = Y[0] / R[0] ^ η

#     # Taylor rule
# 	# R[0] = 1 / β * π[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(ν[0])

# 	R[0] = R[-1] ^ ϕᴿ * (R̄ * π[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ) ^ (1 - ϕᴿ) * exp(ν[0])

#     π̂[0] = log(π[0] / π[ss])

#     Ŷ[0] = log(Y[0] / Y[ss])

#     ΔR[0] = log(R[0] / R[-1])

#     # Shocks
# 	log(A[0]) = ρ_a * log(A[-1]) + σ_a[0] * ε_a[x]

# 	log(Z[0]) = ρ_z * log(Z[-1]) - σ_z[0] * ε_z[x]

# 	ν[0] = ρ_ν * ν[-1] + σ_ν[0] * ε_ν[x]

#     # Stochastic volatility
#     log(σ_a[0]) = (1 - ρ_σ_a) * log(σ_ā) + ρ_σ_a * log(σ_a[-1]) + σ_σ_a * ε_σ_a[x]

#     log(σ_z[0]) = (1 - ρ_σ_z) * log(σ_z̄) + ρ_σ_z * log(σ_z[-1]) + σ_σ_z * ε_σ_z[x]

#     log(σ_ν[0]) = (1 - ρ_σ_ν) * log(σ_ν̄) + ρ_σ_ν * log(σ_ν[-1]) + σ_σ_ν * ε_σ_ν[x]

# end


# @parameters Gali_2015_chapter_3_nonlinear begin
#     R̄ = 1 / β

# 	σ = 1

# 	φ = 5

# 	ϕᵖⁱ = 1.5
	
# 	ϕʸ = 0.125
    
#     ϕᴿ = 0.75

# 	θ = 0.75

# 	ρ_ν = 0.5

# 	ρ_z = 0.5

# 	ρ_a = 0.9

# 	β = 0.99

# 	η = 3.77

# 	α = 0.25

# 	ϵ = 9

# 	τ = 0


#     σ_ā = .01

#     σ_z̄ = .05

#     σ_ν̄ = .0025


#     ρ_σ_a = 0.75

#     ρ_σ_z = 0.75

#     ρ_σ_ν = 0.75


#     σ_σ_a = 0.1

#     σ_σ_z = 0.1

#     σ_σ_ν = 0.1
# end



include("../models/Smets_Wouters_2007 copy.jl")

# US SW07 sample estims
estimated_par_vals = [0.4818650901000989, 0.24054470291311028, 0.5186956692202958, 0.4662413867655003, 0.23136135922950385, 0.13132950287219664, 0.2506090809487915, 0.9776707755474057, 0.2595790622654468, 0.9727418060187103, 0.687330720531337, 0.1643636762401503, 0.9593771388356938, 0.9717966717403557, 0.8082505346152592, 0.8950643861525535, 5.869499350284732, 1.4625899840952736, 0.724649200081708, 0.7508616008157103, 2.06747381157293, 0.647865359908012, 0.585642549132298, 0.22857733002230182, 0.4476375712834215, 1.6446238878581076, 2.0421854715489007, 0.8196744223749656, 0.10480818163546246, 0.20376610336806866, 0.7312462829038883, 0.14032972276989308, 1.1915345520903131, 0.47172181998770146, 0.5676468533218533, 0.2071701728019517]

# EA long sample
# estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# EA tighter priors (no crdy)
estimated_par_vals = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]
estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# find optimal loss coefficients
# Problem definition, find the loss coefficients such that the derivatives of the Taylor rule coefficients wrt the loss are 0
# lbs = [0,0]
# ubs = [1e6, 1e6] #, 1e6]
# initial_values = [.3 ,.3] # ,0.2347]

# var = get_variance(Smets_Wouters_2007, derivatives = false)



# Define the given vector
# SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

# loss_function_weights = [1, 1, .1]
# get_parameters(Smets_Wouters_2007, values = true)
# lbs = [eps(),eps(),eps()] #,eps()]
# ubs = [1-eps(), 1e6, 1e6] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
# regularisation = [1e-7,1e-5,1e-5]  #,1e-5]

# US
optimal_taylor_coefficients = [0.8196744223749656, 2.0421854715489007, 0.10480818163546246, 0.20376610336806866]

# EA
# optimal_taylor_coefficients = [0.935445005053725, 1.0500350793944067, 0.14728806935911198, 0.05076653598648485]


out = get_statistics(Smets_Wouters_2007,   
                    optimal_taylor_coefficients,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = true)


# out[:variance]' * loss_function_weights + abs2.(initial_values)' * regularisation





# function calculate_loss(loss_function_weights,regularisation; verbose = false)
#     out = get_statistics(Smets_Wouters_2007,   
#                     [0.824085387718046, 1.9780022172135707, 4.095695818850862],
#                     # [0.935445005053725, 1.0500350793944067, 0.14728806935911198, 0.05076653598648485, 0],
#                     parameters = [:crr, :crpi, :cry, :crdy],
#                     variance = [:ygap, :pinfobs, :drobs],
#                     algorithm = :first_order,
#                     verbose = verbose)

#     return out[:variance]' * loss_function_weights + abs2.([0.824085387718046, 1.9780022172135707, 4.095695818850862,0])' * regularisation
# end

function calculate_cb_loss(parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    # println(parameter_inputs)
    out = get_statistics(Smets_Wouters_2007,   
                    parameter_inputs,
                    parameters = [:crr, :crpi, :cry],#, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + abs2.(parameter_inputs)' * regularisation
end

optimal_taylor_coefficients = [0.824085387718046, 1.9780022172135707, 4.095695818850862]

loss_function_weights = [1, .1,1]

regularisation = [1e-7,1e-5,1e-5]  #,1e-5]

function find_weights(loss_function_weights, optimal_taylor_coefficients)
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (loss_function_weights / sum(loss_function_weights), regularisation * 100)), optimal_taylor_coefficients)) #, 0.05076653598648485])
end

find_weights(loss_function_weights, optimal_taylor_coefficients)

# get_parameters(Smets_Wouters_2007, values = true)
lbs = fill(0.0,3)
ubs = fill(1.0,3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, fill(.35,3), optimal_taylor_coefficients, ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LN_PRAXIS(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_TNEWTON(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), maxiters = 1000) # this seems to achieve best results

consistent_optimal_weights = sol.u ./ sol.u[1]

find_weights(consistent_optimal_weights, optimal_taylor_coefficients)

ForwardDiff.gradient(x->calculate_cb_loss(x, (sol.u ./ sol.u[1], regularisation * 100)), optimal_taylor_coefficients)




function calculate_cb_loss(parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    # println(parameter_inputs)
    out = get_statistics(Smets_Wouters_2007,   
                    parameter_inputs,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + abs2.(parameter_inputs)' * regularisation
end

# US
optimal_taylor_coefficients = [0.8196744223749656, 2.0421854715489007, 0.10480818163546246, 0.20376610336806866]

# EA
optimal_taylor_coefficients = [0.935445005053725, 1.0500350793944067, 0.14728806935911198, 0.05076653598648485]

loss_function_weights = [1, .1,1]

regularisation = [1e-7, 1e-5, 1e-5, 1e-5]


function find_weights(loss_function_weights, optimal_taylor_coefficients)
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (loss_function_weights / sum(loss_function_weights), regularisation * 100)), optimal_taylor_coefficients)) #, 0.05076653598648485])
end

find_weights(loss_function_weights, optimal_taylor_coefficients)

∇ = ForwardDiff.jacobian(xx->ForwardDiff.gradient(x->calculate_cb_loss(x, (xx.^2 / sum(xx.^2), regularisation * 100)), optimal_taylor_coefficients),loss_function_weights)

# ∇' \ loss_function_weights
∇ \ optimal_taylor_coefficients

# using LinearAlgebra

loss_function_weights += ∇ \ optimal_taylor_coefficients

# get_parameters(Smets_Wouters_2007, values = true)

lbs = fill(0.0,3)
ubs = fill(1.0,3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, [.99,.1,.99], optimal_taylor_coefficients, ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results


calculate_cb_loss(optimal_taylor_coefficients, (sol.u ./ sol.u[1], regularisation * 100))

ForwardDiff.gradient(x->calculate_cb_loss(x, (sol.u ./ sol.u[1], regularisation * 100)), optimal_taylor_coefficients)



function find_weights(loss_function_weights, optimal_taylor_coefficients)
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (loss_function_weights, 100*regularisation)), optimal_taylor_coefficients)) #, 0.05076653598648485])
end

find_weights(loss_function_weights, optimal_taylor_coefficients)

# get_parameters(Smets_Wouters_2007, values = true)

lbs = fill(0.0,2)
ubs = fill(1e6,2)

f = OptimizationFunction((x,p)-> find_weights(vcat(1,x),p), AutoForwardDiff())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, [10,1], optimal_taylor_coefficients, ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LN_PRAXIS(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_TNEWTON(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxiters = 100) # this seems to achieve best results

consistent_optimal_weights = sol.u ./ sol.u[1]

consistent_optimal_weights = vcat(1,sol.u)

find_weights(consistent_optimal_weights, optimal_taylor_coefficients)

ForwardDiff.gradient(x->calculate_cb_loss(x, (consistent_optimal_weights, 100*regularisation)), optimal_taylor_coefficients)



loss_function_weights = [1,6.913136326454511,1.9453221822118556]

vector = [40.0091669196762, 1.042452394619108, 0.023327511003148015]

# Create a model
model = Model(MadNLP.Optimizer)

# Number of weights
n = length(loss_function_weights)

# Define variables: weights must be positive
@variable(model, w[1:n] >= 0)

# Constraint: weights must sum to 1
@constraint(model, sum(w) == 1)

# Objective: minimize the dot product of the weights with the vector
@objective(model, Min, x -> find_weights(x, optimal_taylor_coefficients))

# Solve the model
optimize!(model)

# Check if the model was solved successfully
if termination_status(model) == MOI.OPTIMAL
    optimal_weights = value.(w)
    println("Optimal weights: ", optimal_weights)
    println("Minimum dot product value: ", objective_value(model))
else
    println("The optimization problem was not solved successfully.")
end


f = OptimizationFunction((x,p)-> vcat(1,x)' * p, AutoForwardDiff())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, initial_values, var([:ygap, :pinfobs, :drobs]), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

# Optimal simple rule

loss_function_weights = [1, .3, .4]

# loss_function_weights = [1, 1, .1]

lbs = [eps(),eps(),eps()] #,eps()]
ubs = [1-eps(), 1e6, 1e6] #, 1e6]
initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
regularisation = [1e-7,1e-5,1e-5]  #,1e-5]

get_statistics(Smets_Wouters_2007,   
                    initial_values,
                    parameters = [:crr, :crpi, :cry],#, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = true)

function calculate_cb_loss(parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    # println(parameter_inputs)
    out = get_statistics(Smets_Wouters_2007,   
                    parameter_inputs,
                    parameters = [:crr, :crpi, :cry],#, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + abs2.(parameter_inputs)' * regularisation
end

calculate_cb_loss(initial_values,(loss_function_weights, regularisation), verbose = true)

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, initial_values, (loss_function_weights, regularisation * 100), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
# sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000)

# sol = solve(prob, Optimization.LBFGS(), maxiters = 10000)

# sol = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

# sol = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 1000)

# calculate_cb_loss(sol.u, regularisation)

# abs2.(sol.u)' * regularisation

# sol.objective
# loop across different levels of std

get_parameters(Smets_Wouters_2007, values = true)

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0))
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 4-element view(::Vector{Symbol},...)
# →   Standard_deviation_and_∂standard_deviation∂parameter ∈ 11-element view(::Vector{Symbol},...)
# And data, 4×11 view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 20, 19, 21, 22]) with eltype Float64:
#               (:z_ea)     (:z_eb)      (:z_eg)      (:z_em)     (:z_ew)    (:z_eqs)      (:z_epinf)   (:crr)    (:crpi)     (:cry)       (:crdy)
#   (:ygap)      0.0173547   0.151379     0.00335788   0.303146    4.07402    0.0062702     1.15872    -60.0597    0.0553865  -0.0208848   -0.116337
#   (:pinfobs)   0.0112278   2.48401e-5   9.97486e-5   0.134175    8.97266    0.000271719   3.75081     90.5474   -0.0832394   0.0312395    0.105122
#   (:drobs)     0.289815    3.7398       0.0452899    0.0150356   0.731132   0.0148536     0.297607     4.20058  -0.0045197   0.00175464   0.121104
#   (:robs)      0.216192    1.82174      0.0424333    0.115266    7.89551    0.0742737     2.57712     80.8386   -0.0743082   0.0273497    0.14874

Zygote.gradient(x->calculate_cb_loss(x,regularisation * 1),sol.u)[1]


SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)

Zygote.gradient(x->calculate_cb_loss(x,regularisation * 1),sol.u)[1]

SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1] * 1.05, derivatives = false)

using FiniteDifferences

FiniteDifferences.hessian(x->calculate_cb_loss(x,regularisation * 0),sol.u)[1]


SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)



SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

# nms = []

k_range = 1:1:10
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 5)


ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(vl, 1.5 * vl, length = n_σ_range)


        prob = OptimizationProblem(f, initial_values, ([1,.3, k], regularisation * 1), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(k,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, surface(vec(coeff[:,ii,:,1]), vec(coeff[:,ii,:,2]), vec(coeff[:,ii,:,3]), label = "", xlabel = "Loss weight: r", ylabel = "Std($nm)", zlabel = "crr", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,1]), vec(coeff[:,ii,:,2]), vec((1 .- coeff[:,ii,:,3]) .* coeff[:,ii,:,4]), label = "", xlabel = "Loss weight: r", ylabel = "Std($nm)", zlabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,1]), vec(coeff[:,ii,:,2]), vec((1 .- coeff[:,ii,:,3]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: r", ylabel = "Std($nm)", zlabel = "(1 - crr) * cry", colorbar=false))
    
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm)_surface.png")
    ii += 1
end

coeff[:,1,:,4]
((1 .- coeff[:,1,:,3]) .* coeff[:,1,:,4])[10,:]
((1 .- coeff[:,1,:,3]) .* coeff[:,1,:,5])[1,:]
coeff[1,1,:,2]

surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,3]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "crr")
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "crpi")
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "cry")

shck = 7
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec((1 .- coeff[:,1,:,3]) .* coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * crpi")
surface(vec(coeff[:,shck,:,1]), vec(coeff[:,shck,:,2]), vec((1 .- coeff[:,shck,:,3]) .* coeff[:,shck,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * cry")


surface(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr")

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [i[2] for i in coeffs], label = "", ylabel = "crpi"))
    push!(plots, plot(σ_range, [i[3] for i in coeffs], label = "", ylabel = "cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_direct_$nm.png")
    # display(p)
end

# Demand shocks (Y↑ - pi↑ - R↑)
# z_eb	# risk-premium shock
# z_eg	# government shock
# z_eqs	# investment-specific shock


# Monetary policy (Y↓ - pi↓ - R↑)
# z_em	# interest rate shock

# Supply shock (Y↓ - pi↑ - R↑)
# z_ea	# technology shock

## Mark-up/cost-push shocks (Y↓ - pi↑ - R↑)
# z_ew	# wage mark-up shock
# z_epinf	# price mark-up shock



# demand shock (Y↑ - pi↑ - R↑): more aggressive on all three measures
# irf: GDP and inflation in same direction so you can neutralise this shocks at the cost of higher rate volatility

# supply shocks (Y↓ - pi↑ - R↑): more aggressive on inflation and GDP and less so on inflation
# trade off betwen GDP and inflation will probably dampen interest rate voltility so you can allow yourself to smooth less

# mark-up shocks (Y↓ - pi↑ - R↑): less aggressive on inflation and GDP but more smoothing
# low effectiveness wrt inflation, high costs, inflation less sticky



# try with EA parameters from estimation
estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)


prob = OptimizationProblem(f, initial_values, regularisation / 100, ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0));
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 4-element view(::Vector{Symbol},...)
# →   Standard_deviation_and_∂standard_deviation∂parameter ∈ 11-element view(::Vector{Symbol},...)
# And data, 4×11 view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 20, 19, 21, 22]) with eltype Float64:
#               (:z_ea)     (:z_eb)      (:z_eg)      (:z_em)     (:z_ew)    (:z_eqs)      (:z_epinf)   (:crr)    (:crpi)     (:cry)       (:crdy)
#   (:ygap)      0.0173547   0.151379     0.00335788   0.303146    4.07402    0.0062702     1.15872    -60.0597    0.0553865  -0.0208848   -0.116337
#   (:pinfobs)   0.0112278   2.48401e-5   9.97486e-5   0.134175    8.97266    0.000271719   3.75081     90.5474   -0.0832394   0.0312395    0.105122
#   (:drobs)     0.289815    3.7398       0.0452899    0.0150356   0.731132   0.0148536     0.297607     4.20058  -0.0045197   0.00175464   0.121104
#   (:robs)      0.216192    1.82174      0.0424333    0.115266    7.89551    0.0742737     2.57712     80.8386   -0.0743082   0.0273497    0.14874

Smets_Wouters_2007.parameter_values[indexin([:crr, :crpi, :cry, :crdy],Smets_Wouters_2007.parameters)]

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[2] for i in coeffs], label = "", ylabel = "(1 - crr) * crpi"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[3] for i in coeffs], label = "", ylabel = "(1 - crr) * cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_EA_$nm.png")
    # display(p)
end





solopt = solve(prob, Optimization.LBFGS(), maxiters = 10000)
soloptjl = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

sol_mlsl = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)


f_zyg = OptimizationFunction(calculate_cb_loss, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# 32.749
@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

sol_zyg = solve(prob_zyg, NLopt.LN_PRAXIS(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_SBPLX(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_NELDERMEAD(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_SLSQP(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_MMA(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_CCSAQ(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_TNEWTON(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxiters = 1000)
sol_zyg.u

f_for = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)


ForwardDiff.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

Zygote.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

sol = Optim.optimize(x->calculate_cb_loss(x,[]), 
                    lbs, 
                    ubs, 
                    initial_values, 
                    # LBFGS(),
                    # NelderMead(),
                    # Optim.Fminbox(NelderMead()), 
                    # Optim.Fminbox(LBFGS()), 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                           show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ))#,ad = AutoZgote())

pars = Optim.minimizer(sol)


get_statistics(Smets_Wouters_2007,   
                    sol.u,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = true)

## Central bank loss function: Loss = θʸ * var(Ŷ) + θᵖⁱ * var(π̂) + θᴿ * var(ΔR)
loss_function_weights = [1, .3, .4]

lbs = [eps(),eps(),eps()]
ubs = [1e2,1e2,1-eps()]
initial_values = [1.5, 0.125, 0.75]

function calculate_cb_loss(parameter_inputs; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

calculate_cb_loss(initial_values, verbose = true)


@time sol = Optim.optimize(calculate_cb_loss, 
                    lbs, 
                    ubs, 
                    initial_values, 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    # LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                        #    show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ));

pars = Optim.minimizer(sol)


out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                initial_values,
                parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                variance = [:Ŷ,:π̂,:ΔR],
                algorithm = :first_order,
                verbose = true)
out[:variance]
dd = Dict{Symbol,Array{<:Real}}()
dd[:variance] = out[1]

init_params = copy(Gali_2015_chapter_3_nonlinear.parameter_values)

function calculate_cb_loss_Opt(parameter_inputs,p; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

f_zyg = OptimizationFunction(calculate_cb_loss_Opt, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

f_for = OptimizationFunction(calculate_cb_loss_Opt, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)

# Import a solver package and solve the optimization problem
# sol = solve(prob, NLopt.LN_PRAXIS());
# sol.u
@time sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

@benchmark sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

using ForwardDiff
ForwardDiff.gradient(calculate_cb_loss,initial_values)
Zygote.gradient(calculate_cb_loss,initial_values)[1]
# SS(Gali_2015_chapter_3_nonlinear, parameters = :std_std_a => .00001)

SS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std³ = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std² = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
std¹ = get_std(Gali_2015_chapter_3_nonlinear)

std³([:π,:Y,:R],:)
std²(:π,:)
std¹(:π,:)

plot_solution(Gali_2015_chapter_3_nonlinear, :ν, algorithm = [:pruned_second_order, :pruned_third_order])

mean² = get_mean(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
mean¹ = get_mean(Gali_2015_chapter_3_nonlinear)

mean²(:π,:)
mean¹(:π,:)

get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => 1.5, :σ_σ_a => 2.0), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)

get_parameters(Gali_2015_chapter_3_nonlinear, values = true)

n = 5
res = zeros(3,n^2)

SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_π
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :σ_π => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_π", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Pi.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_Y
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_Y => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_Y", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Y.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.8,n) # std_std_a
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_a => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_a", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_tfp.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_std_z
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_z => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_z", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_z.png")




SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_θ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_θ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_θ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_calvo.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_ϕᵖⁱ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ϕᵖⁱ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ϕᵖⁱ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_infl_reaction.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_ā
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ā => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (60,65),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ā", 
                    zlabel = "std(Inflation)")

savefig("tfp_std_dev.png")
