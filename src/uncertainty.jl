using Revise
using MacroModelling
using StatsPlots
using Optim, LineSearches
using Optimization, OptimizationNLopt
using Zygote, ForwardDiff
using BenchmarkTools


@model Gali_2015_chapter_3_nonlinear begin
	# W_real[0] = C[0] ^ σ * N[0] ^ φ

	# Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / π[1]

	1 / R[0] = β * (Y[1] / Y[0]) ^ (-σ) * Z[1] / Z[0] / π[1]

	# R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

	# R[0] = π[1] * realinterest[0]

	# C[0] = Y[0]
    
	# MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

	# MC[0] = Y[0] ^ σ * N[0] ^ φ / (S[0] * Y[0] * (1 - α) / N[0])

	1 = θ * π[0] ^ (ϵ - 1) + (1 - θ) * π_star[0] ^ (1 - ϵ)

	S[0] = (1 - θ) * π_star[0] ^ (( - ϵ) / (1 - α)) + θ * π[0] ^ (ϵ / (1 - α)) * S[-1]

	π_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

	x_aux_1[0] = Y[0] ^ σ * N[0] ^ φ / (S[0] * Y[0] * (1 - α) / N[0]) * Y[0] * Z[0] * Y[0] ^ (-σ) + β * θ * π[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * Y[0] ^ (-σ) + β * θ * π[1] ^ (ϵ - 1) * x_aux_2[1]

	# log_y[0] = log(Y[0])

	# log_W_real[0] = log(W_real[0])

	# log_N[0] = log(N[0])

	# π_ann[0] = 4 * log(π[0])

	# i_ann[0] = 4 * log(R[0])

	# r_real_ann[0] = 4 * log(realinterest[0])

	# M_real[0] = Y[0] / R[0] ^ η

    # Taylor rule
	# R[0] = 1 / β * π[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(ν[0])

	R[0] = R[-1] ^ ϕᴿ * (R̄ * π[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ) ^ (1 - ϕᴿ) * exp(ν[0])

    π̂[0] = log(π[0] / π[ss])

    Ŷ[0] = log(Y[0] / Y[ss])

    ΔR[0] = log(R[0] / R[-1])

    # Shocks
	log(A[0]) = ρ_a * log(A[-1]) + σ_a[0] * ε_a[x]

	log(Z[0]) = ρ_z * log(Z[-1]) - σ_z[0] * ε_z[x]

	ν[0] = ρ_ν * ν[-1] + σ_ν[0] * ε_ν[x]

    # Stochastic volatility
    log(σ_a[0]) = (1 - ρ_σ_a) * log(σ_ā) + ρ_σ_a * log(σ_a[-1]) + σ_σ_a * ε_σ_a[x]

    log(σ_z[0]) = (1 - ρ_σ_z) * log(σ_z̄) + ρ_σ_z * log(σ_z[-1]) + σ_σ_z * ε_σ_z[x]

    log(σ_ν[0]) = (1 - ρ_σ_ν) * log(σ_ν̄) + ρ_σ_ν * log(σ_ν[-1]) + σ_σ_ν * ε_σ_ν[x]

end


@parameters Gali_2015_chapter_3_nonlinear begin
    R̄ = 1 / β

	σ = 1

	φ = 5

	ϕᵖⁱ = 1.5
	
	ϕʸ = 0.125
    
    ϕᴿ = 0.75

	θ = 0.75

	ρ_ν = 0.5

	ρ_z = 0.5

	ρ_a = 0.9

	β = 0.99

	η = 3.77

	α = 0.25

	ϵ = 9

	τ = 0


    σ_ā = .01

    σ_z̄ = .05

    σ_ν̄ = .0025


    ρ_σ_a = 0.75

    ρ_σ_z = 0.75

    ρ_σ_ν = 0.75


    σ_σ_a = 0.1

    σ_σ_z = 0.1

    σ_σ_ν = 0.1
end



include("../models/Smets_Wouters_2007 copy.jl")

# Optimal simple rule

loss_function_weights = [1, .3, .4]

loss_function_weights = [1, 1, .1]

lbs = [eps(),eps(),eps(),eps()]
ubs = [1-eps(), 1e6, 1e6, 1e6]
initial_values = [0.8762 ,1.488 ,0.0593 ,0.2347]

get_statistics(Smets_Wouters_2007,   
                    initial_values,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :robs],
                    algorithm = :first_order,
                    verbose = true)

function calculate_cb_loss(parameter_inputs,p; verbose = false)
    # println(parameter_inputs)
    out = get_statistics(Smets_Wouters_2007,   
                    parameter_inputs,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :robs],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

calculate_cb_loss(initial_values,[], verbose = true)


f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, initial_values, [], ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)


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
sol_zyg = solve(prob_zyg,  NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxiters = 10000)
sol_zyg.u

f_for = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)


ForwardDiff.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

Zygote.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

sol = Optim.optimize(x->calculate_cb_loss(x,[]), 
                    # lbs, 
                    # ubs, 
                    initial_values, 
                    LBFGS(),
                    # NelderMead(),
                    # Optim.Fminbox(NelderMead()), 
                    # Optim.Fminbox(LBFGS()), 
                    # Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 2))), 
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
