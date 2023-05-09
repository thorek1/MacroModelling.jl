using MacroModelling

Gali_2015_chapter_3_nonlinear = nothing
include("models/Gali_2015_chapter_3_nonlinear.jl")



@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = [:c,:y], Periods = 1:2)
conditions[1,1] = .01
conditions[2,2] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

plot_conditional_forecast(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)




# Gali_2015_chapter_3_nonlinear.solution.outdated_algorithms |>collect|>sort
# Gali_2015_chapter_3_nonlinear.solution.algorithms |>collect|>sort


irf2 = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :second_order)
irf2 = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :second_order, parameters = :std_nu => 1)
irf2p = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
irf1 = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :first_order)
irf3 = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :third_order)
irf3 = get_irf(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)


𝓂 = Gali_2015_chapter_3_nonlinear
solution_matrix = 𝓂.solution.perturbation.third_order.solution_matrix
reshape(solution_matrix,23,8,8,8)[:,1,1,1]
solution_matrix[:,1]

solution_matrix = 𝓂.solution.perturbation.first_order.solution_matrix
solution_mat = permutedims(reshape(solution_matrix,23,8,8,8),[2,1,3,4]);
permutedims(solution_mat,[2,1,3,4]);


KeyedArray(permutedims(reshape(solution_matrix,23,8,8,8),[2,1,3,4]);
States__Shocks¹ = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)],
Variables = 𝓂.var,
States__Shocks² = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)],
States__Shocks³ = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)])




get_SSS(Gali_2015_chapter_3_nonlinear)
get_SSS(Gali_2015_chapter_3_nonlinear, parameters = :std_nu => 1)
get_SSS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order, parameters = :std_nu => 1)
get_SSS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order, parameters = :std_nu => 1)
get_SSS(Gali_2015_chapter_3_nonlinear, algorithm = :third_order, parameters = :std_nu => 1)

get_SSS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order, parameters = :std_nu => 1)

Gali_2015_chapter_3_nonlinear.solution.perturbation.pruned_second_order.stochastic_steady_state


get_SSS(Caldara_et_al_2012)

get_SSS(Caldara_et_al_2012, algorithm = :pruned_second_order)

import StatsPlots
plot_irf(Gali_2015_chapter_3_nonlinear, algorithm = :second_order, shocks = :eps_nu, variables = [:Y,:Pi,:R,:W_real])
plot_irf(Gali_2015_chapter_3_nonlinear, algorithm = :second_order, shocks = :eps_a, variables = [:Y,:Pi,:R,:W_real], parameters = :std_nu => 1)
get_SSS(Gali_2015_chapter_3_nonlinear)
plot_solution(Caldara_et_al_2012,:k,algorithm = [:first_order,:second_order, :pruned_second_order,:third_order, :pruned_third_order])
plot_solution(Caldara_et_al_2012,:k,algorithm = [:first_order,:second_order, :third_order])
plot_solution(Caldara_et_al_2012,:k,algorithm = [:first_order,:second_order, :pruned_third_order])
plot_solution(Caldara_et_al_2012,:k,algorithm = [:first_order,:second_order, :third_order, :pruned_third_order])
plot_solution(Caldara_et_al_2012,:k,algorithm = [:second_order, :third_order])

plot_solution(Caldara_et_al_2012,:k,algorithm = [:second_order])

get_SSS(Caldara_et_al_2012, algorithm = :third_order)

get_SSS(Caldara_et_al_2012, algorithm = :pruned_third_order)

import ComponentArrays as 𝒞
𝓂 = Caldara_et_al_2012
verbose = true
parameters = 𝓂.parameter_values



SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

𝐒₂ = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁; T = 𝓂.timings)

𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

# state, converged = second_order_stochastic_steady_state_iterative_solution(𝒞.ComponentArray(; 𝐒₁, 𝐒₂), SS, 𝓂)

import LinearAlgebra as ℒ
using SpeedMapping
tol = eps()


state = zero(SS)
pruned_state = zero(SS)
shock = zeros(𝓂.timings.nExo)

aug_state .= [state[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]


aug_pruned_state = [pruned_state[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]


sol_pruned = speedmapping(state; 
            m! = (SSS, sss) -> begin 
                                aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]


                                SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_pruned_state, aug_pruned_state) / 2
                                pruned_state .= 𝐒₁ * aug_pruned_state
            end, 
tol = tol, maps_limit = 10000)

sol_pruned.minimizer


𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_pruned_state, aug_pruned_state) / 2

state = zero(SS)
pruned_state = zero(SS)

aug_state .= [state[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]


aug_pruned_state = [pruned_state[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]

sol = speedmapping(state; 
            m! = (SSS, sss) -> begin 
                                aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]

                                SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
            end, 
tol = tol, maps_limit = 10000)

sol.minimizer

isapprox(sol_pruned.minimizer,sol.minimizer, rtol = eps(Float32))