using Revise
using MacroModelling
using StatsPlots
using Zygote
using ForwardDiff
using Random
import Optim, LineSearches
using BenchmarkTools
import LinearAlgebra as ℒ

@model Gali_2015_chapter_3_nonlinear begin
	W_real[0] = C[0] ^ σ * N[0] ^ φ

	Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]

	log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]

	nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]

	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

	1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)

	S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]

	Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]

	log_y[0] = log(Y[0])

	log_W_real[0] = log(W_real[0])

	log_N[0] = log(N[0])

	pi_ann[0] = 4 * log(Pi[0])

	i_ann[0] = 4 * log(R[0])

	r_real_ann[0] = 4 * log(realinterest[0])

	M_real[0] = Y[0] / R[0] ^ η

end


@parameters Gali_2015_chapter_3_nonlinear begin
	σ = 1

	φ = 5

	ϕᵖⁱ = 1.5
	
	ϕʸ = 0.125

	θ = 0.75

	ρ_ν = 0.5

	ρ_z = 0.5

	ρ_a = 0.9

	β = 0.99

	η = 3.77

	α = 0.25

	ϵ = 9

	τ = 0

    std_a = .01

    std_z = .05

    std_nu = .0025

end



Random.seed!(1)
data = simulate(Gali_2015_chapter_3_nonlinear)([:pi_ann,:W_real], :, :simulate)

get_loglikelihood(Gali_2015_chapter_3_nonlinear, data, Gali_2015_chapter_3_nonlinear.parameter_values)

Zygote.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)[1]
ForwardDiff.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)



𝓂 = Gali_2015_chapter_3_nonlinear

import MacroModelling: get_and_check_observables, check_bounds, minimize_distance_to_initial_data, get_relevant_steady_state_and_state_update, replace_indices, minimize_distance_to_data, match_data_sequence!, match_initial_data!

parameter_values = 𝓂.parameter_values
algorithm = :first_order
filter = :inversion
warmup_iterations = 0
presample_periods = 0
initial_covariance = :diagonal
tol = 1e-12
verbose = false

observables = get_and_check_observables(𝓂, data)

solve!(𝓂, verbose = verbose, algorithm = algorithm)

bounds_violated = check_bounds(parameter_values, 𝓂)

NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

obs_indices = convert(Vector{Int}, indexin(observables, NSSS_labels))

TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)

if collect(axiskeys(data,1)) isa Vector{String}
    data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
end

dt = collect(data(observables))

# prepare data
data_in_deviations = dt .- SS_and_pars[obs_indices]



T = 𝓂.timings
# state::Vector{Vector{Float64}}, 
#                                                     𝐒::Union{Matrix{Float64}, Vector{AbstractMatrix{Float64}}}, 
#                                                     data_in_deviations::Matrix{Float64}, 
#                                                     observables::Union{Vector{String}, Vector{Symbol}},
#                                                     T::timings; 
#                                                     warmup_iterations::Int = 0,
#                                                     presample_periods::Int = 0)


function first_order_state_update(state::Vector{U}, shock::Vector{S}) where {U <: Real,S <: Real}
# state_update = function(state::Vector{T}, shock::Vector{S}) where {T <: Real,S <: Real}
    aug_state = [state[T.past_not_future_and_mixed_idx]
                shock]
    return 𝐒 * aug_state # you need a return statement for forwarddiff to work
end

state_update = first_order_state_update

state = state[1]

pruning = false

    
precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))



warmup_iterations = 3
state *=0

data_in_deviations[:,1] - (𝐒 * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[4:6]))[cond_var_idx]

data_in_deviations[:,1] - (𝐒 * (vcat((𝐒[T.past_not_future_and_mixed_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3]))), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))[cond_var_idx]


data_in_deviations[:,1] - (𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))


𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])))

𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])


𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * (𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]) + 𝐒[cond_var_idx,end-T.nExo+1:end] * x[4:6] - data_in_deviations[:,1]




𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3] +
𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[4:6] +
𝐒[cond_var_idx,end-T.nExo+1:end] * x[7:9] -
data_in_deviations[:,1]

hcat(   
    𝐒[cond_var_idx,end-T.nExo+1:end] , 
    𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], 
    𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
    ) \ data_in_deviations[:,1]


warmup_iterations = 5

state *= 0
logabsdets = 0
shocks² = 0

if warmup_iterations > 0
    if warmup_iterations >= 1
        to_be_inverted = 𝐒[cond_var_idx,end-T.nExo+1:end]
        if warmup_iterations >= 2
            to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
            if warmup_iterations >= 3
                Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                for e in 1:warmup_iterations-2
                    to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
                    Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
                end
            end
        end
    end

    x = to_be_inverted \ data_in_deviations[:,1]

    warmup_shocks = reshape(x, T.nExo, warmup_iterations)

    for i in 1:warmup_iterations-1
        state = state_update(state, warmup_shocks[:,i])
    end

    jacc = -to_be_inverted'

    for i in 1:warmup_iterations
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
        end
    end

    shocks² += sum(abs2,x)
end



data_in_deviations[:,1] - 𝐒[cond_var_idx,:] * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])



data_in_deviations[:,1]

(𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[1:3]))[cond_var_idx]


        x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))





state_copy = deepcopy(state)

XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

for i in 1:warmup_iters
    state_copy = state_update(state_copy, XX[:,i])
end

return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])



shocks² = 0.0
logabsdets = 0.0

if warmup_iterations > 0
    res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                        zeros(T.nExo * warmup_iterations), 
                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward)

    matched = Optim.minimum(res) < 1e-12

    if !matched # for robustness try other linesearch
        res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
                        zeros(T.nExo * warmup_iterations), 
                        Optim.LBFGS(), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward)
    
        matched = Optim.minimum(res) < 1e-12
    end

    if !matched return -Inf end

    x = Optim.minimizer(res)

    warmup_shocks = reshape(x, T.nExo, warmup_iterations)

    for i in 1:warmup_iterations-1
        state = state_update(state, warmup_shocks[:,i])
    end
    
    res = zeros(0)

    jacc = zeros(T.nExo * warmup_iterations, length(observables))

    match_initial_data!(res, x, jacc, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor), zeros(size(data_in_deviations, 1))

    for i in 1:warmup_iterations
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
        end
    end

    shocks² += sum(abs2,x)
end




# i = 2
# res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
#                     zeros(T.nExo), 
#                     Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
#                     Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                     autodiff = :forward)

#                     res.minimizer
# # data_in_deviations[:,i] - 𝐒[cond_var_idx,end-T.nExo+1:end] * x

# @benchmark x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ data_in_deviations[:,i]
# @profview for k in 1:1000 𝐒[cond_var_idx,end-T.nExo+1:end] \ data_in_deviations[:,i] end


# @profview for k in 1:1000
@benchmark begin
    state = zeros(23)
    # statetmp = zeros(23)
    shocks² = 0.0
    logabsdets = 0.0
    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)
    # state_reduced = zeros(T.nPast_not_future_and_mixed + T.nExo)

    jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
        jacdecomp = ℒ.lu!(jac)
        invjac = inv(jacdecomp)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
        jacdecomp = ℒ.svd!(jac)
        invjac = inv(jacdecomp)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods
    
    @views 𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x,invjac,y)
        # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,x)
        end

        # # copyto!(state_reduced, 1, state, T.past_not_future_and_mixed_idx)
        # for (i,v) in enumerate(T.past_not_future_and_mixed_idx)
        #     state_reduced[i] = state[v]
        # end
        # copyto!(state_reduced, T.nPast_not_future_and_mixed + 1, x, 1, T.nExo)
        
        ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))
        # state = state_update(state, x)
    end

    -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

pruning = false

jac = ForwardDiff.jacobian( xx -> precision_factor .* abs.(data_in_deviations[:,i] - (pruning ? sum(state_update(state, xx)) : state_update(state, xx))[cond_var_idx]), x)'


res = precision_factor .* abs.(data_in_deviations[:,i] - (pruning ? sum(state_update(state, x)) : state_update(state, x))[cond_var_idx])


state = state_update(state, x)


x = Optim.minimizer(res)

res  = zeros(0)

jacc = zeros(T.nExo, length(observables))

match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)


match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)


@benchmark begin
shocks² = 0.0
logabsdets = 0.0
state = zeros(23)

for i in axes(data_in_deviations,2)
    res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                    zeros(T.nExo), 
                    Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                    Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                    autodiff = :forward)

    matched = Optim.minimum(res) < 1e-12

    if !matched # for robustness try other linesearch
        res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                        zeros(T.nExo), 
                        Optim.LBFGS(), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward)
    
        matched = Optim.minimum(res) < 1e-12
    end

    if !matched return -Inf end

    x = Optim.minimizer(res)

    res  = zeros(0)

    jacc = zeros(T.nExo, length(observables))

    match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x)
    end

    state = state_update(state, x)
end

# See: https://pcubaborda.net/documents/CGIZ-final.pdf
 -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end