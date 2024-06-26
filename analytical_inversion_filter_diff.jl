using Revise
using MacroModelling
using StatsPlots
using Zygote, ForwardDiff, FiniteDiff
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

precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))





state = [zeros(T.nVars) for _ in 1:size(data_in_deviations,2)+1]
# statetmp = zeros(23)
shocks² = 0.0
logabsdets = 0.0
y = zeros(length(cond_var_idx))
x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]
# state_reduced = zeros(T.nPast_not_future_and_mixed + T.nExo)

jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

if T.nExo == length(observables)
    logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
    jacdecomp = ℒ.lu(jac)
    invjac = inv(jacdecomp)
else
    logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
    jacdecomp = ℒ.svd(jac)
    invjac = inv(jacdecomp)
end

logabsdets *= size(data_in_deviations,2) - presample_periods

@views 𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

for i in axes(data_in_deviations,2)
    @views ℒ.mul!(y, 𝐒obs, state[i][T.past_not_future_and_mixed_idx])
    @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
    ℒ.mul!(x[i],invjac,y)
    # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])
    # x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

    if i > presample_periods
        shocks² += sum(abs2,x[i])
    end

    # # copyto!(state_reduced, 1, state, T.past_not_future_and_mixed_idx)
    # for (i,v) in enumerate(T.past_not_future_and_mixed_idx)
    #     state_reduced[i] = state[v]
    # end
    # copyto!(state_reduced, T.nPast_not_future_and_mixed + 1, x, 1, T.nExo)
    
    ℒ.mul!(state[i+1], 𝐒, vcat(state[i][T.past_not_future_and_mixed_idx], x[i]))
    # state[i+1] =  𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], x[i])
    # state = state_update(state, x)
end

-(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2



# for i in axes(data_in_deviations,2)
#     x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])
#     state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], x[i])
#     shocks² += sum(abs2,x[i])
# end
# return shocks²
𝐒endo = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒exo = 𝐒[cond_var_idx, end-T.nExo+1:end]


# ∂state = zero(state[1])
∂x = zero(x[1])
∂𝐒 = zero(𝐒)
∂v = zero(data_in_deviations[:,1])
∂data_in_deviations∂x = zero(data_in_deviations)


for i in 2:-1:1 # reverse(axes(data_in_deviations,2))
    # ∂∂data_in_deviations∂shock²
    ∂x = 2*x[i]
 
    ∂v = invjac' * ∂x

    ∂data_in_deviations∂x[:,i] = ∂v

    if i < size(data_in_deviations,2)
        ∂data_in_deviations∂x[:,i] -= invjac' * (𝐒[T.past_not_future_and_mixed_idx,:]' * 𝐒endo' * invjac' * 2 * x[i+1])[end-T.nExo+1:end]
    end


    ∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i] * x[i]' # [cond_var_idx, end-T.nExo+1:end]

    # ∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i+1] * x[i+1]' # [cond_var_idx, end-T.nExo+1:end]

    if i > 1
        ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (𝐒[st,:] * vcat(state[i-1][st], x[i-1]))' .* (invjac' * 2 * x[i])
        ∂𝐒[cond_var_idx, 1:end-T.nExo] -= invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i])[end-T.nExo+1:end] * state[i-1][st]'

        ∂𝐒[st,:] -= vcat(state[i-1][st], x[i-1])' .* (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i])

        ∂𝐒[cond_var_idx, end-T.nExo+1:end] += invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i])[end-T.nExo+1:end] * x[i-1]'

    end
    # ∂𝐒∂shock²
    # v = (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])
    
    # ∂𝐒[cond_var_idx,end-T.nExo+1:end] -= ∂v * x[i]'# - (v - jac * x[i]) * ∂v' * invjac' - invjac' * x[i] * (∂x' - ∂v' * jac)

    ### state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)

    # if i < size(data_in_deviations,2)
    #     ∂𝐒 += ∂state * vcat(state[i][T.past_not_future_and_mixed_idx], x[i+1])'
    # end

    # ∂state[T.past_not_future_and_mixed_idx] += 𝐒[:,1:end-T.nExo]' * ∂state

    # ∂x += 𝐒[:,end-T.nExo+1:end]' * ∂state

    ### x = ∂𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx]))


    # ∂𝐒[cond_var_idx,1:end-T.nExo] -= invjac' * ∂x * state[i][T.past_not_future_and_mixed_idx]'

    # ∂state[T.past_not_future_and_mixed_idx] -= 𝐒[cond_var_idx,1:end-T.nExo]' * invjac' * ∂x
end



res = FiniteDiff.finite_difference_gradient(𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:2 # axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, 𝐒)#_in_deviations[:,1:2])

isapprox(res, ∂𝐒, rtol = eps(Float32))

res - ∂𝐒


FiniteDiff.finite_difference_gradient(𝐒exo -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])), 𝐒exo)# + ∂v


# there are multiple parts to it. first the effect of the previous iteration through this one and then the direct effect

uuuu = (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])

uuu = vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ uuuu)

uu = 𝐒[st,:] * uuu

u = data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * uu

X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ u

sum(abs2, X)


sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))))


Zygote.jacobian(x -> vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.jacobian(x ->  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]


Zygote.gradient(x -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]


Zygote.gradient(x -> sum(abs2,  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1] + Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]



Zygote.gradient(x -> sum(abs2,  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]



∂𝐒 = zero(𝐒)

i = 1
∂x∂shocks² = 2 * x[i]

∂u∂x = invjac'

∂𝐒∂shocks² = ∂u∂x * ∂x∂shocks² * x[i]' # [cond_var_idx, end-T.nExo+1:end]

∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i] * x[i]' # [cond_var_idx, end-T.nExo+1:end]

∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i+1] * x[i+1]' # [cond_var_idx, end-T.nExo+1:end]


# next S
∂x∂shocks² = 2 * x[i+1]

∂𝐒∂u = - (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))'

∂𝐒2∂shocks² = ∂𝐒∂u .* (∂u∂x * ∂x∂shocks²)

∂𝐒2∂shocks² = - (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))' .* (∂u∂x * ∂x∂shocks²)

∂𝐒[cond_var_idx, 1:end-T.nExo] -= (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))' .* (invjac' * 2 * x[i+1])


# next S
∂uu∂u = - 𝐒[cond_var_idx, 1:end-T.nExo]'

∂𝐒∂uu = uuu'

∂𝐒2∂shocks² = ∂𝐒∂uu .* (∂uu∂u * ∂u∂x * ∂x∂shocks²)

∂𝐒[st,:] += (vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))' .* (- 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])


# next S
i = 1
∂x∂shocks² = 2 * x[i+1]

∂u∂x = invjac'

∂uu∂u = 𝐒[cond_var_idx, 1:end-T.nExo]'

∂uuu∂uu = 𝐒[st,:]'

∂𝐒∂shocks² = invjac' * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * x[i]'

∂𝐒[cond_var_idx, end-T.nExo+1:end] += invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])[end-T.nExo+1:end] * x[i]'



# next S
i = 1
∂x∂shocks² = 2 * x[i+1]

∂u∂x = invjac'

∂uu∂u = 𝐒[cond_var_idx, 1:end-T.nExo]'

∂uuu∂uu = 𝐒[st,:]'

∂uuuu∂uuu = invjac'

∂𝐒∂uuuu = - state[i][st]'

∂uuuu∂uuu * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * ∂𝐒∂uuuu

# ∂𝐒[cond_var_idx, 1:end-T.nExo] += ∂uuuu∂uuu * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * ∂𝐒∂uuuu

∂𝐒[cond_var_idx, 1:end-T.nExo] -= invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])[end-T.nExo+1:end] * state[i][st]'




# u = 

ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])


ForwardDiff.jacobian(XX -> -XX * uu, 𝐒[cond_var_idx, 1:end-T.nExo][:,:])

ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])

Zygote.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])[1]


Zygote.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ XX), (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * uu))[1]


∂data_in_deviations∂x[:,i] -= invjac' * (𝐒[T.past_not_future_and_mixed_idx,:]' * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) * 𝐒endo' * invjac' * 2 * x[i+1])[end-T.nExo+1:end]



FiniteDiff.finite_difference_jacobian(x -> data_in_deviations[:,i+1] - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])



ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - XX * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))))), 𝐒[cond_var_idx, 1:end-T.nExo])

ForwardDiff.jacobian(x -> data_in_deviations[:,i+1] - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])

xxx = ForwardDiff.jacobian(x -> - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])


sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒 * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))


# starting with the iterated indirect effect



FiniteDiff.finite_difference_gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒 * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st])), 𝐒)# + ∂v

FiniteDiff.finite_difference_gradient(𝐒exo2 -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo2 \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st])), 𝐒exo)# + ∂v

invjac' * (𝐒[T.past_not_future_and_mixed_idx,:]' * 𝐒endo'  * invjac' * ∂𝐒[cond_var_idx,end-T.nExo+1:end]')[end-T.nExo+1:end,:]

res = FiniteDiff.finite_difference_gradient(𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:2 # axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, 𝐒)#_in_deviations[:,1:2])

isapprox(res, ∂𝐒, rtol = eps(Float32))

res - ∂𝐒

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:10 # axes(data_in_deviations,2)
        X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data_in_deviations[:,1:10])


∂state *= 0

∂x = 2*x[1]

∂state[T.past_not_future_and_mixed_idx] += 𝐒[:,1:end-T.nExo]' * ∂state

∂x += 𝐒[:,end-T.nExo+1:end]' * ∂state

∂v = invjac' * ∂x

∂data_in_deviations∂x[:,1] = ∂v


∂data_in_deviations = zero(data_in_deviations)

∂shocks² = 2*x[1]

# x = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])



# x = sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st]))

i = 1
st = T.past_not_future_and_mixed_idx

∂x∂shocks² = 2 * x[1]

∂v∂x = invjac'

∂v∂shocks² = ∂v∂x * ∂x∂shocks²

vsub = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])

vvv = vcat(state[i][st], vsub)

vvv = vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st]))

vv =  (𝐒[st,:] * vvv)

v = (data_in_deviations[:,i+1] - 𝐒endo * vv)

∂vv∂v = - 𝐒endo'

∂vv∂shocks² = ∂vv∂v * ∂v∂x * ∂x∂shocks²

∂vvv∂vv = 𝐒[st,:]'

∂vvv∂shocks² = ∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²

∂vsubu∂vvv = ℒ.I(size(𝐒,2))[:,end-T.nExo+1:end]'

∂vsub∂shocks² = ∂vsubu∂vvv * ∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²
∂vsub∂shocks² = (∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²)[end-T.nExo+1:end]

∂dat∂shocks² = invjac' * (∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²)[end-T.nExo+1:end]

∂dat∂shocks² = -invjac' * (𝐒[st,:]' * 𝐒endo' * invjac' * 2 * x[2])[end-T.nExo+1:end]

∂dat∂shocks² = -𝐒exo' \ (𝐒[st,:]' * 𝐒endo' / 𝐒exo' * 2 * x[2])[end-T.nExo+1:end]

∂dat∂shocks² = -𝐒exo' \ (2 * x[1])


# ∂x∂v = 

# shocks² = sum(abs2, 𝐒exo \ v)

invjac' * 𝐒[st,1:end-T.nExo] * 𝐒endo' * invjac' * 2 * x[1]

∂shocks² = 2 * (𝐒exo \ v)' * ∂shocks²
∂v = (∂shocks² / shocks²) * (𝐒exo \ v)'

# x = sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st]))
# i = 1

FiniteDiff.finite_difference_gradient(x -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (x - 𝐒endo * state[i][st])))[st])), data_in_deviations[:,i])# + ∂v

FiniteDiff.finite_difference_gradient(x -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * 𝐒[st,:] * vcat(state[i][st], 𝐒exo \ (x - 𝐒endo * state[i][st])))), data_in_deviations[:,i])# + ∂v


2 * (𝐒exo \ (data_in_deviations[:, 2] - 𝐒endo * (𝐒 * vcat(state[1][st], 𝐒exo \ (data_in_deviations[:, 1] - 𝐒endo * state[1][st])))[st]))

2 * (𝐒exo \ (data_in_deviations[:, 2] - 𝐒endo * (𝐒 * vcat(state[1][st], 𝐒exo \ (data_in_deviations[:, 1] - 𝐒endo * state[1][st])))[st])) * (-invjac * 𝐒endo[:, st] * 𝐒[cond_var_idx,:]' * invjac')

-2 * x[1] * 𝐒endo * invjac * 𝐒
X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], x))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * state[i+1][st])

∂data_in_deviations[:,1] = invjac' * ∂shocks²

# ∂state[end-T.nExo+1:end] = 

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:10 # axes(data_in_deviations,2)
        X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data_in_deviations[:,1:10])


data = copy(data_in_deviations);
data[:,2:3] *= 0;

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data)#_in_deviations[:,1:2])



i = 1


st = T.past_not_future_and_mixed_idx

# data has an impact because of the difference between today and tomorrow, as in the data matters for two periods
# fisrt period:
X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])

# second period:
state[i+1] = 𝐒 * vcat(state[i][st], X)

# here it matters because it is part of X -> state and thereby pushes around the deterministic part of the system (via the state)
# X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])
X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], X))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * state[i+1][st])

1

# state::Vector{Vector{Float64}}, 
#                                                     𝐒::Union{Matrix{Float64}, Vector{AbstractMatrix{Float64}}}, 
#                                                     data_in_deviations::Matrix{Float64}, 
#                                                     observables::Union{Vector{String}, Vector{Symbol}},
#                                                     T::timings; 
#                                                     warmup_iterations::Int = 0,
#                                                     presample_periods::Int = 0)


# function first_order_state_update(state::Vector{U}, shock::Vector{S}) where {U <: Real,S <: Real}
# # state_update = function(state::Vector{T}, shock::Vector{S}) where {T <: Real,S <: Real}
#     aug_state = [state[T.past_not_future_and_mixed_idx]
#                 shock]
#     return 𝐒 * aug_state # you need a return statement for forwarddiff to work
# end

# state_update = first_order_state_update

# state = state[1]

# pruning = false

    
# precision_factor = 1.0

# n_obs = size(data_in_deviations,2)

# cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))



# warmup_iterations = 3
# state *=0

# data_in_deviations[:,1] - (𝐒 * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[4:6]))[cond_var_idx]

# data_in_deviations[:,1] - (𝐒 * (vcat((𝐒[T.past_not_future_and_mixed_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3]))), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))[cond_var_idx]


# data_in_deviations[:,1] - (𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))


# 𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])))

# 𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])


# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * (𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]) + 𝐒[cond_var_idx,end-T.nExo+1:end] * x[4:6] - data_in_deviations[:,1]




# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3] +
# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[4:6] +
# 𝐒[cond_var_idx,end-T.nExo+1:end] * x[7:9] -
# data_in_deviations[:,1]

# hcat(   
#     𝐒[cond_var_idx,end-T.nExo+1:end] , 
#     𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], 
#     𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
#     ) \ data_in_deviations[:,1]


# warmup_iterations = 5

# state *= 0
# logabsdets = 0
# shocks² = 0

# if warmup_iterations > 0
#     if warmup_iterations >= 1
#         to_be_inverted = 𝐒[cond_var_idx,end-T.nExo+1:end]
#         if warmup_iterations >= 2
#             to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
#             if warmup_iterations >= 3
#                 Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
#                 for e in 1:warmup_iterations-2
#                     to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
#                     Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
#                 end
#             end
#         end
#     end

#     x = to_be_inverted \ data_in_deviations[:,1]

#     warmup_shocks = reshape(x, T.nExo, warmup_iterations)

#     for i in 1:warmup_iterations-1
#         state = state_update(state, warmup_shocks[:,i])
#     end

#     jacc = -to_be_inverted'

#     for i in 1:warmup_iterations
#         if T.nExo == length(observables)
#             logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
#         else
#             logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
#         end
#     end

#     shocks² += sum(abs2,x)
# end



# data_in_deviations[:,1] - 𝐒[cond_var_idx,:] * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])



# data_in_deviations[:,1]

# (𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[1:3]))[cond_var_idx]


#         x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

#         ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))





# state_copy = deepcopy(state)

# XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

# for i in 1:warmup_iters
#     state_copy = state_update(state_copy, XX[:,i])
# end

# return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])



# shocks² = 0.0
# logabsdets = 0.0

# if warmup_iterations > 0
#     res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
#                         zeros(T.nExo * warmup_iterations), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
#                         Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                         autodiff = :forward)

#     matched = Optim.minimum(res) < 1e-12

#     if !matched # for robustness try other linesearch
#         res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
#                         zeros(T.nExo * warmup_iterations), 
#                         Optim.LBFGS(), 
#                         Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                         autodiff = :forward)
    
#         matched = Optim.minimum(res) < 1e-12
#     end

#     if !matched return -Inf end

#     x = Optim.minimizer(res)

#     warmup_shocks = reshape(x, T.nExo, warmup_iterations)

#     for i in 1:warmup_iterations-1
#         state = state_update(state, warmup_shocks[:,i])
#     end
    
#     res = zeros(0)

#     jacc = zeros(T.nExo * warmup_iterations, length(observables))

#     match_initial_data!(res, x, jacc, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor), zeros(size(data_in_deviations, 1))

#     for i in 1:warmup_iterations
#         if T.nExo == length(observables)
#             logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
#         else
#             logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
#         end
#     end

#     shocks² += sum(abs2,x)
# end



# jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

# jacdecomp = ℒ.svd(jac)
# invjac = inv(jacdecomp)

# using FiniteDiff
# FiniteDiff.finite_difference_jacobian(xx -> sum(x -> log(abs(x)), ℒ.svdvals(xx)),𝐒[cond_var_idx,end-T.nExo+1:end])

# ForwardDiff.jacobian(xx -> sum(x -> log(abs(x)), ℒ.svdvals(xx)),𝐒[cond_var_idx,end-T.nExo+1:end])


# ForwardDiff.gradient(x-> x'*x,[1,2,3])
# [1,2,3]'*[1,2,3]


# ∂det = -inv(ℒ.svd(𝐒[cond_var_idx,end-T.nExo+1:end]))






state = zeros(T.nVars)
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
    x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

    if i > presample_periods
        shocks² += sum(abs2,x)
    end

    state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
end
shocks²

-(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2




inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]))

inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed]))' * inv( ℒ.svd( 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]) )'

𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * inv( ℒ.svd( 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]) )'

inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed]))' *  𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]


inv(ℒ.svd(𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end])) * inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed])) 

FiniteDiff.finite_difference_gradient(x->begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    X = zeros(eltype(x),T.nExo)

    for i in 1:2#xes(data_in_deviations,2)
        X = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (x[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,X)
        end

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)
    end
    return shocks²
end, data_in_deviations[:,1:2])



ForwardDiff.gradient(x->sum(abs2, x[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - x[cond_var_idx,1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])), 𝐒)

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (x - 𝐒[cond_var_idx,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])), data_in_deviations[:,i])

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - x * state[i][T.past_not_future_and_mixed_idx])), 𝐒[cond_var_idx,1:end-T.nExo])

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * x)), state[i][T.past_not_future_and_mixed_idx])





(jac' * jac)' \ jac'

invjac' * jac' * invjac'
# ∂jac =  (jac)' * (jac *  ∂x)

FiniteDiff.finite_difference_jacobian(x->sum(abs2,x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac)

ForwardDiff.gradient(x->sum(abs2, x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac)

@profview for i in 1:10000 Zygote.gradient(x->sum(abs2, x \ (data_in_deviations[:,1] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac) end

vec(jac) * vec(jac)'
ForwardDiff.gradient(x->ℒ.det(inv(x)),jac[:,1:2])

(jac[:,1:2]) * (jac[:,1:2])'


vec(jac[:,1:2]) * vec(jac[:,1:2])'

∂data_in_deviations∂x = invjac' * ∂x

∂𝐒[cond_var_idx,1:end-T.nExo] = -invjac' * ∂x * state[T.past_not_future_and_mixed_idx]'

∂state[T.past_not_future_and_mixed_idx] = -𝐒[cond_var_idx,1:end-T.nExo]' * invjac' * ∂x

# state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
∂𝐒 += ∂state * vcat(state[T.past_not_future_and_mixed_idx], x)'

∂state[T.past_not_future_and_mixed_idx] += 𝐒[:,1:end-T.nExo]' * ∂state





if i > presample_periods
    shocks² += sum(abs2,x)
end

state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)



ForwardDiff.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), jac)

ForwardDiff.gradient(x -> sum(abs2, x * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), invjac)

ForwardDiff.gradient(x -> sum(abs2, invjac * (x - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), data_in_deviations[:,i])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - x * state[T.past_not_future_and_mixed_idx])), 𝐒[cond_var_idx,1:end-T.nExo])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * x)), state[T.past_not_future_and_mixed_idx])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - x)), 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

ForwardDiff.gradient(x -> sum(abs2, invjac * x), data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])


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