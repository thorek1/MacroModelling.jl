using Revise
using MacroModelling
# using StatsPlots
using Random
import Optim, LineSearches
using BenchmarkTools
import LinearAlgebra as ℒ
import Random
import FiniteDifferences
import Zygote
import ForwardDiff
import CSV
using DataFrames

include("./models/Smets_Wouters_2007_linear.jl")

# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
sample_idx = 47:230 # 1960Q1-2004Q4

data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)



get_loglikelihood(Smets_Wouters_2007_linear, data, Smets_Wouters_2007_linear.parameter_values)

get_loglikelihood(Smets_Wouters_2007_linear, data, Smets_Wouters_2007_linear.parameter_values, filter = :inversion)

kalman_zyg = Zygote.gradient(x->get_loglikelihood(Smets_Wouters_2007_linear, data,x), Smets_Wouters_2007_linear.parameter_values)[1]

kalman_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x->get_loglikelihood(Smets_Wouters_2007_linear, data,x), Smets_Wouters_2007_linear.parameter_values)[1]
maximum(abs,kalman_zyg - kalman_fin)
isapprox(kalman_zyg, kalman_fin, rtol = 1e-6)

inversion_zyg = Zygote.gradient(x->get_loglikelihood(Smets_Wouters_2007_linear, data, x, filter = :inversion), Smets_Wouters_2007_linear.parameter_values)[1]

inversion_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x->get_loglikelihood(Smets_Wouters_2007_linear, data, x, filter = :inversion), Smets_Wouters_2007_linear.parameter_values)[1]

isapprox(inversion_zyg, inversion_fin, rtol = eps(Float32))

inversion_zyg - inversion_fin



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

get_loglikelihood(Gali_2015_chapter_3_nonlinear, data, Gali_2015_chapter_3_nonlinear.parameter_values, filter = :inversion)

kalman_zyg = Zygote.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

kalman_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

isapprox(kalman_zyg, kalman_fin, rtol = eps(Float32))

inversion_zyg = Zygote.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data, x, filter = :inversion, presample_periods = 30), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

inversion_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data, x, filter = :inversion, presample_periods = 30), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

isapprox(inversion_zyg, inversion_fin, rtol = eps(Float32))
inversion_fin
inversion_zyg

FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)[1]


ForwardDiff.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data,x), Gali_2015_chapter_3_nonlinear.parameter_values)




inversion_zyg = Zygote.gradient(x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data[:,1:2], x, filter = :inversion, presample_periods = 1), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

inversion_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), x->get_loglikelihood(Gali_2015_chapter_3_nonlinear, data[:,1:2], x, filter = :inversion, presample_periods = 1), Gali_2015_chapter_3_nonlinear.parameter_values)[1]

isapprox(inversion_zyg, inversion_fin, rtol = eps(Float32))










# 𝓂 = Gali_2015_chapter_3_nonlinear
𝓂 = Smets_Wouters_2007_linear

import MacroModelling: get_and_check_observables, check_bounds, minimize_distance_to_initial_data, get_relevant_steady_state_and_state_update, replace_indices, minimize_distance_to_data, match_data_sequence!, match_initial_data!,calculate_loglikelihood

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

presample_periods = 0


get_loglikelihood(𝓂, data, 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods)


Zygote.gradient(𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


Zygote.gradient(data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations)[1]

res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations)[1]


Zygote.gradient(state -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), state)[1][1][TT.past_not_future_and_mixed_idx]

FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), stt -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, stt, warmup_iterations), state)[1][1][TT.past_not_future_and_mixed_idx]



zygS = Zygote.gradient(𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


finS = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


isapprox(zygS, finS, rtol = eps(Float32))

T = 𝓂.timings

precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))




warmup_iterations = 0



state = copy(state[1])

precision_factor = 1.0

n_obs = size(data_in_deviations,2)

obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

t⁻ = T.past_not_future_and_mixed_idx

shocks² = 0.0
logabsdets = 0.0

if warmup_iterations > 0
    if warmup_iterations >= 1
        jac = 𝐒[obs_idx,end-T.nExo+1:end]
        if warmup_iterations >= 2
            jac = hcat(𝐒[obs_idx,1:T.nPast_not_future_and_mixed] * 𝐒[t⁻,end-T.nExo+1:end], jac)
            if warmup_iterations >= 3
                Sᵉ = 𝐒[t⁻,1:T.nPast_not_future_and_mixed]
                for _ in 1:warmup_iterations-2
                    jac = hcat(𝐒[obs_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[t⁻,end-T.nExo+1:end], jac)
                    Sᵉ *= 𝐒[t⁻,1:T.nPast_not_future_and_mixed]
                end
            end
        end
    end

    jacdecomp = ℒ.svd(jac)


    x = jacdecomp \ data_in_deviations[:,1]

    warmup_shocks = reshape(x, T.nExo, warmup_iterations)

    for i in 1:warmup_iterations-1
        ℒ.mul!(state, 𝐒, vcat(state[t⁻], warmup_shocks[:,i]))
        # state = state_update(state, warmup_shocks[:,i])
    end

    for i in 1:warmup_iterations
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor))
        end
    end

    shocks² += sum(abs2,x)
end



state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]
shocks² = 0.0
logabsdets = 0.0
y = zeros(length(obs_idx))
x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

jac = 𝐒[obs_idx,end-T.nExo+1:end]

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

@views 𝐒obs = 𝐒[obs_idx,1:end-T.nExo]

for i in axes(data_in_deviations,2)
    @views ℒ.mul!(y, 𝐒obs, state[i][t⁻])
    @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
    ℒ.mul!(x[i],invjac,y)
    
    # x = invjac * (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])
    # x = 𝐒[obs_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])

    if i > presample_periods
        shocks² += sum(abs2,x[i])
    end

    # # copyto!(state_reduced, 1, state, t⁻)
    # for (i,v) in enumerate(t⁻)
    #     state_reduced[i] = state[v]
    # end
    # copyto!(state_reduced, T.nPast_not_future_and_mixed + 1, x, 1, T.nExo)
    
    ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
    # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    # state = state_update(state, x)
end

llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2






obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

t⁻ = T.past_not_future_and_mixed_idx

# precomputed matrices
M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹
M3  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]'
M⁴  = M² * M¹




∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

for t in reverse(axes(data_in_deviations,2))
    ∂data_in_deviations[:,t]        -= invjac' * x[t]

    if t > 1
        ∂data[:,t:end]              .= M² * ∂data[:,t:end]
        
        ∂data[:,t-1]                += M¹ * x[t]

        ∂data_in_deviations[:,t-1]  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)
    end
end 

∂data_in_deviations

maximum(abs, ∂data_in_deviations - res)

isapprox(∂data_in_deviations, res, rtol = eps(Float32))

# invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data * ones(size(data_in_deviations,2))


∂data_in_deviations[:,5] -= invjac' * x[5]

∂data_in_deviations[:,4] -= invjac' * x[4]
∂data_in_deviations[:,4] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[5]

∂data_in_deviations[:,3] -= invjac' * x[3]
∂data_in_deviations[:,3] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[4]
∂data_in_deviations[:,3] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[5]

∂data_in_deviations[:,2] -= invjac' * x[2]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[3]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[4]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M¹ * x[5]

∂data_in_deviations[:,1] -= invjac' * x[1]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[2]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[3]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M¹ * x[4]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M² * M¹ * x[5]
res3
for t in 3:-1:1 # reverse(axes(data_in_deviations,2))
    ∂data_in_deviations[:,t] -= invjac' * x[t]

    if t > 1
        ∂data_in_deviations[:,t-1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[t]
    end

    if t > 2
        ∂data_in_deviations[:,t-2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[t]

        # ∂data[:,t-2]    += M¹ * x[t]
        # ∂data = M² * ∂data
        # ∂data_in_deviations[:,1:end-1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,2:end]
    end
    
    if t > 3
        ∂data[:,t-3]    += M² * M¹ * x[t]
        # ∂data¹[:,t-3]   += M² * 𝐒[t⁻,1:end-T.nExo]' * M¹ * x[t]

        # ∂data²[:,t-3]   -= M² * 𝐒[obs_idx, 1:end-T.nExo]' * M³ * x[t]

        ∂data = M² * ∂data

        # ∂data¹ = M² * ∂data¹

        # ∂data² = M² * ∂data²

        ∂data_in_deviations += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data
    end
end

∂data_in_deviations

(𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
M²^2 * M¹ * x[4]


# -2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
# -2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')^3  * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[5]
(𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')^2 

invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (M² * (∂data¹ + ∂data²) + (∂data¹ + ∂data²))

∂data_in_deviations

res5 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:5])[1]

res4 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:4])[1]

res3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:3])[1]

res2 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:2])[1]

res5[:,1:4] - res4
res4[:,1:3] - res3

invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]'

∂data¹ + ∂data²



# i = 4

-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# i = 3
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]


# i = 2
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]


# i = 1
2 * invjac' * x[1]



invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]


invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]


invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

- invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] 



invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+ 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+ 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2]

invjac' * x[1]




M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹
M⁴  = M² * M¹


N = 2

∂𝐒 = zero(𝐒)
    
∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

∂state = zero(state[1])

for t in N:-1:1 # reverse(axes(data_in_deviations,2))
    ∂state[t⁻]                                  .= M² * ∂state[t⁻]

    if t > presample_periods
        ∂state[t⁻]                              += M¹ * x[t]

        ∂data_in_deviations[:,t]                -= invjac' * x[t]

        ∂𝐒[obs_idx, end-T.nExo + 1:end]         += invjac' * x[t] * x[t]'

        if t > 1
            ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
            
            ∂data[:,t-1]                        += M¹ * x[t]
    
            ∂data_in_deviations[:,t-1]          += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

            ∂𝐒[obs_idx, 1:end-T.nExo]           += invjac' * x[t] * state[t][t⁻]'
            ∂𝐒[obs_idx, end-T.nExo + 1:end]     -= M³ * x[t] * x[t-1]'
            ∂𝐒[t⁻,end-T.nExo + 1:end]           += M¹ * x[t] * x[t-1]'
        end

        if t > 2
            ∂𝐒[t⁻,1:end-T.nExo]                 += M¹ * x[t] * state[t-1][t⁻]'
            ∂𝐒[obs_idx, 1:end-T.nExo]           -= M³ * x[t] * state[t-1][t⁻]'
        end
    end

    if t > 2
        ∂𝐒ᵗ⁻        .= 𝐒[t⁻,1:end-T.nExo]' * ∂𝐒ᵗ⁻ / vcat(state[t-1][t⁻], x[t-1])' * vcat(state[t-2][t⁻], x[t-2])'
        
        if t > presample_periods
            ∂𝐒ᵗ⁻    += M⁴ * x[t] * vcat(state[t-2][t⁻], x[t-2])'
        end

        ∂𝐒[t⁻,:]    += ∂𝐒ᵗ⁻
    end
end

∂𝐒[obs_idx,end-T.nExo+1:end] -= (N - presample_periods) * invjac' / 2


res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:N], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

maximum(abs, ∂𝐒 - res)

∂𝐒 - res

finS - ∂𝐒

data = data_in_deviations[:,1:N]

res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:N # axes(data,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        if i > presample_periods
            shocks² += sum(abs2,X)
        end
        # shocks² += sum(abs2,state[i+1])
    end

    return -shocks²/2
end, 𝐒)[1]





# derivatives wrt to s

obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

𝐒¹ = 𝐒[obs_idx, end-T.nExo+1:end]
𝐒² = 𝐒[obs_idx, 1:end-T.nExo]
𝐒³ = 𝐒[t⁻,:]



res4 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:4], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:3], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res2 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res1 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:1], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res3 - res2
res2 - res1


hcat(𝐒[obs_idx, 1:end-T.nExo]' * invjac', 𝐒[obs_idx, 1:end-T.nExo]' * invjac') * vcat(x[3] * vcat(state[1][t⁻], x[1])', x[3] * vcat(state[2][t⁻], x[2])')


iterator = 

# t = 1
# ∂𝐒[obs_idx, :]                  += invjac' * x[1] * vcat(state[1][t⁻], x[1])'

# t = 2
# ∂𝐒[obs_idx, :]                  += invjac' * x[2] * vcat(state[2][t⁻], x[2])'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2] * vcat(state[1][t⁻], x[1])'
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2] * vcat(state[1][t⁻], x[1])'

# t = 3
# ∂𝐒[obs_idx, :]                  += invjac' * x[3] * vcat(state[3][t⁻], x[3])'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' - 𝐒[t⁻,1:end-T.nExo]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[1][t⁻], x[1])'

∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                        += (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[1][t⁻], x[1])'




N = size(data_in_deviations,2)

∂𝐒 = zero(𝐒)
    
∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

∂Stmp = [M¹ for _ in 1:size(data_in_deviations,2)]

for t in 2:size(data_in_deviations,2)
    ∂Stmp[t] = M² * ∂Stmp[t-1]
end

∂state = zero(state[1])

for t in reverse(axes(data_in_deviations,2))
    if t > presample_periods
        ∂𝐒[obs_idx, :]         += invjac' * x[t] * vcat(state[t][t⁻], x[t])'

        if t > 1
            ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
            
            ∂data[:,t-1]                        += M¹ * x[t]
    
            ∂data_in_deviations[:,t-1]          += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

            M²mult = ℒ.I(size(M²,1))

            for tt in t-1:-1:1
                ∂𝐒[obs_idx, :]                      -= invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂Stmp[t-tt] * x[t] * vcat(state[tt][t⁻], x[tt])'
    
                ∂𝐒[t⁻,:]                            += ∂Stmp[t-tt] * x[t] * vcat(state[tt][t⁻], x[tt])'

                M²mult                              *= M²
            end

        end
    end
end

∂𝐒[obs_idx,end-T.nExo+1:end] -= (N - presample_periods) * invjac' / 2





NN = 3

res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:N], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

maximum(abs, ∂𝐒 - res)


∂𝐒 = zero(𝐒)

∂𝐒[obs_idx,end-T.nExo+1:end] -= (NN - presample_periods) * invjac' / 2

i = 1

t = 1
# ForwardDiff.gradient(𝐒¹ -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹)
∂𝐒[obs_idx, end-T.nExo+1:end]       += invjac' * x[t] * x[t]'

# ForwardDiff.gradient(𝐒² -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒²)
# zero because the initial state is 0


t = 2
# ForwardDiff.gradient(x -> -.5 * sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]       += invjac' * x[t] * x[t]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - x * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒²)
# invjac' * x[t] * state[t][t⁻]'

∂𝐒[obs_idx, :]                  += invjac' * x[t] * vcat(state[t][t⁻], x[t])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒¹)
∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-1]'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒³)
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'



t = 3

# tmpres = ForwardDiff.gradient(𝐒 -> -.5 * sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[t⁻,:] * vcat(𝐒[t⁻,:] * vcat(state[i][t⁻], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][t⁻])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[t⁻,:] * vcat(state[i][t⁻], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][t⁻])))))), 𝐒)


# ForwardDiff.gradient(x -> -.5 * sum(abs2, x \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - x * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)

∂𝐒[obs_idx, :]                  += invjac' * x[t] * vcat(state[t][t⁻], x[t])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), x \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-1]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - x * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-1][t⁻]'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-2]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - x * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-2][t⁻]'

# ∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-2]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - x * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-2][t⁻]'

# ∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' - 𝐒[t⁻,1:end-T.nExo]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * x * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * x * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'

# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ∂𝐒[t⁻,:]                        += 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'

# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ∂𝐒[t⁻,:]                        += -𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'
∂𝐒[t⁻,:]                        += (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'

# res3-res2

maximum(abs, ∂𝐒 - tmpres)
maximum(abs, ∂𝐒 - res)


# for i in axes(data_in_deviations,2)
#     x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])
#     state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], x[i])
#     shocks² += sum(abs2,x[i])
# end
# return shocks²

st = T.past_not_future_and_mixed_idx
𝐒endo = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒exo = 𝐒[cond_var_idx, end-T.nExo+1:end]


# ∂state = zero(state[1])

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

∂data_in_deviations = zero(data_in_deviations)

∂state¹ = zero(state[1][st])


for i in reverse(axes(data_in_deviations,2))
    ∂state¹ .= (𝐒[st,1:end-T.nExo] - 𝐒[st,end-T.nExo+1:end] * invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * ∂state¹
    ∂state¹ -= (invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * 2 * x[i]

    if i < size(data_in_deviations,2)
        ∂data_in_deviations[:,i] -= invjac' * ((invjac * 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:])' * 2 * x[i+1])[end-T.nExo+1:end]
    end
    ∂data_in_deviations[:,i] += invjac' * 2 * x[i]

    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                .= 𝐒[st,1:end-T.nExo]' * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]') * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end



T = TT
cond_var_idx = indexin(observables,sort(union(TT.aux,TT.var,TT.exo_present)))
res = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    state[1] .= stat
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[1]#_in_deviations[:,1:2])


isapprox(res, ∂data_in_deviations, rtol = eps(Float32))
isapprox(res, ∂𝐒, rtol = eps(Float32))

res - ∂𝐒

i = 1

𝐒¹ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒² = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒³ = 𝐒[st,:]
sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))
sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))




state[i+1] = 𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))

state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))


state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   
+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]   

+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   
+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))



𝐒[:,1:end-T.nExo] * 𝐒[st,1:end-T.nExo] * state[i][st]   
+  𝐒[:,1:end-T.nExo] * 𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) 

+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[st,1:end-T.nExo] * state[i][st]   
+   𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))




# res = FiniteDiff.finite_difference_gradient(stat -> begin
ForwardDiff.gradient(stat->begin
shocks² = 0.0
# stat = zero(state[1])
for i in 1:2 # axes(data_in_deviations,2)
    stat = 𝐒[:,1:end-T.nExo] * stat[T.past_not_future_and_mixed_idx] + 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * stat[T.past_not_future_and_mixed_idx]))

    # shocks² += sum(abs2,X)
    shocks² += sum(abs2,stat)
end

return shocks²
end, state[1])[st]#_in_deviations[:,1:2])


# i = 4
ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# i = 3
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), data_in_deviations[:,i])

-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))), data_in_deviations[:,i])

2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

# i = 2
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]

# i = 1
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (x - 𝐒² * state[i][st])), data_in_deviations[:,i])
2 * invjac'  * x[1]





(ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))), data_in_deviations[:,i]) + ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), data_in_deviations[:,i]) + ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (x - 𝐒² * state[i][st])), data_in_deviations[:,i])) / 2




xxx =  𝐒[st,1:end-T.nExo]' * 𝐒[:,1:end-T.nExo]' * 2 * state[i+2]
xxx += 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end - T.nExo + 1:end]' * 𝐒[:,1:end-T.nExo]' * 2 * state[i+2]
xxx -= 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+2]
xxx -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end - T.nExo + 1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+2]


xxx +=  𝐒[:,1:end-T.nExo]' * 2 * state[i+1]
xxx -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+1]


xxx * ∂state∂shocks²
# 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[st,1:end-T.nExo] * state[i][st]   +   𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))




state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st])[st] 

∂state∂X * ∂X∂stt * ∂state∂state * ∂state∂shocks²



∂state = zero(state[i])

∂state∂shocks² = 2 * state[i+1]#[st]


∂state∂state = 𝐒[:,1:end-T.nExo]'# * ∂state∂shocks²

∂state∂state -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end-T.nExo+1:end]'

∂state[st] += ∂state∂state * ∂state∂shocks²

∂state[st] += ∂state∂state * ∂state

∂state∂state * ∂state∂shocks²


∂state∂state * 2 * state[i+1] + ∂stt∂stt * ∂state∂state * 2 * state[i+2]
∂state∂shocks² += 2 * state[i+2]#[st]

out = zero(state[i+2][st])

for i in 2:-1:1
    out .= ∂stt∂stt * out
    out += (∂state∂state * 2 * state[i+1])
end


𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]


out = zero(state[i][st])

for i in 1:-1:1
    out .= 𝐒[st,1:end-T.nExo]' * out
    out -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]
end
out



∂state = zero(state[i][st])

for i in reverse(axes(data_in_deviations,2))
    # out .= (𝐒[st,1:end-T.nExo]' - 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]') * out
    ∂state .= (𝐒[st,1:end-T.nExo] - 𝐒[st,end-T.nExo+1:end] * invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * ∂state
    ∂state -= (invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * 2 * x[i]
end

∂state



∂data_in_deviations = zero(data_in_deviations)

for i in reverse(axes(data_in_deviations,2))
    if i < size(data_in_deviations,2)
        ∂data_in_deviations[:,i] -= invjac' * ((invjac * 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:])' * 2 * x[i+1])[end-T.nExo+1:end]
    end
    ∂data_in_deviations[:,i] += invjac' * 2 * x[i]
end



𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[st,1:end-T.nExo]'
𝐒³̂ = 𝐒[st,end-T.nExo+1:end]'

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

for i in reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                 = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end




∂𝐒 = zero(𝐒)

for i in 2:-1:1#reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i] * x[i]' # [cond_var_idx, end-T.nExo+1:end]

    if i > 1
        ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (𝐒[st,:] * vcat(state[i-1][st], x[i-1]))' .* (invjac' * 2 * x[i])
        ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * x[i-1]'
        
        ∂𝐒[st,:] -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'

        # ∂𝐒[cond_var_idx, :] += invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i])[end-T.nExo+1:end] * vcat(-state[i-1][st], x[i-1])'
    end
end

maximum(abs, (res - ∂𝐒) ./ res)

unique((res - ∂𝐒) ./ ∂𝐒) .|> abs |> sort

(𝐒[st,:] * vcat(state[i-1][st], x[i-1]))' .* (invjac' * 2 * x[i])




∂𝐒 = zero(𝐒)

for i in 3:-1:1#reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]         -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])' # [cond_var_idx, end-T.nExo+1:end]

    if i > 1
        ∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]                -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒[st,:]    += 2 * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]') * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[1][st], x[1])'
    end
end



𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[st,1:end-T.nExo]'
𝐒³̂ = 𝐒[st,end-T.nExo+1:end]'

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

for i in reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                 = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end


maximum(abs, res - ∂𝐒)

maximum(abs, Base.filter(isfinite, (res - ∂𝐒) ./ res))


maximum(abs, Base.filter(isfinite, (res5 - ∂𝐒) ./ res5))


# i = 5

∂𝐒st = zero(∂𝐒[st,:])
∂𝐒stl = zero(∂𝐒[st,:])

i = 5
# ∂𝐒st                += 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

i = 4
∂𝐒st                  = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

i = 3
∂𝐒st                  = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

∂𝐒st + ∂𝐒stl


∂𝐒 = zero(𝐒)
∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[3][st], x[3])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[2][st], x[2])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[1][st], x[1])'


∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[2][st], x[2])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[1][st], x[1])'

∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[3] * vcat(state[1][st], x[1])'

∂𝐒 - res5



∂𝐒 = zero(𝐒)


𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[t⁻,1:end-T.nExo]'
𝐒³̂ = 𝐒[t⁻,end-T.nExo+1:end]'

# terms for i = 5
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[5] * vcat(state[4][t⁻], x[4])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[3][t⁻], x[3])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[5] * vcat(state[5][t⁻], x[5])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[5] * vcat(state[4][t⁻], x[4])'




# terms for i = 4
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[4] * vcat(state[3][t⁻], x[3])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[4] * vcat(state[4][t⁻], x[4])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[4] * vcat(state[3][t⁻], x[3])'


# terms for i = 3
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[3] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[3] * vcat(state[3][t⁻], x[3])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[3] * vcat(state[2][t⁻], x[2])'


# terms for i = 2
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[2] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[2] * vcat(state[2][t⁻], x[2])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[2] * vcat(state[1][t⁻], x[1])'


# terms for i = 1
∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[1] * vcat(state[1][t⁻], x[1])'


maximum(abs, res - ∂𝐒/2)

maximum(abs, Base.filter(isfinite, (res - ∂𝐒) ./ ∂𝐒))


32
31

21


𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]'

maximum(abs, res - ∂𝐒)

∂𝐒 = zero(𝐒)

i = 1

∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= invjac' * 2 * x[i] * x[i]'


∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+1] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * vcat(state[i][st], x[i])'


∂𝐒[cond_var_idx, :] -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'

# 𝐒³
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'



using FiniteDifferences
res = FiniteDifferences.grad(central_fdm(4,1), 𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, 𝐒)[1]#_in_deviations[:,1:2])

res4 = -res+rest
res5 = res-rest

maximum(abs, res - ∂𝐒)

∂state




Xx = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,1] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])

Xx = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:] * 
vcat(
    state[1][T.past_not_future_and_mixed_idx], 
    𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,1] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])
    ))



- 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]   +  𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[2] -  𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[2] 
# 𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]


out = zero(state[i][st])

for i in reverse(axes(data_in_deviations,2))
    out .= (𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' - 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]') * out
    out -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i]
end
out


(invjac * 𝐒[cond_var_idx, 1:end-T.nExo])'



res = FiniteDiff.finite_difference_gradient(stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    state[1] = stat
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])

        state[i+1] = 𝐒 * vcat(state[i][st], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[st]#_in_deviations[:,1:2])




(∂state∂X * ∂X∂state + ∂state∂state) * ∂state∂shocks²


∂state∂state[:,st] * ∂state∂state * ∂state∂shocks²

∂stt∂stt = 𝐒[st,1:end-T.nExo]'# * ∂state∂shocks²

∂X∂state = 𝐒[:,end-T.nExo+1:end]'# * ∂state∂shocks²

∂X∂stt = 𝐒[st,end-T.nExo+1:end]'# * ∂state∂shocks²

∂state∂X = -𝐒[cond_var_idx, 1:end-T.nExo]' * invjac'

∂stt∂stt*(∂state∂X * ∂X∂state * ∂state∂shocks² + ∂state∂state * ∂state∂shocks²)

∂state∂X * ∂X∂stt * ∂state∂state * ∂state∂shocks² + ∂stt∂stt * ∂state∂state * ∂state∂shocks² + (∂state∂X * ∂X∂state * ∂state∂shocks² + ∂state∂state * ∂state∂shocks²)




∂stt∂stt * ∂state∂X * ∂X∂state * ∂state∂shocks²

∂stt∂stt * ∂state∂state * ∂state∂shocks²

res = FiniteDiff.finite_difference_gradient(stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    state[1] .= stat
    for i in 1:2 # axes(data_in_deviations,2)
        # X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        # state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        state[i+1] = 𝐒[:,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx] + 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx]))

        # shocks² += sum(abs2,X)
        shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[st]#_in_deviations[:,1:2])



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

Zygote.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒)[1]

# derivative wrt S for two periods
ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒) - ∂𝐒



ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)



res = FiniteDiff.finite_difference_gradient(𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:3 # axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, 𝐒) - ∂𝐒#_in_deviations[:,1:2])

res3 = res1-res2


st = T.past_not_future_and_mixed_idx
𝐒¹ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒² = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒³ = 𝐒[st,:]

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))
, 𝐒²)

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st]))))
, 𝐒²)


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))
, 𝐒[cond_var_idx, end-T.nExo+1:end])

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))
, 𝐒[cond_var_idx, end-T.nExo+1:end])


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)


∂𝐒 = zero(𝐒)

i = 1

# 𝐒¹
∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+2] * x[i+2]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])


# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i]'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])



∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i+1]'
Zygote.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]



# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i]'
# Zygote.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]




# 𝐒²
∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * x[i+2] * state[i+2][st]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])

# 0
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])


∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * state[i+1][st]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²)

# 0
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²) + ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))), 𝐒²)


# 𝐒³
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)


∂𝐒[st,:]                            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)




∂𝐒[st,:]                            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)



# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'


∂𝐒 = zero(𝐒)

i = 1

∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= invjac' * 2 * x[i] * x[i]'

∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+1] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * vcat(state[i][st], x[i])'



# 𝐒¹
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+2] * x[i+2]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i+1]'

# 𝐒²
# ∂𝐒[cond_var_idx, 1:end-T.nExo]      -= 2 * invjac' * x[i+2] * state[i+2][st]'
# ∂𝐒[cond_var_idx, 1:end-T.nExo]      -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * state[i+1][st]'


∂𝐒[cond_var_idx, :] -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'

# 𝐒³
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'







# i = 4
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒²)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))))), 𝐒²)



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * x[i+3] * x[i+3]'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i+1]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i+2]'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# cancels out



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# cancels out


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)




∂𝐒 = zero(𝐒)

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+2][st], x[i+2])'




# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
# ∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            += 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
# cancels out
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+2][st], x[i+2])'

∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'

∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'

∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'

∂𝐒[st,:]            += 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³) - ∂𝐒[st,:]






𝐒[st,1:end-T.nExo]'

2 * invjac' * x[i+1] * x[i+1]' * inv(ℒ.svd(x[i+1]))'

2 * invjac' * x[i+1] * x[i+1]' / x[i+1]' * x[i]'

invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * 2 * invjac' * x[i+1] * x[i+1]' * (x[i+1]' \ x[i]')


2 * invjac' * x[i+1] * x[i]'

2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]' - 2 * invjac' * x[i+1] * x[i+1]'

2 * invjac' * (𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]' - x[i+1] * x[i+1]')

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

# ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (state[i+1][st] * (invjac' * 2 * x[i+1])')'
# ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (state[i+1][st] * x[i+1]' * invjac * 2)'
∂𝐒[cond_var_idx, 1:end-T.nExo] -= invjac' * x[i+1] * state[i+1][st]' * 2

# next S
∂uu∂u = - 𝐒[cond_var_idx, 1:end-T.nExo]'

∂𝐒∂uu = uuu'

∂𝐒2∂shocks² = ∂𝐒∂uu .* (∂uu∂u * ∂u∂x * ∂x∂shocks²)

# ∂𝐒[st,:] -= (vcat(state[i][st], x[i]) * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])')'
∂𝐒[st,:] -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1] * vcat(state[i][st], x[i])'




# next S
i = 1
∂x∂shocks² = 2 * x[i+1]

∂u∂x = invjac'

∂uu∂u = 𝐒[cond_var_idx, 1:end-T.nExo]'

∂uuu∂uu = 𝐒[st,:]'

∂𝐒∂shocks² = invjac' * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * x[i]'

∂𝐒[cond_var_idx, end-T.nExo+1:end] += invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1] * x[i]'


ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒) - ∂𝐒




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