

using MacroModelling
import Turing, StatsPlots , Plots, Random
import LinearAlgebra as ℒ

@model RBC begin
	K[0] = (1 - δ) * K[-1] + I[0]
	Y[0] = Z[0] * K[-1]^α
	Y[0] = C[0] + I[0]
	1 / C[0]^γ = β / C[1]^γ * (α * Y[1] / K[0] + (1 - δ))
	Z[0] = (1 - ρ) + ρ * Z[-1] + σ * ϵ[x]
end


@parameters RBC verbose = true begin 
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.
end
solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)

zsim = simulate(RBC)
zsim1 = hcat(zsim([:K,:Z],:,:)...)
zdata = ℒ.reshape(zsim1,2,40)

# z_rbc1 = hcat(zsim...)
# z_rbc1 = ℒ.reshape(z_rbc1,size(RBC.var,1),40)

# Simulate T observations from a random initial condition
m= RBC

T = 20
Random.seed!(12345) #Fix seed to reproduce data
ϵ = randn(T+1)'  #Shocks are normal can be made anything e.g.  student-t

calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])

σ = 0.01
α = 0.5
β = 0.95
ρ = 0.2
δ = 0.02
γ = 1.

SS = get_steady_state(m,   parameters = (:σ => σ, :α => α, :β => β, :ρ => ρ, :δ => δ, :γ  => γ ), algorithm = :second_order)
Random.seed!(12345) #Fix seed to reproduce data
initial_conditions_dist = Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) #Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 
initial_conditions = ℒ.diag(rand(initial_conditions_dist, m.timings.nPast_not_future_and_mixed))
# long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * randn(m.timings.nPast_not_future_and_mixed)
state = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)
state_predictions = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)

aug_state = [initial_conditions
1 
0]

𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
state[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
state_predictions[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2

for t in 2:T+1
    aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                ϵ[:,t]]
    state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

observables_index = sort(indexin([:K, :Z], m.timings.var))
data = state[observables_index,2:end]

aug_state = [initial_conditions
1 
0]
for t in 2:T+1
    aug_state = [state_predictions[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                0]
    state_predictions[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

state_deviations = data[:,1:end] - state_predictions[observables_index,2:end]
sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)])
##


Turing.@model function loglikelihood_scaling_function(m, data, observables)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    #α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    #β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    #δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    #γ     ~ Turing.Normal(1, 0.05)
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.

    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)
    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
        #initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    #xnought ~ Turing.filldist(Turing.Normal(0.,1.),m.timings.nPast_not_future_and_mixed) #Initial shocks
    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    long_run_covariance = calculate_covariance_(solution[2])
    #initial_conditions = long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * xnought
    # SS = get_steady_state(m,   parameters = (:σ => σ, :α => α, :β => β, :ρ => ρ, :δ => δ, :γ  => γ ), algorithm = :second_order)
    initial_conditions ~  Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions  # Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    # ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    ϵ_draw ~ Turing.filldist(Turing.Normal(0,1), m.timings.nExo * size(data, 2)) #Shocks are Normally - distributed!

    ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    state = zeros(typeof(initial_conditions[1]),m.timings.nVars, size(data, 2)+1)

    # state[m.timings.past_not_future_and_mixed_idx,1] .= initial_conditions

    aug_state = [initial_conditions
    1 
    zeros( m.timings.nExo)]
    state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    for t in 2:size(data, 2)+1
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t-1]]
        state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data[:,1:end] - state[observables_index,2:end]
    #println(sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)] ))

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling = loglikelihood_scaling_function(RBC, data,[:K,:Z])

n_samples = 300
n_adapts = 50
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)

samps = Turing.sample(loglikelihood_scaling, alg, n_samples, progress = true)#, init_params = sol)



#Plot true and estimated latents to see how well we backed them out
noise = ϵ[:,2:end]

symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(samps[:, [Symbol("ϵ_draw[$a]") for a in 1:20], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = Turing.describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
Plots.plot(ϵ_mean[1:end], ribbon=1.96 * ϵ_std[1:end], label="Posterior mean", title = "First-Order Joint: Estimated Latents")
Plots.plot!(noise', label="True values")
