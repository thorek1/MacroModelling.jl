using MacroModelling
import Turing, StatsPlots, Random, Statistics
import LinearAlgebra as ℒ

@model RBC begin
    1 / (- k[0]  + (1 - δ ) * k[-1] + (exp(z[-1]) * k[-1]^α)) = (β   / (- k[+1]  + (1 - δ) * k[0] +(exp(z[0]) * k[0]^α))) * (α* exp(z[0]) * k[0] ^(α - 1) + (1 - δ))  ;
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] - σ* EPSz[x]
end

@parameters RBC verbose = true begin 
    σ = 0.01
    α = 0.25
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.
end
solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)


# draw shocks
periods = 10
shocks = randn(1,periods)
shocks /= Statistics.std(shocks)  # antithetic shocks
shocks .-= Statistics.mean(shocks) # antithetic shocks

# get simulation
simulated_data = get_irf(RBC,shocks = shocks, periods = 0)[:,:,1] |>collect



StatsPlots.plot(simulated_data')
StatsPlots.plot(shocks')



Turing.@model function loglikelihood_scaling_function(m, data, observables,Ω)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    # α     ~ MacroModelling.Beta(0.25, 0.15, 0.1, .4, μσ = true)
    # β     ~ MacroModelling.Beta(0.95, 0.05, .9, .9999, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    # δ     ~ MacroModelling.Beta(0.02, 0.05, 0.0, .1, μσ = true)
    # γ     ~ Turing.Normal(1, 0.05)
    # σ     ~ MacroModelling.InverseGamma(0.01, 0.05, μσ = true)
    α ~ Turing.Uniform(0.1, 0.4)
    β ~ Turing.Uniform(0.9, 0.9999)
    # δ ~ Turing.Uniform(0.0001, 0.05)
    # σ ~ Turing.Uniform(0.0, 0.1)
    # ρ ~ Turing.Uniform(0.0, 1.0)
    # γ ~ Turing.Uniform(0.0, 2.0)

    σ = 0.01
    # α = 0.5
    # β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.

    # solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)
    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :first_order)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
    
    x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    
    ϵ_draw ~ Turing.filldist(Turing.Normal(), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    long_run_covariance = calculate_covariance_(solution[2])
    
    initial_conditions = long_run_covariance * x0
    # initial_conditions = x0

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(Turing.Normal(), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

    ϵ = reshape(ϵ_draw, m.timings.nExo, size(data, 2))

    state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))

    aug_state = [initial_conditions
    1 
    ϵ[:,1]]

    state[:,1] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    for t in 2:size(data, 2)
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t]]

        state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data - state[observables_index,:]
    
    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
end


Ω = sqrt(eps())
loglikelihood_scaling = loglikelihood_scaling_function(RBC, simulated_data, [:k,:z], Ω)

n_samples = 1000

samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)

StatsPlots.plot(samps)

#Plot true and estimated latents to see how well we backed them out
estimated_parameters = Turing.describe(samps)[1].nt.parameters
estimated_parameters_indices = indexin([Symbol("ϵ_draw[$a]") for a in 1:periods], estimated_parameters)
estimated_means = Turing.describe(samps)[1].nt.mean
estimated_std = Turing.describe(samps)[1].nt.std


StatsPlots.plot(estimated_means[estimated_parameters_indices],
                ribbon = 1.96 * estimated_std[estimated_parameters_indices], 
                label = "Posterior mean", 
                title = "First-Order Joint: Estimated Latents")
StatsPlots.plot!(shocks', label = "True values")

