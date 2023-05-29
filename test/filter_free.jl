using MacroModelling
import Turing, StatsPlots, Random, Statistics, DynamicHMC
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

# draw shocks
periods = 10
shocks = randn(1,periods)
shocks /= Statistics.std(shocks)  # antithetic shocks
shocks .-= Statistics.mean(shocks) # antithetic shocks

# get simulation
simulated_data = get_irf(RBC,shocks = shocks, periods = 0, levels = true)#(:k,:,:) |>collect

# plot simulation
plot_irf(RBC,shocks = shocks, periods = 0)
StatsPlots.plot(shocks')


function ϵ_loss(Δ; ϵ = .01, p = 2)
    abs(Δ) > ϵ ? abs(Δ)^p : 0
end

# define loglikelihood model
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    # α     ~ MacroModelling.Beta(0.25, 0.15, 0.1, .4, μσ = true)
    # β     ~ MacroModelling.Beta(0.95, 0.05, .9, .9999, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    # δ     ~ MacroModelling.Beta(0.02, 0.05, 0.0, .1, μσ = true)
    # γ     ~ Turing.Normal(1, 0.05)
    # σ     ~ MacroModelling.InverseGamma(0.01, 0.05, μσ = true)

    α ~ Turing.Uniform(0.15, 0.45)
    # β ~ Turing.Uniform(0.92, 0.9999)
    # δ ~ Turing.Uniform(0.0001, 0.05)
    # σ ~ Turing.Uniform(0.0, 0.1)
    # ρ ~ Turing.Uniform(0.0, 1.0)
    # γ ~ Turing.Uniform(0.5, 1.5)

    # α = 0.25
    σ = 0.01
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.

    algorithm = :first_order
    parameters = [σ, α, β, ρ, δ, γ]
    shock_distribution = Turing.Normal()

    # Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

    solution = get_solution(m, parameters, algorithm = algorithm)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
    # draw_shocks(m)
    x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    
    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    long_run_covariance = calculate_covariance_(solution[2])
    
    initial_conditions = long_run_covariance * x0
    # initial_conditions = x0

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2))

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
    
    state_deviations = data - state[observables_index,:] .- solution[1][observables_index...]

    for (i,o) in enumerate(observables_index)
        if solution[1][o] != 0 && (all(state[o,:] .+ solution[1][o] .> 0) || all(state[o,:] .+ solution[1][o] .< 0))
            state_deviations[i,:] /= solution[1][o]
        end
    end

    # println(sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)]))
    # println(-sum(abs.(state_deviations).^5) / length(data) * 1e3)

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.I(size(data,1))), state_deviations[:,t] .* Ω) for t in 1:size(data, 2)])
    # Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.I(size(data,1))), state_deviations[:,t] .^ 3 .* Ω) for t in 1:size(data, 2)])

    # Turing.@addlogprob! -sum(abs.(state_deviations .* 1e4).^4) / length(data)
    # Turing.@addlogprob! -sum(ϵ_loss.(state_deviations)) / length(data) * 2e6
end

Ω = 1e4#eps()
# loglikelihood_scaling = loglikelihood_scaling_function(RBC, simulated_data(:k,:,:Shock_matrix), [:k], Ω) # Kalman
loglikelihood_scaling = loglikelihood_scaling_function(RBC, collect(simulated_data(:k,:,:Shock_matrix))', [:k], Ω) # Filter free

n_samples = 1000

# solution = get_solution(RBC, RBC.parameter_values, algorithm = :first_order)[1]

# simulated_data(:k,:,:Shock_matrix) .- solution[1][observables_index...]

samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)
samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(1000, .65; init_ϵ = .01), n_samples, progress = true)#, init_params = sol)

interval = -.01:.0001:.01
interval = -1:.01:1

StatsPlots.plot(x->abs(x)^4,interval)
StatsPlots.plot!(x->abs(x)^3,interval)
StatsPlots.plot!(x->abs(x)^2,interval)



StatsPlots.plot(x->4*x^3,interval)
StatsPlots.plot!(x->3*x*abs(x),interval)
StatsPlots.plot!(x->2*x,interval)

interval = -.01:.0001:.01

StatsPlots.plot(x->abs(x)^4,interval)
StatsPlots.plot!(x->abs(x)^3,interval)
StatsPlots.plot!(x->abs(x)^2,interval)



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





# testing functions

function calculate_filter_free_llh(m, parameters, data, observables; algorithm = :first_order, shock_distribution = Turing.Normal(), Ω::Float64 = sqrt(eps()))
    solution = get_solution(m, parameters, algorithm = algorithm)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
    
    x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    
    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    long_run_covariance = calculate_covariance_(solution[2])
    
    initial_conditions = long_run_covariance * x0
    # initial_conditions = x0

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

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

function draw_shocks(m)
    x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    return x0
end
