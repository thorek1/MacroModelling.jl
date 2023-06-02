using MacroModelling
import Turing, StatsPlots, Random, Statistics
import LinearAlgebra as ℒ

using HypothesisTests, Distributions

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
Random.seed!(1)
periods = 20
shockdist = Turing.TDist(3) #  Turing.Beta(10,1) #
shocks = rand(shockdist,1,periods) #  shocks = randn(1,periods)

#shocks /= Statistics.std(shocks)  # antithetic shocks
#shocks .-= Statistics.mean(shocks) # antithetic shocks
# Test for non-normality
        HypothesisTests.ExactOneSampleKSTest(shocks[1,:],Turing.Normal(0,1))
        StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
        StatsPlots.density!(shocks')
# get simulation
simulated_data = get_irf(RBC,shocks = shocks, periods = 0, levels = true)#(:k,:,:) |>collect

# plot simulation
MacroModelling.plot_irf(RBC,shocks = shocks, periods = 0)
#StatsPlots.plot(shocks')
Ω = 10^(-5)# eps()
n_samples = 1000


# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    # α     ~ MacroModelling.Beta(0.25, 0.15, 0.1, .4, μσ = true)
    # β     ~ MacroModelling.Beta(0.95, 0.05, .9, .9999, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    # δ     ~ MacroModelling.Beta(0.02, 0.05, 0.0, .1, μσ = true)
    # γ     ~ Turing.Normal(1, 0.05)
    # σ     ~ MacroModelling.InverseGamma(0.01, 0.05, μσ = true)

    α ~ Turing.Uniform(0.15, 0.45)
    β ~ Turing.Uniform(0.92, 0.9999)
    δ ~ Turing.Uniform(0.0001, 0.1)
    σ ~ Turing.Uniform(0.0, 0.1)
    ρ ~ Turing.Uniform(0.0, 1.0)
    γ ~ Turing.Uniform(0.0, 1.5)

    # α = 0.25
    # β = 0.95
    # σ = 0.01
    # ρ = 0.2
    # δ = 0.02
    # γ = 1.

    algorithm = :first_order
    parameters = [σ, α, β, ρ, δ, γ]
    shock_distribution = Turing.Normal()

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

    # solution = get_solution(m, parameters, algorithm = algorithm)

    # if solution[end] != true
    #     return Turing.@addlogprob! Inf
    # end
    # # draw_shocks(m)
    # x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    
    # calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    # long_run_covariance = calculate_covariance_(solution[2])
    
    # initial_conditions = long_run_covariance * x0
    # # initial_conditions = x0

    # 𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    # ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2))

    # ϵ = reshape(ϵ_draw, m.timings.nExo, size(data, 2))

    # state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))

    # aug_state = [initial_conditions
    #             1 
    #             ϵ[:,1]]

    # state[:,1] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    # for t in 2:size(data, 2)
    #     aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
    #                 1 
    #                 ϵ[:,t]]

    #     state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    # end

    # observables_index = sort(indexin(observables, m.timings.var))
    
    # state_deviations = data - state[observables_index,:] .- solution[1][observables_index...]

    # Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling = loglikelihood_scaling_function(RBC, simulated_data(:,:,:Shock_matrix), [:k], Ω) # Kalman
samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol


StatsPlots.plot(samps)
kf_estimated_parameters = Turing.describe(samps)[1].nt.parameters
kf_estimated_means = Turing.describe(samps)[1].nt.mean
kf_estimated_std = Turing.describe(samps)[1].nt.std
kfmean= kf_estimated_means
kfstd = kf_estimated_std 
Turing.@model function loglikelihood_scaling_function_ff(m, data, observables, Ω) #, kfmean, kfstd
     
    #  σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    #  α     ~ MacroModelling.Beta(0.25, 0.15, 0.1, .4, μσ = true)
    #  β     ~ MacroModelling.Beta(0.95, 0.05, .9, .9999, μσ = true)
    #  ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    #  δ     ~ MacroModelling.Beta(0.02, 0.05, 0.0, .1, μσ = true)
    #  γ     ~ Turing.Normal(1, 0.05)
    #σ     ~ MacroModelling.InverseGamma(0.01, 0.05, μσ = true)

    α ~ Turing.Uniform(0.15, 0.45)
    β ~ Turing.Uniform(0.92, 0.9999)
    δ ~ Turing.Uniform(0.0001, 0.1)
    σ ~ Turing.Uniform(0.0, 0.1)
    ρ ~ Turing.Uniform(0.0, 1.0)
    γ ~ Turing.Uniform(0.0, 1.5)

    #α ~ Turing.Uniform(kfmean[1]-2*kfstd[1], kfmean[1]+2*kfstd[1])
    #β ~ Turing.Uniform(kfmean[2]-2*kfstd[2], kfmean[2]+2*kfstd[2])
    #δ ~ Turing.Uniform(kfmean[3]-2*kfstd[3], kfmean[3]+2*kfstd[3])
    #σ ~ Turing.Uniform(0.0, kfmean[4]+2*kfstd[4])
    #ρ ~ Turing.Uniform(0.0, kfmean[5]+2*kfstd[5])
    #γ ~ Turing.Uniform(0.0, kfmean[6]+2*kfstd[6])


    # α = 0.25
    # β = 0.95
    # σ = 0.01
    # ρ = 0.2
    # δ = 0.02
    # γ = 1.

    algorithm = :first_order
    parameters = [σ, α, β, ρ, δ, γ]
    # skewness
    shock_distribution = Turing.TDist(3.0)

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
    # # initial_conditions = x0

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

     Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(RBC, collect(simulated_data(:k,:,:Shock_matrix))', [:k], Ω) # ,kf_estimated_means, kf_estimated_std  # Filter free

n_samples = 1000
samps_ff = Turing.sample(loglikelihood_scaling_ff, Turing.NUTS(), n_samples, progress = true)#, init_params = sol
StatsPlots.plot(samps_ff)

ff_estimated_parameters = Turing.describe(samps_ff)[1].nt.parameters
ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std


ff_bias= ( ff_estimated_means[1:6]- RBC.parameter_values[[2, 3, 5, 1, 4,6]])
kf_bias= ( kf_estimated_means[1:6]- RBC.parameter_values[[2, 3, 5, 1, 4,6]])

ff_z = (ff_bias)./ff_estimated_std[1:6] 
kf_z = ( kf_bias)./kf_estimated_std[1:6] 

grouplabel = repeat(["KF", "FF"], inner = 6)

StatsPlots.groupedbar( repeat(kf_estimated_parameters, outer =2) , [kf_bias ff_bias], group = grouplabel, xlabel = "Structural Parameters Biases")
StatsPlots.groupedbar( repeat(kf_estimated_parameters, outer =2), [kf_z ff_z], group = grouplabel, xlabel = "Structural Parameter z-scores")
data = KeyedArray(Array(collect(simulated_data(:k,:,:Shock_matrix)))',row = [:k], col = 1:1:20)



kf_filtered_shocks = MacroModelling.get_estimated_shocks(RBC, data, parameters = kf_estimated_means[[4, 1, 2, 5, 3,6]])


ff_estimated_parameters_indices = indexin([Symbol("ϵ_draw[$a]") for a in 1:periods], ff_estimated_parameters )
StatsPlots.plot(ff_estimated_means[ff_estimated_parameters_indices],
                ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices], 
                label = "Posterior mean", 
                title = "Joint: Estimated Latents")
StatsPlots.plot!(shocks', label = "True values")
StatsPlots.plot!(collect(kf_filtered_shocks'), label = "KF filtered shocks")


# samps = Turing.sample(loglikelihood_scaling, Turing.HMCDA(1000,.65,.75;init_ϵ = .05), n_samples, progress = true)#, init_params = sol)
# samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(1000, .65; init_ϵ = .01), n_samples, progress = true)#, init_params = sol)


# filter free generates good parameter estimates but does not necessarily nail the latent states and shocks
# std of MvNormal determines speed, convergence (too small no convergence), accuracy (smaller value will get more info on latent shocks and shock related parameters [shock size and persistence])
# the logic behind it appears to be: ideally the data is perfectly matched so you want the MvNormal to have a very small std but then the sampler will focus on matching the data since thats where the gradients are very large. the gradients regarding parameter priors are too small to have any influence and the sampler gets stuck if parameters do change because the gradients to match the data dominate.

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



StatsPlots.plot(samps_ff)

#Plot true and estimated latents to see how well we backed them out





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
