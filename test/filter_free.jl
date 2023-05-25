using MacroModelling
import Turing, StatsPlots , Plots, Random
import LinearAlgebra as ℒ

@model RBC begin
    1 / (- k[0]  + (1 - δ ) * k[-1] + (exp(z[-1]) * k[-1]^α)) = (β   / (- k[+1]  + (1 - δ) * k[0] +(exp(z[0]) * k[0]^α))) * (α* exp(z[0]) * k[0] ^(α - 1) + (1 - δ))  ;
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] - σ* EPSz[x]
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

# draw from t scaled by approximate invariant variance) for the initial condition
m =RBC
calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])

T =20
ddof = 4
shockdist = Turing.TDist(ddof) #Shocks are student-t
Random.seed!(12345) #Fix seed to reproduce data
initial_conditions = long_run_covariance * rand(shockdist,m.timings.nPast_not_future_and_mixed)
#nitial_conditions_dist = Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) #Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 
#initial_conditions = ℒ.diag.(rand(initial_conditions_dist, m.timings.nPast_not_future_and_mixed))
# long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * randn(m.timings.nPast_not_future_and_mixed)

Random.seed!(12345) #Fix seed to reproduce data
# Generate noise sequence
noiseshocks = rand(shockdist,T)
noise = Matrix(noiseshocks') # the ϵ shocks are "noise" in DifferenceEquations for SciML compatibility 

#ϵ = [-0.369555723973723 0.47827032464044467 0.2567178329209457 -1.1127581634083954 1.779713752762057 -1.3694068387087652 0.4598600006094857 0.1319461357213755 0.21210992474923543 0.37965007742056217 -0.36234330914698276 0.04507575971259013 0.2562242956767027 -1.4425668844506196 -0.2559534237970267 -0.40742710317783837 1.5578503125015226 0.05971261026086091 -0.5590041386255554 -0.1841854411460526]
ϵ = noise

# Initialize states
state = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)
state_predictions = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)

aug_state = [initial_conditions
1 
ϵ[:,1]]


𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
state[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
state_predictions[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2

for t in 2:T
    aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                ϵ[:,t]]
    state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

observables_index = sort(indexin([:k, :z], m.timings.var))
data_sim = state[observables_index,1:end]

aug_state = [initial_conditions
1 
0]
for t in 2:T
    aug_state = [state_predictions[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                0]
    state_predictions[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

state_deviations = data_sim[:,1:end] - state_predictions[observables_index,1:end]
sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data_sim, 2)])

dataFV = [-0.02581238618841974 -0.024755946984579915 -0.0007518239655738511 -0.02582984321259188 -0.04567755888428696 0.021196857503906794 -0.0772465811707222 -0.008386388700111276 -0.02347363396607608 -0.033743271643453004 -0.04771401523417986 -0.0723137820802147 -0.052024995108031956 -0.04914479042856236 -0.0628064692912924 0.026322291179482583 0.05836273680164356 0.08777750705366681 -0.006357303764844118 -0.027859850762631953 0.0036979646377400615; -9.300233770305984e-6 0.0036936971929831686 -0.004043963807807812 -0.0033759710907710194 0.010452387415929751 -0.01570666004443462 0.010552736378200728 -0.0024880527304547108 -0.0018170719033046975 -0.002484513628153294 -0.004293403499836281 0.002764752391502571 0.00010219288117461296 -0.0025418043805321045 0.013917307968399776 0.005342995831650222 0.005142870198108429 -0.014549929085393539 -0.003507111919687318 0.0048886190023180905 0.0028195782119241446]
state_deviations_FV = dataFV[:,1:end] - state_predictions[observables_index,1:end]

sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations_FV,1)))), state_deviations_FV[:,t]) for t in 1:size(data_sim, 2)])

sum([Turing.logpdf(Turing.MvNormal(zeros(size(data_sim)[1]),Matrix(0.0000001*ℒ.I, size(data_sim)[1], size(data_sim)[1])), state_deviations_FV[:,t]) for t in 1:size(data_sim, 2)])



Plots.plot(data_sim[:,1:end]')
Plots.plot!(dataFV[:,2:end]')



Turing.@model function loglikelihood_scaling_function(m, data, observables,Ω)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    #α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    #β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    #δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    #γ     ~ Turing.Normal(1, 0.05)
    σ = 0.01
    #α ~ Turing.Uniform(0.2, 0.8)
    #β ~ Turing.Uniform(0.5, 0.99)

    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.

    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)
    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
    
    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    long_run_covariance = calculate_covariance_(solution[2])
    
    x0 ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 
    ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

    initial_conditions = long_run_covariance * x0

    #xnought ~ Turing.filldist(Turing.Normal(0.,1.),m.timings.nPast_not_future_and_mixed) #Initial shocks
    #calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    # long_run_covariance = calculate_covariance_(solution[2])
    # initial_conditions = long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * xnought
    #SS = get_steady_state(m,   parameters = (:σ => σ, :α => α, :β => β, :ρ => ρ, :δ => δ, :γ  => γ ), algorithm = :second_order)
    # initial_conditions ~  Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions  # Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    #ϵ_draw ~ Turing.filldist(Turing.Normal(0,1), m.timings.nExo * size(data, 2)) #Shocks are Normally - distributed!

    ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    state = zeros(typeof(initial_conditions[1]),m.timings.nVars, size(data, 2))

    # state[m.timings.past_not_future_and_mixed_idx,1] .= initial_conditions

    aug_state = [initial_conditions
    1 
    ϵ[:,1]]
    state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    for t in 2:size(data, 2)
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t]]
        state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data[:,1:end] - state[observables_index,1:end]
    
   # println(sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(Ω*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)]))

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(Ω*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)])


    # Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)])
end


Ω = 0.0001
loglikelihood_scaling = loglikelihood_scaling_function(RBC, data_sim,[:k,:z], Ω)

n_samples = 300
n_adapts = 50
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)

samps = Turing.sample(loglikelihood_scaling, alg, n_samples, progress = true)#, init_params = sol)

Plots.plot(samps[["x0[1]"]]; colordim=:parameter, legend=true)

#Plots.plot(samps[["β"]]; colordim=:parameter, legend=true)

#Plots.plot(samps[["α"]]; colordim=:parameter, legend=true)

#Plot true and estimated latents to see how well we backed them out

symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(samps[:, [Symbol("ϵ_draw[$a]") for a in 1:20], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = Turing.describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
Plots.plot(ϵ_mean[1:end], ribbon=1.96 * ϵ_std[1:end], label="Posterior mean", title = "First-Order Joint: Estimated Latents")
Plots.plot!(noise', label="True values")

