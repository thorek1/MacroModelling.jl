using MacroModelling
import Turing, StatsPlots
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
    γ = 1
end

get_SS(RBC)

# plot_irf(RBC)

get_solution(RBC)



Turing.@model function loglikelihood_function(m)
    σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    γ     ~ Turing.Normal(1, 0.05)
    
    Turing.@addlogprob! sum(get_solution(m,[σ, α, β, ρ, δ, γ])[2]) / 1e8
end

# using LinearAlgebra

# Z₁₁ = randn(10,10)
# Ẑ₁₁ = svd(Z₁₁)
# Ẑ₁₁ |>inv

# Ẑ₁₁.S .|> inv
# Ẑ₁₁.Vt |> inv

# (Ẑ₁₁.U * inv(diagm(Ẑ₁₁.S)) * Ẑ₁₁.Vt)'
# inv(Z₁₁)

# Z₂₁ = randn(10,10)

# D      = Z₂₁ / Ẑ₁₁
# D      = Z₂₁ / Z₁₁



loglikelihood = loglikelihood_function(RBC)


n_samples = 10

Turing.setadbackend(:forwarddiff)

# using Zygote
# Turing.setadbackend(:zygote)
samps = Turing.sample(loglikelihood, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)




Turing.@model function loglikelihood_second_order_function(m)
    σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    γ     ~ Turing.Normal(1, 0.05)
    soll = get_solution(m,[σ, α, β, ρ, δ, γ], algorithm = :second_order)
    println(soll[end])
    Turing.@addlogprob! sum(soll[3]) / 1e6
end


loglikelihood_second_order = loglikelihood_second_order_function(RBC)

samps = Turing.sample(loglikelihood_second_order, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)




solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)

Turing.@model function loglikelihood_scaling_function(m, data, observables)
    σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    γ     ~ Turing.Normal(1, 0.05)
    
    initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

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
                    ϵ[:,t-1]]
        state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data[:,2:end] - state[observables_index,2:end]

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)-1])
end


data=[ 0.062638   0.053282    0.00118333  0.442814   0.300381  0.150443  0.228132   0.382626   -0.0122483   0.0848671  0.0196158   0.197779    0.782655  0.751345   0.911694   0.754197   0.493297    0.0265917   0.209705    0.0876804;
-0.0979824  0.0126432  -0.12628     0.161212  -0.109357  0.120232  0.0316766  0.0678017  -0.0371438  -0.162375  0.0574594  -0.0564989  -0.18021   0.0749526  0.132553  -0.135002  -0.0143846  -0.0770139  -0.0295755  -0.0943254]



# AA = spzeros(10,10)
# AA[1:3,5:7] .= 1

# AA * Real[rand(10)...]


n_samples = 100

loglikelihood_scaling = loglikelihood_scaling_function(RBC, data,[:K,:Z])

samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)

# m = RBC



# solution = get_solution(m, m.parameter_values, algorithm = :second_order)

# 𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

# t=2


# state = zeros(Real,m.timings.nVars, size(data, 2)+1)
# ϵ = zeros( m.timings.nExo,  size(data, 2))
# aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
# 1 
# ϵ[:,t-1]]
# state[:,t] =  𝐒₁ * aug_state + solution[3] * Real[ℒ.kron(aug_state, aug_state)...] / 2 


# observables = :K

# solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)
# solution[3]