using MacroModelling
import Turing, StatsPlots

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





Turing.@model function loglikelihood_scaling_function(m, data)
    σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    γ     ~ Turing.Normal(1, 0.05)
    
    initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nVars) # Initial conditions 

    solution = get_solution(m,[σ, α, β, ρ, δ, γ], algorithm = :second_order)

    if !solution[end]
        return Turing.@addlogprob! Inf
    end

    calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    long_run_covariance = calculate_covariance_(solution[2])

    x_iv = long_run_covariance * initial_conditions #scale initial condition with ergodic variance

    ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

    Turing.@addlogprob! sum(x_iv) / 1e6
end

data = randn(1,10)


loglikelihood_scaling = loglikelihood_scaling_function(RBC, data)

samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)


