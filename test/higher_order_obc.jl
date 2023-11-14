
using MacroModelling

@model testmax_obc begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    # r̂[0] = max(r̄, r[0])
    # 0 = max(r̄ - r[0], r̂[0] - r[0])
    r[0] = max(r̄, r̂[0])

    # r̂[0] = r[0] + ϵll[x-3]

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

    # ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]

    # ϵll⁻²[0] = ϵll⁻³[-1] + ϵll⁻²[x]

    # ϵll⁻³[0] = ϵll⁻³[x]

end

@parameters testmax_obc begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# SS(testmax_obc)
# SSS(testmax_obc)

import StatsPlots

plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all)
plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all, algorithm = :second_order)

plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all)
plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all, ignore_obc = true)
plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all, algorithm = :second_order, ignore_obc = true)

plot_irf(testmax_obc, negative_shock = true, parameters = :σᶻ => 1.1, variables = :all)


testmax_obc.dyn_equations
@model testmax begin
    1  /  c[0] = (β  /  c[1]) * (r[1] + (1 - δ))

    r̂[0] = α * exp(z[0]) * k[-1]^(α - 1)

    # r̂[0] = max(r̄, r[0])
    # 0 = max(r̄ - r[0], r̂[0] - r[0])
    # r[0] = max(r̄, r̂[0])

    r̂[0] = r[0]# + ϵll[x-3]

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

    # ϵll⁻¹[0] = ϵll⁻²[-1] + ϵll⁻¹[x]

    # ϵll⁻²[0] = ϵll⁻³[-1] + ϵll⁻²[x]

    # ϵll⁻³[0] = ϵll⁻³[x]

end

@parameters testmax begin
    r̄ = 0
    σᶻ= 1#0.01
    ρᶻ= 0.8#2
    δ = 0.02
    α = 0.5
    β = 0.95
end

SSS(testmax)

function get_sol(𝓂)
    parameters = 𝓂.parameter_values

    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, true)
        
    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings)

    𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]

    return 𝐒₁, 𝐒₂
end

get_solution(testmax, algorithm = :pruned_second_order)

get_solution(testmax_obc, algorithm = :pruned_second_order)


S1, S2 = get_sol(testmax)
S1obc, S2obc = get_sol(testmax_obc)

import LinearAlgebra as ℒ
𝓂 = testmax_obc

nᵉ = 𝓂.timings.nExo
𝓂.timings.exo
# .!contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")
# .!contains.(string.(𝓂.timings.past_not_future_and_mixed),"ᵒᵇᶜ")
nˢ = 𝓂.timings.nPast_not_future_and_mixed
# 𝓂.var

# .!contains.(string.(𝓂.var),"ᵒᵇᶜ")

# s_in_s⁺ = BitVector(vcat(.!contains.(string.(𝓂.timings.past_not_future_and_mixed),"ᵒᵇᶜ"), zeros(Bool, nᵉ + 1)))
# e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), .!contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")))
# v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))
s_and_e_in_s⁺ = BitVector(vcat(.!contains.(string.(𝓂.timings.past_not_future_and_mixed), "ᵒᵇᶜ"), 1, .!contains.(string.(𝓂.timings.exo), "ᵒᵇᶜ")))

# s_and_e_in_s⁺ = BitVector(vcat(ones(Bool, nˢ + 1), .!contains.(string.(𝓂.timings.exo), "ᵒᵇᶜ")))

# kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
# kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
# kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
# kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

# kron_states     = ℒ.kron(s_in_s⁺, s_in_s⁺)

# first order
# 𝐒₁ = S1obc[:, s_and_e_in_s⁺]
𝐒₁ = S1obc[.!contains.(string.(𝓂.var),"ᵒᵇᶜ"), s_and_e_in_s⁺]



kron_s_s = ℒ.kron(s_and_e_in_s⁺, s_and_e_in_s⁺)
# second order
# 𝐒₂        = S2obc[:, kron_s_s]
𝐒₂        = S2obc[.!contains.(string.(𝓂.var),"ᵒᵇᶜ"), kron_s_s]
# S2

# e_to_y₁ = S1obc[:, (nˢ + 1):end]

# s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
# e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]




using SpeedMapping


# state = zeros(𝓂.timings.nVars)
state = zeros(sum(.!contains.(string.(𝓂.var),"ᵒᵇᶜ")))
shock = zeros(sum(.!contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")))

states = filter(x -> !contains(string(x), "ᵒᵇᶜ"), 𝓂.timings.past_not_future_and_mixed)
vars = filter(x -> !contains(string(x), "ᵒᵇᶜ"), 𝓂.var)
state_idx = indexin(states, vars)
# 𝓂.timings.past_not_future_and_mixed


# 𝓂.timings.past_not_future_and_mixed_idx
aug_state = [state[state_idx]
1
shock]

sol = speedmapping(state; 
            m! = (SSS, sss) -> begin 
                                aug_state .= [sss[state_idx]
                                            1
                                            shock]

                                SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
            end, 
tol = eps(), maps_limit = 10000)

testmax.solution.perturbation.second_order.solution_matrix |> collect
𝐒₂ |> collect
SSS(testmax, algorithm = :second_order)

function second_order_stochastic_steady_state_iterative_solution_forward(𝐒₁𝐒₂::SparseVector{Float64};  dims::Vector{Tuple{Int,Int}},  𝓂::ℳ, tol::AbstractFloat = eps(), ignore_obc = true)
    len𝐒₁ = dims[1][1] * dims[1][2]

    𝐒₁ = reshape(𝐒₁𝐒₂[1 : len𝐒₁],dims[1])
    𝐒₂ = sparse(reshape(𝐒₁𝐒₂[len𝐒₁ + 1 : end],dims[2]))
        
    state = zeros(𝓂.timings.nVars)
    shock = zeros(𝓂.timings.nExo)

    aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
    1
    shock]

    sol = speedmapping(state; 
                m! = (SSS, sss) -> begin 
                                    aug_state .= [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                                1
                                                shock]

                                    SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
                end, 
    tol = tol, maps_limit = 10000)
    
    return sol.minimizer, sol.converged
end
