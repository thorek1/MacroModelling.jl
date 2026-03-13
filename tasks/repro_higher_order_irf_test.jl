using MacroModelling
using Random
using Test
import LinearAlgebra as LA

# Isolated reproduction of the higher-order IRF assertions from
# test/test_standalone_function.jl (without running the full test file).

include("../test/models/RBC_CME.jl")

Random.seed!(3)

SS_and_pars, _ = MacroModelling.get_NSSS_and_parameters(m, m.parameter_values)
get_irf(m, algorithm = :third_order)
get_irf(m, algorithm = :pruned_third_order)
get_irf(m, algorithm = :pruned_second_order)

∇₁ = calculate_jacobian(m.parameter_values, SS_and_pars, m.caches, m.functions.jacobian, m.workspaces)
∇₂ = calculate_hessian(m.parameter_values, SS_and_pars, m.caches, m.functions.hessian, m.workspaces)
∇₃ = calculate_third_order_derivatives(m.parameter_values, SS_and_pars, m.caches, m.functions.third_order_derivatives, m.workspaces)

T = m.constants.post_model_macro

first_order_solution, _, _ = calculate_first_order_solution(∇₁, m.constants, m.workspaces, m.caches)
second_order_solution, _ = calculate_second_order_solution(∇₁, ∇₂, first_order_solution, m.constants, m.workspaces, m.caches)
third_order_solution, _ = calculate_third_order_solution(∇₁, ∇₂, ∇₃, first_order_solution, second_order_solution, m.constants, m.workspaces, m.caches)

second_order_solution = sparse(second_order_solution * m.constants.second_order.𝐔₂)
third_order_solution = sparse(third_order_solution * m.constants.third_order.𝐔₃)

Tz = [first_order_solution[:, 1:T.nPast_not_future_and_mixed] zeros(T.nVars) first_order_solution[:, T.nPast_not_future_and_mixed+1:end]]

second_order_state_update = function(state::Vector{Float64}, shock::Vector{Float64})
    aug_state = [state[T.past_not_future_and_mixed_idx]
                 1
                 shock]
    return Tz * aug_state + second_order_solution * kron(aug_state, aug_state) / 2
end

third_order_state_update = function(state::Vector{Float64}, shock::Vector{Float64})
    aug_state = [state[T.past_not_future_and_mixed_idx]
                 1
                 shock]
    return Tz * aug_state +
           second_order_solution * kron(aug_state, aug_state) / 2 +
           third_order_solution * kron(kron(aug_state, aug_state), aug_state) / 6
end

pruned_second_order_state_update = function(pruned_states::Vector{Vector{Float64}}, shock::Vector{Float64})
    aug_state₁ = [pruned_states[1][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 1; shock]
    aug_state₂ = [pruned_states[2][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 0; zero(shock)]
    return [Tz * aug_state₁,
            Tz * aug_state₂ + second_order_solution * LA.kron(aug_state₁, aug_state₁) / 2]
end

pruned_third_order_state_update = function(pruned_states::Vector{Vector{Float64}}, shock::Vector{Float64})
    aug_state₁ = [pruned_states[1][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 1; shock]
    aug_state₁̂ = [pruned_states[1][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 0; shock]
    aug_state₂ = [pruned_states[2][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 0; zero(shock)]
    aug_state₃ = [pruned_states[3][m.constants.post_model_macro.past_not_future_and_mixed_idx]; 0; zero(shock)]

    kron_aug_state₁ = LA.kron(aug_state₁, aug_state₁)
    return [Tz * aug_state₁,
            Tz * aug_state₂ + second_order_solution * kron_aug_state₁ / 2,
            Tz * aug_state₃ + second_order_solution * LA.kron(aug_state₁̂, aug_state₂) + third_order_solution * LA.kron(kron_aug_state₁, aug_state₁) / 6]
end

# Reproduce exactly the four IRF checks that are currently failing in CI.
SSS_delta_2 = m.caches.non_stochastic_steady_state[1:length(m.constants.post_model_macro.var)] - m.caches.second_order_stochastic_steady_state
initial_state_2 = zeros(m.constants.post_model_macro.nVars) - SSS_delta_2
iirrff2 = irf(second_order_state_update, initial_state_2 + SSS_delta_2, zeros(T.nVars), m.constants)

SSS_delta_3 = m.caches.non_stochastic_steady_state[1:length(m.constants.post_model_macro.var)] - m.caches.third_order_stochastic_steady_state
initial_state_3 = zeros(m.constants.post_model_macro.nVars) - SSS_delta_3
iirrff3 = irf(third_order_state_update, initial_state_3 + SSS_delta_3, zeros(T.nVars), m.constants)

iirrffp2 = irf(pruned_second_order_state_update,
               [zeros(m.constants.post_model_macro.nVars), zeros(m.constants.post_model_macro.nVars)],
               zeros(T.nVars),
               m.constants)

iirrffp3 = irf(pruned_third_order_state_update,
               [zeros(m.constants.post_model_macro.nVars), zeros(m.constants.post_model_macro.nVars), zeros(m.constants.post_model_macro.nVars)],
               zeros(T.nVars),
               m.constants)

expected_iirrff2 = [-0.0004547347878067665, 0.0020831426377533636]
expected_iirrff3 = [-0.00045473149068020854, 0.002083198241302615]
expected_iirrffp2 = [-0.00045473478780675195, 0.002083142637753389]
expected_iirrffp3 = [-0.0004547315171573783, 0.0020831990353127696]

actual_iirrff2 = vec(iirrff2[4, 1, :])
actual_iirrff3 = vec(iirrff3[4, 1, :])
actual_iirrffp2 = vec(iirrffp2[4, 1, :])
actual_iirrffp3 = vec(iirrffp3[4, 1, :])

println("Higher-order IRF isolated repro")
println("iirrff2  actual=$(actual_iirrff2) expected=$(expected_iirrff2)")
println("iirrff3  actual=$(actual_iirrff3) expected=$(expected_iirrff3)")
println("iirrffp2 actual=$(actual_iirrffp2) expected=$(expected_iirrffp2)")
println("iirrffp3 actual=$(actual_iirrffp3) expected=$(expected_iirrffp3)")

@test isapprox(actual_iirrff2, expected_iirrff2, rtol = 1e-6)
@test isapprox(actual_iirrff3, expected_iirrff3, rtol = 1e-6)
@test isapprox(actual_iirrffp2, expected_iirrffp2, rtol = 1e-6)
@test isapprox(actual_iirrffp3, expected_iirrffp3, rtol = 1e-6)

println("HIGHER_ORDER_IRF_REPRO=PASS")
