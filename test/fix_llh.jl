using Revise
using MacroModelling
import Random
using Zygote
using ForwardDiff
import MacroModelling: calculate_second_order_stochastic_steady_state
import LinearAlgebra as ℒ
include("../models/Gali_2015_chapter_3_nonlinear.jl")

m = Gali_2015_chapter_3_nonlinear

algorithm = :first_order

algorithm = :third_order

old_params = copy(m.parameter_values)
# sol = get_solution(m)
        
#         if length(m.exo) > 3
#             n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
#             var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]
#         elseif length(m.var) == 17
#             var_idxs = [5]
#         else
var_idxs = [1]
        # end

Random.seed!(10)

simulation = simulate(m, algorithm = algorithm)

stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(m.parameter_values, m) # , timer = timer)

ℒ.norm(𝐒₂) # 1162.7622359722268

state_update₂ = function(state::Vector{T}, shock::Vector{S}) where {T,S}
    aug_state = [state[m.timings.past_not_future_and_mixed_idx]
                1
                shock]
    return 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2
end

init = stochastic_steady_state - SS_and_pars
Random.seed!(10)
for i in 1:40
init = state_update₂(init, randn(m.timings.nExo))
end

state_update₂(stochastic_steady_state - SS_and_pars, zeros(m.timings.nExo)) - (stochastic_steady_state - SS_and_pars)


data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)

algorithm = :pruned_second_order
llh = get_loglikelihood(m, data_in_levels, old_params, algorithm = algorithm)

ZYG_grad_llh = Zygote.gradient(x -> get_loglikelihood(m, data_in_levels, x, algorithm = algorithm), old_params)


FOR_grad_llh = ForwardDiff.gradient(x -> get_loglikelihood(m, data_in_levels, x, algorithm = algorithm), old_params)
