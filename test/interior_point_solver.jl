using Revise
# using Pkg; Pkg.activate(".");
using MacroModelling
# using StatsPlots
using Random
import Optim, LineSearches
using BenchmarkTools
import LinearAlgebra as ℒ
import Random
import FiniteDifferences
import Zygote
import ForwardDiff
import CSV
using DataFrames

include("../models/Smets_Wouters_2007.jl")

# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
sample_idx = 47:230 # 1960Q1-2004Q4

data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)


𝓂 = Smets_Wouters_2007

T = 𝓂.timings
SSS(𝓂, algorithm = :third_order, derivatives = false)#, parameters = :csadjcost => 6.0144)

import MacroModelling: get_and_check_observables, check_bounds, minimize_distance_to_initial_data, get_relevant_steady_state_and_state_update, replace_indices, minimize_distance_to_data, match_data_sequence!, match_initial_data!,calculate_loglikelihood, String_input, calculate_second_order_stochastic_steady_state, expand_steady_state, calculate_third_order_stochastic_steady_state

parameter_values = 𝓂.parameter_values
parameters = 𝓂.parameter_values
algorithm = :pruned_third_order
filter = :inversion
warmup_iterations = 0
presample_periods = 0
initial_covariance = :diagonal
tol = 1e-12
verbose = false


observables = get_and_check_observables(𝓂, data)

solve!(𝓂, verbose = verbose, algorithm = algorithm)

bounds_violated = check_bounds(parameter_values, 𝓂)

NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

obs_indices = convert(Vector{Int}, indexin(observables, NSSS_labels))

TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)

if collect(axiskeys(data,1)) isa Vector{String}
    data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
end

dt = (data(observables))

# prepare data
data_in_deviations = dt .- SS_and_pars[obs_indices]

presample_periods = 0

get_loglikelihood(𝓂, data[1:6,:], 𝓂.parameter_values, filter = :inversion, algorithm = :pruned_second_order, filter_algorithm = :Newton)

get_loglikelihood(𝓂, data[1:6,:], 𝓂.parameter_values, filter = :inversion, algorithm = :second_order, filter_algorithm = :Newton)


get_loglikelihood(𝓂, data[1:6,1:3], 𝓂.parameter_values, filter = :inversion, algorithm = :pruned_third_order, filter_algorithm = :Newton)



precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

shocks² = 0.0
logabsdets = 0.0


s_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed), zeros(Bool, T.nExo + 1)))
sv_in_s⁺ = BitVector(vcat(ones(Bool, T.nPast_not_future_and_mixed + 1), zeros(Bool, T.nExo)))
e_in_s⁺ = BitVector(vcat(zeros(Bool, T.nPast_not_future_and_mixed + 1), ones(Bool, T.nExo)))

tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
shock_idxs = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
shock²_idxs = tmp.nzind

shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)

tmp = ℒ.kron(sv_in_s⁺, sv_in_s⁺) |> sparse
var_vol²_idxs = tmp.nzind

tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
var²_idxs = tmp.nzind

𝐒⁻¹ = 𝐒[1][T.past_not_future_and_mixed_idx,:]
𝐒¹⁻ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed]
𝐒¹⁻ᵛ = 𝐒[1][cond_var_idx, 1:T.nPast_not_future_and_mixed+1]
𝐒¹ᵉ = 𝐒[1][cond_var_idx,end-T.nExo+1:end]

if length(cond_var_idx) == T.nExo
    𝐒¹ᵉfact = ℒ.lu(𝐒[1][cond_var_idx,end-T.nExo+1:end], check = false)

    if !ℒ.issuccess(𝐒¹ᵉfact)
        if ℒ.rank(𝐒[1][cond_var_idx,end-T.nExo+1:end]) < T.nExo
            return -Inf
        end
        𝐒¹ᵉfact = ℒ.svd(𝐒[1][cond_var_idx,end-T.nExo+1:end])
    end
else
    𝐒¹ᵉfact = ℒ.svd(𝐒[1][cond_var_idx,end-T.nExo+1:end])
end

# inv𝐒¹ᵉ = inv(𝐒¹ᵉfact)

𝐒²⁻ᵛ = 𝐒[2][cond_var_idx,var_vol²_idxs]
𝐒²⁻ = 𝐒[2][cond_var_idx,var²_idxs]
𝐒²⁻ᵉ = 𝐒[2][cond_var_idx,shockvar²_idxs]
𝐒²ᵉ = 𝐒[2][cond_var_idx,shock²_idxs]
𝐒⁻² = 𝐒[2][T.past_not_future_and_mixed_idx,:]

𝐒²⁻ᵛ    = length(𝐒²⁻ᵛ.nzval)    / length(𝐒²⁻ᵛ)  > .1 ? collect(𝐒²⁻ᵛ)    : 𝐒²⁻ᵛ
𝐒²⁻     = length(𝐒²⁻.nzval)     / length(𝐒²⁻)   > .1 ? collect(𝐒²⁻)     : 𝐒²⁻
𝐒²⁻ᵉ    = length(𝐒²⁻ᵉ.nzval)    / length(𝐒²⁻ᵉ)  > .1 ? collect(𝐒²⁻ᵉ)    : 𝐒²⁻ᵉ
𝐒²ᵉ     = length(𝐒²ᵉ.nzval)     / length(𝐒²ᵉ)   > .1 ? collect(𝐒²ᵉ)     : 𝐒²ᵉ
𝐒⁻²     = length(𝐒⁻².nzval)     / length(𝐒⁻²)   > .1 ? collect(𝐒⁻²)     : 𝐒⁻²

state[1] = state[1][T.past_not_future_and_mixed_idx]
state[2] = state[2][T.past_not_future_and_mixed_idx]

if length(state) == 3
    state[3] = state[3][T.past_not_future_and_mixed_idx]

    tmp = ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺)) |> sparse
    var_vol³_idxs = tmp.nzind

    tmp = ℒ.kron(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1), zero(e_in_s⁺) .+ 1) |> sparse
    shock_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shock³_idxs = tmp.nzind

    tmp = ℒ.kron(zero(e_in_s⁺) .+ 1, ℒ.kron(e_in_s⁺, e_in_s⁺)) |> sparse
    shockvar1_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)) |> sparse
    shockvar2_idxs = tmp.nzind

    tmp = ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)) |> sparse
    shockvar3_idxs = tmp.nzind

    shockvar³_idxs = setdiff(shock_idxs, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
    
    𝐒³⁻ᵛ = 𝐒[3][cond_var_idx,var_vol³_idxs]
    𝐒³⁻ᵉ = 𝐒[3][cond_var_idx,shockvar³_idxs]
    𝐒³ᵉ  = 𝐒[3][cond_var_idx,shock³_idxs]
    𝐒⁻³  = 𝐒[3][T.past_not_future_and_mixed_idx,:]

    𝐒³⁻ᵛ    = length(𝐒³⁻ᵛ.nzval)    / length(𝐒³⁻ᵛ)  > .1 ? collect(𝐒³⁻ᵛ)    : 𝐒³⁻ᵛ
    𝐒³⁻ᵉ    = length(𝐒³⁻ᵉ.nzval)    / length(𝐒³⁻ᵉ)  > .1 ? collect(𝐒³⁻ᵉ)    : 𝐒³⁻ᵉ
    𝐒³ᵉ     = length(𝐒³ᵉ.nzval)     / length(𝐒³ᵉ)   > .1 ? collect(𝐒³ᵉ)     : 𝐒³ᵉ
    𝐒⁻³     = length(𝐒⁻³.nzval)     / length(𝐒⁻³)   > .1 ? collect(𝐒⁻³)     : 𝐒⁻³
end

kron_buffer = zeros(T.nExo^2)

J = zeros(T.nExo, T.nExo)

kron_buffer2 = ℒ.kron(J, zeros(T.nExo))


i = 1


state¹⁻ = state[1]#[T.past_not_future_and_mixed_idx]
state¹⁻_vol = vcat(state¹⁻, 1)

if length(state) > 1
    state²⁻ = state[2]#[T.past_not_future_and_mixed_idx]
end

if length(state) == 3
    state³⁻ = state[3]#[T.past_not_future_and_mixed_idx]
end

shock_independent = copy(data_in_deviations[:,i])
ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
if length(state) > 1
    ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
end
ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
if length(state) == 3
    ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)
    ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1/2, 1)
end
if length(𝐒) == 3
    ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
end 

# shock_independent = 𝐒¹ᵉfact \ shock_independent

if length(𝐒) == 2
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)    
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
    # 𝐒ⁱ = 𝐒¹² \ 𝐒²ᵉ / 2
elseif length(𝐒) == 3
    𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(state¹⁻_vol, state¹⁻_vol))  
    𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
    # 𝐒ⁱ = 𝐒¹²³ \ 𝐒²ᵉ / 2
end





# x, matched = find_shocks(Val(filter_algorithm), 
#                         kron_buffer,
#                         kron_buffer2,
#                         J,
#                         𝐒ⁱ,
#                         𝐒ⁱ²ᵉ,
#                         shock_independent)


nExo = Int(sqrt(length(kron_buffer)))

x = zeros(nExo)

max_iter = 1000

for i in 1:max_iter
    ℒ.kron!(kron_buffer, x, x)

    ℒ.lmul!(0, J)
    ℒ.axpy!(1, ℒ.I(nExo), J)
    ℒ.kron!(kron_buffer2, J, x)

    # ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer)
    # ℒ.axpby!(1, shock_independent, -1, res)
    Δx = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron_buffer2) \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer + ℒ.kron(x,ℒ.kron(x,x)))
    # println(ℒ.norm(Δx))
    if i > 6 && ℒ.norm(Δx) < tol
        # println(i)
        break
    end
    
    ℒ.axpy!(1, Δx, x)
    # x += Δx
end

maximum(abs, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x))

xsol1 = copy(x)
# x ≈ xsol





# Interior point solver
using ForwardDiff



jacc(x) = ForwardDiff.jacobian(x-> begin
    aug_state₁ = [state[1]; 1; x]
    aug_state₁̂ = [state[1]; 0; x]
    aug_state₂ = [state[2]; 0; zero(x)]
    aug_state₃ = [state[3]; 0; zero(x)]

    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

    𝐒⁻¹ * aug_state₁ + 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2 + 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6
end,x)

jacc(x)


shock_independent = collect(shock_independent)

x = zeros(nExo)
α = 1
β = (1/eps())^2
for i in 1:1000
    ∂x = jacc(x)
    
    aug_state₁ = [state[1]; 1; x]
    aug_state₁̂ = [state[1]; 0; x]
    aug_state₂ = [state[2]; 0; zero(x)]
    aug_state₃ = [state[3]; 0; zero(x)]

    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

    tmp = 𝐒⁻¹ * aug_state₁ + 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2 + 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6 - 

    Δx = -∂x \ tmp

    aug_state₁ = [state[1]; 1; tmp]
    aug_state₁̂ = [state[1]; 0; tmp]

    kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

    println(ℒ.norm(-tmp + 𝐒⁻¹ * aug_state₁ + 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2 + 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6))
    # println(ℒ.norm(Δx))

    # β = (ℒ.norm(shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x)) / tol)^.9

    if i > 3 && ℒ.norm(Δx) < tol
        println(i)
        break
    end
    
    ℒ.axpy!(1, Δx, x)
    # x += Δx
end

maximum(abs, shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * ℒ.kron!(kron_buffer, x, x))
sum(abs2,xsol)
sum(abs2,xsol1)

xsol = copy(x)

xsol1 - xsol





nExo = Int(sqrt(length(kron_buffer)))

# res = zero(shock_independent) .+ 1

x = zeros(nExo)

max_iter = 1000

for i in 1:max_iter
    ℒ.kron!(kron_buffer, x, x)

    ℒ.lmul!(0, J)
    ℒ.axpy!(1, ℒ.I(nExo), J)
    ℒ.kron!(kron_buffer2, J, x)

    ∂x = (𝐒ⁱ + 2 * 𝐒ⁱ²ᵉ * kron_buffer2)

    ∂x̂ = ℒ.lu!(∂x, check = false)

    if !ℒ.issuccess(∂x̂) 
        return x, false
    end
    # ℒ.mul!(res, 𝐒ⁱ²ᵉ, kron_buffer)
    # ℒ.axpby!(1, shock_independent, -1, res)
    Δx = ∂x̂ \ (shock_independent - 𝐒ⁱ * x - 𝐒ⁱ²ᵉ * kron_buffer)
    # println(ℒ.norm(Δx))
    if i > 6 && ℒ.norm(Δx) < tol
        # println(i)
        break
    end
    
    ℒ.axpy!(1, Δx, x)
    # x += Δx

    if !all(isfinite.(x))
        return x, false
    end
end


using SpeedMapping


sol = speedmapping(zeros(Int(sqrt(length(kron_buffer)))); 
m! = (x̂, x) ->  begin
                    ℒ.kron!(kron_buffer, x, x)
                    ℒ.mul!(x̂, 𝐒ⁱ, kron_buffer)
                    ℒ.axpby!(1, shock_independent, -1, x̂)
                end, tol = tol, maps_limit = 10000)#, stabilize = true, σ_min = 1)

# println(sol.maps)

x = sol.minimizer

return x, maximum(abs, shock_independent - 𝐒ⁱ * ℒ.kron!(kron_buffer, x, x) - x) < tol




for i in axes(data_in_deviations,2)
    state¹⁻ = state[1]#[T.past_not_future_and_mixed_idx]
    state¹⁻_vol = vcat(state¹⁻, 1)

    if length(state) > 1
        state²⁻ = state[2]#[T.past_not_future_and_mixed_idx]
    end

    if length(state) == 3
        state³⁻ = state[3]#[T.past_not_future_and_mixed_idx]
    end
    
    shock_independent = copy(data_in_deviations[:,i])
    ℒ.mul!(shock_independent, 𝐒¹⁻ᵛ, state¹⁻_vol, -1, 1)
    if length(state) > 1
        ℒ.mul!(shock_independent, 𝐒¹⁻, state²⁻, -1, 1)
    end
    ℒ.mul!(shock_independent, 𝐒²⁻ᵛ, ℒ.kron(state¹⁻_vol, state¹⁻_vol), -1/2, 1)
    if length(state) == 3
        ℒ.mul!(shock_independent, 𝐒¹⁻, state³⁻, -1, 1)
        ℒ.mul!(shock_independent, 𝐒²⁻, ℒ.kron(state¹⁻, state²⁻), -1/2, 1)
    end
    if length(𝐒) == 3
        ℒ.mul!(shock_independent, 𝐒³⁻ᵛ, ℒ.kron(state¹⁻_vol, ℒ.kron(state¹⁻_vol, state¹⁻_vol)), -1/6, 1)   
    end 

    # shock_independent = 𝐒¹ᵉfact \ shock_independent
    
    if length(𝐒) == 2
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol)    
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2 
        # 𝐒ⁱ = 𝐒¹² \ 𝐒²ᵉ / 2
    elseif length(𝐒) == 3
        𝐒ⁱ = 𝐒¹ᵉ + 𝐒²⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), state¹⁻_vol) + 𝐒³⁻ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(state¹⁻_vol, state¹⁻_vol))  
        𝐒ⁱ²ᵉ = 𝐒²ᵉ / 2
        # 𝐒ⁱ = 𝐒¹²³ \ 𝐒²ᵉ / 2
    end

    # x, jacc, matchd = find_shocks(Val(:fixed_point), state isa Vector{Float64} ? [state] : state, 𝐒, data_in_deviations[:,i], observables, T)
    x, matched = find_shocks(Val(filter_algorithm), 
                                kron_buffer,
                                kron_buffer2,
                                J,
                                𝐒ⁱ,
                                𝐒ⁱ²ᵉ,
                                shock_independent)

    if length(𝐒) == 2
        jacc = -(𝐒ⁱ + 𝐒²ᵉ * ℒ.kron(ℒ.I(T.nExo), x))
    elseif length(𝐒) == 3
        jacc = -(𝐒ⁱ + 𝐒²ᵉ * ℒ.kron(ℒ.I(T.nExo), x) + 𝐒³ᵉ * ℒ.kron(ℒ.I(T.nExo), ℒ.kron(x, x)))
    end

    if !matched 
        return -Inf # it can happen that there is no solution. think of a = bx + cx² where a is negative, b is zero and c is positive  
    end

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x)
    end

    if length(𝐒) == 2
        if state isa Vector{Float64}
            aug_state = [state; 1; x]

            state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2
        else
            aug_state₁ = [state[1]; 1; x]
            aug_state₂ = [state[2]; 0; zero(x)]

            state = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
        end
    elseif length(𝐒) == 3
        if state isa Vector{Float64}
            aug_state = [state; 1; x]

            state = 𝐒⁻¹ * aug_state + 𝐒⁻² * ℒ.kron(aug_state, aug_state) / 2 + 𝐒⁻³ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
        else
            aug_state₁ = [state[1]; 1; x]
            aug_state₁̂ = [state[1]; 0; x]
            aug_state₂ = [state[2]; 0; zero(x)]
            aug_state₃ = [state[3]; 0; zero(x)]

            kron_aug_state₁ = ℒ.kron(aug_state₁, aug_state₁)

            state = [𝐒⁻¹ * aug_state₁, 𝐒⁻¹ * aug_state₂ + 𝐒⁻² * kron_aug_state₁ / 2, 𝐒⁻¹ * aug_state₃ + 𝐒⁻² * ℒ.kron(aug_state₁̂, aug_state₂) + 𝐒⁻³ * ℒ.kron(kron_aug_state₁,aug_state₁) / 6]
        end
    end
    # state = state_update(state, x)
end
