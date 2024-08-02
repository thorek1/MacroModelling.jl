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




# 𝓂 = Gali_2015_chapter_3_nonlinear
𝓂 = Smets_Wouters_2007
SSS(𝓂, algorithm = :third_order, parameters = :csadjcost => 6.0144)
import MacroModelling: get_and_check_observables, check_bounds, minimize_distance_to_initial_data, get_relevant_steady_state_and_state_update, replace_indices, minimize_distance_to_data, match_data_sequence!, match_initial_data!,calculate_loglikelihood, String_input, calculate_second_order_stochastic_steady_state, expand_steady_state, calculate_third_order_stochastic_steady_state

parameter_values = 𝓂.parameter_values
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



get_loglikelihood(𝓂, data[1:6,1:10], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :pruned_second_order)
get_loglikelihood(𝓂, data[1:6,1:10], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :second_order)
get_loglikelihood(𝓂, data[1:6,1:10], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :third_order)
get_loglikelihood(𝓂, data[1:6,1:10], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :pruned_third_order)
get_loglikelihood(𝓂, data[:,1:50], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :pruned_second_order)
@benchmark get_loglikelihood(𝓂, data[:,1:50], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :pruned_second_order)
@profview for i in 1:10 get_loglikelihood(𝓂, data[:,1:50], 𝓂.parameter_values, filter = :inversion, presample_periods = presample_periods, algorithm = :pruned_second_order) end

# LBFGS
# BenchmarkTools.Trial: 9 samples with 1 evaluation.
#  Range (min … max):  531.746 ms … 663.865 ms  ┊ GC (min … max):  8.58% … 22.30%
#  Time  (median):     590.161 ms               ┊ GC (median):     9.69%
#  Time  (mean ± σ):   593.482 ms ±  46.505 ms  ┊ GC (mean ± σ):  10.43% ±  5.18%

#   █      █     █  █         █   █                 █   █       █  
#   █▁▁▁▁▁▁█▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁█ ▁
#   532 ms           Histogram: frequency by time          664 ms <

#  Memory estimate: 500.38 MiB, allocs estimate: 430979.

# Newton 
# BenchmarkTools.Trial: 42 samples with 1 evaluation.
#  Range (min … max):   88.842 ms … 244.718 ms  ┊ GC (min … max):  0.00% … 51.86%
#  Time  (median):     108.004 ms               ┊ GC (median):     7.39%
#  Time  (mean ± σ):   121.536 ms ±  35.704 ms  ┊ GC (mean ± σ):  10.84% ± 11.15%

#   ▂ ▂▂▅ █   ▂                                                    
#   █▅█████▁▅██▅█▁▁▁█▅█▁▁▅▁▁▅▁▁▁▁▁▁▅▁▁▁▁▁▅▁▁▁▁▁▁▁▅▁▁▁▁▁▅▁▁▁▁▁▁▁▁▅ ▁
#   88.8 ms          Histogram: frequency by time          245 ms <

#  Memory estimate: 109.55 MiB, allocs estimate: 34516.

# fixed point
# BenchmarkTools.Trial: 48 samples with 1 evaluation.
#  Range (min … max):   84.736 ms … 206.011 ms  ┊ GC (min … max):  0.00% … 55.83%
#  Time  (median):     100.113 ms               ┊ GC (median):     8.15%
#  Time  (mean ± σ):   105.939 ms ±  25.324 ms  ┊ GC (mean ± σ):  12.58% ± 12.29%

#   ▅ ▂▂ ▅█▅▅▂▂                                                    
#   ████▅███████▅█▁▅▅▁▁▁▁▅▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▅▁▁▁▁▁▅ ▁
#   84.7 ms          Histogram: frequency by time          206 ms <

#  Memory estimate: 109.15 MiB, allocs estimate: 30671.

# speedmapping
# BenchmarkTools.Trial: 29 samples with 1 evaluation.
#  Range (min … max):  114.272 ms … 442.587 ms  ┊ GC (min … max):  3.86% … 62.60%
#  Time  (median):     164.811 ms               ┊ GC (median):     7.31%
#  Time  (mean ± σ):   179.922 ms ±  66.736 ms  ┊ GC (mean ± σ):  14.94% ± 14.43%

#      ▂    ▂▅█                                                    
#   ▅▅▅██▅▁▁███▁▅▅▁▁▁▅▅▁▅▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
#   114 ms           Histogram: frequency by time          443 ms <

#  Memory estimate: 109.26 MiB, allocs estimate: 32771.
# -4103.49488937482
# -4290.895952421868
1
# Zygote.gradient(𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


# Zygote.gradient(data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations)[1]

# res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations)[1]


# Zygote.gradient(state -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), state)[1][1][TT.past_not_future_and_mixed_idx]

# FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), stt -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, stt, warmup_iterations), state)[1][1][TT.past_not_future_and_mixed_idx]



# zygS = Zygote.gradient(𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


# finS = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]


# isapprox(zygS, finS, rtol = eps(Float32))


# 𝓂
# data_in_deviations
# algorithm
warmup_iterations = 0
verbose = false
tol = 1e-12

observables = collect(axiskeys(data_in_deviations,1))

data_in_deviations = collect(data_in_deviations)

# @assert observables isa Vector{String} || observables isa Vector{Symbol} "Make sure that the data has variables names as rows. They can be either Strings or Symbols."

sort!(observables)

observables = observables isa String_input ? observables .|> Meta.parse .|> replace_indices : observables

# solve model given the parameters
# sss, converged, SS_and_pars, solution_error, ∇₁, ∇₂, 𝐒₁, 𝐒₂ = calculate_second_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)
stochastic_steady_state, converged, SS_and_pars, solution_error, ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝐒₃ = calculate_third_order_stochastic_steady_state(𝓂.parameter_values, 𝓂, pruning = true)
# if !converged 
#     @error "No solution for these parameters."
# end

all_SS = expand_steady_state(SS_and_pars,𝓂)

state = [zeros(𝓂.timings.nVars), collect(stochastic_steady_state) - all_SS]

state_update = function(pruned_states::Vector{Vector{T}}, shock::Vector{S}) where {T,S}
    aug_state₁ = [pruned_states[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shock]
    aug_state₂ = [pruned_states[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shock)]
    
    return [𝐒₁ * aug_state₁, 𝐒₁ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2] # strictly following Andreasen et al. (2018)
end


if state isa Vector{Float64}
    pruning = false
else
    pruning = true
end

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))

states = zeros(𝓂.timings.nVars, n_obs)
shocks = zeros(𝓂.timings.nExo, n_obs)

precision_factor = 1.0



# initial_state = [[0.026462623232903052, -8.690540263878549e-15, -0.02466178444578176, 0.038920237079948794, 8.171564493434605e-15, 0.06221520072047692, -0.01487102043134231, -0.005489084211789118, -0.00465341417282116, -0.01260702415366692, -0.08714261744439267, 0.0739543481375812, 0.09563482699565397, -0.3648732584477732, -0.4084193055733367, -4.3107210401037366e-15, 0.04188989455476198, -0.014850432395287808, -1.1528425396387512, 0.9381305847123859, 4.910787206503637, 1.1313176432632328, 1.4236668644542847, 0.4705573446432324, 0.008500169113901885, 15.165620747215335, 0.03900047975356873, 0.08136747422644752, 0.10888458738068275, 0.15535640769299064, 1.4289989122392746, 2.087086235856355, 4.779760092277518, 4.980507644470384, -0.03525898928037083, 0.006190527473671918, -2.7123613124229866, -0.037620387206518585, 6.0286576324786694e-5, 4.0458527259480715e-15, 2.9223946326176366e-15, 0.0333263537672178, 3.3326353767217807, -0.12879505388545912, -0.1546707296192402, 0.31053943215745217, 0.22250150314081116, 0.04444310316829846, -0.0054890842117900854, -0.004653414172820972, 4.4443103168298475, 0.10475302634852636, 0.9381305847123859, 4.910787206503637, 0.04188989455478643, -1.2463514938548194e-14, 1.84381323641558e-14, 0.09545248958249573, 0.1167670691294581, -1.1864490541495645, -1.2953874848624545, 0.16724243551150106, 0.2546108442669791, -6.426213965997142, -0.38682990629370095, -0.32793808565597793], [-0.030724299078812092, 5.056228471212394e-5, 0.1028555319605677, 0.026176837306986805, 0.00012592983941755588, -2.5306722535971616e-31, -0.001017205840616109, -0.0003754637111896405, 0.000830778853496218, 0.0022507450837867746, 8.99378894374074e-18, -0.025560875722479927, 0.000904295350352368, 0.5288005776118725, 1.703371669147931, 0.006091904849782325, 0.03587679637651427, 0.16186543510513962, 0.7912411386017792, 0.0, 0.0, 0.6169905909719112, 0.33585945073414075, 0.007653648707276972, 16.33868668936927, -1.3508555765026236, 0.5276307711292133, -2.0185468637834722e-17, 0.05289358517647352, 0.042104314556099635, 0.003705814749613179, 0.0721630798352782, 2.3151014224144966, 1.3977201948788966, -0.03733193753422527, -0.0004901742791423181, -2.8718265938089926, -0.003177839117218464, 0.0, -0.004264333394837714, -0.0042643333948268304, -0.04136769165245463, -4.1367691653461085, 0.05412501358358899, 0.017351501971688364, 1.7001473852109648e-15, 7.24042202088713e-16, -0.05418877847670353, -0.0003754637111912907, 0.0008307788534961673, -5.4188778475596635, 0.01089301061950517, 0.0, 0.0, -0.002560752496949795, -0.0338503533411338, -0.033850353341562923, 0.00012166817301492543, -0.07412394143294794, 0.25888569058073596, 0.1772781505124731, -0.05137763040540752, 0.0007094257978505283, -2.834180341484686, -0.05407643157469602, 0.038699304922573605]]


# data_in_deviations[:,1] = [0.21467675206250503, 3.91268098112642, -0.150055162067323, -0.127983944132123, -5.35688015089181, 1.1084829202150113, -0.8354075740313684]


shocks[:,1]

aug_state₁ = [state[1][𝓂.timings.past_not_future_and_mixed_idx]; 1; shocks[:,1]]
aug_state₂ = [state[2][𝓂.timings.past_not_future_and_mixed_idx]; 0; zero(shocks[:,1])]


data_in_deviations[:,1] - (𝐒₁ * aug_state₁ + 𝐒₁ * aug_state₂ + 𝐒₂ * ℒ.kron(aug_state₁, aug_state₁) / 2)[cond_var_idx]


data_in_deviations[:,1] - (𝐒₁[cond_var_idx,:] * aug_state₁ + 𝐒₁[cond_var_idx,:] * aug_state₂ + 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2)


data_in_deviations[:,1] - 𝐒₁[cond_var_idx,:] * aug_state₂ - (𝐒₁[cond_var_idx,:] * aug_state₁ + 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2)


data_in_deviations[:,1] - 𝐒₁[cond_var_idx,:] * aug_state₂ - 𝐒₁[cond_var_idx,1:𝓂.timings.nPast_not_future_and_mixed+1] * aug_state₁[1:𝓂.timings.nPast_not_future_and_mixed+1] - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2)


nᵉ = 𝓂.timings.nExo
s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
e_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), ones(Bool, nᵉ)))

tmp = ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1) |> sparse
shock_idxs = tmp.nzind

tmp = ℒ.kron(e_in_s⁺, e_in_s⁺) |> sparse
shock²_idxs = tmp.nzind

shockvar_idxs = setdiff(shock_idxs, shock²_idxs)

tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
var_idxs = tmp.nzind

cond_var_idx = cond_var_idx[1:6]

shock_independent = data_in_deviations[1:6,1] - 𝐒₁[cond_var_idx,:] * aug_state₂ - 𝐒₁[cond_var_idx,1:𝓂.timings.nPast_not_future_and_mixed+1] * aug_state₁[1:𝓂.timings.nPast_not_future_and_mixed+1] - 𝐒₂[cond_var_idx,var_idxs] * ℒ.kron(aug_state₁[1:𝓂.timings.nPast_not_future_and_mixed+1], aug_state₁[1:𝓂.timings.nPast_not_future_and_mixed+1]) / 2


# shock_independent - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2)
# inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * shock_independent
# inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * 𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end]
# inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * 𝐒₂[cond_var_idx,:]

# inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * shock_independent - (aug_state₁[end-𝓂.timings.nExo+1:end] + inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * 𝐒₂[cond_var_idx,shock_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁) / 2 )


shock_independent - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shock_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁) / 2 )


shock_independent - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[1:end-𝓂.timings.nExo]) / 2 + 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 )


ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[1:end-𝓂.timings.nExo])

𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * vec(aug_state₁[1:end-𝓂.timings.nExo] * aug_state₁[end-𝓂.timings.nExo+1:end]')

(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo]) )  * aug_state₁[end-𝓂.timings.nExo+1:end]




shock_independent - ((𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo]) )  * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 )




shock_independent - ((𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) * aug_state₁[end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 )


shock_independent 
- (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) * aug_state₁[end-𝓂.timings.nExo+1:end] 
- 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 


A = (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) \ shock_independent

𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 

𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[end-𝓂.timings.nExo+1:end]) / 2 * aug_state₁[end-𝓂.timings.nExo+1:end]


X = aug_state₁[end-𝓂.timings.nExo+1:end]

shock_independent - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) * X - 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(X, X) / 2 

A = shock_independent
B = (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo]))
C = 𝐒₂[cond_var_idx,shock²_idxs] / 2

A - B * X - C * ℒ.kron(X, X)

B\A - B\C * ℒ.kron(X, X) - X
B\A - B\C * ℒ.kron(X, X) - X
ℒ.kron(X, X)
vec(X * X')


Y = A - B * X - C * ℒ.kron(X, X)

B*∂X - C * ℒ.kron(X, ∂X) - ∂X
- B - 2 * C * ℒ.kron(ℒ.I(𝓂.timings.nExo), X)
ℒ.kron(ℒ.I(𝓂.timings.nExo), X)
ℒ.kron(ℒ.I(𝓂.timings.nExo), ones(𝓂.timings.nExo)) .* X'

X' * ℒ.I(𝓂.timings.nExo)
2 * C * vec(ℒ.I(𝓂.timings.nExo)) * X'



A = shock_independent
B = (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo]))
C = 𝐒₂[cond_var_idx,shock²_idxs] / 2

XX = zeros(𝓂.timings.nExo)

for i in 1:100
    ΔX = (B + 2 * C * ℒ.kron(ℒ.I(𝓂.timings.nExo), XX)) \ (A - B * XX - C * ℒ.kron(XX, XX))
    # ΔX = (ℒ.I(𝓂.timings.nExo) + 2 * B \ C * ℒ.kron(ℒ.I(𝓂.timings.nExo), XX)) \ (B \ A - XX - B \ C * ℒ.kron(XX, XX))
    # ΔX = (C \ B + 2 * ℒ.kron(ℒ.I(𝓂.timings.nExo), XX)) \ (C \ A - C \ B * XX - ℒ.kron(XX, XX))
    if ℒ.norm(ΔX) < 1e-14
        println(i)
        break
    end
    XX += ΔX
end
ℒ.norm(XX)

SSState = zeros(𝓂.timings.nPast_not_future_and_mixed)

for i in 1:10000
    SSStateold = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] * SSState + 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,var_idxs] * ℒ.kron(vcat(SSState,1), vcat(SSState,1)) / 2
    # println(ℒ.norm(SSStateold - SSState))
    if ℒ.norm(SSStateold - SSState) < 1e-16
        println(i)
        break
    end
    SSState = SSStateold
end


isapprox(𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed] * SSState + 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,var_idxs] * ℒ.kron(vcat(SSState,1), vcat(SSState,1)) / 2, SSState, rtol = 1e-14)
# same for stochastic steady state
# second order

nᵉ = 𝓂.timings.nExo
s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))

tmp = ℒ.kron(s_in_s⁺, s_in_s) |> sparse
var_idxs2 = tmp.nzind

tmp = ℒ.kron(s_in_s⁺, s_in_s⁺) |> sparse
var_idxs = tmp.nzind


A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,var_idxs2]
B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,var_idxs]

A + B * ℒ.kron(vcat(SSState,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)




XX = zeros(𝓂.timings.nPast_not_future_and_mixed)
XX = SSState

jacc = ForwardDiff.jacobian(XX->(B̂ * ℒ.kron(vcat(XX,1), vcat(XX,1)) / 2), XX)
ℒ.norm(jacc - B * ℒ.kron(vcat(XX,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))


jacc = ForwardDiff.jacobian(XX->(A * XX + B̂ * ℒ.kron(vcat(XX,1), vcat(XX,1)) / 2 - XX), XX)
ℒ.norm(jacc - (A + B * ℒ.kron(vcat(XX,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))

for i in 1:100
    ΔX = -(A + B * ℒ.kron(vcat(XX,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) \ (A * XX + B̂ * ℒ.kron(vcat(XX,1), vcat(XX,1)) / 2 - XX)
    println(ℒ.norm(ΔX))
    if ℒ.norm(ΔX) < 1e-14
        println(i)
        break
    end
    XX += ΔX
end


ℒ.norm(SSState - XX)

ℒ.norm(A * XX + B̂ * ℒ.kron(vcat(XX,1), vcat(XX,1)) / 2 - XX)


ℒ.norm(A * SSState + B̂ * ℒ.kron(vcat(SSState,1), vcat(SSState,1)) / 2 - SSState)


nᵉ = 𝓂.timings.nExo

s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))

kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)

kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)

kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)

kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)

kron_s_s_s⁺ = ℒ.kron(ℒ.kron(s_in_s, s_in_s⁺), s_in_s)

A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
C = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
Ĉ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]
C̄ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s_s⁺]


x = zeros(𝓂.timings.nPast_not_future_and_mixed)

max_iters = 100
# SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
for i in 1:max_iters
    Δx = -(A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2 - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x)
    println(ℒ.norm(Δx))
    if i > 6 && ℒ.norm(Δx) < tol
        println(i)
        break
    end
    x += Δx
end


ℒ.norm(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x)

XX = ones(𝓂.timings.nPast_not_future_and_mixed)
x = ones(𝓂.timings.nPast_not_future_and_mixed)

jacc = ForwardDiff.jacobian(x->A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6 - x, XX)

jacc = ForwardDiff.jacobian(x->A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 - x, XX)

ℒ.norm(jacc - (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))



jacc = ForwardDiff.jacobian(x-> Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, XX)

jacc - C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) / 2

vec(vec(x * x') * x')


hess = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> Ĉ * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), vcat(x,1)) / 6, y), XX)


hess = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), x-> Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6, XX)[1]

# hess = Zygote.jacobian(x -> Zygote.jacobian(x-> Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) / 6,x), XX)

C̄ * ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed))) / 2

ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1) == ℒ.kron(vcat(x,1), vcat(x,1))
ℒ.kron(vcat(x,1), (ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1))) == ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

vec(vcat(x,1) * (ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1))') ≈ ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

vec(vcat(x,1) * (ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1))') ≈ ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1), ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1)) * vcat(x,1) ≈ ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))


ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1), ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1)) * vcat(x,1) ≈ ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1)))

vec(vec(ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * (ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed + 1)) * vcat(x,1))')


C̄ * (ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), vcat(x,1))) + ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed))) + ℒ.kron(vcat(x,1), ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.I(𝓂.timings.nPast_not_future_and_mixed))))  - hess'

𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,ℒ.kron(ℒ.kron(s_in_s, s_in_s), s_in_s⁺)] * ℒ.kron(ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)), vcat(x,1)) * 3
𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,ℒ.kron(ℒ.kron(s_in_s, s_in_s⁺), s_in_s)] * ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))
𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,ℒ.kron(ℒ.kron(s_in_s⁺, s_in_s), s_in_s)] * ℒ.kron(vcat(x,1), ℒ.kron(ℒ.I(𝓂.timings.nPast_not_future_and_mixed), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))




ℒ.norm(jacc - (A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))


jacc = ForwardDiff.jacobian(XX->(A * XX + B̂ * ℒ.kron(vcat(XX,1), vcat(XX,1)) / 2 - XX), XX)
ℒ.norm(jacc - (A + B * ℒ.kron(vcat(XX,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)))



function calculate_third_order_stochastic_steady_state(::Val{:Newton}, 
                                                        𝐒₁::AbstractSparseMatrix{Float64}, 
                                                        𝐒₂::AbstractSparseMatrix{Float64}, 
                                                        𝐒₃::AbstractSparseMatrix{Float64},
                                                        𝓂::ℳ;
                                                        tol::AbstractFloat = 1e-14)
    nᵉ = 𝓂.timings.nExo

    s_in_s⁺ = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), zeros(Bool, nᵉ)))
    s_in_s = BitVector(vcat(ones(Bool, 𝓂.timings.nPast_not_future_and_mixed ), zeros(Bool, nᵉ + 1)))

    kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
    
    kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)
    
    kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
    
    kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s, kron_s⁺_s⁺)
    
    A = 𝐒₁[𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nPast_not_future_and_mixed]
    B = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s]
    B̂ = 𝐒₂[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺]
    C = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s_s⁺_s⁺]
    Ĉ = 𝐒₃[𝓂.timings.past_not_future_and_mixed_idx,kron_s⁺_s⁺_s⁺]

    x = zeros(𝓂.timings.nPast_not_future_and_mixed)

    max_iters = 100
    # SSS .= 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
    for i in 1:max_iters
        Δx = -(A + B * ℒ.kron(vcat(x,1), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) + C * ℒ.kron(ℒ.kron(vcat(x,1), vcat(x,1)), ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) - ℒ.I(𝓂.timings.nPast_not_future_and_mixed)) \ (A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2 + Ĉ * ℒ.kron(vcat(x,1), ℒ.kron(vcat(x,1), vcat(x,1))) - x)
        if i > 6 && ℒ.norm(Δx) < tol
            break
        end
        x += Δx
    end

    return x, isapprox(A * x + B̂ * ℒ.kron(vcat(x,1), vcat(x,1)) / 2, x, rtol = tol)
end




ℒ.kron(ℒ.I(𝓂.timings.nExo), X)
# aug_state₁[end-𝓂.timings.nExo+1:end] - inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * shock_independent + inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2
# inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * shock_independent + inv(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end]) * 𝐒₂[cond_var_idx,:] * ℒ.kron(aug_state₁, aug_state₁) / 2

shock_independent_future =     𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] \ shock_independent
inv𝐒₁𝐒₂ = (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) \ 𝐒₂[cond_var_idx,shock²_idxs] / 2


𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] \ shock_independent - (𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) \ 𝐒₂[cond_var_idx,shock²_idxs] / 2 * ℒ.kron(x, x) - x



shock_independent - 𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * ((𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) \ 𝐒₂[cond_var_idx,shock²_idxs] / 2 * ℒ.kron(x, x)) - 𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] * x


Shock = zeros(𝓂.timings.nExo)
# ℒ.kron(Shock, aug_state₁)
kron_buffer = ℒ.kron(Shock, Shock)

# @profview for i in 1:1000 
# @benchmark begin
    aug_state₁[end-𝓂.timings.nExo+1:end] .= 0

    i = 0
    max_update = 1.0
    while max_update > 1e-14
        i += 1
        Shock = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron!(kron_buffer, aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end])
        if i % 1 == 0
            max_update = maximum(abs, Shock - aug_state₁[end-𝓂.timings.nExo+1:end])
            # println(max_update)
        end
        # println(Shock)
        @views aug_state₁[end-𝓂.timings.nExo+1:end] = Shock
    end
# end

J = [-0.2077485365475398 -0.12139726426848271 -0.13341459155391763 -0.3660228077477204 0.27427153855442354 -0.08780621567930127; -7.2225311691756096 -3.960394277853059 -1.1850921677420012 -6.300417083854628 -4.351040030663629 0.5005573731190642; 0.130021571630954 0.027030373377584432 -0.017022502756414013 -0.598842947339625 -0.4288813028289572 0.02144335025495961; 0.3107942790860842 0.25358457803503165 0.08525325543642537 0.29469280347568333 0.20142711850907727 -0.08287269894243712; 0.03146449678470407 0.06805478688367494 0.19737511286527035 0.06649352545730573 -0.005486502192908558 0.19932025200494496; -0.0009687983288967529 -1.3042538585691665 -0.07663917501852041 -0.284232348573774 -0.19610075197359755 0.06347429430145608; -0.024242925542140626 0.04253426375942233 -0.43069336064014985 -0.0806511112870578 0.020130320062903953 0.08894580401068636]



shock_independent - ((𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) * Shock + 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(Shock, Shock) / 2 )

J' 
(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) +  𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), Shock)


inv𝐒₁𝐒₂ * ℒ.kron(ℒ.I(𝓂.timings.nExo)[:,1:length(cond_var_idx)], Shock) * 2 + ℒ.I(𝓂.timings.nExo)[:,1:length(cond_var_idx)]

inv(ℒ.svd(𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])))

𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])


shock_independent - ((𝐒₁[cond_var_idx,end-𝓂.timings.nExo+1:end] + 𝐒₂[cond_var_idx,shockvar_idxs] * ℒ.kron(ℒ.I(𝓂.timings.nExo), aug_state₁[1:end-𝓂.timings.nExo])) * Shock + 𝐒₂[cond_var_idx,shock²_idxs] * ℒ.kron(Shock, Shock) / 2 )



shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(zeros(𝓂.timings.nExo), zeros(𝓂.timings.nExo))

using Optimization, OptimizationNLopt

prob²(x, _) = sqrt(sum(abs2, shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(x, x) - x))
u0 = zero(Shock)
p = [1.0, 100.0]

prob²(u0, p)
f = OptimizationFunction(prob², AutoForwardDiff())
prob = OptimizationProblem(f, u0, p, ub = zero(u0) .+ 1e2, lb = zero(u0) .- 1e2)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)

@benchmark sol = solve(prob, NLopt.LD_TNEWTON_PRECOND_RESTART(), maxiters = 10000)

sol = solve(prob, NLopt.LN_SBPLX(), maxiters = 1000000)

sol = solve(prob, NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_LBFGS(), maxiters = 100000)

sol = solve(prob, NLopt.GN_ISRES(), maxiters = 100000)

maximum(abs, shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(sol.minimizer, sol.minimizer) - sol.minimizer)


using OptimizationNOMAD

sol = solve(prob, NOMADOpt(), maxiters = 100000)

maximum(abs, shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(sol.minimizer, sol.minimizer) - sol.minimizer)


using OptimizationOptimisers

prob = OptimizationProblem(f, u0, p)

sol = solve(prob, Lion(), maxiters = 10000000)

maximum(abs, shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(sol.minimizer, sol.minimizer) - sol.minimizer)

prob = OptimizationProblem(f, sol.minimizer, p)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)




using SpeedMapping

C = zero(Shock)
C̄ = zero(Shock)

@benchmark sol = speedmapping(zeros(𝓂.timings.nExo); m! = (C̄, C) -> begin
                                                            ℒ.kron!(kron_buffer, C, C)
                                                            ℒ.mul!(C̄, inv𝐒₁𝐒₂, kron_buffer)
                                                            ℒ.axpby!(1, shock_independent_future, -1, C̄)
                                                        end,
        tol = 1e-14, maps_limit = 10000)


@benchmark begin
    # aug_state₁[end-𝓂.timings.nExo+1:end] *= 0

    # C = zero(Shock)
    # C̄ = zero(Shock)
    # i = 0
    # Shck = zero($Shock)
    # shck = zero($Shock)
    max_update = 1.0
    while max_update > 1e-14
        ℒ.kron!($kron_buffer, shck, shck)
        ℒ.mul!(Shck, $inv𝐒₁𝐒₂, $kron_buffer)
        ℒ.axpby!(1, $shock_independent_future, -1, Shck)
        i += 1
        # Shock = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron!(kron_buffer, aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end])
        if i % 10 == 0
            ℒ.axpy!(-1,Shck, shck)
            max_update = maximum(abs, shck)
        end
        # println(shck)
        copyto!(shck, Shck)
    end
end  setup = begin Shck = zero(Shock); shck = zero(Shock); i = 0 end



i = 0
Shck = zero(Shock)
shck = zero(Shock)
max_update = 1.0

while max_update > 1e-13
    ℒ.kron!(kron_buffer, shck, shck)
    ℒ.mul!(Shck, inv𝐒₁𝐒₂, kron_buffer)
    ℒ.axpby!(1, shock_independent_future, -1, Shck)
    # Shock = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(aug_state₁[end-𝓂.timings.nExo+1:end], aug_state₁[end-𝓂.timings.nExo+1:end])

    i += 1
    if i % 1 == 0
        ℒ.axpy!(-1, Shck, shck)
        max_update = maximum(abs, shck)
        println(max_update)
    end
    
    copyto!(shck, Shck)
end



max_iter = 1000
# @benchmark begin
x = zero(Shock)
i = 0
fx = ones(𝓂.timings.nExo)
while maximum(abs, fx) > 1e-14
    fx = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(x, x) - x
    J = inv𝐒₁𝐒₂ * 2 * ℒ.kron(ℒ.I(𝓂.timings.nExo), x) + ℒ.I(𝓂.timings.nExo)
println(fx)
    Δx = J \ fx
    
    x_new = x + Δx

    x = x_new
    i += 1
end
# end

kron_buffer2 = ℒ.kron(ℒ.I(𝓂.timings.nExo), x)

@benchmark begin

res = zero($shock_independent_future) .+ 1
J = zeros($𝓂.timings.nExo, length($shock_independent_future))
x = zero($Shock)
x̂ = zero($Shock)
Δx = zero($Shock)
# max_iter = 1000
while maximum(abs, res) > 1e-14
    ℒ.kron!($kron_buffer, x, x)
    ℒ.mul!(res, $inv𝐒₁𝐒₂, $kron_buffer)
    ℒ.axpby!(1, $shock_independent_future, -1, res)
    ℒ.axpy!(-1, x, res)
    # res = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(x, x) - x

    J .= ℒ.I(𝓂.timings.nExo)
    ℒ.kron!($kron_buffer2, J, x)
    ℒ.mul!(J, inv𝐒₁𝐒₂, $kron_buffer2, 2, 1)
    # J = inv𝐒₁𝐒₂ * 2 * ℒ.kron(ℒ.I(𝓂.timings.nExo), x) + ℒ.I(𝓂.timings.nExo)

    # Δx = J \ res
    ℒ.ldiv!(Δx, ℒ.factorize(J), res)
    ℒ.axpy!(1, Δx, x)
    # x̂ = x + Δx
    # x = x̂
end

end
shck

# sol = speedmapping(zeros(𝓂.timings.nExo); m! = (C̄, C) -> C̄ = shock_independent_future - inv𝐒₁𝐒₂ * ℒ.kron(C, C),
        # tol = tol, maps_limit = 10000)
# sol.minimizer

Shock - shocks[:,1]

initial_state = state


for i in axes(data_in_deviations,2)
    res = begin Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                        zeros(𝓂.timings.nExo), 
                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward) end

    matched = Optim.minimum(res) < 1e-12

    if !matched
        res = begin Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                        zeros(𝓂.timings.nExo), 
                        Optim.LBFGS(), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward) end

        matched = Optim.minimum(res) < 1e-12
    end

    @assert matched "Numerical stabiltiy issues for restrictions in period $i."

    x = Optim.minimizer(res)

    state = state_update(state, x)

    states[:,i] = pruning ? sum(state) : state
    shocks[:,i] = x
end
    



T = 𝓂.timings

precision_factor = 1.0

n_obs = size(data_in_deviations,2)

cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))




warmup_iterations = 0



state = copy(state[1])

precision_factor = 1.0

n_obs = size(data_in_deviations,2)

obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

t⁻ = T.past_not_future_and_mixed_idx

shocks² = 0.0
logabsdets = 0.0

if warmup_iterations > 0
    if warmup_iterations >= 1
        jac = 𝐒[obs_idx,end-T.nExo+1:end]
        if warmup_iterations >= 2
            jac = hcat(𝐒[obs_idx,1:T.nPast_not_future_and_mixed] * 𝐒[t⁻,end-T.nExo+1:end], jac)
            if warmup_iterations >= 3
                Sᵉ = 𝐒[t⁻,1:T.nPast_not_future_and_mixed]
                for _ in 1:warmup_iterations-2
                    jac = hcat(𝐒[obs_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[t⁻,end-T.nExo+1:end], jac)
                    Sᵉ *= 𝐒[t⁻,1:T.nPast_not_future_and_mixed]
                end
            end
        end
    end

    jacdecomp = ℒ.svd(jac)


    x = jacdecomp \ data_in_deviations[:,1]

    warmup_shocks = reshape(x, T.nExo, warmup_iterations)

    for i in 1:warmup_iterations-1
        ℒ.mul!(state, 𝐒, vcat(state[t⁻], warmup_shocks[:,i]))
        # state = state_update(state, warmup_shocks[:,i])
    end

    for i in 1:warmup_iterations
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jac[:,(i - 1) * T.nExo+1:i*T.nExo] ./ precision_factor))
        end
    end

    shocks² += sum(abs2,x)
end



state = [copy(state) for _ in 1:size(data_in_deviations,2)+1]
shocks² = 0.0
logabsdets = 0.0
y = zeros(length(obs_idx))
x = [zeros(T.nExo) for _ in 1:size(data_in_deviations,2)]

jac = 𝐒[obs_idx,end-T.nExo+1:end]

if T.nExo == length(observables)
    logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
    jacdecomp = ℒ.lu(jac)
    invjac = inv(jacdecomp)
else
    logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
    jacdecomp = ℒ.svd(jac)
    invjac = inv(jacdecomp)
end

logabsdets *= size(data_in_deviations,2) - presample_periods

@views 𝐒obs = 𝐒[obs_idx,1:end-T.nExo]

for i in axes(data_in_deviations,2)
    @views ℒ.mul!(y, 𝐒obs, state[i][t⁻])
    @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
    ℒ.mul!(x[i],invjac,y)
    
    # x = invjac * (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])
    # x = 𝐒[obs_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[obs_idx,1:end-T.nExo] * state[t⁻])

    if i > presample_periods
        shocks² += sum(abs2,x[i])
    end

    # # copyto!(state_reduced, 1, state, t⁻)
    # for (i,v) in enumerate(t⁻)
    #     state_reduced[i] = state[v]
    # end
    # copyto!(state_reduced, T.nPast_not_future_and_mixed + 1, x, 1, T.nExo)
    
    ℒ.mul!(state[i+1], 𝐒, vcat(state[i][t⁻], x[i]))
    # state[i+1] =  𝐒 * vcat(state[i][t⁻], x[i])
    # state = state_update(state, x)
end

llh = -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2






obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

t⁻ = T.past_not_future_and_mixed_idx

# precomputed matrices
M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹
M3  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]'
M⁴  = M² * M¹




∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

for t in reverse(axes(data_in_deviations,2))
    ∂data_in_deviations[:,t]        -= invjac' * x[t]

    if t > 1
        ∂data[:,t:end]              .= M² * ∂data[:,t:end]
        
        ∂data[:,t-1]                += M¹ * x[t]

        ∂data_in_deviations[:,t-1]  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)
    end
end 

∂data_in_deviations

maximum(abs, ∂data_in_deviations - res)

isapprox(∂data_in_deviations, res, rtol = eps(Float32))

# invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data * ones(size(data_in_deviations,2))


∂data_in_deviations[:,5] -= invjac' * x[5]

∂data_in_deviations[:,4] -= invjac' * x[4]
∂data_in_deviations[:,4] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[5]

∂data_in_deviations[:,3] -= invjac' * x[3]
∂data_in_deviations[:,3] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[4]
∂data_in_deviations[:,3] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[5]

∂data_in_deviations[:,2] -= invjac' * x[2]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[3]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[4]
∂data_in_deviations[:,2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M¹ * x[5]

∂data_in_deviations[:,1] -= invjac' * x[1]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[2]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[3]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M¹ * x[4]
∂data_in_deviations[:,1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M² * M² * M¹ * x[5]
res3
for t in 3:-1:1 # reverse(axes(data_in_deviations,2))
    ∂data_in_deviations[:,t] -= invjac' * x[t]

    if t > 1
        ∂data_in_deviations[:,t-1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹ * x[t]
    end

    if t > 2
        ∂data_in_deviations[:,t-2] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M² * M¹ * x[t]

        # ∂data[:,t-2]    += M¹ * x[t]
        # ∂data = M² * ∂data
        # ∂data_in_deviations[:,1:end-1] += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,2:end]
    end
    
    if t > 3
        ∂data[:,t-3]    += M² * M¹ * x[t]
        # ∂data¹[:,t-3]   += M² * 𝐒[t⁻,1:end-T.nExo]' * M¹ * x[t]

        # ∂data²[:,t-3]   -= M² * 𝐒[obs_idx, 1:end-T.nExo]' * M³ * x[t]

        ∂data = M² * ∂data

        # ∂data¹ = M² * ∂data¹

        # ∂data² = M² * ∂data²

        ∂data_in_deviations += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data
    end
end

∂data_in_deviations

(𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
M²^2 * M¹ * x[4]


# -2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
# -2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]
2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')^3  * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[5]
(𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]')^2 

invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (M² * (∂data¹ + ∂data²) + (∂data¹ + ∂data²))

∂data_in_deviations

res5 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:5])[1]

res4 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:4])[1]

res3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:3])[1]

res2 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), data_in_deviations -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations), data_in_deviations[:,1:2])[1]

res5[:,1:4] - res4
res4[:,1:3] - res3

invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]'

∂data¹ + ∂data²



# i = 4

-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]

-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# i = 3
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]


# i = 2
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]


# i = 1
2 * invjac' * x[1]



invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]


invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]


invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

- invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] 



invjac'  * x[1] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2] + invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] - invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+ 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+ 2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]'   *   𝐒[t⁻,1:end-T.nExo]'   *   𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2]

invjac' * x[1]




M¹  = 𝐒[obs_idx, 1:end-T.nExo]' * invjac' 
M²  = 𝐒[t⁻,1:end-T.nExo]' - M¹ * 𝐒[t⁻,end-T.nExo+1:end]'
M³  = invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * M¹
M⁴  = M² * M¹


N = 2

∂𝐒 = zero(𝐒)
    
∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

∂state = zero(state[1])

for t in N:-1:1 # reverse(axes(data_in_deviations,2))
    ∂state[t⁻]                                  .= M² * ∂state[t⁻]

    if t > presample_periods
        ∂state[t⁻]                              += M¹ * x[t]

        ∂data_in_deviations[:,t]                -= invjac' * x[t]

        ∂𝐒[obs_idx, end-T.nExo + 1:end]         += invjac' * x[t] * x[t]'

        if t > 1
            ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
            
            ∂data[:,t-1]                        += M¹ * x[t]
    
            ∂data_in_deviations[:,t-1]          += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

            ∂𝐒[obs_idx, 1:end-T.nExo]           += invjac' * x[t] * state[t][t⁻]'
            ∂𝐒[obs_idx, end-T.nExo + 1:end]     -= M³ * x[t] * x[t-1]'
            ∂𝐒[t⁻,end-T.nExo + 1:end]           += M¹ * x[t] * x[t-1]'
        end

        if t > 2
            ∂𝐒[t⁻,1:end-T.nExo]                 += M¹ * x[t] * state[t-1][t⁻]'
            ∂𝐒[obs_idx, 1:end-T.nExo]           -= M³ * x[t] * state[t-1][t⁻]'
        end
    end

    if t > 2
        ∂𝐒ᵗ⁻        .= 𝐒[t⁻,1:end-T.nExo]' * ∂𝐒ᵗ⁻ / vcat(state[t-1][t⁻], x[t-1])' * vcat(state[t-2][t⁻], x[t-2])'
        
        if t > presample_periods
            ∂𝐒ᵗ⁻    += M⁴ * x[t] * vcat(state[t-2][t⁻], x[t-2])'
        end

        ∂𝐒[t⁻,:]    += ∂𝐒ᵗ⁻
    end
end

∂𝐒[obs_idx,end-T.nExo+1:end] -= (N - presample_periods) * invjac' / 2


res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:N], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

maximum(abs, ∂𝐒 - res)

∂𝐒 - res

finS - ∂𝐒

data = data_in_deviations[:,1:N]

res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:N # axes(data,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        if i > presample_periods
            shocks² += sum(abs2,X)
        end
        # shocks² += sum(abs2,state[i+1])
    end

    return -shocks²/2
end, 𝐒)[1]





# derivatives wrt to s

obs_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))

𝐒¹ = 𝐒[obs_idx, end-T.nExo+1:end]
𝐒² = 𝐒[obs_idx, 1:end-T.nExo]
𝐒³ = 𝐒[t⁻,:]



res4 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:4], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:3], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res2 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:2], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res1 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:1], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

res3 - res2
res2 - res1


hcat(𝐒[obs_idx, 1:end-T.nExo]' * invjac', 𝐒[obs_idx, 1:end-T.nExo]' * invjac') * vcat(x[3] * vcat(state[1][t⁻], x[1])', x[3] * vcat(state[2][t⁻], x[2])')


iterator = 

# t = 1
# ∂𝐒[obs_idx, :]                  += invjac' * x[1] * vcat(state[1][t⁻], x[1])'

# t = 2
# ∂𝐒[obs_idx, :]                  += invjac' * x[2] * vcat(state[2][t⁻], x[2])'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2] * vcat(state[1][t⁻], x[1])'
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[2] * vcat(state[1][t⁻], x[1])'

# t = 3
# ∂𝐒[obs_idx, :]                  += invjac' * x[3] * vcat(state[3][t⁻], x[3])'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' - 𝐒[t⁻,1:end-T.nExo]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[1][t⁻], x[1])'

∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                        += (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3] * vcat(state[1][t⁻], x[1])'




N = size(data_in_deviations,2)

∂𝐒 = zero(𝐒)
    
∂𝐒ᵗ⁻ = copy(∂𝐒[t⁻,:])

∂data_in_deviations = zero(data_in_deviations)

∂data = zeros(length(t⁻), size(data_in_deviations,2) - 1)

∂Stmp = [M¹ for _ in 1:size(data_in_deviations,2)]

for t in 2:size(data_in_deviations,2)
    ∂Stmp[t] = M² * ∂Stmp[t-1]
end

∂state = zero(state[1])

for t in reverse(axes(data_in_deviations,2))
    if t > presample_periods
        ∂𝐒[obs_idx, :]         += invjac' * x[t] * vcat(state[t][t⁻], x[t])'

        if t > 1
            ∂data[:,t:end]                      .= M² * ∂data[:,t:end]
            
            ∂data[:,t-1]                        += M¹ * x[t]
    
            ∂data_in_deviations[:,t-1]          += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂data[:,t-1:end] * ones(size(data_in_deviations,2) - t + 1)

            M²mult = ℒ.I(size(M²,1))

            for tt in t-1:-1:1
                ∂𝐒[obs_idx, :]                      -= invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * ∂Stmp[t-tt] * x[t] * vcat(state[tt][t⁻], x[tt])'
    
                ∂𝐒[t⁻,:]                            += ∂Stmp[t-tt] * x[t] * vcat(state[tt][t⁻], x[tt])'

                M²mult                              *= M²
            end

        end
    end
end

∂𝐒[obs_idx,end-T.nExo+1:end] -= (N - presample_periods) * invjac' / 2





NN = 3

res = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 𝐒 -> calculate_loglikelihood(Val(filter), observables, 𝐒, data_in_deviations[:,1:N], TT, presample_periods, initial_covariance, state, warmup_iterations), 𝐒)[1]

maximum(abs, ∂𝐒 - res)


∂𝐒 = zero(𝐒)

∂𝐒[obs_idx,end-T.nExo+1:end] -= (NN - presample_periods) * invjac' / 2

i = 1

t = 1
# ForwardDiff.gradient(𝐒¹ -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹)
∂𝐒[obs_idx, end-T.nExo+1:end]       += invjac' * x[t] * x[t]'

# ForwardDiff.gradient(𝐒² -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒²)
# zero because the initial state is 0


t = 2
# ForwardDiff.gradient(x -> -.5 * sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]       += invjac' * x[t] * x[t]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - x * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒²)
# invjac' * x[t] * state[t][t⁻]'

∂𝐒[obs_idx, :]                  += invjac' * x[t] * vcat(state[t][t⁻], x[t])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒¹)
∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-1]'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))), 𝐒³)
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'



t = 3

# tmpres = ForwardDiff.gradient(𝐒 -> -.5 * sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[t⁻,:] * vcat(𝐒[t⁻,:] * vcat(state[i][t⁻], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][t⁻])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[t⁻,:] * vcat(state[i][t⁻], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][t⁻])))))), 𝐒)


# ForwardDiff.gradient(x -> -.5 * sum(abs2, x \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - x * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)

∂𝐒[obs_idx, :]                  += invjac' * x[t] * vcat(state[t][t⁻], x[t])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), x \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-1]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - x * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-1][t⁻]'

∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-2]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - x * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-2][t⁻]'

# ∂𝐒[obs_idx, :]                  += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], x \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒¹)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * x[t-2]'

# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - x * state[i][t⁻])))))), 𝐒²)
# ∂𝐒[obs_idx, end-T.nExo+1:end]    += -invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * state[t-2][t⁻]'

# ∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


∂𝐒[obs_idx, :]                  += invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * (𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' - 𝐒[t⁻,1:end-T.nExo]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'


# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * x * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * x * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
∂𝐒[t⁻,:]                        += 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-1][t⁻], x[t-1])'

# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ∂𝐒[t⁻,:]                        += 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'

# ∂𝐒[t⁻,:]                        += ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ForwardDiff.gradient(x -> -.5 * sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * x * vcat(state[i][t⁻], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][t⁻])))))), 𝐒³)
# ∂𝐒[t⁻,:]                        += -𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'
∂𝐒[t⁻,:]                        += (𝐒[t⁻,1:end-T.nExo]' - 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]') * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[t] * vcat(state[t-2][t⁻], x[t-2])'

# res3-res2

maximum(abs, ∂𝐒 - tmpres)
maximum(abs, ∂𝐒 - res)


# for i in axes(data_in_deviations,2)
#     x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])
#     state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], x[i])
#     shocks² += sum(abs2,x[i])
# end
# return shocks²

st = T.past_not_future_and_mixed_idx
𝐒endo = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒exo = 𝐒[cond_var_idx, end-T.nExo+1:end]


# ∂state = zero(state[1])

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

∂data_in_deviations = zero(data_in_deviations)

∂state¹ = zero(state[1][st])


for i in reverse(axes(data_in_deviations,2))
    ∂state¹ .= (𝐒[st,1:end-T.nExo] - 𝐒[st,end-T.nExo+1:end] * invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * ∂state¹
    ∂state¹ -= (invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * 2 * x[i]

    if i < size(data_in_deviations,2)
        ∂data_in_deviations[:,i] -= invjac' * ((invjac * 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:])' * 2 * x[i+1])[end-T.nExo+1:end]
    end
    ∂data_in_deviations[:,i] += invjac' * 2 * x[i]

    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                .= 𝐒[st,1:end-T.nExo]' * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]') * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end



T = TT
cond_var_idx = indexin(observables,sort(union(TT.aux,TT.var,TT.exo_present)))
res = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1), stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    state[1] .= stat
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[1]#_in_deviations[:,1:2])


isapprox(res, ∂data_in_deviations, rtol = eps(Float32))
isapprox(res, ∂𝐒, rtol = eps(Float32))

res - ∂𝐒

i = 1

𝐒¹ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒² = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒³ = 𝐒[st,:]
sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))
sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))




state[i+1] = 𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))

state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   +   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))


state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   
+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]   

+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st]   
+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))



𝐒[:,1:end-T.nExo] * 𝐒[st,1:end-T.nExo] * state[i][st]   
+  𝐒[:,1:end-T.nExo] * 𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) 

+   𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[st,1:end-T.nExo] * state[i][st]   
+   𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))




# res = FiniteDiff.finite_difference_gradient(stat -> begin
ForwardDiff.gradient(stat->begin
shocks² = 0.0
# stat = zero(state[1])
for i in 1:2 # axes(data_in_deviations,2)
    stat = 𝐒[:,1:end-T.nExo] * stat[T.past_not_future_and_mixed_idx] + 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * stat[T.past_not_future_and_mixed_idx]))

    # shocks² += sum(abs2,X)
    shocks² += sum(abs2,stat)
end

return shocks²
end, state[1])[st]#_in_deviations[:,1:2])


# i = 4
ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]+2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]


# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[t⁻,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[4]



# i = 3
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), data_in_deviations[:,i])

-2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))), data_in_deviations[:,i])

2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac' * x[3]

# i = 2
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), data_in_deviations[:,i])
-2 * invjac' * 𝐒[t⁻,end-T.nExo+1:end]' * 𝐒[obs_idx, 1:end-T.nExo]' * invjac'  * x[2]

# i = 1
# ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (x - 𝐒² * state[i][st])), data_in_deviations[:,i])
2 * invjac'  * x[1]





(ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))))), data_in_deviations[:,i]) + ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (x - 𝐒² * state[i][st])))), data_in_deviations[:,i]) + ForwardDiff.gradient(x -> sum(abs2, 𝐒¹ \ (x - 𝐒² * state[i][st])), data_in_deviations[:,i])) / 2




xxx =  𝐒[st,1:end-T.nExo]' * 𝐒[:,1:end-T.nExo]' * 2 * state[i+2]
xxx += 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end - T.nExo + 1:end]' * 𝐒[:,1:end-T.nExo]' * 2 * state[i+2]
xxx -= 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+2]
xxx -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end - T.nExo + 1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+2]


xxx +=  𝐒[:,1:end-T.nExo]' * 2 * state[i+1]
xxx -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end - T.nExo + 1:end]' * 2 * state[i+1]


xxx * ∂state∂shocks²
# 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒[st,1:end-T.nExo] * state[i][st]   +   𝐒[st,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))




state[i+2] = 𝐒[:,1:end-T.nExo] * (𝐒[:,1:end-T.nExo] * state[i][st])[st] 

∂state∂X * ∂X∂stt * ∂state∂state * ∂state∂shocks²



∂state = zero(state[i])

∂state∂shocks² = 2 * state[i+1]#[st]


∂state∂state = 𝐒[:,1:end-T.nExo]'# * ∂state∂shocks²

∂state∂state -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[:,end-T.nExo+1:end]'

∂state[st] += ∂state∂state * ∂state∂shocks²

∂state[st] += ∂state∂state * ∂state

∂state∂state * ∂state∂shocks²


∂state∂state * 2 * state[i+1] + ∂stt∂stt * ∂state∂state * 2 * state[i+2]
∂state∂shocks² += 2 * state[i+2]#[st]

out = zero(state[i+2][st])

for i in 2:-1:1
    out .= ∂stt∂stt * out
    out += (∂state∂state * 2 * state[i+1])
end


𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]


out = zero(state[i][st])

for i in 1:-1:1
    out .= 𝐒[st,1:end-T.nExo]' * out
    out -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]
end
out



∂state = zero(state[i][st])

for i in reverse(axes(data_in_deviations,2))
    # out .= (𝐒[st,1:end-T.nExo]' - 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]') * out
    ∂state .= (𝐒[st,1:end-T.nExo] - 𝐒[st,end-T.nExo+1:end] * invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * ∂state
    ∂state -= (invjac * 𝐒[cond_var_idx, 1:end-T.nExo])' * 2 * x[i]
end

∂state



∂data_in_deviations = zero(data_in_deviations)

for i in reverse(axes(data_in_deviations,2))
    if i < size(data_in_deviations,2)
        ∂data_in_deviations[:,i] -= invjac' * ((invjac * 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:])' * 2 * x[i+1])[end-T.nExo+1:end]
    end
    ∂data_in_deviations[:,i] += invjac' * 2 * x[i]
end



𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[st,1:end-T.nExo]'
𝐒³̂ = 𝐒[st,end-T.nExo+1:end]'

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

for i in reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                 = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end




∂𝐒 = zero(𝐒)

for i in 2:-1:1#reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i] * x[i]' # [cond_var_idx, end-T.nExo+1:end]

    if i > 1
        ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (𝐒[st,:] * vcat(state[i-1][st], x[i-1]))' .* (invjac' * 2 * x[i])
        ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * x[i-1]'
        
        ∂𝐒[st,:] -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'

        # ∂𝐒[cond_var_idx, :] += invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i])[end-T.nExo+1:end] * vcat(-state[i-1][st], x[i-1])'
    end
end

maximum(abs, (res - ∂𝐒) ./ res)

unique((res - ∂𝐒) ./ ∂𝐒) .|> abs |> sort

(𝐒[st,:] * vcat(state[i-1][st], x[i-1]))' .* (invjac' * 2 * x[i])




∂𝐒 = zero(𝐒)

for i in 3:-1:1#reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]         -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])' # [cond_var_idx, end-T.nExo+1:end]

    if i > 1
        ∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]                -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒[st,:]    += 2 * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]') * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i] * vcat(state[1][st], x[1])'
    end
end



𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[st,1:end-T.nExo]'
𝐒³̂ = 𝐒[st,end-T.nExo+1:end]'

∂𝐒 = zero(𝐒)
∂𝐒st = copy(∂𝐒[st,:])

for i in reverse(axes(data_in_deviations,2))
    ∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[i] * vcat(state[i][st], x[i])'
    
    if i > 1
        ∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
        ∂𝐒[st,:]            -= 2 * 𝐒²̂ * invjac' * x[i] * vcat(state[i-1][st], x[i-1])'
    end

    if i > 2
        ∂𝐒st                 = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
        ∂𝐒st                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
        ∂𝐒[st,:]            += ∂𝐒st
    end
end


maximum(abs, res - ∂𝐒)

maximum(abs, Base.filter(isfinite, (res - ∂𝐒) ./ res))


maximum(abs, Base.filter(isfinite, (res5 - ∂𝐒) ./ res5))


# i = 5

∂𝐒st = zero(∂𝐒[st,:])
∂𝐒stl = zero(∂𝐒[st,:])

i = 5
# ∂𝐒st                += 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

i = 4
∂𝐒st                  = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

i = 3
∂𝐒st                  = 𝐒³̃ * ∂𝐒st * (vcat(state[i-1][st], x[i-1])' \ vcat(state[i-2][st], x[i-2])')
∂𝐒st                 += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[i] * vcat(state[i-2][st], x[i-2])'
∂𝐒stl                += ∂𝐒st

∂𝐒st + ∂𝐒stl


∂𝐒 = zero(𝐒)
∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[3][st], x[3])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[2][st], x[2])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[1][st], x[1])'


∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[2][st], x[2])'
∂𝐒[st,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[1][st], x[1])'

∂𝐒[st,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[3] * vcat(state[1][st], x[1])'

∂𝐒 - res5



∂𝐒 = zero(𝐒)


𝐒¹̂ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒²̂ = 𝐒[cond_var_idx, 1:end-T.nExo]'
𝐒³̃ = 𝐒[t⁻,1:end-T.nExo]'
𝐒³̂ = 𝐒[t⁻,end-T.nExo+1:end]'

# terms for i = 5
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[5] * vcat(state[4][t⁻], x[4])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[3][t⁻], x[3])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[5] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[5] * vcat(state[5][t⁻], x[5])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[5] * vcat(state[4][t⁻], x[4])'




# terms for i = 4
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[4] * vcat(state[3][t⁻], x[3])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * 𝐒³̃ * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[4] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[4] * vcat(state[4][t⁻], x[4])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[4] * vcat(state[3][t⁻], x[3])'


# terms for i = 3
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[3] * vcat(state[2][t⁻], x[2])'
∂𝐒[t⁻,:]                += 2 * (𝐒²̂ * invjac' * 𝐒³̂ - 𝐒³̃) * 𝐒²̂ * invjac' * x[3] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[3] * vcat(state[3][t⁻], x[3])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[3] * vcat(state[2][t⁻], x[2])'


# terms for i = 2
∂𝐒[t⁻,:]                -= 2 * 𝐒²̂ * invjac' * x[2] * vcat(state[1][t⁻], x[1])'

∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[2] * vcat(state[2][t⁻], x[2])'
∂𝐒[cond_var_idx, :]     += 2 * invjac' * 𝐒³̂ * 𝐒²̂ * invjac' * x[2] * vcat(state[1][t⁻], x[1])'


# terms for i = 1
∂𝐒[cond_var_idx, :]     -= 2 * invjac' * x[1] * vcat(state[1][t⁻], x[1])'


maximum(abs, res - ∂𝐒/2)

maximum(abs, Base.filter(isfinite, (res - ∂𝐒) ./ ∂𝐒))


32
31

21


𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' - 𝐒[st,1:end-T.nExo]'

maximum(abs, res - ∂𝐒)

∂𝐒 = zero(𝐒)

i = 1

∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= invjac' * 2 * x[i] * x[i]'


∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+1] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * vcat(state[i][st], x[i])'


∂𝐒[cond_var_idx, :] -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'

# 𝐒³
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'



using FiniteDifferences
res = FiniteDifferences.grad(central_fdm(4,1), 𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, 𝐒)[1]#_in_deviations[:,1:2])

res4 = -res+rest
res5 = res-rest

maximum(abs, res - ∂𝐒)

∂state




Xx = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,1] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])

Xx = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[T.past_not_future_and_mixed_idx,:] * 
vcat(
    state[1][T.past_not_future_and_mixed_idx], 
    𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,1] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])
    ))



- 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]   +  𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[2] -  𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[2] 
# 𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[1]


out = zero(state[i][st])

for i in reverse(axes(data_in_deviations,2))
    out .= (𝐒[T.past_not_future_and_mixed_idx,1:end-T.nExo]' - 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]') * out
    out -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i]
end
out


(invjac * 𝐒[cond_var_idx, 1:end-T.nExo])'



res = FiniteDiff.finite_difference_gradient(stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    state[1] = stat
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])

        state[i+1] = 𝐒 * vcat(state[i][st], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[st]#_in_deviations[:,1:2])




(∂state∂X * ∂X∂state + ∂state∂state) * ∂state∂shocks²


∂state∂state[:,st] * ∂state∂state * ∂state∂shocks²

∂stt∂stt = 𝐒[st,1:end-T.nExo]'# * ∂state∂shocks²

∂X∂state = 𝐒[:,end-T.nExo+1:end]'# * ∂state∂shocks²

∂X∂stt = 𝐒[st,end-T.nExo+1:end]'# * ∂state∂shocks²

∂state∂X = -𝐒[cond_var_idx, 1:end-T.nExo]' * invjac'

∂stt∂stt*(∂state∂X * ∂X∂state * ∂state∂shocks² + ∂state∂state * ∂state∂shocks²)

∂state∂X * ∂X∂stt * ∂state∂state * ∂state∂shocks² + ∂stt∂stt * ∂state∂state * ∂state∂shocks² + (∂state∂X * ∂X∂state * ∂state∂shocks² + ∂state∂state * ∂state∂shocks²)




∂stt∂stt * ∂state∂X * ∂X∂state * ∂state∂shocks²

∂stt∂stt * ∂state∂state * ∂state∂shocks²

res = FiniteDiff.finite_difference_gradient(stat -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    state[1] .= stat
    for i in 1:2 # axes(data_in_deviations,2)
        # X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        # state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        state[i+1] = 𝐒[:,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx] + 𝐒[:,end - T.nExo + 1:end] * (𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx]))

        # shocks² += sum(abs2,X)
        shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, state[1])[st]#_in_deviations[:,1:2])



FiniteDiff.finite_difference_gradient(𝐒exo -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])), 𝐒exo)# + ∂v


# there are multiple parts to it. first the effect of the previous iteration through this one and then the direct effect

uuuu = (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])

uuu = vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ uuuu)

uu = 𝐒[st,:] * uuu

u = data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * uu

X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ u

sum(abs2, X)


sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))))


Zygote.jacobian(x -> vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.jacobian(x ->  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]


Zygote.gradient(x -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]


Zygote.gradient(x -> sum(abs2,  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1] + Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]



Zygote.gradient(x -> sum(abs2,  x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]

Zygote.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒)[1]

# derivative wrt S for two periods
ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒) - ∂𝐒



ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)



res = FiniteDiff.finite_difference_gradient(𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:3 # axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
        # shocks² += sum(abs2,state[i+1])
    end

    return shocks²
end, 𝐒) - ∂𝐒#_in_deviations[:,1:2])

res3 = res1-res2


st = T.past_not_future_and_mixed_idx
𝐒¹ = 𝐒[cond_var_idx, end-T.nExo+1:end]
𝐒² = 𝐒[cond_var_idx, 1:end-T.nExo]
𝐒³ = 𝐒[st,:]

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))

sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))
, 𝐒²)

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st]))))
, 𝐒²)


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))
, 𝐒[cond_var_idx, end-T.nExo+1:end])

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st]))))))
, 𝐒[cond_var_idx, end-T.nExo+1:end])


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+2] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])), 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * 𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))))), 𝐒)


∂𝐒 = zero(𝐒)

i = 1

# 𝐒¹
∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+2] * x[i+2]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])


# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i]'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])



∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i+1]'
Zygote.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]



# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i]'
# Zygote.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, end-T.nExo+1:end])[1]




# 𝐒²
∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * x[i+2] * state[i+2][st]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])

# 0
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])


∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * state[i+1][st]'
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²)

# 0
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²) + ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒[cond_var_idx, 1:end-T.nExo])

ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒 * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒 * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))), 𝐒²)


# 𝐒³
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)


∂𝐒[st,:]                            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)




∂𝐒[st,:]                            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒³)



# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'


∂𝐒 = zero(𝐒)

i = 1

∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= invjac' * 2 * x[i] * x[i]'

∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+1] * x[i+1]'
∂𝐒[cond_var_idx,:]                  -= 2 * invjac' * x[i+1] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]                            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * vcat(state[i][st], x[i])'



# 𝐒¹
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  -= 2 * invjac' * x[i+2] * x[i+2]'
# ∂𝐒[cond_var_idx, end-T.nExo+1:end]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * x[i+1]'

# 𝐒²
# ∂𝐒[cond_var_idx, 1:end-T.nExo]      -= 2 * invjac' * x[i+2] * state[i+2][st]'
# ∂𝐒[cond_var_idx, 1:end-T.nExo]      -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * state[i+1][st]'


∂𝐒[cond_var_idx, :] -= 2 * invjac' * x[i+2] * vcat(state[i+2][st], x[i+2])'
∂𝐒[cond_var_idx, :] += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'

# 𝐒³
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i+1][st], x[i+1])'
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+2] * vcat(state[i][st], x[i])'







# i = 4
ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))), 𝐒²)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒²)



ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒 * state[i][st])))))))), 𝐒²)



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * x[i+3] * x[i+3]'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i+1]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
∂𝐒[cond_var_idx, 1:end-T.nExo]  += 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i+2]'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# ∂𝐒[cond_var_idx, 1:end-T.nExo]  -= 2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * x[i]'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# cancels out



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)
# cancels out


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒 \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒 \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒 \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒 \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒¹)




∂𝐒 = zero(𝐒)

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+2][st], x[i+2])'




# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
# ∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'
# cancels out

# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'


# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
∂𝐒[st,:]            += 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'



# ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒³ * vcat(𝐒³ * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³)
# cancels out
∂𝐒[st,:]            -= 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+2][st], x[i+2])'

∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'

∂𝐒[st,:]            -= 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'

∂𝐒[st,:]            += 2 * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]'  * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i+1][st], x[i+1])'

∂𝐒[st,:]            += 2 * 𝐒[st,1:end-T.nExo]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+3] * vcat(state[i][st], x[i])'


ForwardDiff.gradient(𝐒 -> sum(abs2, 𝐒¹ \ (data_in_deviations[:,i+3] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))), 𝐒¹ \ (data_in_deviations[:,i+2] - 𝐒² * 𝐒 * vcat(𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])), 𝐒¹ \ (data_in_deviations[:,i+1] - 𝐒² * 𝐒 * vcat(state[i][st], 𝐒¹ \ (data_in_deviations[:,i] - 𝐒² * state[i][st])))))))), 𝐒³) - ∂𝐒[st,:]






𝐒[st,1:end-T.nExo]'

2 * invjac' * x[i+1] * x[i+1]' * inv(ℒ.svd(x[i+1]))'

2 * invjac' * x[i+1] * x[i+1]' / x[i+1]' * x[i]'

invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * 2 * invjac' * x[i+1] * x[i+1]' * (x[i+1]' \ x[i]')


2 * invjac' * x[i+1] * x[i]'

2 * invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]' - 2 * invjac' * x[i+1] * x[i+1]'

2 * invjac' * (𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * x[i+1] * x[i]' - x[i+1] * x[i+1]')

∂𝐒 = zero(𝐒)

i = 1
∂x∂shocks² = 2 * x[i]

∂u∂x = invjac'

∂𝐒∂shocks² = ∂u∂x * ∂x∂shocks² * x[i]' # [cond_var_idx, end-T.nExo+1:end]

∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i] * x[i]' # [cond_var_idx, end-T.nExo+1:end]

∂𝐒[cond_var_idx, end-T.nExo+1:end] -= invjac' * 2 * x[i+1] * x[i+1]' # [cond_var_idx, end-T.nExo+1:end]


# next S
∂x∂shocks² = 2 * x[i+1]

∂𝐒∂u = - (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))'

∂𝐒2∂shocks² = ∂𝐒∂u .* (∂u∂x * ∂x∂shocks²)

∂𝐒2∂shocks² = - (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))' .* (∂u∂x * ∂x∂shocks²)

# ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (state[i+1][st] * (invjac' * 2 * x[i+1])')'
# ∂𝐒[cond_var_idx, 1:end-T.nExo] -= (state[i+1][st] * x[i+1]' * invjac * 2)'
∂𝐒[cond_var_idx, 1:end-T.nExo] -= invjac' * x[i+1] * state[i+1][st]' * 2

# next S
∂uu∂u = - 𝐒[cond_var_idx, 1:end-T.nExo]'

∂𝐒∂uu = uuu'

∂𝐒2∂shocks² = ∂𝐒∂uu .* (∂uu∂u * ∂u∂x * ∂x∂shocks²)

# ∂𝐒[st,:] -= (vcat(state[i][st], x[i]) * (𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])')'
∂𝐒[st,:] -= 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1] * vcat(state[i][st], x[i])'




# next S
i = 1
∂x∂shocks² = 2 * x[i+1]

∂u∂x = invjac'

∂uu∂u = 𝐒[cond_var_idx, 1:end-T.nExo]'

∂uuu∂uu = 𝐒[st,:]'

∂𝐒∂shocks² = invjac' * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * x[i]'

∂𝐒[cond_var_idx, end-T.nExo+1:end] += invjac' * 𝐒[st,end-T.nExo+1:end]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1] * x[i]'


ForwardDiff.gradient(𝐒 -> sum(abs2,  𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) + sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] *  𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))), 𝐒) - ∂𝐒




# next S
i = 1
∂x∂shocks² = 2 * x[i+1]

∂u∂x = invjac'

∂uu∂u = 𝐒[cond_var_idx, 1:end-T.nExo]'

∂uuu∂uu = 𝐒[st,:]'

∂uuuu∂uuu = invjac'

∂𝐒∂uuuu = - state[i][st]'

∂uuuu∂uuu * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * ∂𝐒∂uuuu

# ∂𝐒[cond_var_idx, 1:end-T.nExo] += ∂uuuu∂uuu * (∂uuu∂uu * ∂uu∂u * ∂u∂x * ∂x∂shocks²)[end-T.nExo+1:end] * ∂𝐒∂uuuu

∂𝐒[cond_var_idx, 1:end-T.nExo] -= invjac' * (𝐒[st,:]' * 𝐒[cond_var_idx, 1:end-T.nExo]' * invjac' * 2 * x[i+1])[end-T.nExo+1:end] * state[i][st]'




# u = 

ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])


ForwardDiff.jacobian(XX -> -XX * uu, 𝐒[cond_var_idx, 1:end-T.nExo][:,:])

ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])

Zygote.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] -XX * uu)), 𝐒[cond_var_idx, 1:end-T.nExo][:,:])[1]


Zygote.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ XX), (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * uu))[1]


∂data_in_deviations∂x[:,i] -= invjac' * (𝐒[T.past_not_future_and_mixed_idx,:]' * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])) * 𝐒endo' * invjac' * 2 * x[i+1])[end-T.nExo+1:end]



FiniteDiff.finite_difference_jacobian(x -> data_in_deviations[:,i+1] - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])



ForwardDiff.gradient(XX -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - XX * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))))), 𝐒[cond_var_idx, 1:end-T.nExo])

ForwardDiff.jacobian(x -> data_in_deviations[:,i+1] - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])

xxx = ForwardDiff.jacobian(x -> - x * (𝐒[st,:] * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st]))), 𝐒[cond_var_idx, 1:end-T.nExo])


sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒 * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st]))


# starting with the iterated indirect effect



FiniteDiff.finite_difference_gradient(𝐒 -> sum(abs2, 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i+1] - 𝐒[cond_var_idx, 1:end-T.nExo] * (𝐒 * vcat(state[i][st], 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][st])))[st])), 𝐒)# + ∂v

FiniteDiff.finite_difference_gradient(𝐒exo2 -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo2 \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st])), 𝐒exo)# + ∂v

invjac' * (𝐒[T.past_not_future_and_mixed_idx,:]' * 𝐒endo'  * invjac' * ∂𝐒[cond_var_idx,end-T.nExo+1:end]')[end-T.nExo+1:end,:]

res = FiniteDiff.finite_difference_gradient(𝐒 -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:2 # axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, 𝐒)#_in_deviations[:,1:2])

isapprox(res, ∂𝐒, rtol = eps(Float32))

res - ∂𝐒

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:10 # axes(data_in_deviations,2)
        X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data_in_deviations[:,1:10])


∂state *= 0

∂x = 2*x[1]

∂state[T.past_not_future_and_mixed_idx] += 𝐒[:,1:end-T.nExo]' * ∂state

∂x += 𝐒[:,end-T.nExo+1:end]' * ∂state

∂v = invjac' * ∂x

∂data_in_deviations∂x[:,1] = ∂v


∂data_in_deviations = zero(data_in_deviations)

∂shocks² = 2*x[1]

# x = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])



# x = sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st]))

i = 1
st = T.past_not_future_and_mixed_idx

∂x∂shocks² = 2 * x[1]

∂v∂x = invjac'

∂v∂shocks² = ∂v∂x * ∂x∂shocks²

vsub = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])

vvv = vcat(state[i][st], vsub)

vvv = vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st]))

vv =  (𝐒[st,:] * vvv)

v = (data_in_deviations[:,i+1] - 𝐒endo * vv)

∂vv∂v = - 𝐒endo'

∂vv∂shocks² = ∂vv∂v * ∂v∂x * ∂x∂shocks²

∂vvv∂vv = 𝐒[st,:]'

∂vvv∂shocks² = ∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²

∂vsubu∂vvv = ℒ.I(size(𝐒,2))[:,end-T.nExo+1:end]'

∂vsub∂shocks² = ∂vsubu∂vvv * ∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²
∂vsub∂shocks² = (∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²)[end-T.nExo+1:end]

∂dat∂shocks² = invjac' * (∂vvv∂vv * ∂vv∂v * ∂v∂x * ∂x∂shocks²)[end-T.nExo+1:end]

∂dat∂shocks² = -invjac' * (𝐒[st,:]' * 𝐒endo' * invjac' * 2 * x[2])[end-T.nExo+1:end]

∂dat∂shocks² = -𝐒exo' \ (𝐒[st,:]' * 𝐒endo' / 𝐒exo' * 2 * x[2])[end-T.nExo+1:end]

∂dat∂shocks² = -𝐒exo' \ (2 * x[1])


# ∂x∂v = 

# shocks² = sum(abs2, 𝐒exo \ v)

invjac' * 𝐒[st,1:end-T.nExo] * 𝐒endo' * invjac' * 2 * x[1]

∂shocks² = 2 * (𝐒exo \ v)' * ∂shocks²
∂v = (∂shocks² / shocks²) * (𝐒exo \ v)'

# x = sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st]))
# i = 1

FiniteDiff.finite_difference_gradient(x -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (x - 𝐒endo * state[i][st])))[st])), data_in_deviations[:,i])# + ∂v

FiniteDiff.finite_difference_gradient(x -> sum(abs2, 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * 𝐒[st,:] * vcat(state[i][st], 𝐒exo \ (x - 𝐒endo * state[i][st])))), data_in_deviations[:,i])# + ∂v


2 * (𝐒exo \ (data_in_deviations[:, 2] - 𝐒endo * (𝐒 * vcat(state[1][st], 𝐒exo \ (data_in_deviations[:, 1] - 𝐒endo * state[1][st])))[st]))

2 * (𝐒exo \ (data_in_deviations[:, 2] - 𝐒endo * (𝐒 * vcat(state[1][st], 𝐒exo \ (data_in_deviations[:, 1] - 𝐒endo * state[1][st])))[st])) * (-invjac * 𝐒endo[:, st] * 𝐒[cond_var_idx,:]' * invjac')

-2 * x[1] * 𝐒endo * invjac * 𝐒
X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], x))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * state[i+1][st])

∂data_in_deviations[:,1] = invjac' * ∂shocks²

# ∂state[end-T.nExo+1:end] = 

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in 1:10 # axes(data_in_deviations,2)
        X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data_in_deviations[:,1:10])


data = copy(data_in_deviations);
data[:,2:3] *= 0;

FiniteDiff.finite_difference_gradient(data_in_deviations -> begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    for i in axes(data_in_deviations,2)
        X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)

        shocks² += sum(abs2,X)
    end

    return shocks²
end, data)#_in_deviations[:,1:2])



i = 1


st = T.past_not_future_and_mixed_idx

# data has an impact because of the difference between today and tomorrow, as in the data matters for two periods
# fisrt period:
X = 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])

# second period:
state[i+1] = 𝐒 * vcat(state[i][st], X)

# here it matters because it is part of X -> state and thereby pushes around the deterministic part of the system (via the state)
# X = 𝐒[cond_var_idx, end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx, 1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])
X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], 𝐒exo \ (data_in_deviations[:,i] - 𝐒endo * state[i][st])))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * (𝐒 * vcat(state[i][st], X))[st])

X = 𝐒exo \ (data_in_deviations[:,i+1] - 𝐒endo * state[i+1][st])

1

# state::Vector{Vector{Float64}}, 
#                                                     𝐒::Union{Matrix{Float64}, Vector{AbstractMatrix{Float64}}}, 
#                                                     data_in_deviations::Matrix{Float64}, 
#                                                     observables::Union{Vector{String}, Vector{Symbol}},
#                                                     T::timings; 
#                                                     warmup_iterations::Int = 0,
#                                                     presample_periods::Int = 0)


# function first_order_state_update(state::Vector{U}, shock::Vector{S}) where {U <: Real,S <: Real}
# # state_update = function(state::Vector{T}, shock::Vector{S}) where {T <: Real,S <: Real}
#     aug_state = [state[T.past_not_future_and_mixed_idx]
#                 shock]
#     return 𝐒 * aug_state # you need a return statement for forwarddiff to work
# end

# state_update = first_order_state_update

# state = state[1]

# pruning = false

    
# precision_factor = 1.0

# n_obs = size(data_in_deviations,2)

# cond_var_idx = indexin(observables,sort(union(T.aux,T.var,T.exo_present)))



# warmup_iterations = 3
# state *=0

# data_in_deviations[:,1] - (𝐒 * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[4:6]))[cond_var_idx]

# data_in_deviations[:,1] - (𝐒 * (vcat((𝐒[T.past_not_future_and_mixed_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3]))), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))[cond_var_idx]


# data_in_deviations[:,1] - (𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])) + vcat(state[T.past_not_future_and_mixed_idx], x[4:6])))


# 𝐒[cond_var_idx,:] * (vcat((𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]), zero(x[1:3])))

# 𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])


# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * (𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3]) + 𝐒[cond_var_idx,end-T.nExo+1:end] * x[4:6] - data_in_deviations[:,1]




# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[1:3] +
# 𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end] * x[4:6] +
# 𝐒[cond_var_idx,end-T.nExo+1:end] * x[7:9] -
# data_in_deviations[:,1]

# hcat(   
#     𝐒[cond_var_idx,end-T.nExo+1:end] , 
#     𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], 
#     𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]
#     ) \ data_in_deviations[:,1]


# warmup_iterations = 5

# state *= 0
# logabsdets = 0
# shocks² = 0

# if warmup_iterations > 0
#     if warmup_iterations >= 1
#         to_be_inverted = 𝐒[cond_var_idx,end-T.nExo+1:end]
#         if warmup_iterations >= 2
#             to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
#             if warmup_iterations >= 3
#                 Sᵉ = 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
#                 for e in 1:warmup_iterations-2
#                     to_be_inverted = hcat(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * Sᵉ * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end], to_be_inverted)
#                     Sᵉ *= 𝐒[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed]
#                 end
#             end
#         end
#     end

#     x = to_be_inverted \ data_in_deviations[:,1]

#     warmup_shocks = reshape(x, T.nExo, warmup_iterations)

#     for i in 1:warmup_iterations-1
#         state = state_update(state, warmup_shocks[:,i])
#     end

#     jacc = -to_be_inverted'

#     for i in 1:warmup_iterations
#         if T.nExo == length(observables)
#             logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
#         else
#             logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
#         end
#     end

#     shocks² += sum(abs2,x)
# end



# data_in_deviations[:,1] - 𝐒[cond_var_idx,:] * vcat((𝐒 * vcat(state[T.past_not_future_and_mixed_idx], (x[1:3])))[T.past_not_future_and_mixed_idx], zero(x[1:3])) + 𝐒[cond_var_idx,:] * vcat(state[T.past_not_future_and_mixed_idx], x[4:6])



# data_in_deviations[:,1]

# (𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x[1:3]))[cond_var_idx]


#         x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

#         ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))





# state_copy = deepcopy(state)

# XX = reshape(X, length(X) ÷ warmup_iters, warmup_iters)

# for i in 1:warmup_iters
#     state_copy = state_update(state_copy, XX[:,i])
# end

# return precision_factor .* sum(abs2, data - (pruning ? sum(state_copy) : state_copy)[cond_var_idx])



# shocks² = 0.0
# logabsdets = 0.0

# if warmup_iterations > 0
#     res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
#                         zeros(T.nExo * warmup_iterations), 
#                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
#                         Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                         autodiff = :forward)

#     matched = Optim.minimum(res) < 1e-12

#     if !matched # for robustness try other linesearch
#         res = Optim.optimize(x -> minimize_distance_to_initial_data(x, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor, pruning), 
#                         zeros(T.nExo * warmup_iterations), 
#                         Optim.LBFGS(), 
#                         Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                         autodiff = :forward)
    
#         matched = Optim.minimum(res) < 1e-12
#     end

#     if !matched return -Inf end

#     x = Optim.minimizer(res)

#     warmup_shocks = reshape(x, T.nExo, warmup_iterations)

#     for i in 1:warmup_iterations-1
#         state = state_update(state, warmup_shocks[:,i])
#     end
    
#     res = zeros(0)

#     jacc = zeros(T.nExo * warmup_iterations, length(observables))

#     match_initial_data!(res, x, jacc, data_in_deviations[:,1], state, state_update, warmup_iterations, cond_var_idx, precision_factor), zeros(size(data_in_deviations, 1))

#     for i in 1:warmup_iterations
#         if T.nExo == length(observables)
#             logabsdets += ℒ.logabsdet(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor)[1]
#         else
#             logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc[(i - 1) * T.nExo .+ (1:2),:] ./ precision_factor))
#         end
#     end

#     shocks² += sum(abs2,x)
# end



# jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

# jacdecomp = ℒ.svd(jac)
# invjac = inv(jacdecomp)

# using FiniteDiff
# FiniteDiff.finite_difference_jacobian(xx -> sum(x -> log(abs(x)), ℒ.svdvals(xx)),𝐒[cond_var_idx,end-T.nExo+1:end])

# ForwardDiff.jacobian(xx -> sum(x -> log(abs(x)), ℒ.svdvals(xx)),𝐒[cond_var_idx,end-T.nExo+1:end])


# ForwardDiff.gradient(x-> x'*x,[1,2,3])
# [1,2,3]'*[1,2,3]


# ∂det = -inv(ℒ.svd(𝐒[cond_var_idx,end-T.nExo+1:end]))






state = zeros(T.nVars)
# statetmp = zeros(23)
shocks² = 0.0
logabsdets = 0.0
y = zeros(length(cond_var_idx))
x = zeros(T.nExo)
# state_reduced = zeros(T.nPast_not_future_and_mixed + T.nExo)

jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

if T.nExo == length(observables)
    logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
    jacdecomp = ℒ.lu!(jac)
    invjac = inv(jacdecomp)
else
    logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
    jacdecomp = ℒ.svd!(jac)
    invjac = inv(jacdecomp)
end

logabsdets *= size(data_in_deviations,2) - presample_periods

@views 𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

for i in axes(data_in_deviations,2)
    x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

    if i > presample_periods
        shocks² += sum(abs2,x)
    end

    state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
end
shocks²

-(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2




inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]))

inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed]))' * inv( ℒ.svd( 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]) )'

𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed] * inv( ℒ.svd( 𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]) )'

inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed]))' *  𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end]


inv(ℒ.svd(𝐒[T.past_not_future_and_mixed_idx,end-T.nExo+1:end])) * inv(ℒ.svd(𝐒[cond_var_idx,1:T.nPast_not_future_and_mixed])) 

FiniteDiff.finite_difference_gradient(x->begin
# ForwardDiff.gradient(x->begin
# Zygote.gradient(x->begin
    shocks² = 0.0
    X = zeros(eltype(x),T.nExo)

    for i in 1:2#xes(data_in_deviations,2)
        X = 𝐒[cond_var_idx,end-T.nExo+1:end] \ (x[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,X)
        end

        state[i+1] = 𝐒 * vcat(state[i][T.past_not_future_and_mixed_idx], X)
    end
    return shocks²
end, data_in_deviations[:,1:2])



ForwardDiff.gradient(x->sum(abs2, x[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - x[cond_var_idx,1:end-T.nExo] * state[1][T.past_not_future_and_mixed_idx])), 𝐒)

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (x - 𝐒[cond_var_idx,1:end-T.nExo] * state[i][T.past_not_future_and_mixed_idx])), data_in_deviations[:,i])

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - x * state[i][T.past_not_future_and_mixed_idx])), 𝐒[cond_var_idx,1:end-T.nExo])

ForwardDiff.gradient(x->sum(abs2, 𝐒[cond_var_idx,end-T.nExo+1:end] \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * x)), state[i][T.past_not_future_and_mixed_idx])





(jac' * jac)' \ jac'

invjac' * jac' * invjac'
# ∂jac =  (jac)' * (jac *  ∂x)

FiniteDiff.finite_difference_jacobian(x->sum(abs2,x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac)

ForwardDiff.gradient(x->sum(abs2, x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac)

@profview for i in 1:10000 Zygote.gradient(x->sum(abs2, x \ (data_in_deviations[:,1] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])),jac) end

vec(jac) * vec(jac)'
ForwardDiff.gradient(x->ℒ.det(inv(x)),jac[:,1:2])

(jac[:,1:2]) * (jac[:,1:2])'


vec(jac[:,1:2]) * vec(jac[:,1:2])'

∂data_in_deviations∂x = invjac' * ∂x

∂𝐒[cond_var_idx,1:end-T.nExo] = -invjac' * ∂x * state[T.past_not_future_and_mixed_idx]'

∂state[T.past_not_future_and_mixed_idx] = -𝐒[cond_var_idx,1:end-T.nExo]' * invjac' * ∂x

# state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)
∂𝐒 += ∂state * vcat(state[T.past_not_future_and_mixed_idx], x)'

∂state[T.past_not_future_and_mixed_idx] += 𝐒[:,1:end-T.nExo]' * ∂state





if i > presample_periods
    shocks² += sum(abs2,x)
end

state = 𝐒 * vcat(state[T.past_not_future_and_mixed_idx], x)



ForwardDiff.gradient(x -> sum(abs2, x \ (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), jac)

ForwardDiff.gradient(x -> sum(abs2, x * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), invjac)

ForwardDiff.gradient(x -> sum(abs2, invjac * (x - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])), data_in_deviations[:,i])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - x * state[T.past_not_future_and_mixed_idx])), 𝐒[cond_var_idx,1:end-T.nExo])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * x)), state[T.past_not_future_and_mixed_idx])

ForwardDiff.gradient(x -> sum(abs2, invjac * (data_in_deviations[:,i] - x)), 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

ForwardDiff.gradient(x -> sum(abs2, invjac * x), data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])


# i = 2
# res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
#                     zeros(T.nExo), 
#                     Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
#                     Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
#                     autodiff = :forward)

#                     res.minimizer
# # data_in_deviations[:,i] - 𝐒[cond_var_idx,end-T.nExo+1:end] * x

# @benchmark x = 𝐒[cond_var_idx,end-T.nExo+1:end] \ data_in_deviations[:,i]
# @profview for k in 1:1000 𝐒[cond_var_idx,end-T.nExo+1:end] \ data_in_deviations[:,i] end


# @profview for k in 1:1000
@benchmark begin
    state = zeros(23)
    # statetmp = zeros(23)
    shocks² = 0.0
    logabsdets = 0.0
    y = zeros(length(cond_var_idx))
    x = zeros(T.nExo)
    # state_reduced = zeros(T.nPast_not_future_and_mixed + T.nExo)

    jac = 𝐒[cond_var_idx,end-T.nExo+1:end]

    if T.nExo == length(observables)
        logabsdets = ℒ.logabsdet(-jac' ./ precision_factor)[1]
        jacdecomp = ℒ.lu!(jac)
        invjac = inv(jacdecomp)
    else
        logabsdets = sum(x -> log(abs(x)), ℒ.svdvals(-jac' ./ precision_factor))
        jacdecomp = ℒ.svd!(jac)
        invjac = inv(jacdecomp)
    end

    logabsdets *= size(data_in_deviations,2) - presample_periods
    
    @views 𝐒obs = 𝐒[cond_var_idx,1:end-T.nExo]

    for i in axes(data_in_deviations,2)
        @views ℒ.mul!(y, 𝐒obs, state[T.past_not_future_and_mixed_idx])
        @views ℒ.axpby!(1, data_in_deviations[:,i], -1, y)
        ℒ.mul!(x,invjac,y)
        # x = invjac * (data_in_deviations[:,i] - 𝐒[cond_var_idx,1:end-T.nExo] * state[T.past_not_future_and_mixed_idx])

        if i > presample_periods
            shocks² += sum(abs2,x)
        end

        # # copyto!(state_reduced, 1, state, T.past_not_future_and_mixed_idx)
        # for (i,v) in enumerate(T.past_not_future_and_mixed_idx)
        #     state_reduced[i] = state[v]
        # end
        # copyto!(state_reduced, T.nPast_not_future_and_mixed + 1, x, 1, T.nExo)
        
        ℒ.mul!(state, 𝐒, vcat(state[T.past_not_future_and_mixed_idx], x))
        # state = state_update(state, x)
    end

    -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end

pruning = false

jac = ForwardDiff.jacobian( xx -> precision_factor .* abs.(data_in_deviations[:,i] - (pruning ? sum(state_update(state, xx)) : state_update(state, xx))[cond_var_idx]), x)'


res = precision_factor .* abs.(data_in_deviations[:,i] - (pruning ? sum(state_update(state, x)) : state_update(state, x))[cond_var_idx])


state = state_update(state, x)


x = Optim.minimizer(res)

res  = zeros(0)

jacc = zeros(T.nExo, length(observables))

match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)


match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)


@benchmark begin
shocks² = 0.0
logabsdets = 0.0
state = zeros(23)

for i in axes(data_in_deviations,2)
    res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                    zeros(T.nExo), 
                    Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                    Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                    autodiff = :forward)

    matched = Optim.minimum(res) < 1e-12

    if !matched # for robustness try other linesearch
        res = Optim.optimize(x -> minimize_distance_to_data(x, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor, pruning), 
                        zeros(T.nExo), 
                        Optim.LBFGS(), 
                        Optim.Options(f_abstol = eps(), g_tol= 1e-30); 
                        autodiff = :forward)
    
        matched = Optim.minimum(res) < 1e-12
    end

    if !matched return -Inf end

    x = Optim.minimizer(res)

    res  = zeros(0)

    jacc = zeros(T.nExo, length(observables))

    match_data_sequence!(res, x, jacc, data_in_deviations[:,i], state, state_update, cond_var_idx, precision_factor)

    if i > presample_periods
        # due to change of variables: jacobian determinant adjustment
        if T.nExo == length(observables)
            logabsdets += ℒ.logabsdet(jacc ./ precision_factor)[1]
        else
            logabsdets += sum(x -> log(abs(x)), ℒ.svdvals(jacc ./ precision_factor))
        end

        shocks² += sum(abs2,x)
    end

    state = state_update(state, x)
end

# See: https://pcubaborda.net/documents/CGIZ-final.pdf
 -(logabsdets + shocks² + (length(observables) * (warmup_iterations + n_obs - presample_periods)) * log(2 * 3.141592653589793)) / 2
end