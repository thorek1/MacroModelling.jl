
using MacroModelling
# import Turing: NUTS, HMC, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using CSV, DataFrames, AxisKeys
import Zygote
import ForwardDiff
import ChainRulesCore: @ignore_derivatives, ignore_derivatives, rrule, NoTangent, @thunk
using Random
import BenchmarkTools: @benchmark
Random.seed!(1)
# ]add CSV, DataFrames, Zygote, AxisKeys, MCMCChains, Turing, DynamicPPL, Pigeons, StatsPlots
println("Threads used: ", Threads.nthreads())

smpler = "nuts" #
mdl = "linear" # 
fltr = :kalman
algo = :first_order

sample_idx = 47:230
# sample_idx = 47:47
dat = CSV.read("benchmark/usmodel.csv", DataFrame)

# Initialize a DataFrame to store the data
df = DataFrame(iteration = Float64[])

if mdl == "linear"
    include("../models/Smets_Wouters_2007_linear.jl")
    Smets_Wouters_2007 = Smets_Wouters_2007_linear
elseif mdl == "nonlinear"
    include("../models/Smets_Wouters_2007.jl")
end


# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
data = data(observables_old, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01], algorithm = algo)(observables)

𝓂 = Smets_Wouters_2007


parameters = [  0.5295766584252728
0.25401999781328677
0.5555813987579575
0.3654903601830364
0.2294564856713931
0.12294028349908431
0.20767050150368016
0.9674674841230338
0.20993223738088435
0.9888169549988175
0.8669340301385475
0.07818383624087137
0.6105112778170307
0.37671694996404337
0.2187231627543815
0.1362385298510586
6.3886101979474015
1.6678696241559958
0.6799655079831786
0.9424292929726574
2.502826072472096
0.6570767721691694
0.6729083298930368
0.23408903978575385
0.6457362272648652
1.4738116352107862
2.088069269612668
0.8655409607264644
0.0895375194503755
0.18792207697672325
0.696046453737325
0.1899464169442222
-0.5748023731804703
0.3683194328119635
0.5101771887138438
0.17425592648706756]


z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = parameters

ctou, clandaw, cg, curvp, curvw = Smets_Wouters_2007.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007.parameters)]

parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

get_loglikelihood(𝓂, data, parameters_combined, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal) # -1082.8088568705207
#old order -1087.2659101980191


import LinearAlgebra: mul!, transpose!, rmul!, logdet
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import MacroModelling: get_and_check_observables, solve!, check_bounds, get_relevant_steady_state_and_state_update, calculate_loglikelihood, get_initial_covariance
parameter_values = parameters_combined
algorithm = :first_order
filter = :kalman
warmup_iterations = 0
presample_periods = 0
initial_covariance = :diagonal
tol = 1e-12
verbose = false
T = 𝓂.timings

observables = @ignore_derivatives get_and_check_observables(𝓂, data)

@ignore_derivatives solve!(𝓂, verbose = verbose, algorithm = algorithm)

bounds_violated = @ignore_derivatives check_bounds(parameter_values, 𝓂)

NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

obs_indices = @ignore_derivatives convert(Vector{Int}, indexin(observables, NSSS_labels))

# TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)


SS_and_pars, (solution_error, iters) = get_non_stochastic_steady_state(𝓂, parameter_values)

state = [zeros(𝓂.timings.nVars)]

TT = 𝓂.timings

sp∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂)# |> Matrix

∇₁ = Matrix{Float64}(sp∇₁)

𝐒, solved = calculate_first_order_solution(∇₁; T = TT)






# prepare data
data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

observables_index = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

observables_and_states = @ignore_derivatives sort(union(T.past_not_future_and_mixed_idx,observables_index))

A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

𝐁 = B * B'

# Gaussian Prior
coordinates = @ignore_derivatives Tuple{Vector{Int}, Vector{Int}}[]

dimensions = @ignore_derivatives [size(A),size(𝐁)]


# working code -  optimized
import RecursiveFactorization as RF
# import Octavian: matmul!
P = get_initial_covariance(Val(:theoretical), vcat(vec(A), vec(collect(-𝐁))), coordinates, dimensions)
presample_periods = 4


T = size(data_in_deviations, 2) + 1

z = zeros(size(data_in_deviations, 1))

ū = zeros(size(C,2))

P̄ = deepcopy(P) 

temp_N_N = similar(P)

PCtmp = similar(C')

F = similar(C * C')

u = [similar(ū) for _ in 1:T] # used in backward pass

P = [deepcopy(P̄) for _ in 1:T] # used in backward pass

CP = [zero(C) for _ in 1:T] # used in backward pass

K = [similar(C') for _ in 1:T] # used in backward pass

invF = [similar(F) for _ in 1:T] # used in backward pass

v = [zeros(size(data_in_deviations, 1)) for _ in 1:T] # used in backward pass

loglik = 0.0

for t in 2:T
    v[t] .= data_in_deviations[:, t-1] .- z#[t-1]

    # CP[t] .= C * P̄[t-1]
    mul!(CP[t], C, P̄)#[t-1])

    # F[t] .= CP[t] * C'
    mul!(F, CP[t], C')

    luF = RF.lu(F, check = false)

    if !ℒ.issuccess(luF)
        return -Inf, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    Fdet = ℒ.det(luF)

    # Early return if determinant is too small, indicating numerical instability.
    if Fdet < eps(Float64)
        return -Inf, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    # invF[t] .= inv(luF)
    copy!(invF[t], inv(luF))
    
    if t - 1 > presample_periods
        loglik += log(Fdet) + ℒ.dot(v[t], invF[t], v[t])
    end

    # K[t] .= P̄[t-1] * C' * invF[t]
    mul!(PCtmp, P̄, C')
    mul!(K[t], PCtmp, invF[t])

    # P[t] .= P̄[t-1] - K[t] * CP[t]
    mul!(P[t], K[t], CP[t], -1, 0)
    P[t] .+= P̄

    # P̄[t] .= A * P[t] * A' + 𝐁
    mul!(temp_N_N, P[t], A')
    mul!(P̄, A, temp_N_N)
    P̄ .+= 𝐁

    # u[t] .= K[t] * v[t] + ū[t-1]
    mul!(u[t], K[t], v[t])
    u[t] .+= ū
    
    # ū[t] .= A * u[t]
    mul!(ū, A, u[t])

    # z[t] .= C * ū[t]
    mul!(z, C, ū)
end

llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 

# initialise derivative variables
∂A = zero(A)
∂F = zero(F)
∂Faccum = zero(F)
∂P = zero(P̄)
∂ū = zero(ū)
∂v = zero(v[1])
∂𝐁 = zero(𝐁)
∂data_in_deviations = zero(data_in_deviations)
vtmp = zero(v[1])
Ptmp = zero(P[1])



∂llh = 1


ℒ.rmul!(∂A, 0)
ℒ.rmul!(∂Faccum, 0)
ℒ.rmul!(∂P, 0)
ℒ.rmul!(∂ū, 0)
ℒ.rmul!(∂𝐁, 0)

for t in T:-1:2
    if t > presample_periods + 1
        # ∂llh∂F
        # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
        # ∂F = invF[t]' - invF[t]' * v[t] * v[t]' * invF[t]'
        mul!(∂F, v[t], v[t]')
        mul!(invF[1], invF[t]', ∂F) # using invF[1] as temporary storage
        mul!(∂F, invF[1], invF[t]')
        ℒ.axpby!(1, invF[t]', -1, ∂F)

        # ∂llh∂ū
        # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
        # z[t] .= C * ū[t]
        # ∂v = (invF[t]' + invF[t]) * v[t]
        copy!(invF[1], invF[t]' + invF[t]) # using invF[1] as temporary storage
        mul!(∂v, invF[1], v[t])
        # mul!(∂ū∂v, C', v[1])
    else
        ℒ.rmul!(∂F, 0)
        ℒ.rmul!(∂v, 0)
    end

    # ∂F∂P
    # F[t] .= C * P̄[t-1] * C'
    # ∂P += C' * (∂F + ∂Faccum) * C
    ℒ.axpy!(1, ∂Faccum, ∂F)
    mul!(PCtmp, C', ∂F) 
    mul!(∂P, PCtmp, C, 1, 1) 

    # ∂ū∂P
    # K[t] .= P̄[t-1] * C' * invF[t]
    # u[t] .= K[t] * v[t] + ū[t-1]
    # ū[t] .= A * u[t]
    # ∂P += A' * ∂ū * v[t]' * invF[t]' * C
    mul!(CP[1], invF[t]', C) # using CP[1] as temporary storage
    mul!(PCtmp, ∂ū , v[t]')
    mul!(P[1], PCtmp , CP[1]) # using P[1] as temporary storage
    mul!(∂P, A', P[1], 1, 1) 

    # ∂ū∂data
    # v[t] .= data_in_deviations[:, t-1] .- z
    # z[t] .= C * ū[t]
    # ∂data_in_deviations[:,t-1] = -C * ∂ū
    mul!(u[1], A', ∂ū)
    mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
    ℒ.axpy!(1, ∂v, v[1])
    ∂data_in_deviations[:,t-1] .= v[1]
    # mul!(∂data_in_deviations[:,t-1], C, ∂ū, -1, 0) # cannot assign to columns in matrix, must be whole matrix 

    # ∂ū∂ū
    # z[t] .= C * ū[t]
    # v[t] .= data_in_deviations[:, t-1] .- z
    # K[t] .= P̄[t-1] * C' * invF[t]
    # u[t] .= K[t] * v[t] + ū[t-1]
    # ū[t] .= A * u[t]
    # step to next iteration
    # ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
    mul!(u[1], A', ∂ū) # using u[1] as temporary storage
    mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
    mul!(∂ū, C', v[1])
    mul!(u[1], C', v[1], -1, 1)
    copy!(∂ū, u[1])

    # ∂llh∂ū
    # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
    # v[t] .= data_in_deviations[:, t-1] .- z
    # z[t] .= C * ū[t]
    # ∂ū -= ∂ū∂v
    mul!(u[1], C', ∂v) # using u[1] as temporary storage
    ℒ.axpy!(-1, u[1], ∂ū)

    if t > 2
        # ∂ū∂A
        # ū[t] .= A * u[t]
        # ∂A += ∂ū * u[t-1]'
        mul!(∂A, ∂ū, u[t-1]', 1, 1)

        # ∂P̄∂A and ∂P̄∂𝐁
        # P̄[t] .= A * P[t] * A' + 𝐁
        # ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
        mul!(P[1], A, P[t-1]')
        mul!(Ptmp ,∂P, P[1])
        mul!(P[1], A, P[t-1])
        mul!(Ptmp ,∂P', P[1], 1, 1)
        ℒ.axpy!(1, Ptmp, ∂A)

        # ∂𝐁 += ∂P
        ℒ.axpy!(1, ∂P, ∂𝐁)

        # ∂P∂P
        # P[t] .= P̄[t-1] - K[t] * C * P̄[t-1]
        # P̄[t] .= A * P[t] * A' + 𝐁
        # step to next iteration
        # ∂P = A' * ∂P * A
        mul!(P[1], ∂P, A) # using P[1] as temporary storage
        mul!(∂P, A', P[1])

        # ∂P̄∂P
        # K[t] .= P̄[t-1] * C' * invF[t]
        # P[t] .= P̄[t-1] - K[t] * CP[t]
        # ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 
        mul!(PCtmp, ∂P, K[t-1])
        mul!(CP[1], K[t-1]', ∂P) # using CP[1] as temporary storage
        mul!(∂P, PCtmp, C, -1, 1)
        mul!(∂P, C', CP[1], -1, 1)

        # ∂ū∂F
        # K[t] .= P̄[t-1] * C' * invF[t]
        # u[t] .= K[t] * v[t] + ū[t-1]
        # ū[t] .= A * u[t]
        # ∂Faccum = -invF[t-1]' * CP[t-1] * A' * ∂ū * v[t-1]' * invF[t-1]'
        mul!(u[1], A', ∂ū) # using u[1] as temporary storage
        mul!(v[1], CP[t-1], u[1]) # using v[1] as temporary storage
        mul!(vtmp, invF[t-1]', v[1], -1, 0)
        mul!(invF[1], vtmp, v[t-1]') # using invF[1] as temporary storage
        mul!(∂Faccum, invF[1], invF[t-1]')

        # ∂P∂F
        # K[t] .= P̄[t-1] * C' * invF[t]
        # P[t] .= P̄[t-1] - K[t] * CP[t]
        # ∂Faccum -= invF[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invF[t-1]'
        mul!(CP[1], invF[t-1]', CP[t-1]) # using CP[1] as temporary storage
        mul!(PCtmp, CP[t-1]', invF[t-1]')
        mul!(K[1], ∂P, PCtmp) # using K[1] as temporary storage
        mul!(∂Faccum, CP[1], K[1], -1, 1)

    end
end

ℒ.rmul!(∂P, -∂llh/2)
ℒ.rmul!(∂A, -∂llh/2)
ℒ.rmul!(∂𝐁, -∂llh/2)
ℒ.rmul!(∂data_in_deviations, -∂llh/2)



# calculate covariance
import MatrixEquations
import MacroModelling: solve_matrix_equation_forward, riccati_AD_direct, run_kalman_iterations

coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

dimensions =  [size(A),size(∂P)]

values = vcat(vec(A'), vec(-∂P))

S, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :doubling)

∂𝐁 += S

∂B = ∂𝐁 * B + (B' * ∂𝐁)'# ≈ for_diff

P = get_initial_covariance(Val(:theoretical), vcat(vec(A), vec(collect(-𝐁))), coordinates, dimensions)

∂A += S * A * P' + S' * A * P

# ∂A ≈ for_diff

𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

rev_diff_𝐒 = Zygote.gradient(𝐒 -> begin
                # 𝐒ᵗ, solved = riccati_forward(∇₁; T = TT, explosive = false)

                # Jm = @view(ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx,:])

                # ∇₊ = @views ∇₁[:,1:TT.nFuture_not_past_and_mixed] * ℒ.diagm(ones(TT.nVars))[TT.future_not_past_and_mixed_idx,:]
                # ∇₀ = @view ∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
                # ∇ₑ = @view ∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]
                    
                # 𝐒ᵉ = -(∇₊ * 𝐒ᵗ * Jm + ∇₀) \ ∇ₑ # otherwise Zygote doesnt diff it

                # 𝐒 = hcat(𝐒ᵗ, 𝐒ᵉ)

                A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
                B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

                C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                
                presample_periods = 4

                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
                
            end, 𝐒)[1]

import MacroModelling: riccati_forward
𝐒ᵗ, solved = riccati_forward(∇₁; T = TT, explosive = false)

Jm = @view(ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx,:])

∇₊ = @views ∇₁[:,1:TT.nFuture_not_past_and_mixed] * ℒ.diagm(ones(TT.nVars))[TT.future_not_past_and_mixed_idx,:]
∇₀ = @view ∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
∇ₑ = @view ∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]
    
𝐒ᵉ = -(∇₊ * 𝐒ᵗ * Jm + ∇₀) \ ∇ₑ # otherwise Zygote doesnt diff it

# return hcat(𝐒ᵗ, 𝐒ᵉ), solved

∂𝐒 = (rev_diff_𝐒,true)

∂𝐒ᵗ = rev_diff_𝐒[:,1:TT.nPast_not_future_and_mixed]
∂𝐒ᵉ = rev_diff_𝐒[:,TT.nPast_not_future_and_mixed + 1:end]

M = inv(∇₊ * 𝐒ᵗ * Jm + ∇₀)

∂∇ₑ = -M' * ∂𝐒ᵉ
# ∂∇ₑ ≈ rev_diff_∇ₑ

∂∇₀ = M' * ∂𝐒ᵉ * ∇ₑ' * M'
# ∂∇₀ ≈ rev_diff_∇₀

∂𝐒ᵗ += ∇₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * Jm'
# ∂𝐒ᵗ ≈ rev_diff_𝐒ᵗ

∂∇₊ = M' * ∂𝐒ᵉ * ∇ₑ' * M' * Jm' * 𝐒ᵗ'
# ∂∇₊ ≈ rev_diff_∇₊



T = TT
# function rrule(::typeof(calculate_first_order_solution), ∇₁; T, explosive = false)
    # Forward pass to compute the output and intermediate values needed for the backward pass
    𝐒ᵗ, solved = riccati_forward(∇₁, T = T, explosive = false)

    # if !solved
    #     return (hcat(𝐒ᵗ, zeros(size(𝐒ᵗ,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    # end

    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
                    ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    M̂ = RF.lu(∇₊ * 𝐒ᵗ * expand[2] + ∇₀, check = false)
    
    # if !ℒ.issuccess(M̂)
    #     return (hcat(𝐒ᵗ, zeros(size(𝐒ᵗ,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    # end
    
    M = inv(M̂)
    
    𝐒ᵉ = -M * ∇ₑ # otherwise Zygote doesnt diff it

    𝐒̂ᵗ = 𝐒ᵗ * expand[2]
    
    ∇̂₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    ∂∇₁ = zero(∇₁)
    
    invtmp = inv(-𝐒̂ᵗ * ∇̂₊' - ∇₀')
    
    tmp2 = invtmp * ∇̂₊'

    function first_order_solution_pullback(∂𝐒) 
        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1:end] .= -M' * ∂𝐒ᵉ

        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .= M' * ∂𝐒ᵉ * ∇ₑ' * M'

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]' * 𝐒ᵗ')[:,T.future_not_past_and_mixed_idx]

        ∂𝐒ᵗ += ∇̂₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]'

        tmp1 = invtmp * ∂𝐒ᵗ * expand[2]

        coordinates = Tuple{Vector{Int}, Vector{Int}}[]

        values = vcat(vec(tmp2), vec(𝐒̂ᵗ'), vec(-tmp1))
        
        dimensions = Tuple{Int, Int}[]
        push!(dimensions,size(tmp2))
        push!(dimensions,size(𝐒̂ᵗ'))
        push!(dimensions,size(tmp1))
        
        ss, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres)
        
        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= (ss * 𝐒̂ᵗ' * 𝐒̂ᵗ')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .+= ss * 𝐒̂ᵗ'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent()
    end

    return (hcat(𝐒ᵗ, 𝐒ᵉ), solved), first_order_solution_pullback
# end

explosive = false


function rrule(::typeof(calculate_first_order_solution), ∇₁; T, explosive = false)
    # Forward pass to compute the output and intermediate values needed for the backward pass
    A, solved = riccati_forward(∇₁, T = T, explosive = explosive)

    if !solved
        return (hcat(A, zeros(size(A,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    end

    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
                    ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    M̂ = RF.lu(∇₊ * A * expand[2] + ∇₀, check = false)
    
    if !ℒ.issuccess(M̂)
        return (hcat(A, zeros(size(A,1),T.nExo)), solved), x -> NoTangent(), NoTangent(), NoTangent()
    end
    
    M = inv(M̂)
    
    B = -M * ∇ₑ # otherwise Zygote doesnt diff it

    Â = A * expand[2]
    
    ∇̂₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    ∂∇₁ = zero(∇₁)

    tmp2 = -M' * ∇̂₊'

    function first_order_solution_pullback(∂𝐒) 
        ∂𝐒ᵗ = ∂𝐒[1][:,1:T.nPast_not_future_and_mixed]
        ∂𝐒ᵉ = ∂𝐒[1][:,T.nPast_not_future_and_mixed + 1:end]

        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1:end] = -M' * ∂𝐒ᵉ

        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] = M' * ∂𝐒ᵉ * ∇ₑ' * M'

        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .= (M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]' * 𝐒ᵗ')[:,T.future_not_past_and_mixed_idx]

        ∂𝐒ᵗ += ∇̂₊' * M' * ∂𝐒ᵉ * ∇ₑ' * M' * expand[2]'

        tmp1 = -M' * ∂𝐒ᵗ * expand[2]

        coordinates = Tuple{Vector{Int}, Vector{Int}}[]

        values = vcat(vec(tmp2), vec(Â'), vec(-tmp1))
        
        dimensions = Tuple{Int, Int}[]
        push!(dimensions,size(tmp2))
        push!(dimensions,size(Â'))
        push!(dimensions,size(tmp1))
        
        ss, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres)
        
        ∂∇₁[:,1:T.nFuture_not_past_and_mixed] .+= (ss * Â' * Â')[:,T.future_not_past_and_mixed_idx]
        ∂∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] .+= ss * Â'
        ∂∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] .+= ss[:,T.past_not_future_and_mixed_idx]

        return NoTangent(), ∂∇₁, NoTangent()
    end

    return (hcat(A, B), solved), first_order_solution_pullback
end

import MacroModelling: run_kalman_iterations

rev_diff_∇₁ = Zygote.gradient(∇₁ -> begin 
                𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

                A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
                B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

                C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                
                presample_periods = 4

                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
                
            end, ∇₁)[1]

        #     40×79 Matrix{Float64}:
        #     37.0986       53.7827        9.52726      16.3199     -21.9992   …   -11.3947       -4.7704       0.228485     -7.02388     1.43183       1.50236
        #     -0.53164      -7.12795       3.96538       3.57407      5.71089        4.41053       4.18724     -0.386566      4.17207    -2.40934      -0.957483
        #    -10.5995      -13.3692       -2.89561      -5.17062      4.68735        0.434112      1.46593     -0.172263      1.06493    -1.0723       -0.263499
        #     -2.71337      -4.11749     -10.6171       -8.35937      1.58953        4.41314      -0.377532     0.205815      1.25675     1.2227       -0.241633
        #   -392.822      -381.575      -497.036      -493.58        74.4534         7.09026       3.26691      0.209451      1.62577    -0.634989     -1.59555
        #   -525.209      -585.029      -237.206      -332.603      164.683    …     0.552048      0.254362     0.0163079     0.126582   -0.0494403    -0.12423
        #   4555.95       5074.87       2057.66       2885.18     -1428.55          -4.78877      -2.20648     -0.141464     -1.09805     0.428872      1.07764
        #     12.5203      -17.2761       83.6844       68.4822      23.6512        -0.0149987    26.1958      -3.39949      16.7301    -20.8431       -4.10804
        #    -51.8372      -68.0892      -52.6155      -52.6827      24.4407        18.8741        4.23798      0.130645      9.04008     0.5856       -1.96687
        #     20.0345       31.0416        4.97147       8.30546    -13.4784        -8.97498      -2.4732       0.0164069    -4.735       0.110023      0.977066
        #      ⋮                                                               ⋱                                ⋮
        # -23976.6      -34555.8      -32497.6      -46563.1      11020.8            3.22134     144.931       58.4435       26.9744      3.19094    -134.058
        #  -1414.22      -1180.68      -1150.42       -982.669      157.083         -0.333678      8.02544      5.99611      -3.41535    -5.04097       4.37137
        #   7128.82       1940.16      -5229.67      -9016.51      1172.58         210.272       100.722      423.228      -209.651     -12.9319     -534.082
        #   1605.3        -121.044     -3553.99      -5074.0        502.679        -91.3893       76.5817     172.745      -117.96      -80.7101     -241.368
        #  -1281.11       -296.491      1187.79       1906.1       -231.157    …   -41.4761      -20.6298     -88.3789       48.5492     -6.64073      91.1058
        #   -345.923      -585.011      -967.454     -1617.85       275.427        103.533       -71.5974      88.3228     -206.891     -50.7558     -115.146
        #  -1169.54       -704.598     -1667.92      -1580.36       -84.6905      -124.246      -131.468        4.27502    -131.611      80.7644       31.2212
        #  -7543.03      -2142.64       4647.66       8810.48     -1256.77        -247.146        -0.882851  -471.866       186.413      38.5441      503.763
        #    297.08        180.272       403.962       479.286       23.072        175.875       109.438       21.0084       38.7868    -18.4063       86.0075



@benchmark Zygote.gradient(∇₁ -> begin
            𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

            A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
            B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

            C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

            𝐁 = B * B'
            coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

            dimensions =  [size(A),size(𝐁)]

            values = vcat(vec(A), vec(collect(-𝐁)))

            P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
            
            presample_periods = 4

            return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
            
        end, ∇₁)[1]


        
@benchmark begin
    𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

    A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

    𝐁 = B * B'
    coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

    dimensions =  [size(A),size(𝐁)]

    values = vcat(vec(A), vec(collect(-𝐁)))

    P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

    presample_periods = 4

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
end


@profview for i in 1:1000 Zygote.gradient(∇₁ -> begin 
        𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

        A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
        B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

        C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

        𝐁 = B * B'
        coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

        dimensions =  [size(A),size(𝐁)]

        values = vcat(vec(A), vec(collect(-𝐁)))

        P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
        
        presample_periods = 4

        return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
        
    end, ∇₁)[1]
end
# BenchmarkTools.Trial: 75 samples with 1 evaluation.
#  Range (min … max):  43.996 ms … 127.982 ms  ┊ GC (min … max): 0.00% … 11.69%
#  Time  (median):     61.436 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   67.343 ms ±  18.361 ms  ┊ GC (mean ± σ):  3.21% ±  6.25%

#       ▃▁ ▃    █ ▁▁▁ ▁▁ ▃  ▁
#   ▄▁▇▇██▇█▇▄▇▁█▄███▄██▁█▁▄█▁▄▁▁▄▄▁▇▁▁▁▁▄▁▄▁▁▄▁▇▁▇▄▁▇▄▁▁▁▄▄▁▁▄▄ ▁
#   44 ms           Histogram: frequency by time          108 ms <

#  Memory estimate: 38.90 MiB, allocs estimate: 6148.

# calculate first order solution
using SpeedMapping
expand = @views [ℒ.diagm(ones(TT.nVars))[TT.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx,:]] 


∇₊ = @views ∇₁[:,1:TT.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
∇₋ = @views ∇₁[:,TT.nFuture_not_past_and_mixed + TT.nVars .+ range(1,TT.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]

∇̂₀ =  RF.lu(∇₀)

AA = ∇̂₀ \ ∇₋
BB = ∇̂₀ \ ∇₊

C = similar(AA)
C̄ = similar(AA)

E = similar(C)

sol = speedmapping(zero(AA); m! = (C̄, C) -> begin 
                                    ℒ.mul!(E, C, C)
                                    ℒ.mul!(C̄, BB, E)
                                    ℒ.axpy!(1, AA, C̄)
                                end,
                                # C̄ .=  A + B * C^2, 
tol = tol, maps_limit = 10000)


CC = -sol.minimizer

DD = -(∇₊ * CC + ∇₀) \ ∇ₑ

tmp = ∇₊ * CC +  CC * ∇₊ + ∇₀
inv(tmp)


rev_diff_∇₁
diff_∇₊ = @views rev_diff_∇₁[:,1:TT.nFuture_not_past_and_mixed] * expand[1]
diff_∇₀ = @views rev_diff_∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
diff_∇₋ = @views rev_diff_∇₁[:,TT.nFuture_not_past_and_mixed + TT.nVars .+ range(1,TT.nPast_not_future_and_mixed)] * expand[2]
diff_∇ₑ = @views rev_diff_∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]



# rev_diff_A

# [:,TT.past_not_future_and_mixed_idx]

# rev_diff_A[:,TT.nFuture_not_past_and_mixed + TT.nVars .+ range(1,TT.nPast_not_future_and_mixed)]
# for_diff_A[:,TT.nFuture_not_past_and_mixed + TT.nVars .+ range(1,TT.nPast_not_future_and_mixed)]

import FiniteDifferences
a, _ = riccati_AD_direct(∇₁; T = TT, explosive = false)
#fin_diff_A = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1),
# for_diff_A = ForwardDiff.gradient(∇₁ -> begin 
# rev_diff_A_B - rev_diff_A



rev_diff_∇₁ = Zygote.gradient(x -> begin 
                # 𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

                aa, solved = riccati_AD_direct(x; T = TT, explosive = false)
            
                Jm = (ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx,:])
                
                ∇₊ =  ∇₁[:,1:TT.nFuture_not_past_and_mixed] * ℒ.diagm(ones(TT.nVars))[TT.future_not_past_and_mixed_idx,:]
                ∇₀ =  ∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
                ∇ₑ =  ∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]

                # aa  = CC[:,TT.past_not_future_and_mixed_idx]

                bb = -((∇₊ * aa * Jm + ∇₀) \ ∇ₑ)
            
                𝐒 = hcat(aa, bb)

                A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
                B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

                C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                
                presample_periods = 4

                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
                
            end, ∇₁)[1]



rev_diff_A_B = Zygote.gradient(CC -> begin 
                # 𝐒, solved = calculate_first_order_solution(∇₁; T = TT)

                # aa, solved = riccati_AD_direct(∇₁; T = TT, explosive = false)
            
                Jm = (ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx,:])
                
                # ∇₊ =  ∇₁[:,1:TT.nFuture_not_past_and_mixed] * ℒ.diagm(ones(TT.nVars))[TT.future_not_past_and_mixed_idx,:]
                # ∇₀ =  ∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)]
                # ∇ₑ =  ∇₁[:,(TT.nFuture_not_past_and_mixed + TT.nVars + TT.nPast_not_future_and_mixed + 1):end]

                aa  = CC[:,TT.past_not_future_and_mixed_idx]

                bb = -((∇₊ * aa * Jm + ∇₀) \ ∇ₑ)
            
                𝐒 = hcat(aa, bb)

                A = 𝐒[observables_and_states,1:TT.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(TT.past_not_future_and_mixed_idx,observables_and_states)),:]
                B = 𝐒[observables_and_states,TT.nPast_not_future_and_mixed+1:end]

                C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                
                presample_periods = 4

                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
                
            end, CC)[1]



invtmp = inv(-CC' * ∇₊' - ∇₀')

tmp2 = invtmp * ∇₊'


# sol = speedmapping(zero(CC); m! = (X, x) ->  X .= invtmp * (rev_diff_A_B + ∇₊' * x * CC'), tol = 1e-11, maps_limit = 20000)

# sol = speedmapping(zero(CC); m! = (X, x) ->  X .= -(inv∇₀ * (rev_diff_A_B + ∇₊' * x * CC' + CC' * ∇₊' * x)), tol = 1e-11, maps_limit = 20000)



# inv∇₀ = inv(∇̂₀)'
# ∇̂₀' \  ∇₊' * X * CC' + ∇̂₀' \ CC' * ∇₊' * X 

# invtmp * (rev_diff_A_B + ∇₊' * ss * CC') + ss
tmp1 = invtmp * rev_diff_A_B
@benchmark sol = speedmapping(zero(CC); m! = (X, x) ->  X .= tmp1 + tmp2 * x * CC', tol = 1e-12, maps_limit = 20000)

import LinearOperators , Krylov

@benchmark begin
    function sylvester!(sol,𝐱)
        𝐗 = reshape(𝐱, size(tmp1))
        sol .= vec(tmp2 * 𝐗 * CC' - 𝐗)
        return sol
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(tmp1), length(tmp1), true, true, sylvester!)

    𝐂, info = Krylov.gmres(sylvester, [vec(-tmp1);])
    reshape(𝐂, size(C))
end
tmp2*tmp1

@benchmark begin
    tmp̂ = similar(tmp1)
    tmp̄ = similar(tmp1)
    𝐗 = similar(tmp1)
    function sylvester!(sol,𝐱)
        copyto!(𝐗, 𝐱)
        mul!(tmp̄, 𝐗, CC')
        mul!(tmp̂, tmp2, tmp̄)
        ℒ.axpy!(-1, tmp̂, 𝐗)
        ℒ.rmul!(𝐗, -1)
        copyto!(sol, 𝐗)
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(tmp1), length(tmp1), true, true, sylvester!)

    𝐂, info = Krylov.gmres(sylvester, [vec(-tmp1);])
    reshape(𝐂, size(C))
end



@profview for i in 1:10 
# @profview 
begin
        tmp̂ = similar(tmp1)
        tmp̄ = similar(tmp1)
        𝐗 = similar(tmp1)

        function sylvester!(sol,𝐱)
            copyto!(𝐗, 𝐱)
            mul!(tmp̄, 𝐗, CC')
            mul!(tmp̂, tmp2, tmp̄)
            ℒ.axpy!(-1, tmp̂, 𝐗)
            ℒ.rmul!(𝐗, -1)
            copyto!(sol, 𝐗)
        end

        sylvester = LinearOperators.LinearOperator(Float64, length(tmp1), length(tmp1), true, true, sylvester!)

        𝐂, info = Krylov.gmres(sylvester, [vec(-tmp1);])
        reshape(𝐂, size(C))
    end
end

tmp1 + tmp2 * x * CC'
MatrixEquations.sylvd(-tmp2, (CC'), tmp1)
# elseif solver == :bicgstab
#     𝐂, info = Krylov.bicgstab(sylvester, [vec(C);])
# end
solved = info.solved

coordinates = Tuple{Vector{Int}, Vector{Int}}[]

values = vcat(vec(tmp2), vec(CC'), vec(-tmp1))

dimensions = Tuple{Int, Int}[]
push!(dimensions,size(tmp2))
push!(dimensions,size(CC'))
push!(dimensions,size(tmp1))

ss, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :sylvester)


ss = sol.minimizer

∇₊' * ss * CC' +  CC' * ∇₊' * ss  + rev_diff_A_B   +   ∇₀' * ss 


rev_diff_A_B + ∇₊' * ss * CC' + (CC' * ∇₊' + ∇₀') * ss


maximum(abs,tmp1 + tmp2 * ss * CC' - ss)

sss = zero(ss)
sss[:,TT.past_not_future_and_mixed_idx] .= ss[:,TT.past_not_future_and_mixed_idx]
ss * CC'
ss * CC' * CC'

hcat((ss * CC' * CC')[:,TT.future_not_past_and_mixed_idx], ss * CC', ss[:,TT.past_not_future_and_mixed_idx])
# inv∇₀ * (rev_diff_A + ∇₊' * ss * CC' + CC' * ∇₊' * ss) - ss
# rev_diff_A + ∇₊' * ss * CC' + CC' * ∇₊' * ss - ∇₀' * ss

function first_order_solution_pullback(∂A)
    ∂∇₁ = zero(∇₁)
    invtmp = inv(-A' * ∇₊' - ∇₀')

    tmp1 = invtmp * ∂A
    tmp2 = invtmp * ∇₊'

    sol = speedmapping(zero(CC); m! = (X, x) ->  X .= tmp1 + tmp2 * x * A', tol = 1e-12, maps_limit = 20000)

    ss = sol.minimizer

    ∂∇₁[:,1:TT.nFuture_not_past_and_mixed] .= (ss * A' * A')[:,TT.future_not_past_and_mixed_idx]
    ∂∇₁[:,TT.nFuture_not_past_and_mixed .+ range(1,TT.nVars)] .= ss * A'
    ∂∇₁[:,TT.nFuture_not_past_and_mixed + TT.nVars .+ range(1,TT.nPast_not_future_and_mixed)] .= ss[:,TT.past_not_future_and_mixed_idx]

    return ∂∇₁
    # return NoTangent(), ∂∇₁
end
            
            
            first_order_solution_pullback(rev_diff_A_B)# ≈ rev_diff_∇₁

            

rev_diff_A = Zygote.gradient(CC -> begin
                𝐒 = hcat(CC[:,T.past_not_future_and_mixed_idx],DD)

                A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
                B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

                C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

                presample_periods = 4
                
                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
            end, CC)[1]


rev_diff_B = Zygote.gradient(B -> begin
                # A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones( length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
                # B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

                # C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

                presample_periods = 4
                
                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
            end, B)[1]

            








𝐁 = B * B'

AA = deepcopy(A)



@profview for i in 1:10000 begin
iter = 1
change = 1
A = copy(AA)
𝐂  = copy(𝐁)
𝐂¹ = copy(𝐁)
CA = similar(A)
A² = similar(A)
for iter in 1:500
    # 𝐂¹ .= A * 𝐂 * A' + 𝐂
    mul!(CA, 𝐂, A')
    mul!(𝐂¹, A, CA, 1, 1)

    # A .*= A
    mul!(A², A, A)
    copy!(A, A²)

    if !(A isa DenseMatrix)
        droptol!(A, eps())
    end
    
    solved = true

    if iter > 10
        ℒ.axpy!(-1, 𝐂¹, 𝐂)
        for c in 𝐂
            if abs(c) > eps(Float32) 
                solved = false
                break
            end
        end
    else 
        solved = false
    end

    # 𝐂 = 𝐂¹
    copy!(𝐂, 𝐂¹)

    if solved break end
end
end end


MatrixEquations.lyapd(A, 𝐁)

@benchmark MatrixEquations.lyapd(A, 𝐁)

@benchmark begin
    iter = 1
    change = 1
    A = copy(AA)
    𝐂  = copy(𝐁)
    𝐂¹ = copy(𝐁)
    CA = similar(A)
    A² = similar(A)
    while change > eps(Float32) && iter < 500
        # 𝐂¹ .= A * 𝐂 * A' + 𝐂
        mul!(CA, 𝐂, A')
        mul!(𝐂¹, A, CA, 1, 1)

        # A .*= A
        mul!(A², A, A)
        copy!(A, A²)

        if !(A isa DenseMatrix)
            droptol!(A, eps())
        end
        
        if iter > 10
            ℒ.axpy!(-1, 𝐂¹, 𝐂)
            change = maximum(abs, 𝐂)
        end

        # 𝐂 = 𝐂¹
        copy!(𝐂, 𝐂¹)

        iter += 1
    end
end



@profview for i in 1:10000 begin
    iter = 1
    change = 1
    A = copy(AA)
    𝐂  = copy(𝐁)
    𝐂¹ = copy(𝐁)
    CA = similar(A)
    A² = similar(A)
    while change > eps(Float32) && iter < 500
        # 𝐂¹ .= A * 𝐂 * A' + 𝐂
        mul!(CA, 𝐂, A')
        mul!(𝐂¹, A, CA, 1, 1)

        # A .*= A
        mul!(A², A, A)
        copy!(A, A²)

        if !(A isa DenseMatrix)
            droptol!(A, eps())
        end
        if iter > 10
            ℒ.axpy!(-1, 𝐂¹, 𝐂)
            change = maximum(abs, 𝐂)
        end

        # 𝐂 = 𝐂¹
        copy!(𝐂, 𝐂¹)

        iter += 1
    end
end end
# for_diff ≈ 𝐂



A = copy(AA)
CA = similar(A)

using SpeedMapping

@benchmark soll = speedmapping(collect(𝐁); 
                m! = (X, x) -> begin
                    mul!(CA, x, A')
                    mul!(X, A, CA)
                    ℒ.axpy!(1, 𝐁, X)
                end, stabilize = false)


@benchmark soll = speedmapping(collect(𝐁); 
                m! = (X, x) -> begin
                    mul!(CA, x, A')
                    X .= muladd(A, CA, 𝐁)
                    # mul!(X, A, CA)
                    # ℒ.axpy!(1, 𝐁, X)
                end, stabilize = false)

                # soll.minimizer

import MacroModelling: run_kalman_iterations, riccati_AD_direct


rev_diff = Zygote.gradient(B -> begin
                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                
                return ℒ.tr(P)
            end, B)[1]



for_diff = ForwardDiff.gradient(B -> begin
            𝐁 = B * B'
            coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

            dimensions =  [size(A),size(𝐁)]

            values = vcat(vec(A), vec(collect(-𝐁)))

            P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
            
            return ℒ.tr(P)
        end, B)

for_diff ≈ rev_diff


rev_diff = Zygote.gradient(B -> begin
                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

                presample_periods = 4
                
                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
            end, B)[1]


@benchmark Zygote.gradient(B -> begin
𝐁 = B * B'
coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

dimensions =  [size(A),size(𝐁)]

values = vcat(vec(A), vec(collect(-𝐁)))

P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

presample_periods = 4

return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
end, B)[1]



@profview for i in 1:1000 Zygote.gradient(B -> begin
𝐁 = B * B'
coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

dimensions =  [size(A),size(𝐁)]

values = vcat(vec(A), vec(collect(-𝐁)))

P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

presample_periods = 4

return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
end, B)[1] end



@benchmark  begin
𝐁 = B * B'
coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

dimensions =  [size(A),size(𝐁)]

values = vcat(vec(A), vec(collect(-𝐁)))

P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

presample_periods = 4

return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
end




for_diff = ForwardDiff.gradient(B -> begin
                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)
                # P = copy(Pl)
                presample_periods = 4

                u = zeros(size(C,2))

                z = C * u
            
                loglik = (0.0)
            
                F = similar(C * C')
            
                K = similar(C')
            
                for t in 1:size(data_in_deviations, 2)
                    v = data_in_deviations[:, t] - z
            
                    F = C * P * C'
            
                    luF = ℒ.lu(F, check = false) ###
            
                    if !ℒ.issuccess(luF)
                        return -Inf
                    end
            
                    Fdet = ℒ.det(luF)
            
                    # Early return if determinant is too small, indicating numerical instability.
                    if Fdet < eps(Float64)
                        return -Inf
                    end
            
                    invF = inv(luF) ###
            
                    if t > presample_periods
                        loglik += log(Fdet) + ℒ.dot(v, invF, v)###
                    end
            
                    K = P * C' * invF
            
                    P = A * (P - K * C * P) * A' + 𝐁
            
                    u = A * (u + K * v)
            
                    z = C * u
                end
            
                return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
            
            end, B)

for_diff ≈ rev_diff

import FiniteDifferences
fin_diff = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), B -> begin
                𝐁 = B * B'
                coordinates =  Tuple{Vector{Int}, Vector{Int}}[]

                dimensions =  [size(A),size(𝐁)]

                values = vcat(vec(A), vec(collect(-𝐁)))

                P = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)

                presample_periods = 4

                return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods)
            end, B)[1]



fin_diff ≈ rev_diff

for_diff ≈ fin_diff





# try again but with more elemental operations

TT = 4
presample_periods = 3
∂A = zero(A)
∂K = zero(K[1])
∂V = zero(V[1])
∂Vaccum = zero(V[1])
∂P = zero(PP)
∂u = zero(u[1])
∂u_mid = zero(u[1])
# ∂u_mid∂innovation = zero(u[1])
∂B_prod = zero(B_prod)
∂observables = zero(observables)

for t in TT:-1:2
    # loglik += logdet(V[t]) + innovation[t]' * invV[t] * innovation[t]
    if t > presample_periods
    #     ∂V = invV[t]' - invV[t]' * innovation[t] * innovation[t]' * invV[t]'
    #     # ∂Vaccum *= 0
    # end
        ∂u_mid∂innovation = C' * (invV[t]' + invV[t]) * innovation[t]
    else
        ∂u_mid∂innovation = zero(u[1])
        # ∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
    end
    # ∂V =  - invV[t]' * innovation[t] * innovation[t]' * invV[t]'
    # ∂observables[:,t-1] = (invV[t]' + invV[t]) * innovation[t]
    if t == 2
        ∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
        ∂P += C' * (∂V + ∂Vaccum) * C
        ∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid
        ∂u_mid -= ∂u_mid∂innovation
        ∂observables[:,t-1] = -C * ∂u_mid
    else
        ∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
        ∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

        # innovation[t] .= observables[:, t-1] - z[t-1]
        # z[t] .= C * u_mid[t]
        ∂u_mid -= ∂u_mid∂innovation
        ∂observables[:,t-1] = -C * ∂u_mid
        # u_mid[t] .= A * u[t]
        # innovation[t] .= observables[:, t-1] - C * A * u[t-1]

        # V[t] .= C * P_mid[t-1] * C'
        ∂P += C' * (∂V + ∂Vaccum) * C

        # P_mid[t] .= A * P[t] * A' + B_prod
        ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
        ∂A += ∂u_mid * u[t-1]'
        ∂B_prod += ∂P
        # if t == 3
            # ∂P += A' * ∂P * A
            # ∂K -= ∂P * CP[t-1]'
            # ∂P += ∂K * invV[t-1]'
        # else

        # P[t] .= P_mid[t-1] - K[t] * C * P_mid[t-1]
        ∂P = A' * ∂P * A

        # u[t] .= P_mid[t-1] * C' * invV[t] * innovation[t] + u_mid[t-1]

        # K[t] .= P_mid[t-1] * C' * invV[t]
        ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 


        ∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'

        # if t > 2
            # ∂Vaccum -= invV[t-1]' * (P_mid[t-2] * C')' * ∂P * CP[t-1]' * invV[t-1]'
        ∂Vaccum -= invV[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invV[t-1]'
        # end
        # ∂P -= 2 * ∂P * K[t-1] * C
            # ∂P += A' * ∂P * A
        # end
    end
end

∂P *= -1/2
∂A *= -1/2
∂B_prod *= -1/2
∂observables *= -1/2

zyggrad ≈ ∂P
zyggrad - ∂P
# ∂B_prod ≈ zyggrad
# ∂observables ≈ fingrad

# ∂P += ∂P_mid
# forgrad_P ≈ ∂P

# ∂observables - fingrad

# ΔA, ΔB, NoTangent(), ΔP, Δobservables

t = T
obs = (invV[t]' + invV[t]) * innovation[t]


A * K[t] * obs
-(K[t-1])' * ∂u_mid + (invV[t-1]' + invV[t-1]) * innovation[t-1]



∂A ≈ 2*∂wⁿ⁻⁹₂∂A
∂A ≈ 2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A)
∂A ≈ 2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A) + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A
∂A ≈ 2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A) + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A + ∂wⁿ⁻¹⁵₃²∂A + ∂wⁿ⁻¹⁵₃³∂A
∂A ≈ 2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A) + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A + ∂wⁿ⁻¹⁵₃²∂A + ∂wⁿ⁻¹⁵₃³∂A + ∂wⁿ⁻²⁰₃²∂A + ∂wⁿ⁻²⁰₃³∂A


# figure out P

zyggrad = Zygote.gradient(
    PP -> begin
        CP2 = C * PP
        V2 = CP2 * C'
        K2 = PP * C' * inv(V2)
        innovation2 = observables[:, 1] - z[1]
        u2 = K2 * innovation2 + u_mid[1]
        P2 = PP - K2 * CP2
        u_mid2 = A * u2
        z2 = C * u_mid2
        P_mid2 = A * P2 * A' + B_prod

        CP3 = C * P_mid2
        V3 = CP3 * C'
        innovation3 = observables[:, 2] - z2
        K3 = P_mid2 * C' * inv(V3)
        u3 = K3 * innovation3 + u_mid2
        P3 = P_mid2 - K3 * CP3
        u_mid3 = A * u3
        z3 = C * u_mid3
        P_mid3 = A * P3 * A' + B_prod

        CP4 = C * P_mid3
        V4 = CP4 * C'
        innovation4 = observables[:, 3] - z3

        # return -1/2*(logdet(V[2]) + innovation2' * inv(V[2]) * innovation2)
        # return -1/2*(logdet(V[3]) + innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(logdet(V2) + innovation2' * inv(V2) * innovation2 + logdet(V3) + innovation3' * inv(V3) * innovation3)
        return -1/2*(logdet(V[4]) + innovation4' * inv(V[4]) * innovation4)
        # return -1/2*(logdet(V4) + innovation4' * inv(V4) * innovation4 + logdet(V3) + innovation3' * inv(V3) * innovation3)
    end, 
    PP)[1]

    zyggrad ≈ ∂P
    zyggrad - ∂P



∂A = zero(A)
∂K = zero(K[1])
∂V = zero(V[1])
∂Vaccum = zero(V[1])
∂P = zero(PP)
∂P_mid = zero(PP)
∂u = zero(u[1])
∂u_mid = zero(u[1])
∂u_mid_accum = zero(u[1])
∂B_prod = zero(B_prod)
∂observables = zero(observables)

# t = 5

# ∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
# ∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

# ∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
# ∂observables[:,t-1] = -C * ∂u_mid

# ∂P += C' * (∂V + ∂Vaccum) * C

# ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
# ∂A += ∂u_mid * u[t-1]'
# ∂B_prod += ∂P

# ∂P = A' * ∂P * A
# ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 

# ∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'
# ∂Vaccum -= invV[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invV[t-1]'

t = 4

∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
∂observables[:,t-1] = -C * ∂u_mid

∂P += C' * (∂V + ∂Vaccum) * C

∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
∂A += ∂u_mid * u[t-1]'
∂B_prod += ∂P

∂P = A' * ∂P * A
∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 

∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'
∂Vaccum -= invV[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invV[t-1]'

t = 3

∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

# ∂u_mid -= ∂u_mid∂innovation
∂observables[:,t-1] = -C * ∂u_mid

∂P += C' * (∂V + ∂Vaccum) * C

∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
∂A += ∂u_mid * u[t-1]'
∂B_prod += ∂P

∂P = A' * ∂P * A
∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 

∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'
∂Vaccum -= invV[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invV[t-1]'

t = 2

∂P += A' * ∂u_mid * innovation[t]' * invV[t]' * C
∂P += C' * (∂V + ∂Vaccum) * C
∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid
# ∂u_mid -= ∂u_mid∂innovation
∂observables[:,t-1] = -C * ∂u_mid


∂P/= -2

zyggrad ≈ ∂P


∂V = invV[t]' - invV[t]' * innovation[t] * innovation[t]' * invV[t]'
∂P += C' * ∂V * C
∂P/= -2

    # ∂V =  - invV[t]' * innovation[t] * innovation[t]' * invV[t]'
    # ∂observables[:,t-1] = (invV[t]' + invV[t]) * innovation[t]
    if t == 2
        ∂P_mid += A' * ∂u_mid * innovation[t]' * invV[t]' * C
        ∂P_mid += C' * (∂V + ∂Vaccum) * C
        ∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid
        ∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
        ∂observables[:,t-1] = -C * ∂u_mid
    else
        ∂P_mid += A' * ∂u_mid * innovation[t]' * invV[t]' * C
        ∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

        # innovation[t] .= observables[:, t-1] - z[t-1]
        # z[t] .= C * u_mid[t]
        # u_mid[t] .= A * u[t]
        # innovation[t] .= observables[:, t-1] - C * A * u[t-1]
        # ∂u_mid -= C' * ∂observables[:,t-1]
        ∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
        ∂observables[:,t-1] = -C * ∂u_mid
        # ∂u -= A' * C' * (invV[t]' + invV[t]) * innovation[t]
        # V[t] .= C * P_mid[t-1] * C'
        ∂P_mid += C' * (∂V + ∂Vaccum) * C

        # P_mid[t] .= A * P[t] * A' + B_prod
        ∂A += ∂P_mid * A * P[t-1]' + ∂P_mid' * A * P[t-1]
        ∂A += ∂u_mid * u[t-1]'
        ∂B_prod += ∂P_mid
        # if t == 3
            # ∂P += A' * ∂P_mid * A
            # ∂K -= ∂P_mid * CP[t-1]'
            # ∂P += ∂K * invV[t-1]'
        # else

        # P[t] .= P_mid[t-1] - K[t] * C * P_mid[t-1]
        ∂P_mid = A' * ∂P_mid * A

        # u[t] .= P_mid[t-1] * C' * invV[t] * innovation[t] + u_mid[t-1]

        # K[t] .= P_mid[t-1] * C' * invV[t]
        ∂P_mid -= C' * K[t-1]' * ∂P_mid + ∂P_mid * K[t-1] * C 


        ∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'

        # if t > 2
            # ∂Vaccum -= invV[t-1]' * (P_mid[t-2] * C')' * ∂P_mid * CP[t-1]' * invV[t-1]'
        ∂Vaccum -= invV[t-1]' * CP[t-1] * ∂P_mid * CP[t-1]' * invV[t-1]'
        # end
        # ∂P_mid -= 2 * ∂P_mid * K[t-1] * C
            # ∂P_mid += A' * ∂P_mid * A
        # end
    end


    
# figure out obs
# attempt with u_mid
∂u_mid = zero(u[1])

t = 4
obs3 = (invV[t]' + invV[t]) * innovation[t]

∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

∂u_mid -= C' * obs3

t = 3
obs2 = (invV[t]' + invV[t]) * innovation[t]

∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

∂u_mid -= C' * obs2

# obs2 -= K[t]' * A' * C' * obs3
obs2 = -C * ∂u_mid
t = 2
obs1 = (invV[t]' + invV[t]) * innovation[t]

∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid

∂u_mid -= C' * obs1

obs1 = -C * ∂u_mid

# obs1 -= K[t]' * A' * A' * C' * obs3 - K[t]' * A' * C' * K[t+1]' * A' * C' * obs3 + K[t]' * A' * C' * (invV[t+1]' + invV[t+1]) * innovation[t+1]
# obs1 += 
# - K[t]' * A' * A' * C' * obs3 
# + K[t]' * A' * C' * K[t+1]' * A' * C' * obs3 
# - K[t]' * A' * C' * (invV[t+1]' + invV[t+1]) * innovation[t+1]


obs1 /= -2
obs2 /= -2
obs3 /= -2

hcat(obs1, obs2, obs3)





# this works
t = 4
obs3 = (invV[t]' + invV[t]) * innovation[t]

t = 3
obs2 = (invV[t]' + invV[t]) * innovation[t]

obs2 -= K[t]' * A' * C' * obs3

t = 2
obs1 = (invV[t]' + invV[t]) * innovation[t]

# obs1 -= K[t]' * A' * A' * C' * obs3 - K[t]' * A' * C' * K[t+1]' * A' * C' * obs3 + K[t]' * A' * C' * (invV[t+1]' + invV[t+1]) * innovation[t+1]
obs1 += 
- K[t]' * A' * A' * C' * obs3 
+ K[t]' * A' * C' * K[t+1]' * A' * C' * obs3 
- K[t]' * A' * C' * (invV[t+1]' + invV[t+1]) * innovation[t+1]


obs1 /= -2
obs2 /= -2
obs3 /= -2

hcat(obs1, obs2, obs3)

zyggrad = Zygote.gradient(
    observables -> begin
        CP2 = C * P_mid[1]
        K2 = P_mid[1] * C' * invV[2]
        innovation2 = observables[:, 1] - z[1]
        u2 = K2 * innovation2 + u_mid[1]
        P2 = P_mid[1] - K2 * CP2
        u_mid2 = A * u2
        z2 = C * u_mid2
        P_mid2 = A * P2 * A' + B_prod

        CP3 = C * P_mid2
        V3 = CP3 * C'
        innovation3 = observables[:, 2] - z2
        K3 = P_mid2 * C' * inv(V3)
        u3 = K3 * innovation3 + u_mid2
        P3 = P_mid2 - K3 * CP3
        u_mid3 = A * u3
        z3 = C * u_mid3
        P_mid3 = A * P3 * A' + B_prod

        CP4 = C * P_mid3
        V4 = CP4 * C'
        innovation4 = observables[:, 3] - z3

        # return -1/2*(innovation2' * inv(V[2]) * innovation2)
        # return -1/2*(innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(innovation2' * inv(V[2]) * innovation2 + innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(innovation4' * inv(V[4]) * innovation4)
        return -1/2*(innovation2' * inv(V[2]) * innovation2 + innovation3' * inv(V[3]) * innovation3 + innovation4' * inv(V[4]) * innovation4)
    end, 
    observables[:,1:3])[1]

forgrad_data_in_deviations



∂A = zero(A)
∂K = zero(K[1])
∂V = zero(V[1])
∂Vaccum = zero(V[1])
∂P = zero(PP)
∂P_mid = zero(PP)
∂u = zero(u[1])
∂u_mid = zero(u[1])

t = 4
∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid
∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
∂A += ∂u_mid * u[t-1]'

∂Vaccum = -invV[t-1]' * CP[t-1] * A' * ∂u_mid * innovation[t-1]' * invV[t-1]'

t = 3
∂P_mid += A' * ∂u_mid * innovation[t]' * invV[t]' * C
∂P_mid += C' * (∂V + ∂Vaccum) * C
∂u_mid = A' * ∂u_mid - C' * K[t]' * A' * ∂u_mid
∂u_mid -= C' * (invV[t]' + invV[t]) * innovation[t]
∂A += ∂u_mid * u[t-1]'

∂A += ∂P_mid * A * P[t-1]' + ∂P_mid' * A * P[t-1]

∂A *= -1/2


maximum(abs, ∂A - (2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A) + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A + ∂wⁿ⁻¹⁵₃²∂A + ∂wⁿ⁻¹⁵₃³∂A + ∂wⁿ⁻²⁰₃²∂A + ∂wⁿ⁻²⁰₃³∂A))
∂A ≈ ∂z∂A


zyggrad =   Zygote.gradient(
                x -> begin
                    u_mid2 = A * x
                    z2 = C * u_mid2
                    innovation3 = observables[:, 2] - z2
                    
                    return -1/2*(innovation3' * invV[3] * innovation3)
                end, 
            u[2])[1]

    
            ∂u - zyggrad

zyggrad =   Zygote.gradient(
                x -> begin
                    CP2 = C * P_mid[1]
                    K2 = P_mid[1] * C' * invV[2]
                    u2 = K2 * innovation[2] + u_mid[1]
                    P2 = P_mid[1] - K2 * CP2
                    u_mid2 = A * u2
                    z2 = C * u_mid2
                    P_mid2 = A * P2 * A' + x

                    CP3 = C * P_mid2
                    V3 = CP3 * C'
                    innovation3 = observables[:, 2] - z2

                    # return -1/2*(innovation[3]' * inv(V3) * innovation[3])
                    # return -1/2*(innovation3' * inv(V3) * innovation3)
                    # return -1/2*(logdet(V3) + innovation3' * invV[3] * innovation3)
                    return -1/2*(logdet(V3) + innovation3' * inv(V3) * innovation3)
                end, 
                B_prod)[1]

            zyggrad ≈ ∂A
            zyggrad - ∂A



zyggrad =   Zygote.gradient(
    x -> begin
        CP2 = C * P_mid[1]
        K2 = P_mid[1] * C' * invV[2]
        u2 = K2 * innovation[2] + u_mid[1]
        P2 = P_mid[1] - K2 * CP2
        # u_mid2 = x * (K2 * innovation[2] + u_mid[1])
        u_mid2 = A * u2
        z2 = C * u_mid2
        P_mid2 = A * P2 * A' + x

        CP3 = C * P_mid2
        V3 = CP3 * C'
        innovation3 = observables[:, 2] - z2
        K3 = P_mid2 * C' * inv(V3)
        u3 = K3 * innovation3 + u_mid2
        P3 = P_mid2 - K3 * CP3
        # u_mid3 = x * (P_mid[2] * C' * inv(V[3]) * (observables[:, 2] - C * u_mid2) + u_mid2)
        # u_mid3 = x * (K[3] * (observables[:, 2] - C * u_mid[2]) + u_mid[2])
        u_mid3 = A * u3
        z3 = C * u_mid3
        P_mid3 = A * P3 * A' + x

        CP4 = C * P_mid3
        V4 = CP4 * C'
        innovation4 = observables[:, 3] - z3
        # innovation4 = observables[:, 3] - C * u_mid3

        # return -1/2*(innovation[3]' * inv(V3) * innovation[3])
        # return -1/2*(innovation3' * inv(V3) * innovation3)
        # return -1/2*(logdet(V3) + innovation3' * invV[3] * innovation3)
        # return -1/2*(logdet(V4) + innovation4' * inv(V4) * innovation4 + logdet(V3) + innovation3' * inv(V3) * innovation3)
        # return -1/2*(innovation4' * inv(V4) * innovation4)
        # return -1/2*(innovation[3]' * inv(V3) * innovation[3] + innovation[4]' * inv(V4) * innovation[4])
        # return -1/2*(innovation3' * inv(V[3]) * innovation3 + innovation4' * inv(V[4]) * innovation4)
        # return -1/2*(logdet(V4) + innovation[4]' * inv(V4) * innovation[4] + logdet(V3) + innovation[3]' * inv(V3) * innovation[3])
        # return -1/2*(innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(innovation4' * inv(V[4]) * innovation4)
        # return -1/2*(innovation3' * inv(V3) * innovation3 + innovation4' * inv(V4) * innovation4)
        return -1/2*(logdet(V3) + innovation3' * inv(V3) * innovation3 + logdet(V4) + innovation4' * inv(V4) * innovation4)
    end, 
    B_prod)[1]

zyggrad
zyggrad ≈ ∂A

zyggrad - ∂A

k3effect = zyggrad - ∂A




zyggrad =   Zygote.gradient(
                x -> begin
                    P_mid2 = x * P[2] * x' + B_prod
                    CP3 = C * P_mid2
                    V3 = CP3 * C'
                    K3 = P_mid2 * C' * inv(V3)
                    P3 = P_mid2 - K3 * CP3

                    P_mid3 = x * P3 * x' + B_prod
                    CP4 = C * P_mid3
                    V4 = CP4 * C'
                    K4 = P_mid3 * C' * inv(V4)
                    P4 = P_mid3 - K4 * CP4

                    P_mid4 = x * P4 * x' + B_prod
                    CP5 = C * P_mid4
                    V5 = CP5 * C'
                    # return -1/2*(logdet(V3))
                    # return -1/2*(logdet(V4) + logdet(V3))
                    return -1/2*(logdet(V5) + logdet(V4) + logdet(V3))
                end, 
            A)[1]

isapprox(∂A, zyggrad)
isapprox(∂A, fingrad)

isapprox(fingrad, ∂A)
fingrad - ∂A
isapprox(fingrad, zyggrad)
∂A - zyggrad

(P[3]' * A' *                                              C' * -∂z∂z/ 2 * inv(V[4])' * C    )'
(P[2]' * A' * A' *                                         C' * -∂z∂z/ 2 * inv(V[4])' * C     * A)'





zyggrad =   Zygote.gradient(
                x -> begin
                    P_mid2 = x * P[2] * x' + B_prod
                    CP3 = C * P_mid2
                    V3 = CP3 * C'
                    K3 = P_mid2 * C' * inv(V3)
                    P3 = P_mid2 - K3 * CP3

                    P_mid3 = x * P3 * x' + B_prod
                    CP4 = C * P_mid3
                    V4 = CP4 * C'
                    # return -1/2*(logdet(V3) + innovation[3]' * inv(V3) * innovation[3])
                    return -1/2*(logdet(V4) + innovation[4]' * inv(V4) * innovation[4] + logdet(V3) + innovation[3]' * inv(V3) * innovation[3])
                end, 
            A)[1]

isapprox(∂A, zyggrad)




# ∂A ≈ ∂z∂A

# ForwardDiff

PP = get_initial_covariance(Val(:theoretical), vcat(vec(A), vec(collect(-𝐁))), coordinates, dimensions)

forgrad_A = ForwardDiff.gradient(A -> begin
    u = zeros(size(C,2))

    z = C * u

    P = deepcopy(PP)

    observables = data_in_deviations

    presample_periods = 0

    loglik = 0.0

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + v' * invF * v###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2

    return zz
end, A)

∂A ≈ forgrad_A
maximum(abs, ∂A - forgrad_A)
maximum(abs, (∂A - forgrad_A) ./ forgrad_A)



forgrad_𝐁 = ForwardDiff.gradient(𝐁 -> begin
    u = zeros(size(C,2))

    z = C * u

    P = deepcopy(PP)

    observables = data_in_deviations

    presample_periods = 0

    loglik = 0.0

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + v' * invF * v###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2

    return zz
end, 𝐁)

∂B_prod ≈ forgrad_𝐁
maximum(abs, ∂B_prod - forgrad_𝐁)
maximum(abs, (∂B_prod - forgrad_𝐁) ./ forgrad_𝐁)




forgrad_data_in_deviations = ForwardDiff.gradient(data_in_deviations -> begin
    u = zeros(size(C,2))

    z = C * u

    P = deepcopy(PP)

    observables = data_in_deviations

    presample_periods = 0

    loglik = 0.0

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + v' * invF * v###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2

    return zz
end, data_in_deviations)

forgrad_data_in_deviations ≈ ∂observables
∂observables - forgrad_data_in_deviations
maximum(abs, ∂observables - forgrad_data_in_deviations)
maximum(abs, (∂observables - forgrad_data_in_deviations) ./ forgrad_data_in_deviations)





forgrad_P = ForwardDiff.gradient(P -> begin
    u = zeros(size(C,2))

    z = C * u

    # P = deepcopy(PP)

    observables = data_in_deviations

    presample_periods = 0

    loglik = 0.0

    for t in 1:2#size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            return -Inf
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + v' * invF * v###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2

    return zz
end, PP)

forgrad_P ≈ ∂P
∂P - forgrad_P
maximum(abs, ∂P - forgrad_P)
maximum(abs, (∂P - forgrad_P) ./ forgrad_P)




import FiniteDifferences

fingrad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),
x -> begin
P_mid[1] = deepcopy(PP)
P[1] = deepcopy(PP)
loglik = 0.0
for t in 2:T
    CP[t] .= C * P_mid[t-1]

    V[t] .= CP[t] * C'

    luV = ℒ.lu(V[t], check = false)

    Vdet = ℒ.det(luV)
    
    invV[t] .= inv(luV)
    
    innovation[t] .= x[:, t-1] - z[t-1]
    # if t == 4
    loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
    # end
    K[t] .= P_mid[t-1] * C' * invV[t]

    u[t] .= K[t] * innovation[t] + u_mid[t-1]
    
    P[t] .= P_mid[t-1] - K[t] * CP[t]

    u_mid[t] .= A * u[t]

    z[t] .= C * u_mid[t]

    P_mid[t] .= A * P[t] * A' + B_prod
end
return -1/2*loglik
end, observables)[1]



PP = get_initial_covariance(Val(:theoretical), vcat(vec(A), vec(collect(-𝐁))), coordinates, dimensions)
observables = data_in_deviations

T = size(observables, 2) + 1

u = [zeros(size(C,2)) for _ in 1:T]

u_mid = deepcopy(u)

z = [zeros(size(observables, 1)) for _ in 1:T]

P_mid = [deepcopy(PP) for _ in 1:T]

temp_N_N = similar(PP)

P = deepcopy(P_mid)

B_prod = 𝐁
# Ct = collect(C')
CP = [zero(C) for _ in 1:T]

K = [zero(C') for _ in 1:T]

cc = C * C'

V = [zero(cc) for _ in 1:T]

invV = [zero(cc) for _ in 1:T]

V[1] += ℒ.I
invV[1] = inv(V[1])

innovation = deepcopy(z)

# V[1] .= C * P[1] * C'

loglik = (0.0)



for t in 2:T
    CP[t] .= C * P_mid[t-1]

    V[t] .= CP[t] * C'

    luV = ℒ.lu(V[t], check = false)

    Vdet = ℒ.det(luV)
    
    invV[t] .= inv(luV)
    
    innovation[t] .= observables[:, t-1] - z[t-1]
    
    loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]

    K[t] .= P_mid[t-1] * C' * invV[t]

    u[t] .= K[t] * innovation[t] + u_mid[t-1]
    
    P[t] .= P_mid[t-1] - K[t] * CP[t]

    u_mid[t] .= A * u[t]

    z[t] .= C * u_mid[t]

    P_mid[t] .= A * P[t] * A' + B_prod
end


isapprox(fingrad, ∂A)

maximum(abs, (fingrad - ∂A) ./ ∂A)

isapprox(fingrad, ∂B_prod)

maximum(abs, (fingrad - ∂B_prod) ./ ∂B_prod)

isapprox(fingrad, zyggrad)

fingrad - ∂z∂A
# wⁿ⁻¹³₃ = K[3] * CP[3] = wⁿ⁻¹⁴₃ * wⁿ⁻¹⁵₃
∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃ = ∂wⁿ⁻¹¹₃∂wⁿ⁻¹³₃ * CP[3]'
∂wⁿ⁻¹³₃∂wⁿ⁻¹⁵₃ = K[3]' * ∂wⁿ⁻¹¹₃∂wⁿ⁻¹³₃


# wⁿ⁻¹⁴₃ = K[3] = PC[2] * invV[3] = P_mid[2] * C' * invV[3] = wⁿ⁻¹⁶₃ * wⁿ⁻¹⁷₃
∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁶₃ = ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃ * invV[3]'
∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁷₃ = (P_mid[2] * C')' * ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃

wⁿ⁻¹⁶₃ = P_mid[2] * C'
∂wⁿ⁻¹⁶₃∂P = ∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁶₃ * C

# wⁿ⁻¹⁷₃ = inv(V[3]) = inv(wⁿ⁻¹⁸₃)
∂wⁿ⁻¹⁷₃∂wⁿ⁻¹⁸₃ = -invV[3]' * ∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁷₃ * invV[3]'

# wⁿ⁻¹⁸₃ = V[3] = CP[3] * C' = wⁿ⁻¹⁹₃ * C' = wⁿ⁻⁶₁
# wⁿ⁻¹⁹₃ = CP[3] = C * P_mid[2] = 
∂wⁿ⁻¹⁸₃∂wⁿ⁻¹⁹₃ = ∂wⁿ⁻¹⁷₃∂wⁿ⁻¹⁸₃ * C
∂wⁿ⁻¹⁹₃∂P = C' * ∂wⁿ⁻¹⁸₃∂wⁿ⁻¹⁹₃


# wⁿ⁻¹⁹₃ = wⁿ⁻¹⁵₃
∂wⁿ⁻¹⁵₃∂P = C' * ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁵₃


∂z∂P = ∂wⁿ⁻¹⁵₃∂P + ∂wⁿ⁻¹⁹₃∂P + ∂wⁿ⁻¹⁶₃∂P + ∂wⁿ⁻¹¹₃∂P + ∂wⁿ⁻⁷₁∂P
