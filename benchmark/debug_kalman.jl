
using MacroModelling
import Turing: NUTS, HMC, PG, IS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
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


import Zygote
import ForwardDiff
import FiniteDifferences

back_grad = Zygote.gradient(x -> get_loglikelihood(𝓂, data, x, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal), parameters_combined)[1]

fini_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> get_loglikelihood(𝓂, data, x, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal), parameters_combined)[1]

fini_grad ≈ back_grad

forw_grad = ForwardDiff.gradient(x -> get_loglikelihood(𝓂, data, x, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal), parameters_combined)

forw_grad ≈ back_grad
forw_grad ≈ fini_grad

@benchmark Zygote.gradient(x -> get_loglikelihood(𝓂, data, x, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal), parameters_combined)[1]

@profview for i in 1:100 Zygote.gradient(x -> get_loglikelihood(𝓂, data, x, verbose = false, presample_periods = 4, filter = fltr, algorithm = algo, initial_covariance = :diagonal), parameters_combined)[1] end

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

    TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)

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
    
    values = vcat(vec(A), vec(collect(-𝐁)))


    ####### old ########
    P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

    u = zeros(size(C,2))

    z = C * u

    loglik = (0.0)

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            return -Inf
        end

        Fdet = ℒ.det(luF)

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


    -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 




    #### more explicit


    v = data_in_deviations[:, t] - C * u

    if t > presample_periods
        llh = loglik + logdet(C * P * C') + v' * inv(C * P * C') * v###
    end

    û = A * (u + P * C' * inv(C * P * C') * v)

    P̂ = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁




    ######## new
    P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)
    observables = data_in_deviations

    T = size(observables, 2) + 1

    u = [zeros(size(C,2)) for _ in 1:T]

    u_mid = deepcopy(u)

    z = [zeros(size(observables, 1)) for _ in 1:T]

    P_mid = [deepcopy(P) for _ in 1:T]

    temp_N_N = similar(P)

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
        # Kalman iteration
        mul!(CP[t], C, P_mid[t-1]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t], CP[t], C')
        # V[t].mat .+= R

        luV = ℒ.lu(V[t], check = false)
        Vdet = ℒ.det(luV)
        if Vdet < eps(Float64)
            return -Inf
        end
        invV[t] .= inv(luV)
        
        innovation[t] .= observables[:, t-1] - z[t-1]
        # loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat
        if t - 1 > presample_periods
            loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
        end

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        mul!(K[t], P_mid[t-1] * C', invV[t])

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_mid[t-1])
        mul!(u[t], K[t], innovation[t], 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_mid[t-1])
        mul!(P[t], K[t], CP[t], -1, 1)

        # this was moved down indicating a timing difference between the two approaches
        mul!(u_mid[t], A, u[t]) # u[t] = A u[t-1]
        mul!(z[t], C, u_mid[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t], A')
        mul!(P_mid[t], A, temp_N_N)
        P_mid[t] .+= B_prod
    end

    -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 



    for t in 2:T
        # Kalman iteration
        u[t] = A * u[t-1]
        z[t] = C * u[t]

        P[t] = A * P[t - 1] * A' + B * B'

        CP[t] = C * P[t]

        V[t] = CP[t] * C' + R

        innovation[t] .= observables[:, t-1] - z[t-1]

        # loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat
        if t - 1 > presample_periods
            loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
        end

        K[t] .= CP[t]' / V[t]  # Kalman gain

        u[t] += K[t] * innovation[t]
        P[t] -= K[t] * CP[t]

    end

    for t in T:-1:2
        # pullback
        # Sensitivity accumulation
        # P[t] -= K[t] * CP[t]
        copy!(ΔP_mid, ΔP)
        ΔK = -ΔP * CP[t]'
        ΔCP = - K[t]' * ΔP

        # u[t] += K[t] * innovation[t]
        copy!(Δu_mid, Δu)
        ΔK += Δu * innovation[t]'
        Δinnovation += -K[t]'* Δu

        # K[t] .= CP[t]' / V[t]
        ΔCP += invV[t] * ΔK'
        ΔV .= -invV[t] * CP[t] * ΔK * invV[t]

        # loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
        Δinnovation += 2 * Δlogpdf * invV[t] * innovation[t] # Σ^-1 * (z_obs - z)
        ΔV -= Δlogpdf * (invV[t] - invV[t] * innovation[t] * innovation[t]' * invV[t])

        # innovation[t] .= observables[:, t-1] - z[t-1]
        Δobservables = Δinnovation
        Δz = -Δinnovation

        # V[t] = CP[t] * C' + R
        ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
        ΔP_mid += C' * ΔV * C

        # CP[t] = C * P[t]
        ΔC += ΔCP * P_mid[t]'
        ΔP_mid += C' * ΔCP

        # P[t] = A * P[t - 1] * A' + B
        ΔA += ΔP_mid * A * P[t - 1]
        ΔP .= A' * ΔP_mid * A # pass into next period
        ΔB += ΔP_mid

        # z[t] = C * u[t]
        ΔC += Δz * u_mid[t]'
        Δu_mid += C' * Δz

        # u[t] = A * u[t-1]
        ΔA += Δu_mid * u[t - 1]'
        Δu += A' * Δu_mid
    end



    ### check derivatives fro llh
    
    function calc_llh(c, p)
        # innovation[t]' * invV[t] * innovation[t]
        f = c * p * c'
        return -logdet(f)/2
    end

    function rrule(::typeof(calc_llh), c, p)
        f = c * p * c'

        function pullback(∂llh)
            ∂P = -c' * inv(f) * c * ∂llh/2
            return NoTangent(), NoTangent(), ∂P
        end
        return -logdet(f)/2, pullback
    end

    calc_llh(C, P[1])
    grad1 = Zygote.gradient(x -> calc_llh(C, x), P[1])[1]#|>sparse

    grad2 = Zygote.gradient(x -> -logdet(C * x * C')/2, P[1])[1]#|>sparse
    
    grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> -logdet(C * x * C')/2, P[1])[1]


    isapprox(grad3, grad1)
    isapprox(grad3, grad2)
    ### derivatives for second part of llh
        
    function calc_llh2(P, C, v)
        -v' * inv(C * P * C') * v / 2
    end

    function rrule(::typeof(calc_llh2), P, C, v)
        invF = inv(C * P * C')
        
        function pullback(∂llh)
            ∂F = -invF * (v * v') * invF * ∂llh
            ∂P = C' * ∂F * C
            return NoTangent(), ∂P, NoTangent(), NoTangent()
        end

        return -v' * invF * v / 2, pullback
    end


    calc_llh2(P[1], C, v)
    C' * v * v' * C 
    grad2 = Zygote.gradient(x -> v' * inv(C * x * C') * v, P[1])[1]#|>sparse

    grad1 = Zygote.gradient(x -> calc_llh2(x, C, v), P[1])[1]#|>sparse

    grad1 == grad2

    # derivative for u and data_in_deviations
    v = data_in_deviations[:, t] - C * u
    
    llh = loglik + logdet(C * P * C') + v' * inv(C * P * C') * v###



    function calc_llh3(P, C, u, data)
        -(data - C * u)' * inv(C * P * C') * (data - C * u)/2
    end

    function rrule(::typeof(calc_llh3), P, C, u, data)
        invF = inv(C * P * C')
        
        function pullback(∂llh)
            ∂P̂ = -invF * (v * v') * invF * ∂llh
            ∂P = C' * ∂P̂ * C
            ∂u = -2 * C' * invF * (data - C * u) * ∂llh
            ∂data = 2 * invF * (data - C * u) * ∂llh
            return NoTangent(), ∂P, NoTangent(), ∂u, ∂data
        end

        return - (data - C * u)' * inv(C * P * C') * (data - C * u) / 2, pullback
    end


    calc_llh3(P[1], C, u[1], data_in_deviations[:,t])

    grad2 = Zygote.gradient(x -> (data_in_deviations[:, t] - C * x)' * inv(C * P[1] * C') * (data_in_deviations[:, t] - C * x), u[1])[1]#|>sparse


    grad1 = Zygote.gradient(x -> calc_llh3(P[1], C, x, data_in_deviations[:,t]), u[1])[1]#|>sparse
    


    grad2 == grad1
    
    


    grad2 = Zygote.gradient(x -> (x - C * u[1])' * inv(C * P[1] * C') * (x - C * u[1]), data_in_deviations[:, t])[1]#|>sparse

    grad1 = Zygote.gradient(x -> calc_llh3(P[1], C, u[1], x), data_in_deviations[:,t])[1]#|>sparse
    
    grad2 == grad1
    
    





    function calc_llh4(P, C, u, data)
        F = C * P * C'
        v = data - C * u
        return -(logdet(F) + v' * inv(F) * v) / 2
    end

    function rrule(::typeof(calc_llh4), P, C, u, data)
        F = C * P * C'
        invF = inv(F)
        v = data - C * u

        function pullback(∂llh)
            ∂P̂ = invF * (v * v') * invF * ∂llh/2
            ∂P = C' * ∂P̂ * C - C' * invF * C * ∂llh/2
            ∂u = C' * invF * (data - C * u) * ∂llh
            ∂data = -invF * (data - C * u) * ∂llh
            return NoTangent(), ∂P, NoTangent(), ∂u, ∂data
        end

        return -(logdet(F) + v' * invF * v) / 2, pullback
    end


    
    calc_llh4(P[1], C, u[1], data_in_deviations[:,t])

    grad2 = Zygote.gradient(x -> -(logdet(C * x * C') + (data_in_deviations[:, t] - C * u[1])' * inv(C * x * C') * (data_in_deviations[:, t] - C * u[1])) / 2 , P[1])[1]#|>sparse

    grad1 = Zygote.gradient(x -> calc_llh4(x, C, u[1], data_in_deviations[:,t]), P[1])[1]#|>sparse
    
    grad2 == grad1


# P
grad1 = Zygote.gradient(x -> calc_llh4(x, C, u[1], data_in_deviations[:,t]), P[1])[1]#|>sparse
    
grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> calc_llh4(x, C, u[1], data_in_deviations[:,t]), P[1])[1]


isapprox(grad3, grad1, rtol = 1e-7)

# u
grad1 = Zygote.gradient(x -> calc_llh4(P[1], C, x, data_in_deviations[:,t]), u[1])[1]#|>sparse
    
grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> calc_llh4(P[1], C, x, data_in_deviations[:,t]), u[1])[1]


isapprox(grad3, grad1, rtol = 1e-7)



# data
grad1 = Zygote.gradient(x -> calc_llh4(P[1], C, u[1], x), data_in_deviations[:,t])[1]#|>sparse
    
grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> calc_llh4(P[1], C, u[1], x), data_in_deviations[:,t])[1]

isapprox(grad3, grad1, rtol = 1e-7)


    


# now moving to the next time period



if t > presample_periods
    llh = loglik + logdet(C * P * C') + v' * inv(C * P * C') * v###
end

û = A * (u + P * C' * inv(C * P * C') * v)

P̂ = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁

v = data_in_deviations[:, t] - C * û

llh = loglik + logdet(C * P̂ * C') + v' * inv(C * P̂ * C') * v



logdet(C * (A * (P[1] - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C')

grad2 = Zygote.gradient(x -> logdet(C * (A * (x - x * C' * inv(C * x * C') * C * x) * A' + 𝐁) * C'), P[1])[1]



Zygote.gradient(x -> logdet(C * (A * (P - x * C' * invF * C * x) * A' + 𝐁) * C'), P[1])




function logdet_transform(p, C, A, 𝐁)
    logdet(C * (A * (p - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C')
end

logdet_transformed(P[1], C, A, 𝐁)



function rrule(::typeof(logdet_transformed), P, C, A, B, invF)
    # Step 1: Compute intermediate matrices
    M = P - P * C' * invF * C * P
    N = A * M * A' + 𝐁
    Omega = C * N * C'

    # Step 2: Compute function value
    logdet_value = logdet(Omega)
    
    # Step 3: Define the pullback
    function logdet_transformed_pullback(Δlogdet)
        # Computing gradients using chain rule as derived
        Omega_inv = inv(Omega)
        dN_dM = A' * (C' * Omega_inv * C) * A
        # Gradient of M wrt x involves the derivative of a quadratic form
        dM_dx = ℒ.I - 2 * C' * invF * C * P

        # Pullback calculation
        ∂x = dM_dx * dN_dM * Δlogdet
        return NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return logdet_value, logdet_transformed_pullback
end


using ChainRulesCore

function rrule(::typeof(logdet_transform), P, C, A, B)
    # Intermediate matrix definitions
    Y = C * P * C'
    Y_inv = inv(Y)
    inner_matrix = P - P * C' * Y_inv * C * P
    transformed_matrix = A * inner_matrix * A' + B
    Omega = C * transformed_matrix * C'
    
    # Compute logdet
    logdet_val = logdet(Omega)

    # Define the pullback
    function logdet_custom_pullback(Δlogdet)
        print("yes")
        # Compute gradients based on chain rule
        Omega_inv = inv(Omega)
        dTransformed_dInner = A' * (C' * Omega_inv * C) * A
        dInner_dx = ℒ.I - 2 * C' * Y_inv * C * P
        ∂x = dInner_dx * dTransformed_dInner * Δlogdet
        
        return NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return logdet_val, logdet_custom_pullback
end






gba = Zygote.gradient(x -> logdet(C * (A * (x - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C'), P[1])[1]

gfo = ForwardDiff.gradient(x -> logdet_transform(x, C, A, 𝐁), P[1])

gba2 =  Zygote.gradient(x -> logdet_transform(x, C, A, 𝐁), P[1])[1]

FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> logdet_transform(x, C, A, 𝐁), P[1])[1]

gfi = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> logdet(C * (A * (x - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C'), P[1])[1]


isapprox(gfi, gba)
isapprox(gfi, gba2)
isapprox(gfi, gfo)






∂P = -C' * inv(C * P * C') * C * ∂llh/2

-∂P / ∂llh * 2
Y = C * P * C'
Y_inv = inv(C * P * C')
inner_matrix = P - P * C' * Y_inv * C * P
transformed_matrix = A * inner_matrix * A' + B
Omega = C * (A * (P - P * C' * inv(C * P * C') * C * P) * A' + B) * C'


Omega_inv = inv(Omega)
dTransformed_dInner = A' * (C' * Omega_inv * C) * A
# dInner_dx = ℒ.I - 2 * C' * Y_inv * C * P
dInner_dx = ℒ.I + 2 / ∂llh * 2 * ∂P * P
∂x = -∂llh / 2 * dInner_dx * dTransformed_dInner

dInner_dx = ℒ.I + 2 / ∂llh * 2 * ∂P * P
∂x = -(ℒ.I * ∂llh / 2 + 2 * ∂P * P) * dTransformed_dInner




Omega_inv = inv(C * (A * (P - P * C' * inv(C * P * C') * C * P) * A' + B) * C')
dTransformed_dInner = A' * (C' * Omega_inv * C) * A
dInner_dx = ℒ.I - 2 * C' * inv(C * P * C') * C * P
∂x = dInner_dx * dTransformed_dInner * Δlogdet



# now go for the second part


function vv_transform(p, C, v)
    v' * inv(C * (A * (p - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C') * v
end

vv_transform(P[1], C, v)

gba = Zygote.gradient(x -> logdet(C * (A * (x - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C'), P[1])[1]

gfo = ForwardDiff.gradient(x -> logdet_transform(x, C, A, 𝐁), P[1])

gba2 =  Zygote.gradient(x -> logdet_transform(x, C, A, 𝐁), P[1])[1]

FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> logdet_transform(x, C, A, 𝐁), P[1])[1]

gfi = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> logdet(C * (A * (x - P[1] * C' * inv(C * P[1] * C') * C * P[1]) * A' + 𝐁) * C'), P[1])[1]


isapprox(gfi, gba)
isapprox(gfi, gba2)
isapprox(gfi, gfo)



U = deepcopy(u[1])

# and now trying for th complete part

function calc_llh_complete(x, C, A, 𝐁, data_in_deviations)
    llh = 0.0
    u = zeros(size(A,2))
    P = x



    for data in eachcol(data_in_deviations)
        llh += logdet(C * P * C') + (data - C * u)' * inv(C * P * C') * (data - C * u)###

        u = A * (u + P * C' * inv(C * P * C') * (data - C * u))

        P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁
    end



    return llh
end

for data in eachcol(data_in_deviations)
    println(data)
end

calc_llh_complete(P[1], C, A, 𝐁, data_in_deviations)


function rrule(::typeof(calc_llh_complete), x, C, A, 𝐁, data_in_deviations)
    function likelihood_pullback(Δllh)
        llh = 0.0
        u = zeros(size(A, 2))
        P = x
        ∂P = zero(P)  # Initialize the gradient for P

        # Backward pass must also accumulate changes for gradient calculation
        for data in eachcol(data_in_deviations)
            v = data - C * u
            CPC = C * P * C'
            CPC_inv = inv(CPC)
            llh_contrib = logdet(CPC) + v' * CPC_inv * v
            llh += llh_contrib

            # Gradient calculations for each step
            G = P * C' * CPC_inv
            ∂llh_contrib = Δllh  # From upstream, affect on llh
            ∂P += ∂llh_contrib * G' * C' * CPC_inv * v  # Chain rule application
            ∂P += ∂llh_contrib * CPC_inv  # from logdet derivative

            u = A * (u + G * v)
            P = A * (P - G * C * P) * A' + 𝐁
        end

        return NoTangent(), ∂P, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    llh = 0.0  # Initialization as in the original function
    # Repeat the forward computation to get the final llh
    u = zeros(size(A, 2))
    P = x
    for data in eachcol(data_in_deviations)
        v = data - C * u
        CPC = C * P * C'
        llh += logdet(CPC) + v' * inv(CPC) * v
        u = A * (u + P * C' * inv(CPC) * v)
        P = A * (P - P * C' * inv(CPC) * C * P) * A' + 𝐁
    end
    return llh, likelihood_pullback
end

gradzyg = Zygote.gradient(x -> calc_llh_complete(x, C, A, 𝐁, data_in_deviations), P[1])[1]




using ChainRulesCore

# Define the function whose rrule we are setting up
function kalman_update(C, A, B, data_in_deviations, x)
    llh = 0.0
    u = zeros(size(A, 2))
    P = x
    stored_values = []

    for t in 1:size(data_in_deviations, 2)
        v = data_in_deviations[:, t] - C * u
        CPC = C * P * C'
        invCPC = inv(CPC)
        llh += logdet(CPC) + v' * invCPC * v

        # Store values needed for the backward pass
        push!(stored_values, (copy(u), copy(P), copy(v), copy(CPC), copy(invCPC)))

        u = A * (u + P * C' * invCPC * v)
        P = A * (P - P * C' * invCPC * C * P) * A' + B
    end

    return llh, stored_values
end

# Define the rrule
function rrule(::typeof(kalman_update), C, A, B, data_in_deviations, x)
    llh = 0.0
    u = zeros(size(A, 2))
    P = x

    # First iteration
    v1 = data_in_deviations[:, 1] - C * u
    CPC1 = C * P * C'
    invCPC1 = inv(CPC1)
    llh += logdet(CPC1) + v1' * invCPC1 * v1
    u = A * (u + x * C' * inv(C * x * C') * v1)
    P2 = A * (P - P * C' * invCPC1 * C * P) * A' + B

    # Second iteration
    v2 = data_in_deviations[:, 2] - C * u
    CPC2 = C * P2 * C'
    invCPC2 = inv(CPC2)
    llh += logdet(CPC2) + v2' * invCPC2 * v2

    function compute_llh_pullback(Δllh)
        # Initialize gradient accumulations
        ∂P = zero(x)
        # ∂u = zero(size(A, 2))

        # Backward for the second iteration
        # ∂v2 = 2 * invCPC2 * v2 * Δllh
        ∂CPC2 = invCPC2' * Δllh - invCPC2' * (v2 * v2') * invCPC2' * Δllh
        ∂P += C' * ∂CPC2 * C# + A' * (C' * ∂CPC2 * C) * A
        ∂P += A' * (∂P) * A
        # Backward for the first iteration
        # ∂v1 = 2 * invCPC1 * v1 * Δllh
        ∂CPC1 = invCPC1' * Δllh - invCPC1' * (v1 * v1') * invCPC1' * Δllh
        ∂P += C' * ∂CPC1 * C  # Since P was transformed by A
        # ∂P += A' * ∂P * A
        # ∂P +=   # Since P was transformed by A
        # println("yks")

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂P
    end

    return llh, compute_llh_pullback
end

# PP * A' 

kalman_update(C, A, 𝐁, data_in_deviations[:,1], P[1])
A * C'

gradfin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> kalman_update(C, A, 𝐁, data_in_deviations[:,1:2], x)[1], PP)[1]


gradzyg = Zygote.gradient(x -> kalman_update(C, A, 𝐁, data_in_deviations[:,1:2], x)[1], PP)[1]
gradfor = ForwardDiff.gradient(x -> kalman_update(C, A, 𝐁, data_in_deviations[:,1:2], x)[1], PP)

isapprox(gradfin, gradzyg)
isapprox(gradfin, gradfor)
isapprox(gradzyg, gradfor)

gradfin - gradfor

gradfin - gradzyg

v = C * u[1]
CPC = C * P[1] * C'
CPC_inv = inv(CPC)
# llh_contrib = logdet(CPC) + v' * CPC_inv * v
# llh += llh_contrib

# Gradient calculations for each step
G = P[1] * C' * CPC_inv
# ∂llh_contrib = Δllh  # From upstream, affect on llh
∂P += ∂llh_contrib * G' * C' * CPC_inv * v  # Chain rule application


PP = get_initial_covariance(Val(:theoretical), values, coordinates, dimensions)


gradfin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x->begin
llh = 0.0
u = zeros(size(A,2))
P = x

v = data_in_deviations[:, 1] - C * u

llh += logdet(C * P * C') + v' * inv(C * P * C') * v

u = A * (u + x * C' * inv(C * x * C') * v)

P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁

v = data_in_deviations[:, 2] - C * u

llh += logdet(C * P * C') + v' * inv(C * P * C') * v

return llh
end, PP)[1]

gradfin2 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> kalman_update(C, A, 𝐁, data_in_deviations[:,1:2], x)[1], PP)[1]


isapprox(gradfin, gradfin2)



gradfin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> calc_llh_complete(x, C, A, 𝐁, data_in_deviations), P[1])[1]





grad2 = Zygote.gradient(x->begin
llh = 0.0
u = zeros(size(A,2))
P = x

v = data_in_deviations[:, 1] - C * u

llh += logdet(C * P * C') + v' * inv(C * P * C') * v

u = A * (u + x * C' * inv(C * x * C') * v)

P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁

v = data_in_deviations[:, 2] - C * u

llh += logdet(C * P * C') + v' * inv(C * P * C') * v

return llh
end, P[1])[1]



grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> begin
            llh = 0.0
            u = zeros(size(A,2))
            P = x
            for i in 1:10#size(data_in_deviations,2)
                v = data_in_deviations[:, t] - C * u

                llh += logdet(C * P * C') + v' * inv(C * P * C') * v###

                u = A * (u + P * C' * inv(C * P * C') * v)

                P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁
        end
        return llh
    end, PP)[1]

isapprox(grad3, grad2)


grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
x -> begin
            llh = 0.0
            P = x
            for i in 1:10#size(data_in_deviations,2)
                llh += logdet(C * P * C') 

                P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁
        end
        return llh
    end, PP)[1]


grad2 = Zygote.gradient(x -> begin
llh = 0.0
P = x
for i in 1:10#size(data_in_deviations,2)
    llh += logdet(C * P * C') 

    P = A * (P - P * C' * inv(C * P * C') * C * P) * A' + 𝐁
end
return llh
end, PP)[1]#|>sparse

isapprox(grad3, grad2)




using ChainRulesCore

function compute_llh_recursive(C, A, B, x)
    llh = 0.0
    P = x
    # Ps = [P]  # To store each iteration of P for the backward pass

    for i in 1:10#size(data_in_deviations, 2)
        CPC = C * P * C'
        llh += logdet(CPC)

        invCPC = inv(CPC)
        P = A * (P - P * C' * invCPC * C * P) * A' + B
        # push!(Ps, P)  # Store updated P
    end
    return llh
end

compute_llh_recursive(C, A, 𝐁, P[1])

function rrule(::typeof(compute_llh_recursive), C, A, B, x)
    llh = 0.0
    P = x
    Ps = [P]  # To store each iteration of P for the backward pass

    for i in 1:10
        CPC = C * P * C'
        llh += logdet(CPC)

        invCPC = inv(CPC)
        P = A * (P - P * C' * invCPC * C * P) * A' + B
        push!(Ps, P)  # Store updated P
    end

    function compute_llh_recursive_pullback(Δllh)
        ∂P = zero(P)  # Initialize gradient for P as zero
        
        # Back-propagate through each iteration
        for i in reverse(1:10)
            P = Ps[i]  # Retrieve P from the corresponding forward iteration
            CPC = C * P * C'
            invCPC = inv(CPC)

            # Gradient of logdet contributes to P
            ∂CPC = invCPC' * Δllh  # Gradient of logdet wrt CPC
            ∂P_current = C' * ∂CPC * C  # Chain rule to get ∂P_current

            # Backprop through the update formula for P
            if i > 1  # For all but the first iteration where P isn't dependent on previous Ps
                P_prev = Ps[i]
                # ∂P_next = A' * (∂P - (P_prev * C' * invCPC * C * ∂P + ∂P * C' * invCPC * C * P_prev) + ∂P * A
                ∂P_next = A' * (∂P - ((P_prev * C' * invCPC * C * ∂P) + (∂P * C' * invCPC * C * P_prev))) * A
            
                ∂P = ∂P_next
            end

            # Add current iteration's contribution to ∂P
            ∂P += ∂P_current
        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂P  # Gradients for C, A, B are NO_FIELDS as they are constants
    end

    return llh, compute_llh_recursive_pullback
end



grad2 = Zygote.gradient(x -> compute_llh_recursive(C, A, 𝐁, x),PP)[1]

grad3 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), 
                                x -> compute_llh_recursive(C, A, 𝐁, x),PP)[1]



grad2 = Zygote.gradient(x -> v' * inv(C * x * C') * v, P[1])[1]#|>sparse


    grad1 = Zygote.gradient(x -> calc_llh2(x, C, v), P[1])[1]#|>sparse




    grad2 == grad1
    
    
    
    Zygote.gradient(logdet, F)[1]#|>sparse

    C' * inv(F) * C
C' * 10 * C

C' * inv(F) * C * Δlogdet

    ####### new but old order

    P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)
    observables = data_in_deviations

    T = size(observables, 2) + 1

    u = [zeros(size(C,2)) for _ in 1:T]

    u_mid = deepcopy(u)

    z = [zeros(size(observables, 1)) for _ in 1:T]

    P_mid = [deepcopy(P) for _ in 1:T]

    temp_N_N = similar(P)

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
        # Kalman iteration
        # this was moved down indicating a timing difference between the two approaches
        mul!(u_mid[t], A, u[t-1]) # u[t] = A u[t-1]
        mul!(z[t], C, u_mid[t]) # z[t] = C u[t]

        # P[t] = A * P[t - 1] * A' + B * B'
        mul!(temp_N_N, P[t-1], A')
        mul!(P_mid[t], A, temp_N_N)
        P_mid[t] .+= B_prod

        mul!(CP[t], C, P_mid[t]) # CP[t] = C * P[t]

        # V[t] = CP[t] * C' + R
        mul!(V[t], CP[t], C')
        # V[t].mat .+= R

        luV = ℒ.lu(V[t], check = false)
        Vdet = ℒ.det(luV)
        if Vdet < eps(Float64)
            return -Inf
        end
        invV[t] .= inv(luV)
        
        innovation[t] .= observables[:, t-1] - z[t]
        # loglik += logpdf(MvNormal(V[t]), innovation[t])  # no allocations since V[t] is a PDMat
        if t - 1 > presample_periods
            loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
        end

        # K[t] .= CP[t]' / V[t]  # Kalman gain
        mul!(K[t], P_mid[t] * C', invV[t])

        #u[t] += K[t] * innovation[t]
        copy!(u[t], u_mid[t])
        mul!(u[t], K[t], innovation[t], 1, 1)

        #P[t] -= K[t] * CP[t]
        copy!(P[t], P_mid[t])
        mul!(P[t], K[t], CP[t], -1, 1)
    end

    -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 





    # reverse pass new but old order
    Δlogpdf = 1.0
    temp_L_N = similar(C)
    temp_N_L = similar(C')
    temp_L_L = similar(V[1])
    temp_M = similar(z[1])

    # Buffers
    ΔP = zero(P[1])
    Δu = zero(u[1])
    ΔA = zero(A)
    ΔB = zero(B)
    ΔC = zero(C)
    ΔK = zero(K[1])
    ΔP_mid = zero(ΔP)
    ΔP_mid_sum = zero(ΔP)
    ΔCP = zero(CP[1])
    Δu_mid = zero(u_mid[1])
    Δz = zero(z[1])
    ΔV = zero(V[1])

    for t in T:-1:2+presample_periods
        # Sensitivity accumulation
        copy!(ΔP_mid, ΔP)
        mul!(ΔK, ΔP, CP[t]', -1, 0) # i.e. ΔK = -ΔP * CP[t]'
        mul!(ΔCP, K[t]', ΔP, -1, 0) # i.e. ΔCP = - K[t]' * ΔP
        copy!(Δu_mid, Δu)
        mul!(ΔK, Δu, innovation[t]', 1, 1) # ΔK += Δu * innovation[t]'
        mul!(Δz, K[t]', Δu, -1, 0)  # i.e, Δz = -K[t]'* Δu
        mul!(ΔCP, invV[t], ΔK', 1, 1) # ΔCP += invV[t] * ΔK'

        # ΔV .= -invV[t] * CP[t] * ΔK * invV[t]
        mul!(temp_L_N, invV[t], (P_mid[t] * C')')
        mul!(temp_N_L, ΔK, invV[t])
        mul!(ΔV, temp_L_N, temp_N_L, -1, 0)

        mul!(ΔC, ΔCP, P_mid[t]', 1, 1) # ΔC += ΔCP * P_mid[t]'
        mul!(ΔP_mid, C', ΔCP, 1, 1) # ΔP_mid += C' * ΔCP
        mul!(Δz, invV[t], innovation[t], Δlogpdf, 1) # Δz += Δlogpdf * invV[t] * innovation[t] # Σ^-1 * (z_obs - z)

        #ΔV -= Δlogpdf * 0.5 * (invV[t] - invV[t] * innovation[t] * innovation[t]' * invV[t]) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
        mul!(temp_M, invV[t], innovation[t])
        mul!(temp_L_L, temp_M, temp_M')
        temp_L_L .-= invV[t]
        rmul!(temp_L_L, Δlogpdf * 0.5)
        ΔV += temp_L_L

        #ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
        mul!(temp_L_N, C, P_mid[t])
        transpose!(temp_L_L, ΔV)
        temp_L_L .+= ΔV
        mul!(ΔC, temp_L_L, temp_L_N, 1, 1)

        # ΔP_mid += C' * ΔV * C
        mul!(temp_L_N, ΔV, C)
        mul!(ΔP_mid, C', temp_L_N, 1, 1)

        mul!(ΔC, Δz, u_mid[t]', 1, 1) # ΔC += Δz * u_mid[t]'
        mul!(Δu_mid, C', Δz, 1, 1) # Δu_mid += C' * Δz

        # # Calculates (ΔP_mid + ΔP_mid')
        # transpose!(ΔP_mid_sum, ΔP_mid)
        # ΔP_mid_sum .+= ΔP_mid

        # ΔA += (ΔP_mid + ΔP_mid') * A * P[t - 1]
        mul!(temp_N_N, A, P[t - 1])
        # mul!(ΔA, ΔP_mid_sum, temp_N_N, 1, 1)
        mul!(ΔA, ΔP_mid, temp_N_N, 1, 1)

        # ΔP .= A' * ΔP_mid * A # pass into next period
        mul!(temp_N_N, ΔP_mid, A)
        mul!(ΔP, A', temp_N_N)

        # mul!(ΔB, ΔP_mid_sum, B, 1, 1) # ΔB += ΔP_mid_sum * B
        mul!(ΔB, ΔP_mid, B, 1, 1) # ΔB += ΔP_mid_sum * B
        mul!(ΔA, Δu_mid, u[t - 1]', 1, 1) # ΔA += Δu_mid * u[t - 1]'
        mul!(Δu, A', Δu_mid)
    end




    # reverse pass new but old order
    Δlogpdf = 1.0
    temp_L_N = similar(C)
    temp_N_L = similar(C')
    temp_L_L = similar(V[1])
    temp_M = similar(z[1])

    # Buffers
    ΔP = zero(P[1])
    Δu = zero(u[1])
    ΔA = zero(A)
    ΔB = zero(P[1])
    ΔC = zero(C)
    ΔK = zero(K[1])
    ΔP_mid = zero(ΔP)
    ΔP_mid_sum = zero(ΔP)
    ΔCP = zero(CP[1])
    ΔPC = zero(CP[1])
    Δu_mid = zero(u_mid[1])
    Δz = zero(z[1])
    ΔV = zero(V[1])
    

    for t in T:-1:2
        # pullback
        # Sensitivity accumulation
        # P[t] -= K[t] * CP[t]
        copy!(ΔP_mid, ΔP)
        ΔK .= -ΔP * CP[t]'
        ΔCP .= - K[t]' * ΔP

        # u[t] += K[t] * innovation[t]
        copy!(Δu_mid, Δu)
        ΔK += Δu * innovation[t]'
        Δinnovation = K[t]'* Δu

        # K[t] .= CP[t]' / V[t]
        ΔPC .= invV[t] * ΔK'
        ΔV .= -invV[t] * (P_mid[t] * C')' * ΔK * invV[t]
        
        # PC = P_mid[t] * C'
        ΔP_mid += ΔPC' * C
        ΔC .= (P_mid[t] * ΔPC')'

        # loglik += log(Vdet) + innovation[t]' * invV[t] * innovation[t]
        Δinnovation += 2 * Δlogpdf * invV[t] * innovation[t] # Σ^-1 * (z_obs - z)
        ΔV -= Δlogpdf * (invV[t] - invV[t] * innovation[t] * innovation[t]' * invV[t])

        # innovation[t] .= observables[:, t-1] - z[t-1]
        Δobservables = Δinnovation
        Δz = -Δinnovation

        # V[t] = CP[t] * C' + R
        ΔC .= ΔV * C * P_mid[t]'# + ΔV' * C * P_mid[t]
        ΔP_mid += C' * ΔV * C

        # CP[t] = C * P[t]
        ΔC += ΔCP * P_mid[t]'
        ΔP_mid += C' * ΔCP

        # P[t] = A * P[t - 1] * A' + B
        ΔA .= ΔP_mid * A * P[t - 1]
        ΔP .= A' * ΔP_mid * A # pass into next period
        ΔB .= ΔP_mid

        # z[t] = C * u[t]
        ΔC += Δz * u_mid[t]'
        Δu_mid += C' * Δz

        # u[t] = A * u[t-1]
        ΔA += Δu_mid * u[t - 1]'
        Δu = A' * Δu_mid
    end
    t -= 1

    ΔA






        # If it was a failure, just return and hope the gradients are ignored!
            for t in T:-1:2
                # # Calculates (ΔP_mid + ΔP_mid')
                # transpose!(ΔP_mid_sum, ΔP_mid)
                # ΔP_mid_sum .+= ΔP_mid

                # ΔA += (ΔP_mid + ΔP_mid') * A * P[t - 1]
                mul!(temp_N_N, A, P[t])
                mul!(ΔA, ΔP_mid, temp_N_N, 1, 1)

                # ΔP .= A' * ΔP_mid * A # pass into next period
                mul!(temp_N_N, ΔP_mid, A)
                mul!(ΔP, A', temp_N_N)

                mul!(ΔB, ΔP_mid, B, 1, 1) # ΔB += ΔP_mid_sum * B
                mul!(ΔA, Δu_mid, u[t]', 1, 1) # ΔA += Δu_mid * u[t - 1]'
                mul!(Δu, A', Δu_mid)

                # Sensitivity accumulation
                copy!(ΔP_mid, ΔP)
                mul!(ΔK, ΔP, CP[t]', -1, 0) # i.e. ΔK = -ΔP * CP[t]'
                mul!(ΔCP, K[t]', ΔP, -1, 0) # i.e. ΔCP = - K[t]' * ΔP
                copy!(Δu_mid, Δu)
                mul!(ΔK, Δu, innovation[t]', 1, 1) # ΔK += Δu * innovation[t]'
                mul!(Δz, K[t]', Δu, -1, 0)  # i.e, Δz = -K[t]'* Δu
                mul!(ΔCP, invV[t], ΔK', 1, 1) # ΔCP += invV[t] * ΔK'

                # ΔV .= -invV[t] * CP[t] * ΔK * invV[t]
                mul!(temp_L_N, invV[t], CP[t])
                mul!(temp_N_L, ΔK, invV[t])
                mul!(ΔV, temp_L_N, temp_N_L, -1, 0)

                # mul!(ΔC, ΔCP, P_mid[t]', 1, 1) # ΔC += ΔCP * P_mid[t]'
                mul!(ΔP_mid, C', ΔCP, 1, 1) # ΔP_mid += C' * ΔCP
                mul!(Δz, invV[t], innovation[t], Δlogpdf, 1) # Δz += Δlogpdf * invV[t] * innovation[t] # Σ^-1 * (z_obs - z)

                #ΔV -= Δlogpdf * 0.5 * (invV[t] - invV[t] * innovation[t] * innovation[t]' * invV[t]) # -0.5 * (Σ^-1 - Σ^-1(z_obs - z)(z_obx - z)'Σ^-1)
                mul!(temp_M, invV[t], innovation[t])
                mul!(temp_L_L, temp_M, temp_M')
                temp_L_L .-= invV[t]
                rmul!(temp_L_L, Δlogpdf * 0.5)
                ΔV += temp_L_L

                #ΔC += ΔV * C * P_mid[t]' + ΔV' * C * P_mid[t]
                mul!(temp_L_N, C, P_mid[t])
                transpose!(temp_L_L, ΔV)
                temp_L_L .+= ΔV
                # mul!(ΔC, temp_L_L, temp_L_N, 1, 1)

                # ΔP_mid += C' * ΔV * C
                mul!(temp_L_N, ΔV, C)
                mul!(ΔP_mid, C', temp_L_N, 1, 1)

                # mul!(ΔC, Δz, u_mid[t]', 1, 1) # ΔC += Δz * u_mid[t]'
                mul!(Δu_mid, C', Δz, 1, 1) # Δu_mid += C' * Δz
            end









P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

u = zeros(size(C,2))

loglik = 0.0

# single update function
function kalman_iteration(u, P, loglik, A, 𝐁, C, data_in_deviations, presample_periods, t)
    v = data_in_deviations[:, t] - C * u

    F = C * P * C'

    luF = ℒ.lu(F, check = false) ###

    if !ℒ.issuccess(luF)
        return -Inf
    end

    Fdet = ℒ.det(luF)

    if Fdet < eps(Float64)
        return -Inf
    end

    invF = inv(luF) ###
    

    if t > presample_periods
        llh = loglik +logdet(F) + v' * invF * v ###
    else
        llh = loglik
    end

    û = A * (u + P * C' * invF * v)

    P̂ = A * (P - P * C' * invF * C * P) * A' + 𝐁

    return û, P̂, llh
end

for t in 1:size(data_in_deviations, 2)
    u,P,loglik = kalman_update(u,P,loglik,A,𝐁,C,data_in_deviations,presample_periods,t)
end

-(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 






P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

u = zeros(size(C,2))

loglik = 0.0

v = data_in_deviations[:, t] - C * u
        
F = C * P * C'

luF = ℒ.lu(F, check = false) ###

if !ℒ.issuccess(luF)
    return -Inf
end

Fdet = ℒ.det(luF)

if Fdet < eps(Float64)
    return -Inf
end

invF = inv(luF) ###


if t > presample_periods
    llh = loglik +logdet(F) + v' * invF * v ###
else
    llh = loglik
end

û = A * (u + P * C' * invF * v)

P̂ = A * (P - P * C' * invF * C * P) * A' + 𝐁




function rrule(::typeof(kalman_iteration), u, P, loglik, A, 𝐁, C, data_in_deviations, presample_periods, t)
    # Perform the forward pass
    v = data_in_deviations[:, t] - C * u

    F = C * P * C'

    luF = ℒ.lu(F, check = false) ###

    if !ℒ.issuccess(luF)
        return (u, P, -Inf), (u, P, loglik, A, 𝐁, C, data_in_deviations, presample_periods, t) -> (u, P, loglik)
    end

    Fdet = ℒ.det(luF)

    if Fdet < eps(Float64)
        return (u, P, -Inf), (u, P, loglik, A, 𝐁, C, data_in_deviations, presample_periods, t) -> (u, P, loglik)
    end

    invF = inv(luF) ###
    
    if t > presample_periods
        llh = loglik +logdet(F) + v' * invF * v ###
    else
        llh = loglik
    end

    û = A * (u + P * C' * invF * v)

    P̂ = A * (P - P * C' * invF * C * P) * A' + 𝐁

    # pullback of single update function
    function kalman_pullback(∂û, ∂P̂, ∂llh)
        # Calculate gradients for each input
        ∂loglik = ∂llh
    
        # Gradient w.r.t. P from `P̂`
        ∂P = -A' * ∂P̂ * A  # Simplified reverse derivative, expand with chain rule for full gradient
    
        # Additional contributions to ∂P from llh
        if t > presample_periods
            ∂S_llh = C' * (∂llh * invF - invF * (v * v') * invF) * C
            ∂P += ∂S_llh
        end
        
        # Gradient w.r.t. u
        ∂u = -C' * invF * v  # Derivative contribution from v in the update step
        ∂u += A' * ∂û
        
        # Gradient w.r.t. A
        ∂A = ∂û * (u + P * C' * invF * v)' + ∂P̂ * (P - P * C' * invF * C * P)'
        
        # Gradient w.r.t. B
        ∂B = ∂P̂
    
        return NoTangent(), ∂u, ∂P, ∂loglik, ∂A, ∂B, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    return (û, P̂, llh), kalman_pullback
end


# test
using ForwardDiff, Zygote, FiniteDifferences


fordif = ForwardDiff.gradient(x->begin
                P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

                u = zeros(size(C,2))

                loglik = 0.0

                for t in 1:size(data_in_deviations, 2)
                    u,P,loglik = kalman_update(u,P,loglik,x,𝐁,C,data_in_deviations,presample_periods,t)
                end

                -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
                end, A)


findif = FiniteDifferences.grad(central_fdm(3,1), x->begin
                P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

                u = zeros(size(C,2))

                loglik = 0.0

                for t in 1:size(data_in_deviations, 2)
                    u,P,loglik = kalman_update(u,P,loglik,x,𝐁,C,data_in_deviations,presample_periods,t)
                end

                -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
                end, A)[1]


bacdif = Zygote.gradient(x->begin
                P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

                u = zeros(size(C,2))

                loglik = 0.0

                for t in 1:size(data_in_deviations, 2)
                    u,P,loglik = kalman_update(u,P,loglik,x,𝐁,C,data_in_deviations,presample_periods,t)
                end

                -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
                end, A)[1]
