
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




import LinearAlgebra: mul!, transpose!, rmul!, logdet
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import MacroModelling: get_and_check_observables, solve!, check_bounds, get_relevant_steady_state_and_state_update, calculate_loglikelihood, get_initial_covariance
parameter_values = parameters_combined
algorithm = :first_order
filter = :kalman
warmup_iterations = 0
presample_periods = 4
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
