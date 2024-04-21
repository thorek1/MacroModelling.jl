
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

zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 


# reverse pass
zz = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 
∂z∂z = 1

# z = -(wⁿ⁻¹ + wⁿ⁻²) / 2
∂z∂wⁿ⁻¹ = -∂z∂z/ 2
∂z∂wⁿ⁻² = -∂z∂z/ 2

# wⁿ⁻¹ = loglik = wⁿ⁻³₁ + wⁿ⁻³₂ = for t in 2:4 logdet(V[t]) + innovation[t]' * invV[t] * innovation[t] end
∂wⁿ⁻¹∂wⁿ⁻³₁ = ∂z∂wⁿ⁻¹
∂wⁿ⁻¹∂wⁿ⁻³₂ = ∂z∂wⁿ⁻¹
∂wⁿ⁻¹∂wⁿ⁻³₃ = ∂z∂wⁿ⁻¹

# branch wⁿ⁻³₁
# wⁿ⁻³₁ = wⁿ⁻⁴₁ + wⁿ⁻⁵₁ = logdet(V[2]) + innovation[2]' * invV[2] * innovation[2]
∂wⁿ⁻³₁∂wⁿ⁻⁴₁ = ∂wⁿ⁻¹∂wⁿ⁻³₁
∂wⁿ⁻³₁∂wⁿ⁻⁵₁ = ∂wⁿ⁻¹∂wⁿ⁻³₁

# branch wⁿ⁻⁴₁
wⁿ⁻⁶₁ = C * P[1] * C'#V[2]
wⁿ⁻⁴₁ = logdet(wⁿ⁻⁶₁)
∂wⁿ⁻⁴₁∂wⁿ⁻⁶₁ = ∂wⁿ⁻³₁∂wⁿ⁻⁴₁ * inv(wⁿ⁻⁶₁)'

# wⁿ⁻⁶₁ = V[2] = wⁿ⁻⁷₁ * C' = CP[2] * C'
# wⁿ⁻⁷₁ = CP[2] = C * P_mid[1]
∂wⁿ⁻⁶₁∂wⁿ⁻⁷₁ = ∂wⁿ⁻⁴₁∂wⁿ⁻⁶₁ * C

∂wⁿ⁻⁷₁∂P = C' * ∂wⁿ⁻⁶₁∂wⁿ⁻⁷₁


# ∂z∂P_mid = ∂z∂z * ∂z∂wⁿ⁻¹ * ∂wⁿ⁻¹∂wⁿ⁻³₁ * ∂wⁿ⁻³₁∂wⁿ⁻⁴₁ * ∂wⁿ⁻⁴₁∂wⁿ⁻⁶₁ *  ∂wⁿ⁻⁶₁∂wⁿ⁻⁷₁ * ∂wⁿ⁻⁷₁∂P_mid


# branch wⁿ⁻³₂
# wⁿ⁻³₂ = wⁿ⁻⁴₂ + wⁿ⁻⁵₂ = logdet(V[3]) + innovation[3]' * invV[3] * innovation[3]
∂wⁿ⁻³₂∂wⁿ⁻⁴₂ = ∂wⁿ⁻¹∂wⁿ⁻³₂
∂wⁿ⁻³₂∂wⁿ⁻⁵₂ = ∂wⁿ⁻¹∂wⁿ⁻³₂

# branch wⁿ⁻⁵₂
# wⁿ⁻⁵₂ = wⁿ⁻⁵₂¹ * wⁿ⁻⁵₂² = (innovation[3]' * invV[3]) * innovation[3]
∂wⁿ⁻⁵₂∂wⁿ⁻⁵₂¹ = ∂wⁿ⁻³₂∂wⁿ⁻⁵₂ * innovation[3]'
∂wⁿ⁻⁵₂∂wⁿ⁻⁵₂² = (innovation[3]' * invV[3])' * ∂wⁿ⁻³₂∂wⁿ⁻⁵₂ # ∂innovation

# wⁿ⁻⁵₂¹ = wⁿ⁻⁵₂³ * wⁿ⁻⁵₂⁴ = innovation[3]' * invV[3]
∂wⁿ⁻⁵₂¹∂wⁿ⁻⁵₂⁴ = innovation[3] * ∂wⁿ⁻⁵₂∂wⁿ⁻⁵₂¹
∂wⁿ⁻⁵₂¹∂wⁿ⁻⁵₂³ = (∂wⁿ⁻⁵₂∂wⁿ⁻⁵₂¹ * invV[3]')' # ∂innovation

∂wⁿ⁻⁵₂∂innovation = ∂wⁿ⁻⁵₂∂wⁿ⁻⁵₂² + ∂wⁿ⁻⁵₂¹∂wⁿ⁻⁵₂³

A' * C' * -(invV[3]' * innovation[3] * ∂wⁿ⁻³₂∂wⁿ⁻⁵₂ + invV[3] * innovation[3] *  ∂wⁿ⁻³₂∂wⁿ⁻⁵₂')
(invV[3]' + invV[3]) * innovation[3]
# innovation[t] .= observables[:, t-1] - z[t-1]
# z[t] .= C * u_mid[t]
# u_mid[t] .= A * u[t]
# u[t] .= K[t] * innovation[t] + u_mid[t-1]
# K[t] .= P_mid[t-1] * C' * invV[t]
∂innovation∂z = -∂wⁿ⁻⁵₂∂innovation
∂z∂u_mid = C' * ∂innovation∂z
∂u_mid∂u = A' * ∂z∂u_mid
# ∂u_mid∂A = ∂z∂u_mid * u[t]'
∂u∂innovation = K[3]' * ∂u_mid∂u
∂u∂u_mid = ∂u_mid∂u
∂u∂K = ∂u_mid∂u * innovation[3]'
∂u∂K * C
# wⁿ⁻⁵₂⁴ = inv(V[3]) = inv(wⁿ⁻⁵₂⁴)
∂wⁿ⁻⁵₂⁴∂wⁿ⁻⁵₂⁴ = -invV[3]' * ∂wⁿ⁻⁵₂¹∂wⁿ⁻⁵₂⁴ * invV[3]'


# branch wⁿ⁻⁴₂
# wⁿ⁻⁴₂ = logdet(wⁿ⁻⁶₂)
wⁿ⁻⁶₂ = C * P_mid[2] * C'#V[3]
∂wⁿ⁻⁴₂∂wⁿ⁻⁶₂ = ∂wⁿ⁻³₂∂wⁿ⁻⁴₂ * inv(wⁿ⁻⁶₂)'

# wⁿ⁻⁶₂ = V[3] = wⁿ⁻⁷₂ * C' = CP[3] * C'
# wⁿ⁻⁷₂ = CP[3] = C * P_mid[2] = C * wⁿ⁻⁸₂
∂wⁿ⁻⁶₂∂wⁿ⁻⁷₂ = ∂wⁿ⁻⁴₂∂wⁿ⁻⁶₂ * C

∂wⁿ⁻⁷₂∂wⁿ⁻⁸₂ = C' * ∂wⁿ⁻⁶₂∂wⁿ⁻⁷₂

# wⁿ⁻⁸₂ = P_mid[2] = wⁿ⁻⁹₂ + B_prod = A * P[2] * A' + B_prod

∂wⁿ⁻⁸₂∂wⁿ⁻⁹₂ = ∂wⁿ⁻⁷₂∂wⁿ⁻⁸₂

# wⁿ⁻⁹₂ = A * P[2] * A' = AP[2] * A' = wⁿ⁻¹⁰₂ * A'
wⁿ⁻¹⁰₂ = A * P[2]
∂wⁿ⁻⁹₂∂A = (wⁿ⁻¹⁰₂' * ∂wⁿ⁻⁸₂∂wⁿ⁻⁹₂)'

∂wⁿ⁻⁹₂∂wⁿ⁻¹⁰₂ = ∂wⁿ⁻⁸₂∂wⁿ⁻⁹₂ * A
∂wⁿ⁻¹⁰₂∂A = ∂wⁿ⁻⁹₂∂wⁿ⁻¹⁰₂ * P[2]'

∂z∂A = ∂wⁿ⁻¹⁰₂∂A + ∂wⁿ⁻⁹₂∂A

# ∂z∂A = ∂wⁿ⁻⁷₂∂wⁿ⁻⁸₂ * ∂z∂z * ∂z∂wⁿ⁻¹ * ∂wⁿ⁻¹∂wⁿ⁻³₁ * ∂wⁿ⁻³₂∂wⁿ⁻⁴₂ * ∂wⁿ⁻⁴₂∂wⁿ⁻⁶₂ * ∂wⁿ⁻⁶₂∂wⁿ⁻⁷₂  * ∂wⁿ⁻⁸₂∂wⁿ⁻⁹₂ * (∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₂∂wⁿ⁻¹⁰₂ * ∂wⁿ⁻¹⁰₂∂A)
∂z∂A = -1/2 * C' * inv(C * P_mid[2] * C')' * C * (A * P[2] + A * P[2]')


zyggrad = Zygote.gradient(x -> -1/2*logdet(C * (x * (P[2] - P[2] * C' * invV[3] * C * P[2]) * x' + 𝐁) * C'), A)[1]

isapprox(∂z∂A, zyggrad)

# continue with wⁿ⁻¹⁰₂ derivative wrt P[2]
∂wⁿ⁻⁹₂∂wⁿ⁻¹⁰₂ = ∂wⁿ⁻⁸₂∂wⁿ⁻⁹₂ * A
# AP[2] = A * P[2] = A * wⁿ⁻¹¹₂
∂wⁿ⁻¹⁰₂∂wⁿ⁻¹¹₂ = A' * ∂wⁿ⁻⁹₂∂wⁿ⁻¹⁰₂

# wⁿ⁻¹¹₂ = P[2] =  P_mid[1] - K[2] * CP[2] = wⁿ⁻¹²₂ - wⁿ⁻¹³₂
∂wⁿ⁻¹¹₂∂P = ∂wⁿ⁻¹⁰₂∂wⁿ⁻¹¹₂
∂wⁿ⁻¹¹₂∂wⁿ⁻¹³₂ = -∂wⁿ⁻¹⁰₂∂wⁿ⁻¹¹₂


# wⁿ⁻¹³₂ = K[2] * CP[2] = wⁿ⁻¹⁴₂ * wⁿ⁻¹⁵₂
∂wⁿ⁻¹³₂∂wⁿ⁻¹⁴₂ = ∂wⁿ⁻¹¹₂∂wⁿ⁻¹³₂ * CP[2]'
∂wⁿ⁻¹³₂∂wⁿ⁻¹⁵₂ = K[2]' * ∂wⁿ⁻¹¹₂∂wⁿ⁻¹³₂


# wⁿ⁻¹⁴₂ = K[2] = PC[1] * invV[2] = P_mid[1] * C' * invV[2] = wⁿ⁻¹⁶₂ * wⁿ⁻¹⁷₂
∂wⁿ⁻¹⁴₂∂wⁿ⁻¹⁶₂ = ∂wⁿ⁻¹³₂∂wⁿ⁻¹⁴₂ * invV[2]'
∂wⁿ⁻¹⁴₂∂wⁿ⁻¹⁷₂ = (P_mid[1] * C')' * ∂wⁿ⁻¹³₂∂wⁿ⁻¹⁴₂

wⁿ⁻¹⁶₂ = P_mid[1] * C'
∂wⁿ⁻¹⁶₂∂P = ∂wⁿ⁻¹⁴₂∂wⁿ⁻¹⁶₂ * C

# wⁿ⁻¹⁷₂ = inv(V[2]) = inv(wⁿ⁻¹⁸₂)
∂wⁿ⁻¹⁷₂∂wⁿ⁻¹⁸₂ = -invV[2]' * ∂wⁿ⁻¹⁴₂∂wⁿ⁻¹⁷₂ * invV[2]'

# wⁿ⁻¹⁸₂ = V[2] = CP[2] * C' = wⁿ⁻¹⁹₂ * C' = wⁿ⁻⁶₁
# wⁿ⁻¹⁹₂ = CP[2] = C * P_mid[1]
∂wⁿ⁻¹⁸₂∂wⁿ⁻¹⁹₂ = ∂wⁿ⁻¹⁷₂∂wⁿ⁻¹⁸₂ * C
∂wⁿ⁻¹⁹₂∂P = C' * ∂wⁿ⁻¹⁸₂∂wⁿ⁻¹⁹₂


# wⁿ⁻¹⁹₂ = wⁿ⁻¹⁵₂
∂wⁿ⁻¹⁵₂∂P = C' * ∂wⁿ⁻¹³₂∂wⁿ⁻¹⁵₂


∂z∂P = ∂wⁿ⁻¹⁵₂∂P + ∂wⁿ⁻¹⁹₂∂P + ∂wⁿ⁻¹⁶₂∂P + ∂wⁿ⁻¹¹₂∂P + ∂wⁿ⁻⁷₁∂P

isapprox(∂wⁿ⁻¹⁵₂∂P, C' * K[2]' * -A' * C' * -∂z∂z / 2 * invV[3]' * C * A)

isapprox(∂wⁿ⁻¹⁹₂∂P, C' * -invV[2]' * (P_mid[1] * C')' * -A' * C' * -∂z∂z / 2 * invV[3]' * C * A * CP[2]' * invV[2]' * C)
# isapprox(∂wⁿ⁻¹⁹₂∂P, C' * -K[2]' * -A' * C' * -∂z∂z / 2 * invV[3]' * C * A * K[2] * C)

isapprox(∂wⁿ⁻¹⁶₂∂P, -A' * C' * -∂z∂z / 2 * invV[3]' * C * A * CP[2]' * invV[2]' * C)
# isapprox(∂wⁿ⁻¹⁶₂∂P, -A' * C' * -∂z∂z / 2 * invV[3]' * C * A * K[2] * C)

isapprox(∂wⁿ⁻¹¹₂∂P, A' * C' * -∂z∂z / 2 * invV[3]' * C * A)

isapprox(∂wⁿ⁻⁷₁∂P, C' * -∂z∂z/ 2 * invV[2]' * C)




core = C' * -∂z∂z / 2 * invV[3]' * C
isapprox(∂wⁿ⁻¹⁵₂∂P, C' * K[2]' * -A' * core * A)

isapprox(∂wⁿ⁻¹⁹₂∂P, C' * -invV[2]' * (P_mid[1] * C')' * -A' * core * A * CP[2]' * invV[2]' * C)
# isapprox(∂wⁿ⁻¹⁹₂∂P, C' * -K[2]' * -A' * core * A * K[2] * C)

isapprox(∂wⁿ⁻¹⁶₂∂P, -A' * core * A * CP[2]' * invV[2]' * C)
# isapprox(∂wⁿ⁻¹⁶₂∂P, -A' * core * A * K[2] * C)

isapprox(∂wⁿ⁻¹¹₂∂P, A' * core * A)


core = C' * -∂z∂z / 2 * invV[3]' * C
AcoreA = A' * core * A
AcoreA * (ℒ.I - CP[2]' * invV[2]' * C) + C' * invV[2]' * (P_mid[1] * C')' * AcoreA * CP[2]' * invV[2]' * C - C' * K[2]' * AcoreA


zyggrad = Zygote.gradient(x -> -1/2*logdet(C * x * C'), PP)[1]

isapprox(∂wⁿ⁻⁷₁∂P, zyggrad)

∂wⁿ⁻¹¹₂∂P

zyggrad = Zygote.gradient(x -> -1/2*(logdet(C * (A * (x - PP * C' * inv(C * PP * C') * C * PP) * A' + 𝐁) * C')), PP)[1]
isapprox(∂wⁿ⁻¹¹₂∂P, zyggrad)


zyggrad = Zygote.gradient(x -> -1/2*(logdet(C * (A * (x) * A' + 𝐁) * C')), PP)[1]



zyggrad = Zygote.gradient(x -> -1/2*(logdet(C * (A * (x - x * C' * inv(C * x * C') * C * x) * A' + 𝐁) * C') + logdet(C * x * C')), PP)[1]
forgrad = ForwardDiff.gradient(x -> -1/2*(logdet(C * (A * (x - x * C' * inv(C * x * C') * C * x) * A' + 𝐁) * C') + logdet(C * x * C')), PP)

isapprox(zyggrad, ∂z∂P)
isapprox(zyggrad, forgrad)


# fingrad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4,1),
# x -> begin
# P_mid[1] = deepcopy(x)
# P[1] = deepcopy(x)
# loglik = 0.0
# for t in 2:3
#     CP[t] .= C * P_mid[t-1]

#     V[t] .= CP[t] * C'

#     luV = ℒ.lu(V[t], check = false)

#     Vdet = ℒ.det(luV)
    
#     invV[t] .= inv(luV)
    
#     innovation[t] .= observables[:, t-1] - z[t-1]
    
#     loglik += log(Vdet)# + innovation[t]' * invV[t] * innovation[t]

#     K[t] .= P_mid[t-1] * C' * invV[t]

#     u[t] .= K[t] * innovation[t] + u_mid[t-1]
    
#     P[t] .= P_mid[t-1] - K[t] * CP[t]

#     u_mid[t] .= A * u[t]

#     z[t] .= C * u_mid[t]

#     P_mid[t] .= A * P[t] * A' + B_prod
# end
# return -1/2*loglik
# end, PP)[1]

# zyggrad - fingrad


# core = C' * -∂z∂z / 2 * invV[4]' * C
# AcoreA = A' * core * A
# AcoreA = A' * AcoreA * A
# AcoreA * (ℒ.I - CP[2]' * invV[2]' * C) + C' * invV[2]' * (P_mid[1] * C')' * AcoreA * CP[2]' * invV[2]' * C - C' * K[2]' * AcoreA



# isapprox(∂z∂P, fingrad)
# isapprox(zyggrad, fingrad)
# maximum(abs, zyggrad - fingrad)



# continue with t = 4
# branch wⁿ⁻³₃
# wⁿ⁻³₃ = wⁿ⁻⁴₃ + wⁿ⁻⁵₃ = logdet(V[4]) + innovation[4]' * invV[4] * innovation[4]
∂wⁿ⁻³₃∂wⁿ⁻⁴₃ = ∂wⁿ⁻¹∂wⁿ⁻³₃
∂wⁿ⁻³₃∂wⁿ⁻⁵₃ = ∂wⁿ⁻¹∂wⁿ⁻³₃

# branch wⁿ⁻⁴₃
# wⁿ⁻⁴₃ = logdet(wⁿ⁻⁶₃)
wⁿ⁻⁶₃ = C * P_mid[3] * C'#V[4]
∂wⁿ⁻⁴₃∂wⁿ⁻⁶₃ = ∂wⁿ⁻³₃∂wⁿ⁻⁴₃ * inv(wⁿ⁻⁶₃)'

# wⁿ⁻⁶₃ = V[4] = wⁿ⁻⁷₃ * C' = CP[4] * C'
# wⁿ⁻⁷₃ = CP[4] = C * P_mid[3] = C * wⁿ⁻⁸₃
∂wⁿ⁻⁶₃∂wⁿ⁻⁷₃ = ∂wⁿ⁻⁴₃∂wⁿ⁻⁶₃ * C

∂wⁿ⁻⁷₃∂wⁿ⁻⁸₃ = C' * ∂wⁿ⁻⁶₃∂wⁿ⁻⁷₃

# wⁿ⁻⁸₃ = P_mid[3] = wⁿ⁻⁹₃ + B_prod = A * P[3] * A' + B_prod

∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃ = ∂wⁿ⁻⁷₃∂wⁿ⁻⁸₃

# wⁿ⁻⁹₃ = A * P[3] * A' = AP[3] * A' = wⁿ⁻¹⁰₃ * A'
wⁿ⁻¹⁰₃ = A * P[3]
∂wⁿ⁻⁹₃∂A = (wⁿ⁻¹⁰₃' * ∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃)'

∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃ = ∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃ * A

∂wⁿ⁻¹⁰₃∂A = ∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃ * P[3]'

∂z∂A = ∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A

# ∂z∂A = ∂wⁿ⁻⁷₃∂wⁿ⁻⁸₃ * ∂z∂z * ∂z∂wⁿ⁻¹ * ∂wⁿ⁻¹∂wⁿ⁻³₁ * ∂wⁿ⁻³₃∂wⁿ⁻⁴₃ * ∂wⁿ⁻⁴₃∂wⁿ⁻⁶₃ * ∂wⁿ⁻⁶₃∂wⁿ⁻⁷₃  * ∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃ * (∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃ * ∂wⁿ⁻¹⁰₃∂A)
∂z∂A = -1/2 * C' * inv(C * P_mid[3] * C')' * C * (A * P[3] + A * P[3]')


zyggrad = Zygote.gradient(x -> -1/2*logdet(C * (x * (P[3] - P[3] * C' * invV[4] * C * P[3]) * x' + 𝐁) * C'), A)[1]

isapprox(∂z∂A, zyggrad)

# continue with wⁿ⁻¹⁰₃ derivative wrt P[3]
∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃ = ∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃ * A
# AP[3] = A * P[3] = A * wⁿ⁻¹¹₃
∂wⁿ⁻¹⁰₃∂wⁿ⁻¹¹₃ = A' * ∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃

# wⁿ⁻¹¹₃ = P[3] =  P_mid[2] - K[3] * CP[3] = wⁿ⁻¹²₃ - wⁿ⁻¹³₃
∂wⁿ⁻¹¹₃∂wⁿ⁻¹²₃ = ∂wⁿ⁻¹⁰₃∂wⁿ⁻¹¹₃
∂wⁿ⁻¹¹₃∂wⁿ⁻¹³₃ = -∂wⁿ⁻¹⁰₃∂wⁿ⁻¹¹₃

# wⁿ⁻¹²₃ = P_mid[2] = wⁿ⁻¹²₃¹ + B_prod = A * P[2] * A' + B_prod
∂wⁿ⁻¹²₃∂wⁿ⁻¹²₃¹ = ∂wⁿ⁻¹¹₃∂wⁿ⁻¹²₃

# wⁿ⁻¹²₃¹ = A * P[2] * A' = AP[2] * A' = wⁿ⁻¹²₃² * A'
wⁿ⁻¹²₃² = A * P[2]
∂wⁿ⁻¹²₃¹∂A = (wⁿ⁻¹²₃²' * ∂wⁿ⁻¹¹₃∂wⁿ⁻¹²₃)'
∂wⁿ⁻¹²₃¹∂wⁿ⁻¹²₃² = ∂wⁿ⁻¹¹₃∂wⁿ⁻¹²₃ * A

∂wⁿ⁻¹²₃²∂A = ∂wⁿ⁻¹²₃¹∂wⁿ⁻¹²₃² * P[2]'

# effect through wⁿ⁻¹³₃ = K[3] * CP[3]
# wⁿ⁻¹³₃ = K[3] * CP[3] = wⁿ⁻¹⁴₃ * wⁿ⁻¹⁵₃
∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃ = ∂wⁿ⁻¹¹₃∂wⁿ⁻¹³₃ * CP[3]'
∂wⁿ⁻¹³₃∂wⁿ⁻¹⁵₃ = K[3]' * ∂wⁿ⁻¹¹₃∂wⁿ⁻¹³₃

# wⁿ⁻¹⁴₃ = K[3] = PC[2] * invV[3] = P_mid[2] * C' * invV[3] = wⁿ⁻¹⁶₃ * wⁿ⁻¹⁷₃
∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁶₃ = ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃ * invV[3]'
∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁷₃ = (P_mid[2] * C')' * ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁴₃

# wⁿ⁻¹⁶₃ = P_mid[2] * C' = wⁿ⁻¹⁶₃¹ * C'
∂wⁿ⁻¹⁶₃∂wⁿ⁻¹⁶₃¹ = ∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁶₃ * C

# wⁿ⁻¹⁶₃¹ = P_mid[2] = wⁿ⁻¹⁶₃² + B_prod = A * P[2] * A' + B_prod
# wⁿ⁻¹⁶₃² = A * P[2] * A' = AP[2] * A' = wⁿ⁻¹⁶₃³ * A'
wⁿ⁻¹⁶₃³ = A * P[2]
∂wⁿ⁻¹⁶₃²∂A = (wⁿ⁻¹⁶₃³' * ∂wⁿ⁻¹⁶₃∂wⁿ⁻¹⁶₃¹)'
∂wⁿ⁻¹⁶₃²∂wⁿ⁻¹⁶₃³ = ∂wⁿ⁻¹⁶₃∂wⁿ⁻¹⁶₃¹ * A

∂wⁿ⁻¹⁶₃³∂A = ∂wⁿ⁻¹⁶₃²∂wⁿ⁻¹⁶₃³ * P[2]'

# wⁿ⁻¹⁷₃ = inv(V[3]) = inv(wⁿ⁻¹⁸₃)
∂wⁿ⁻¹⁷₃∂wⁿ⁻¹⁸₃ = -invV[3]' * ∂wⁿ⁻¹⁴₃∂wⁿ⁻¹⁷₃ * invV[3]'

# wⁿ⁻¹⁸₃ = V[3] = CP[3] * C' = wⁿ⁻¹⁹₃ * C' = wⁿ⁻⁶₁
# wⁿ⁻¹⁹₃ = CP[3] = C * P_mid[2] = C * wⁿ⁻²⁰₃
∂wⁿ⁻¹⁸₃∂wⁿ⁻¹⁹₃ = ∂wⁿ⁻¹⁷₃∂wⁿ⁻¹⁸₃ * C
∂wⁿ⁻¹⁹₃∂wⁿ⁻²⁰₃ = C' * ∂wⁿ⁻¹⁸₃∂wⁿ⁻¹⁹₃

# wⁿ⁻²⁰₃ = P_mid[2] = wⁿ⁻²⁰₃² + B_prod = A * P[2] * A' + B_prod
# wⁿ⁻²⁰₃² = A * P[2] * A' = AP[2] * A' = wⁿ⁻²⁰₃³ * A'
wⁿ⁻²⁰₃³ = A * P[2]
∂wⁿ⁻²⁰₃²∂A = (wⁿ⁻²⁰₃³' * ∂wⁿ⁻¹⁹₃∂wⁿ⁻²⁰₃)'
∂wⁿ⁻²⁰₃²∂wⁿ⁻²⁰₃³ = ∂wⁿ⁻¹⁹₃∂wⁿ⁻²⁰₃ * A

∂wⁿ⁻²⁰₃³∂A = ∂wⁿ⁻²⁰₃²∂wⁿ⁻²⁰₃³ * P[2]'



# wⁿ⁻¹⁹₃ = wⁿ⁻¹⁵₃ = CP[3] = C * P_mid[2] = C * wⁿ⁻¹⁵₃¹
∂wⁿ⁻¹⁵₃∂P = C' * ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁵₃

∂wⁿ⁻¹⁵₃∂wⁿ⁻¹⁵₃¹ = C' * ∂wⁿ⁻¹³₃∂wⁿ⁻¹⁵₃

# wⁿ⁻¹⁵₃¹ = P_mid[2] = wⁿ⁻¹⁵₃² + B_prod = A * P[2] * A' + B_prod
# wⁿ⁻¹⁵₃² = A * P[2] * A' = AP[2] * A' = wⁿ⁻¹⁵₃³ * A'
wⁿ⁻¹⁵₃¹ = A * P[2]
∂wⁿ⁻¹⁵₃²∂A = (wⁿ⁻¹⁵₃¹' * ∂wⁿ⁻¹⁵₃∂wⁿ⁻¹⁵₃¹)'
∂wⁿ⁻¹⁵₃²∂wⁿ⁻¹⁵₃³ = ∂wⁿ⁻¹⁵₃∂wⁿ⁻¹⁵₃¹ * A

∂wⁿ⁻¹⁵₃³∂A = ∂wⁿ⁻¹⁵₃²∂wⁿ⁻¹⁵₃³ * P[2]'

∂z∂A₃ = ∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A + ∂wⁿ⁻²⁰₃²∂A + ∂wⁿ⁻²⁰₃³∂A + ∂wⁿ⁻¹⁵₃²∂A + ∂wⁿ⁻¹⁵₃³∂A # this is correct and captues the effect for t = 4

# V[4] -> P_mid[3] -> A * P[3] * A'
∂wⁿ⁻⁹₃∂A   ≈ (P[3]' * A' *                                              C' * -∂z∂z/ 2 * inv(V[4])' * C    )'
# ∂wⁿ⁻⁹₃∂A   ≈ ((A * P[3])' *                                             C' * -∂z∂z/ 2 * inv(V[4])' * C    )'
∂wⁿ⁻¹⁰₃∂A  ≈                                                            C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * P[3]'

# V[4] -> P_mid[3] -> P[3] -> P_mid[2] -> A * P[2] * A'
∂wⁿ⁻¹²₃¹∂A ≈ (P[2]' * A' * A' *                                         C' * -∂z∂z/ 2 * inv(V[4])' * C     * A)'
# ∂wⁿ⁻¹²₃¹∂A ≈ ((A * P[2])' * A' *                                        C' * -∂z∂z/ 2 * inv(V[4])' * C     * A)'
∂wⁿ⁻¹²₃²∂A ≈ A' *                                                       C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * A * P[2]

∂wⁿ⁻¹⁵₃²∂A ≈ (P[2]' * A' * C' * K[3]' * -A' *                           C' * -∂z∂z/ 2 * inv(V[4])' * C     * A)'
∂wⁿ⁻¹⁵₃³∂A ≈ C' * K[3]' * -A' *                                         C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * A * P[2]'
∂wⁿ⁻¹⁶₃²∂A ≈ (P[2]' * A' * -A' *                                        C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * K[3] * C)'
∂wⁿ⁻¹⁶₃³∂A ≈ -A' *                                                      C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * K[3] * C * A * P[2]'

∂wⁿ⁻²⁰₃²∂A ≈ (P[2]' * A' * C' * -K[3]' * -A' *                          C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * K[3] * C)'
∂wⁿ⁻²⁰₃³∂A ≈ C' * -K[3]' * -A' *                                        C' * -∂z∂z/ 2 * inv(V[4])' * C     * A * K[3] * C * A * P[2]'

∂z∂A₂ = ∂wⁿ⁻¹⁰₂∂A + ∂wⁿ⁻⁹₂∂A # this is correct and captues the effect for t = 3

∂wⁿ⁻¹⁰₂∂A ≈ C' * -∂z∂z/ 2 * inv(V[3])' * C * A * P[2]'
∂wⁿ⁻⁹₂∂A  ≈ (P[2]' * A' * C' * -∂z∂z/ 2 * inv(V[3])' * C)'
# ∂z∂A = ∂wⁿ⁻⁷₃∂wⁿ⁻⁸₃ * ∂z∂z * ∂z∂wⁿ⁻¹ * ∂wⁿ⁻¹∂wⁿ⁻³₁ * ∂wⁿ⁻³₃∂wⁿ⁻⁴₃ * ∂wⁿ⁻⁴₃∂wⁿ⁻⁶₃ * ∂wⁿ⁻⁶₃∂wⁿ⁻⁷₃  * ∂wⁿ⁻⁸₃∂wⁿ⁻⁹₃ * (∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻⁹₃∂wⁿ⁻¹⁰₃ * ∂wⁿ⁻¹⁰₃∂A)
# ∂z∂A₂ = -1/2 * C' * inv(C * P_mid[3] * C')' * C * (A * P[3] + A * P[3]')

2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A)
∂z∂A = ∂wⁿ⁻¹⁰₂∂A + ∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A + ∂wⁿ⁻¹⁶₃²∂A + ∂wⁿ⁻¹⁶₃³∂A + ∂wⁿ⁻²⁰₃²∂A + ∂wⁿ⁻²⁰₃³∂A + ∂wⁿ⁻¹⁵₃²∂A + ∂wⁿ⁻¹⁵₃³∂A # this is correct and captues the effect for all t

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
                    return -1/2*(logdet(V3))
                    # return -1/2*(logdet(V4) + logdet(V3))
                end, 
            A)[1]

isapprox(∂z∂A₂, zyggrad)
∂z∂A - zyggrad

zyggrad =   Zygote.gradient(
                x -> begin
                    P_mid2 = x * P[2] * x' + B_prod
                    CP3 = C * P_mid2
                    V3 = CP3 * C'
                    K3 = P_mid2 * C' * inv(V3)
                    P3 = P_mid2 - K3 * CP3

                    P_mid3 = x * P[3] * x' + B_prod
                    CP4 = C * P_mid3
                    V4 = CP4 * C'
                    return -1/2*logdet(V4)
                end, 
            A)[1]

isapprox(zyggrad, ∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A)
zyggrad - (∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A)


zyggrad =   Zygote.gradient(
                x -> begin
                    P_mid2 = x * P[2] * x' + B_prod
                    # CP3 = C * P_mid2
                    # V3 = CP3 * C'
                    # K3 = P_mid2 * C' * inv(V3)
                    P3 = P_mid2 - K[3] * CP[3]

                    P_mid3 = A * P3 * A' + B_prod
                    CP4 = C * P_mid3
                    V4 = CP4 * C'
                    return -1/2*logdet(V4)
                end, 
            A)[1]

isapprox(zyggrad, ∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A)
zyggrad - (∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A)
maximum(abs, zyggrad - (∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A))



zyggrad =   Zygote.gradient(
                x -> begin
                    P_mid2 = x * P[2] * x' + B_prod
                    CP3 = C * P_mid2
                    # V3 = CP3 * C'
                    # K3 = P_mid2 * C' * inv(V3)
                    P3 = P_mid[2] - K[3] * CP3

                    P_mid3 = A * P3 * A' + B_prod
                    CP4 = C * P_mid3
                    V4 = CP4 * C'
                    return -1/2*logdet(V4)
                end, 
            A)[1]





# isapprox(fingrad, ∂z∂A)
# fingrad - ∂z∂A

∂z∂A = ∂wⁿ⁻¹⁰₃∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A + ∂wⁿ⁻¹²₃²∂A

zyggrad = Zygote.gradient(x -> -1/2*logdet(C * (x * (P[3] - P[3] * C' * invV[4] * C * P[3]) * x' + 𝐁) * C'), A)[1]

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
                    return -1/2*logdet(V4)
                end, 
            A)[1]

isapprox(∂z∂A, zyggrad)
∂z∂A - zyggrad


# write function to compute the gradient of the log likelihood for P_mid terms
# forward pass

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


# backward pass
TT = 4
∂A = zero(A)
# for T:-1:2
for t in TT:-1:2
    for h in 2:(t-1)
        ∂A += 2 * (A^(t-h-1))' * C' * invV[t]' * C * A^(t-h) * P[h]'
    end
end

∂A *= -1/2

∂A ≈ 2*(∂wⁿ⁻⁹₂∂A + ∂wⁿ⁻⁹₃∂A + ∂wⁿ⁻¹²₃¹∂A)



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
