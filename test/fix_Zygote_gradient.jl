using SparseArrays
using MacroModelling
import MacroModelling: timings
using ForwardDiff
import LinearAlgebra as ℒ
using FiniteDifferences, Zygote
import Optim, LineSearches
using Test, Random

# add FiniteDifferences, Zygote, Optim, LineSearches, Turing

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end


@parameters RBC_CME verbose = true begin
    # alpha | k[ss] / (4 * y[ss]) = cap_share
    # cap_share = 1.66
    alpha = .157

    # beta | R[ss] = R_ss
    # R_ss = 1.0035
    beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    # Pibar | Pi[ss] = Pi_ss
    # Pi_ss = 1.0025
    Pibar = 1.0008

    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_solution(RBC_CME)

get_irf(RBC_CME; parameters = RBC_CME.parameter_values)


Random.seed!(3)

data = simulate(RBC_CME)[:,:,1]
observables = [:c,:k]
# @test isapprox(425.7689804539224, get_loglikelihood(RBC_CME, data(observables), RBC_CME.parameter_values),rtol = 1e-5)

forw_grad = ForwardDiff.gradient(x -> get_loglikelihood(RBC_CME, data(observables), x), Float64.(RBC_CME.parameter_values))
reverse_grad = Zygote.gradient(x -> get_loglikelihood(RBC_CME, data(observables), x), Float64.(RBC_CME.parameter_values))[1]

fin_grad = FiniteDifferences.grad(central_fdm(4,1),x -> get_loglikelihood(RBC_CME, data(observables), x), RBC_CME.parameter_values)[1]

@test isapprox(fin_grad, reverse_grad, rtol = 1.0e-6)
# @test isapprox(forw_grad, reverse_grad, rtol = 1.0e-6)
# @test isapprox(fin_grad, forw_grad, rtol = 1.0e-6)

# calc the individual gradients
import MacroModelling: get_and_check_observables, get_relevant_steady_state_and_state_update, calculate_loglikelihood, check_bounds, get_initial_covariance, run_kalman_iterations
𝓂 = RBC_CME
data = data(observables)
parameter_values = RBC_CME.parameter_values
algorithm = :first_order
filter = :kalman
warmup_iterations = 0
presample_periods = 0
initial_covariance = :theoretical
tol = 1e-12
verbose = false
T = 𝓂.timings
# checks to avoid errors further down the line and inform the user
@assert filter ∈ [:kalman, :inversion] "Currently only the Kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

# checks to avoid errors further down the line and inform the user
@assert initial_covariance ∈ [:theoretical, :diagonal] "Invalid method to initialise the Kalman filters covariance matrix. Supported methods are: the theoretical long run values (option `:theoretical`) or large values (10.0) along the diagonal (option `:diagonal`)."

if algorithm ∈ [:second_order,:pruned_second_order,:third_order,:pruned_third_order]
    filter = :inversion
end

observables = get_and_check_observables(𝓂, data)

solve!(𝓂, verbose = verbose, algorithm = algorithm)

bounds_violated = check_bounds(parameter_values, 𝓂)

NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

obs_indices = convert(Vector{Int}, indexin(observables, NSSS_labels))

T, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, tol)

# prepare data
data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

observables_index = convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

observables_and_states = sort(union(T.past_not_future_and_mixed_idx,observables_index))

A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

C = ℒ.diagm(ones(length(observables_and_states)))[(indexin(sort(observables_index), observables_and_states)),:]

𝐁 = B * B'

# Gaussian Prior
coordinates = Tuple{Vector{Int}, Vector{Int}}[]

dimensions = [size(A),size(𝐁)]

values = vcat(vec(A), vec(collect(-𝐁)))

P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)




fin_A = FiniteDifferences.grad(central_fdm(3,1),
A -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), A)[1]

for_A = ForwardDiff.gradient(A -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), A)
bac_A = Zygote.gradient(A -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), A)[1]
isapprox(for_A, bac_A, rtol = 1e-6)

fin_𝐁 = FiniteDifferences.grad(central_fdm(4,1),
𝐁 -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), 𝐁)[1]

for_𝐁 = ForwardDiff.gradient(𝐁 -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), 𝐁)
bac_𝐁 = Zygote.gradient(𝐁 -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), 𝐁)[1]
isapprox(for_𝐁, bac_𝐁, rtol = 1e-6)

fin_P = FiniteDifferences.grad(central_fdm(2,1),
P -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), P)[1]

for_P = ForwardDiff.gradient(P -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), P)
bac_P = Zygote.gradient(P -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), P)[1]
isapprox(for_P, bac_P, rtol = 1e-6)

fin_data_in_deviations = FiniteDifferences.grad(central_fdm(3,1),
data_in_deviations -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), data_in_deviations)[1]

for_data_in_deviations = ForwardDiff.gradient(data_in_deviations -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), data_in_deviations)
bac_data_in_deviations = Zygote.gradient(data_in_deviations -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), data_in_deviations)[1]

isapprox(for_data_in_deviations, bac_data_in_deviations, rtol = 1e-6)


Zygote.withgradient(data_in_deviations -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), data_in_deviations)[1]


import LinearAlgebra: mul!, logdet
import RecursiveFactorization as RF

P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

∂llh = 1


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
∂ū∂v = zero(ū)
∂𝐁 = zero(𝐁)
∂data_in_deviations = zero(data_in_deviations)
vtmp = zero(v[1])
Ptmp = zero(P[1])

# pullback
# function kalman_pullback(∂llh)
ℒ.rmul!(∂A, 0)
ℒ.rmul!(∂Faccum, 0)
ℒ.rmul!(∂P, 0)
ℒ.rmul!(∂ū, 0)
ℒ.rmul!(∂𝐁, 0)

# t = T-1
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
        # ∂ū∂v = C' * (invF[t]' + invF[t]) * v[t]
        copy!(invF[1], invF[t]' + invF[t]) # using invF[1] as temporary storage
        mul!(v[1], invF[1], v[t]) # using v[1] as temporary storage
        mul!(∂ū∂v, C', v[1])
    else
        ℒ.rmul!(∂F, 0)
        ℒ.rmul!(∂ū∂v, 0)
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
    ∂data_in_deviations[:,t-1] = C * ∂ū∂v + K[t]' * A' * ∂ū
    # mul!(vtmp, C, ∂ū)
    # ℒ.rmul!(vtmp, -1)
    # ∂data_in_deviations[:,t-1] .= vtmp
    # mul!(∂data_in_deviations[:,t-1], C, ∂ū, -1, 0) # cannot assign to columns in matrix, must be whole matrix 

    # ∂ū∂ū
    # z[t] .= C * ū[t]
    # v[t] .= data_in_deviations[:, t-1] .- z
    # K[t] .= P̄[t-1] * C' * invF[t]
    # u[t] .= K[t] * v[t] + ū[t-1]
    # ū[t] .= A * u[t]
    # step to next iteration
    ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
    # ∂ū = C' * K[t]' * A' * ∂ū
    # mul!(u[1], A', ∂ū) # using u[1] as temporary storage
    # mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
    # mul!(u[1], C', v[1], -1, 1)
    # copy!(∂ū, u[1])

    # ∂llh∂ū
    # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
    # v[t] .= data_in_deviations[:, t-1] .- z
    # z[t] .= C * ū[t]
    # ∂ū -= ∂ū∂v
    ℒ.axpy!(-1, ∂ū∂v, ∂ū)


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

# return NoTangent(), ∂A, ∂𝐁, NoTangent(), ∂P, ∂data_in_deviations, NoTangent()
# end



PP = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

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

llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2 * 3.141592653589793)) / 2 


zyggrad = Zygote.gradient(
    x -> begin
        CP2 = C * PP
        V2 = CP2 * C'
        K2 = PP * C' * inv(V2)
        innovation2 = x[:, 1] - z[1]
        u2 = K2 * innovation2 + u_mid[1]
        P2 = PP - K2 * CP2
        u_mid2 = A * u2
        z2 = C * u_mid2
        P_mid2 = A * P2 * A' + B_prod

        CP3 = C * P_mid2
        V3 = CP3 * C'
        innovation3 = x[:, 2] - z2
        K3 = P_mid2 * C' * inv(V3)
        u3 = K3 * innovation3 + u_mid2
        P3 = P_mid2 - K3 * CP3
        u_mid3 = A * u3
        z3 = C * u_mid3
        P_mid3 = A * P3 * A' + B_prod

        CP4 = C * P_mid3
        V4 = CP4 * C'
        innovation4 = x[:, 3] - z3

        # return -1/2*(logdet(V[2]) + innovation2' * inv(V[2]) * innovation2)
        # return -1/2*(logdet(V[3]) + innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(logdet(V[2]) + innovation2' * inv(V[2]) * innovation2 + logdet(V[3]) + innovation3' * inv(V[3]) * innovation3)
        # return -1/2*(logdet(V[4]) + innovation4' * inv(V[4]) * innovation4)
        # return -1/2*(logdet(V4) + innovation4' * inv(V4) * innovation4 + logdet(V3) + innovation3' * inv(V3) * innovation3)
        return -1/2*(logdet(V2) + innovation2' * inv(V2) * innovation2 + logdet(V3) + innovation3' * inv(V3) * innovation3 + logdet(V4) + innovation4' * inv(V4) * innovation4)
    end, 
    observables)[1]




innovation4 = observables[:, 3] - C * A * (K3 * (observables[:, 2] - C * A * (K2 * (x[:, 1] - z[1]) + u_mid[1])) + A * (K2 * (x[:, 1] - z[1]) + u_mid[1]))

innovation4 = - C * A * (K3 * ( - C * A * (K2 * x)))

K[2]' * A' * C' * K[3]' * A' * C' * (invF[4]' + invF[4]) * v[4]


innovation4 = - C * A * A * K2 * x

-K[2]' * A' * A' * C' * (invF[4]' + invF[4]) * v[4]


K[2]' * A' * C' * K[3]' * A' * C' * (invF[4]' + invF[4]) * v[4] - K[2]' * A' * A' * C' * (invF[4]' + invF[4]) * v[4]


(invF[3]' + invF[3]) * v[3]
for_data_in_deviations

P = get_initial_covariance(Val(initial_covariance), values, coordinates, dimensions)

for_data_in_deviations = ForwardDiff.gradient(data_in_deviations -> run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods), data_in_deviations[:,1:2][:,:])
zyggrad


addd = K[t]' * A' * ∂ū∂v

C * A' * ∂ū∂v3 + K[t]' * A' * ∂ū∂v3

for_data_in_deviations[:,1]


zyggrad

t = 4
-(K[t-1]' * A' * C' * (invF[t-1]' + invF[t-1]) * v[t-1] + (invF[t+0]' + invF[t+0]) * v[t+0])/2

∂ū∂v4 = C' * (invF[t]' + invF[t]) * v[t]

∂llh4∂o2 = K[t]' * A' * ∂ū∂v4

# ∂llh4∂o1 = K[t-1]' * A' * C' * ∂llh4∂o2 - K[t-1]' * A' * A' * C' * (invF[t]' + invF[t]) * v[t]
∂llh4∂o1 = K[t-1]' * A' * C' * K[t]' * A' * ∂ū∂v4 - K[t-1]' * A' * A' * ∂ū∂v4


# K[t-1]' * A' * C' * invF[t-1] * v[t-1]

# K[t-1]' * A' * C' * invF[t-1] * v[t-1] + (v[t-1]' * invF[t-1] * C * A * K[t-1])

# K[t-1]' * A' * C' * (invF[t-1]' + invF[t-1]) * v[t-1]

t = 4
∂ū∂v4 = C' * (invF[t]' + invF[t]) * v[t]
C * ∂ū∂v4 /-2
∂ū = ∂ū∂v4



-(C*∂ū∂v3 - ∂llh4∂o2)/2

t = 3
# ∂ū =  C' * K[t]' * A' * ∂ū
∂ū∂v3 = C' * (invF[t]' + invF[t]) * v[t]
C * ∂ū∂v3 - K[t]' * A' * ∂ū
# ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
# C * ∂ū∂v3 /-2
# ∂ū -= ∂ū∂v3
-∂ū/2

t = 2
# ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
∂ū =  C' * K[t]' * A' * ∂ū
∂ū∂v2 = C' * (invF[t]' + invF[t]) * v[t]
∂ū -= ∂ū∂v2

# -(A' * ∂ū∂v3 - C' * K[t]' * A' * ∂ū∂v3 - ∂ū∂v2)/2
# ∂ū∂v = C' * K[t]' * A' * C' * (invF[t]' + invF[t]) * v[t] - addd

∂ū/-2



for_data_in_deviations[:,1] - C * ∂ū∂v2/2


∂data_in_deviations

zyggrad


# initialise derivative variables
∂A = zero(A)
∂F = zero(F)
∂Faccum = zero(F)
∂P = zero(P̄)
∂ū = zero(ū)
∂v = zero(v[1])
∂v̂ = zero(v[1])
∂û = zero(ū)
∂z = zero(z[1])
∂ū∂v = zero(ū)
∂𝐁 = zero(𝐁)
∂data_in_deviations = zero(data_in_deviations)
vtmp = zero(v[1])
Ptmp = zero(P[1])


for t in 4:-1:2
    if t > presample_periods + 1
        # ∂F = invF[t]' - invF[t]' * v[t] * v[t]' * invF[t]'
        # ∂ū∂v = C' * (invF[t]' + invF[t]) * v[t]
        ∂v = (invF[t]' + invF[t]) * v[t]
    else
        # ℒ.rmul!(∂F, 0)
        ℒ.rmul!(∂ū∂v, 0)
    end

    # ∂z = C' * ∂v̂
    # ∂ū = A' * ∂z
    # ∂u = K[t] * ∂v̂ + ∂ū
    # ∂P += C' * (∂F + ∂Faccum) * C
    # ∂P += A' * ∂ū * v[t]' * invF[t]' * C

    mul!(u[1], A', ∂ū)
    mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
    ℒ.axpy!(1, ∂v, v[1])
    ∂data_in_deviations[:,t-1] .= v[1]
    # ∂data_in_deviations[:,t-1] = K[t]' * A' * ∂ū + ∂v̂

    ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
    ∂ū -= C' * ∂v


    # ∂û = A' * C' * ∂v̂
    # ∂v = - K[t]' * ∂û
    

    if t > 2
        # ∂A += ∂ū * u[t-1]'
        # ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]

        # ∂𝐁 += ∂P

        # ∂P = A' * ∂P * A
        # ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 

        # ∂Faccum = -invF[t-1]' * CP[t-1] * A' * ∂ū * v[t-1]' * invF[t-1]'
        # ∂Faccum -= invF[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invF[t-1]'
    end
end


ℒ.rmul!(∂P, -∂llh/2)
ℒ.rmul!(∂A, -∂llh/2)
ℒ.rmul!(∂𝐁, -∂llh/2)
ℒ.rmul!(∂data_in_deviations, -∂llh/2)

zyggrad

# obs in t = 1 impact on innovation2
(invF[2]' + invF[2]) * v[2] 

# obs in t = 1 impact on innovation3
-K[2]' * A' * C' * (invF[3]' + invF[3]) * v[3]

# obs in t = 1 impact on innovation4
K[2]' * A' * C' * K[3]' * A' * C' * (invF[4]' + invF[4]) * v[4]    - K[2]' * A' * A' * C' * (invF[4]' + invF[4]) * v[4]

(K[2]' * A' * C' * K[3]' - K[2]' * A') * A' * C' * (invF[4]' + invF[4]) * v[4]

K[2]' * A' * (C' * K[3]' - ℒ.I) * A' * C' * (invF[4]' + invF[4]) * v[4]
K[2]' * A' * (C' * K[3]' - ℒ.I) * A' * C' * (invF[4]' + invF[4]) * v[4]


(invF[2]' + invF[2]) * v[2]  + -K[2]' * A' * C' * (invF[3]' + invF[3]) * v[3] + K[2]' * A' * C' * K[3]' * A' * C' * (invF[4]' + invF[4]) * v[4] - K[2]' * A' * A' * C' * (invF[4]' + invF[4]) * v[4]