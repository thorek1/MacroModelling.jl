using MacroModelling
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import LinearAlgebra as ℒ
import RecursiveFactorization as RF


include("models/FS2000.jl")

FS2000 = m
get_SS(FS2000)
get_covariance(FS2000)
# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)

# get_SS(FS2000, parameters = [0.4027212142373724
# 0.9909438997461472
# 0.00455007831270222
# 1.014322728752977
# 0.8457081193818059
# 0.6910339118126667
# 0.0016353140797331237
# 0.013479922353054475
# 0.003257545969294338])

get_SS(FS2000,parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950])



calculate_kalman_filter_loglikelihood(m, data(observables), observables)


𝓂 = FS2000
verbose = true
tol = eps()


sort!(observables)

solve!(𝓂, verbose = verbose)

parameters = [0.403475267025427,0.990923010561409,0.004566214169879,1.014318555099325,0.845538800525148,0.689060025764850,0.001665380385476,0.013570417835562,0.003274145891950]

SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

if solution_error > tol || isnan(solution_error)
    return -Inf
end

NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

obs_indices = indexin(observables,NSSS_labels)

data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

observables_and_states = sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

C = @views ℒ.diagm(ones(length(observables_and_states)))[(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

𝐁 = B * B'

# Gaussian Prior

calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

Pstar1 = calculate_covariance_(sol)
# P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
u = zeros(length(observables_and_states))
# u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
z = C * u












nk = 1
d = 0
decomp = []
# spinf = size(Pinf1)
spstar = size(Pstar1)
v = zeros(size(C,1), size(data_in_deviations,2))
u = zeros(size(A,1), size(data_in_deviations,2)+1)
# û = zeros(size(A,1), size(data_in_deviations,2))
# uK = zeros(nk, size(A,1), size(data_in_deviations,2)+nk)
# PK = zeros(nk, size(A,1), size(A,1), size(data_in_deviations,2)+nk)
iF = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# Fstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFinf = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# K = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
L = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Linf = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Lstar = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Kstar = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
# Kinf = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
P = zeros(size(A,1), size(A,1), size(data_in_deviations,2)+1)
# Pstar = zeros(spstar[1], spstar[2], size(data_in_deviations,2)+1)
# Pstar[:, :, 1] = Pstar1
# Pinf = zeros(spinf[1], spinf[2], size(data_in_deviations,2)+1)
# Pinf[:, :, 1] = Pinf1
rr = size(C,1)
# 𝐁 = R * Q * transpose(R)
ū = zeros(size(A,1), size(data_in_deviations,2))
ϵ̄ = zeros(rr, size(data_in_deviations,2))
ϵ = zeros(rr, size(data_in_deviations,2))
# epsilonhat = zeros(rr, size(data_in_deviations,2))
r = zeros(size(A,1))
# Finf_singular = zeros(1, size(data_in_deviations,2))

V = []

# t = 0

# d = t
P[:, :, 1] = Pstar1
# iFinf = iFinf[:, :, 1:d]
# iFstar= iFstar[:, :, 1:d]
# Linf = Linf[:, :, 1:d]
# Lstar = Lstar[:, :, 1:d]
# Kstar = Kstar[:, :, 1:d]
# Pstar = Pstar[:, :, 1:d]
# Pinf = Pinf[:, :, 1:d]
# K
# û

for t in 1:size(data_in_deviations,2)
    v[:, t] = data_in_deviations[:, t] - C * u[:, t]
    F = C * P[:, :, t] * C'
    iF[:, :, t] = inv(F)
    PCiF = P[:, :, t] * C' * iF[:, :, t]
    û = u[:, t] + PCiF * v[:, t]
    K = A * PCiF
    L[:, :, t] = A - K * C
    P[:, :, t+1] = A * P[:, :, t] * L[:, :, t]' + 𝐁
    u[:, t+1] = A * û
    ϵ[:, t] = B' * C' * iF[:, :, t] * v[:, t]
    # Pf = P[:, :, t]
    # uK[1, :, t+1] = u[:, t+1]
    # for jnk in 1:nk
    #     Pf = A * Pf * A' + 𝐁
    #     PK[jnk, :, :, t+jnk] = Pf
    #     if jnk > 1
    #         uK[jnk, :, t+jnk] = A * uK[jnk-1, :, t+jnk-1]
    #     end
    # end
end

# backward pass; r_T and N_T, stored in entry (size(data_in_deviations,2)+1) were initialized at 0
# t = size(data_in_deviations,2) + 1
for t in size(data_in_deviations,2):-1:1
    r = C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r # compute r_{t-1}, DK (2012), eq. 4.38
    ū[:, t] = u[:, t] + P[:, :, t] * r # DK (2012), eq. 4.35
    ϵ̄[:, t] = B' * r # DK (2012), eq. 4.63
end

# if decomp_flag
    decomp = zeros(nk, size(A,1), rr, size(data_in_deviations,2)+nk)
    CBs = C' * inv(C * 𝐁 * C') * C * B
    for t in 1:size(data_in_deviations,2)
        # : = data_index[t]
        # calculate eta_tm1t
        eta_tm1t = B' * C' * iF[:, :, t] * v[:, t]
        AAA = P[:, :, t] * CBs .* eta_tm1t'
        # calculate decomposition
        decomp[1, :, :, t+1] = AAA
        for h = 2:nk
            AAA = A * AAA
            decomp[h, :, :, t+h] = AAA
        end
    end
# end

epsilonhat = data_in_deviations - C * ū





# outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

# C*outt[1]
# C*outt[3]
# data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*ū)[1,:])
plot!((C*u)[1,2:end])


plot(data_in_deviations[2,:])
plot!((C*ū)[2,:])
plot!((C*u)[2,2:end])

plot(ϵ̄[1,:])
plot(ϵ̄[2,:])

ϵ̄

plot(ϵ[1,:])
plot(ϵ[2,:])

ϵ

decomposition = zeros(size(A,1),size(B,2)+2,size(data_in_deviations,2))
decomposition[:,end,:] = ū


decomposition[:,1:end-2,1] = B .* repeat(ϵ̄[:, 1]', size(A,1))
decomposition[:,end-1,1] = decomposition[:,end,1] - sum(decomposition[:,1:end-2,1],dims=2)

for i in 2:size(data_in_deviations,2)
    decomposition[:,1:end-2,i] = A * decomposition[:,1:end-2,i-1]
    decomposition[:,1:end-2,i] += B .* repeat(ϵ̄[:, i]', size(A,1))
    decomposition[:,end-1,i] = decomposition[:,end,i] - sum(decomposition[:,1:end-2,i],dims=2)
end


# Assuming your 4x192 array is named "data"
data = decomposition[2,:,:]
# sum(data[1:3, :],dims=1)' .- data[4, :]
# Split the data into the relevant components
bar_data = data[1:3, :]
line_data = data[4, :]



# Create the stacked bar plot
bar_plot = groupedbar(bar_data[[end,1:end-1...],:]', label=["Bar1" "Bar2" "Bar3"], xlabel="Time", ylabel="Value", alpha=0.5, title="Stacked Bars with Line Overlay", bar_position = :stack)

plot!(line_data, label="Line", linewidth=2, color=:black, linestyle=:solid, legend=:topright)



ϵ̄
ū











loglik = 0.0

B̂ = RF.lu(C * B , check = false)

@assert ℒ.issuccess(B̂) "Numerical stabiltiy issues for restrictions in period 1."

B̂inv = inv(B̂)

n_timesteps = size(data, 2)
n_states = length(u)
filtered_states = zeros(n_states, n_timesteps)
updated_states = zeros(n_states, n_timesteps)
smoothed_states = zeros(n_states, n_timesteps)
filtered_covariance = zeros(n_states, n_states, n_timesteps)
filtered_shocks = zeros(size(C,1), n_timesteps)
smoothed_shocks = zeros(n_states, n_timesteps)
P_smoothed = copy(P)

# Kalman filter
for t in 1:n_timesteps
    v = data_in_deviations[:, t] - C * u
    filtered_shocks[:, t] = B̂inv * v
    F = C * P * C'
    K = P * C' / F
    P = A * (P - K * C * P) * A' + 𝐁
    filtered_covariance[:, :, t] = P
    u = A * (u + K * v)
    filtered_states[:, t] = u
    updated_states[:,t] = filtered_states[:, t] + K * v
end

smoothed_states = copy(filtered_states)
smoothed_covariance = copy(filtered_covariance)

for t in n_timesteps-1:-1:1
    J = filtered_covariance[:,:, t] * A * filtered_covariance[:,:, t + 1]
    smoothed_states[:, t] = filtered_states[:, t+ 1] + J * (smoothed_states[:, t + 1] - filtered_states[:, t])
    smoothed_covariance[:,:, t] = filtered_covariance[:,:, t] + J * (smoothed_covariance[:,:, t + 1] - filtered_covariance[:,:, t + 1]) * J'
end

v = data_in_deviations[:,t] - z

F = C * P * C'

K = P * C' / F

P = A * (P - K * C * P) * A' + 𝐁

u = A * (u + K * v)

z = C * u 
# Kalman smoother for states
smoothed_states[:, end] = filtered_states[:, end]
for t in (n_timesteps - 1):-1:1
    P_future = A * P * A' + 𝐁
    J = P * A' / P_future
    smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
end





# Kalman smoother for states
smoothed_states[:, end] = filtered_states[:, end]
smoothed_covariances[:, :, end] = filtered_covariances[:, :, end]
for t in (n_timesteps - 1):-1:1
    P_future = A * P * A' + 𝐁
    J = P * A' / P_future
    smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
    P = filtered_covariances[:, :, t] - J * (P_future - smoothed_covariances[:, :, t + 1]) * J'
    smoothed_covariances[:, :, t] = P
end




for t in 1:size(data)[2]
    v = data_in_deviations[:,t] - z

    F = C * P * C'

    # F = (F + F') / 2

    # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
    # K = P * C' * ℒ.pinv(F)

    # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
    Fdet = ℒ.det(F)

    if Fdet < eps() return -Inf end

    loglik += log(Fdet) + v' / F  * v
    
    K = P * C' / F

    P = A * (P - K * C * P) * A' + 𝐁

    u = A * (u + K * v)
    
    z = C * u 
end

return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment




function kalman_filter_and_smoother(data, A, B, C, u, P, 𝐁)
    n_timesteps = size(data, 2)
    n_states = length(u)
    filtered_states = zeros(n_states, n_timesteps)
    smoothed_states = zeros(n_states, n_timesteps)
    filtered_shocks = zeros(n_states, n_timesteps)
    smoothed_shocks = zeros(n_states, n_timesteps)
    P_smoothed = copy(P)

    # Kalman filter
    for t in 1:n_timesteps
        v = data[:, t] - C * u
        F = C * P * C'
        K = P * C' / F
        filtered_shocks[:, t] = K * v
        P = A * (P - K * C * P) * A' + 𝐁
        u = A * (u + filtered_shocks[:, t])
        filtered_states[:, t] = u
    end

    # Kalman smoother for states
    smoothed_states[:, end] = filtered_states[:, end]
    for t in (n_timesteps - 1):-1:1
        P_future = A * P * A' + 𝐁
        J = P * A' / P_future
        smoothed_states[:, t] = filtered_states[:, t] + J * (smoothed_states[:, t + 1] - A * filtered_states[:, t])
    end

    # Kalman smoother for shocks
    smoothed_shocks[:, end] = filtered_shocks[:, end]
    for t in (n_timesteps - 1):-1:1
        P_future = A * P * A' + 𝐁
        J = P * A' / P_future
        smoothed_shocks[:, t] = filtered_shocks[:, t] + J * (smoothed_shocks[:, t + 1] - A * filtered_shocks[:, t])
    end

    return filtered_states, smoothed_states, filtered_shocks, smoothed_shocks
end







using Distributions
using LinearAlgebra

for t in 1:size(data)[2]
    v = data_in_deviations[:,t] - z

    F = C * P * C'
    
    K = P * C' / F

    P = A * (P - K * C * P) * A' + 𝐁

    u = A * (u + K * v)
    u = A * (u + K * (data_in_deviations[:,t] - z))
    
    z = C * u 
end
# Kalman filter implementation
function kalman_filter(data_in_deviations, A, 𝐁, C, u0, P0)
    T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    û = zeros(n, T)
    P = Array{Matrix{Float64}}(undef, T)
    û[:, 1] = u0
    P[1] = P0

    # Update
    F = C * P0 * C'
    K = P0 * C' / F
    û[:, 1] = K * (data_in_deviations[:, 1])
    P[1] = A * (P0 - K * C * P0) * A' + 𝐁

    for t in 2:T
        # Predict
        û⁻ = A * û[:, t - 1]
    
        # Update
        F = C * P[t - 1] * C'
        K = P[t - 1] * C' / F
        û[:, t] = û⁻ + K * (data_in_deviations[:, t] - C * û⁻)
        P[t] = A * (P[t - 1] - K * C * P[t - 1]) * A' + 𝐁
    end

    return û, P
end

# Kalman smoother implementation
function kalman_smoother(data_in_deviations, A, 𝐁, û, P)
    T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    u_smoother = zeros(n, T)
    u_smoother[:, end] = û[:, end]
    P_smoother = Array{Matrix{Float64}}(undef, T)
    P_smoother[end] = P[end]

    for t in T-1:-1:1
        J = P[t] * A' / (A * P[t] * A' + 𝐁)
        u_smoother[:, t] = û[:, t] + J * (u_smoother[:, t + 1] - A * û[:, t])
        P_smoother[t] = P[t] + J * (P_smoother[t + 1] - (A * P[t] * A' + 𝐁)) * J'
    end

    return u_smoother, P_smoother
end



us
Ps

u0 = u
P0 = P

T = size(data_in_deviations, 2)
n = size(A, 1)

û = zeros(n, T)
P = Array{Matrix{Float64}}(undef, T)
û[:, 1] = u0
P[1] = P0


# Update
F = C * P0 * C'
K = P0 * C' / F
û[:, 1] = K * (data_in_deviations[:, 1])




for t in 2:T
    # Predict
    û⁻ = A * û[:, t - 1]

    # Update
    F = C * P[t - 1] * C'
    K = P[t - 1] * C' / F
    û[:, t] = û⁻ + K * (data_in_deviations[:, t] - C * û⁻)
    P[t] = A * (P[t - 1] - K * C * P[t - 1]) * A' + 𝐁
end




for t in 1:T
    v[:,t] = data_in_deviations[:,t] - C * A * û[:,t]
    F[:,:,t] = C * P[:,:,t] * C'
    K = P[:,:,t] * C' / F[:,:,t]
    L[:,:,t] = A - A * K * C
    if t < T
        û[:,t+1] = A * û[:,t] + K * v[:,t]
        P[:,:,t+1] = A * P[:,:,t] * L[:,:,t]' + 𝐁
    end
end



T = size(data_in_deviations, 2)
    n = size(A, 1)
    
    u_smoother = zeros(n, T)
    u_smoother[:, end] = û[:, end]
    P_smoother = Array{Matrix{Float64}}(undef, T)
    P_smoother[end] = P[end]

    for t in T-1:-1:1
        J = P[t] * A' / (A * P[t] * A')
        u_smoother[:, t] = û[:, t] + J * (u_smoother[:, t + 1] - A * û[:, t])
        P_smoother[t] = P[t] + J * (P_smoother[t + 1] - (A * P[t] * A')) * J'
    end

ufilter, Pfilter = kalman_filter(data_in_deviations,A,𝐁,C,u,P)

usmooth , Psmooth = kalman_smoother(data_in_deviations, A, 𝐁, ufilter, Pfilter)


Pfilter[end-1] * A' / (A * Pfilter[end-1] * A' + 𝐁)


using Distributions

function kalman_filter(data_in_deviations, A, 𝐁, C, u0, P0)
    T = size(data_in_deviations,2)
    n = size(u0,1)
    û = zeros(n,T)
    P = zeros(n,n,T)
    v = zeros(size(C,1),T)
    F = zeros(size(C,1),size(C,1),T)
    L = zeros(n,n,T)

    û[:,1] = u0
    P[:,:,1] = P0

    t = 1
    F[:,:,t] = C * P[:,:,t] * C'
    K = P[:,:,t] * C' / F[:,:,t]
    L[:,:,t] = A - A * K * C
    û[:,t] = K * data_in_deviations[:,t]
    P[:,:,t] = A * P[:,:,t] * L[:,:,t]' + 𝐁

    for t in 2:T
        v[:,t] = data_in_deviations[:,t] - C * A * û[:,t-1]
        F[:,:,t] = C * P[:,:,t-1] * C'
        K = P[:,:,t-1] * C' / F[:,:,t]
        L[:,:,t] = A - A * K * C
        û[:,t] = A * û[:,t-1] + K * v[:,t]
        P[:,:,t] = A * P[:,:,t-1] * L[:,:,t]' + 𝐁
    end

    r = zeros(n)
    N = 0.0
    u = zeros(n,T)
    U = zeros(n,n,T)

    for t in T:-1:1
        r = C' / F[:,:,t] * v[:,t] + L[:,:,t]' * r
        N = C' / F[:,:,t] * C + L[:,:,t]' * N * L[:,:,t]
        u[:,t] = û[:,t] + P[:,:,t] * r
        U[:,:,t] = P[:,:,t] - P[:,:,t] * N * P[:,:,t]
    end

    return û,P,u,U
end


out = kalman_filter(data_in_deviations,A,𝐁,C,u,P)

out[1]
out[2]
out[3]
out[4]

ufilter

data_in_deviations = rand(Normal(0,1),100) # example data
C = 1.0
H = 0.5^2
A = 0.8
𝐁 = 0.2^2
u0 = 0.0
P0 = 1.0

v,F,K,û,P,u,U = kalman_filter(data_in_deviations,C,H,A,𝐁,u0,P0)





Sure! Here is an example of a Kalman filter and smoother implemented in Julia:


using Distributions

function kalman_filter(y, Z, H, T, Q, R, a1, P1)
    n = size(y)[2]
    m = size(a1)[1]
    a = zeros(m,n)
    P = zeros(m,m,n)
    v = zeros(n)
    F = zeros(n)
    K = zeros(m,n)
    L = zeros(m,m,n)

    a[:,1] = a1
    P[:,:,1] = P1
    for t in 1:n
        v[t] = y[t] - Z*a[:,t]
        F[t] = Z*P[:,:,t]*Z' + H
        K[:,t] = T*P[:,:,t]*Z'*inv(F[t])
        L[:,:,t] = T - K[:,t]*Z
        if t < n
            a[:,t+1] = T*a[:,t] + K[:,t]*v[t]
            P[:,:,t+1] = T*P[:,:,t]*L[:,:,t]' + R
        end
    end

    r = zeros(m)
    N = 0.0
    u = zeros(n)
    U = zeros(n)
    for t in n:-1:1
        r = Z' / F[t] * v[t] + L[:,:,t]' * r
        N = Z' / F[t] * Z + L[:,:,t]' * N * L[:,:,t]
        u[t] = a[:,t] + P[:,:,t] * r
        U[t] = P[:,:,t] - P[:,:,t]*N*P[:,:,t]
    end

    return v,F,K,a,P,u,U
end

y = rand(Normal(0,1),100) # example data
Z = 1.0
H = 0.5^2
T = 0.8
Q = 0.2^2
R = 0.3^2
a1 = 0.0
P1 = 1.0

v,F,K,a,P,u,U = kalman_filter(y,Z,H,T,Q,R,a1,P1)


# This code defines a function `kalman_filter` that takes in the observed data `y`, the observation matrix `Z`, the observation variance `H`, the state transition matrix `T`, the state transition variance `Q`, the initial state variance `R`, the initial state mean `a1`, and the initial state variance `P1`. The function returns the prediction error `v`, the prediction error variance `F`, the Kalman gain `K`, the filtered state mean `a`, the filtered state variance `P`, the smoothed state mean `u`, and the smoothed state variance `U`.

# I hope this helps! Let me know if you have any questions or need further assistance.






# function calculate_kalman_filter_loglikelihood(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose::Bool = false, tol::AbstractFloat = eps())
#     sort!(observables)

#     solve!(𝓂, verbose = verbose)

#     parameters = 𝓂.parameter_values

#     SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)

#     NSSS_labels = [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

#     obs_indices = indexin(observables,NSSS_labels)

#     data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

# 	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

#     sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

#     observables_and_states = sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

#     A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
#     B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

#     C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

#     𝐁 = B * B'

#     calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

#     P = calculate_covariance_(sol)
#     u = zeros(length(observables_and_states))
#     z = C * u
    
#     loglik = 0.0

#     for t in 1:size(data)[2]
#         v = data_in_deviations[:,t] - z

#         F = C * P * C'
        
#         K = P * C' / F

#         P = A * (P - K * C * P) * A' + 𝐁

#         u = A * (u + K * v)
        
#         z = C * u 
#     end

#     return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment
# end










function kalman_filter_and_smoother(data, A, 𝐁, C, u, P)
    n_timesteps = size(data, 2)
    n_states = length(u)
    filtered_states = zeros(n_states, n_timesteps)
    filtered_covariance = zeros(n_states, n_states, n_timesteps)
    smoothed_states = zeros(n_states, n_timesteps)
    smoothed_covariance = zeros(n_states, n_states, n_timesteps)

    # filtered_shocks = zeros(n_states, n_timesteps)
    # smoothed_shocks = zeros(n_states, n_timesteps)
    # P_smoothed = copy(P)

    # Kalman filter
    for t in 1:n_timesteps
        v = data[:, t] - C * u
        F = C * P * C'
        K = A * P * C' / F
        û = u + P * C' / F * v
        P̂ = P - P * C' / F * C * P
        u = A * û
        L = A - K * C
        P = A * P * L' + 𝐁

        filtered_states[:, t] = u
        filtered_covariance[:,:,t] = P
    end

    # Kalman smoother for states
    smoothed_states[:, end] = filtered_states[:, end]
    smoothed_covariance[:,:, end] = filtered_covariance[:,:, end]
    r = zero(u)

    for t in n_timesteps:-1:1
        u = filtered_states[:, t]
        P = filtered_covariance[:,:,t] 

        v = data[:, t] - C * u
        F = C * P * C'
        K = P * C' / F
        r = (C' * F)' \ v + r
        smoothed_states[:, t] = u + P * r
        smoothed_covariance[:,:, t] = P - r * r' * P'
        r = A' * r
    end


    return filtered_states, filtered_covariance, smoothed_states ,smoothed_covariance #, filtered_shocks, smoothed_shocks
end


outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

C*outt[1]
C*outt[3]
data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*outt[1])[1,:])
plot!((C*outt[3])[1,:])


plot(data_in_deviations[2,:])
plot!((C*outt[1])[2,:])
plot!((C*outt[3])[2,:])







𝐁 = B * B'

# Gaussian Prior

calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

Pstar1 = calculate_covariance_(sol)
# P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
u = zeros(length(observables_and_states))
# u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
z = C * u



using LinearAlgebra


nk = 1
d = 0
decomp = []
# spinf = size(Pinf1)
spstar = size(Pstar1)
v = zeros(size(C,1), size(data_in_deviations,2))
u = zeros(size(A,1), size(data_in_deviations,2)+1)
# û = zeros(size(A,1), size(data_in_deviations,2))
# uK = zeros(nk, size(A,1), size(data_in_deviations,2)+nk)
# PK = zeros(nk, size(A,1), size(A,1), size(data_in_deviations,2)+nk)
iF = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# Fstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFstar = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# iFinf = zeros(size(C,1), size(C,1), size(data_in_deviations,2))
# K = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
L = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Linf = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Lstar = zeros(size(A,1), size(A,1), size(data_in_deviations,2))
# Kstar = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
# Kinf = zeros(size(A,1), size(C,1), size(data_in_deviations,2))
P = zeros(size(A,1), size(A,1), size(data_in_deviations,2)+1)
# Pstar = zeros(spstar[1], spstar[2], size(data_in_deviations,2)+1)
# Pstar[:, :, 1] = Pstar1
# Pinf = zeros(spinf[1], spinf[2], size(data_in_deviations,2)+1)
# Pinf[:, :, 1] = Pinf1
rr = size(C,1)
# 𝐁 = R * Q * transpose(R)
ū = zeros(size(A,1), size(data_in_deviations,2))
ϵ̄ = zeros(rr, size(data_in_deviations,2))
ϵ = zeros(rr, size(data_in_deviations,2))
epsilonhat = zeros(rr, size(data_in_deviations,2))
r = zeros(size(A,1), size(data_in_deviations,2)+1)
Finf_singular = zeros(1, size(data_in_deviations,2))

V = []

# t = 0

# d = t
P[:, :, 1] = Pstar1
# iFinf = iFinf[:, :, 1:d]
# iFstar= iFstar[:, :, 1:d]
# Linf = Linf[:, :, 1:d]
# Lstar = Lstar[:, :, 1:d]
# Kstar = Kstar[:, :, 1:d]
# Pstar = Pstar[:, :, 1:d]
# Pinf = Pinf[:, :, 1:d]
# K
# û

for t in 1:size(data_in_deviations,2)
    v[:, t] = data_in_deviations[:, t] - C * u[:, t]
    F = C * P[:, :, t] * C'
    iF[:, :, t] = inv(F)
    PCiF = P[:, :, t] * C' * iF[:, :, t]
    û = u[:, t] + PCiF * v[:, t]
    K = A * PCiF
    L[:, :, t] = A - K * C
    P[:, :, t+1] = A * P[:, :, t] * L[:, :, t]' + 𝐁
    u[:, t+1] = A * û
    ϵ[:, t] = B' * C' * iF[:, :, t] * v[:, t]
    # Pf = P[:, :, t]
    # uK[1, :, t+1] = u[:, t+1]
    # for jnk in 1:nk
    #     Pf = A * Pf * A' + 𝐁
    #     PK[jnk, :, :, t+jnk] = Pf
    #     if jnk > 1
    #         uK[jnk, :, t+jnk] = A * uK[jnk-1, :, t+jnk-1]
    #     end
    # end
end

# backward pass; r_T and N_T, stored in entry (size(data_in_deviations,2)+1) were initialized at 0
# t = size(data_in_deviations,2) + 1
for t in size(data_in_deviations,2):-1:1
    r[:, t] = C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r[:, t+1] # compute r_{t-1}, DK (2012), eq. 4.38
    ū[:, t] = u[:, t] + P[:, :, t] * r[:, t] # DK (2012), eq. 4.35
    ϵ̄[:, t] = B' * r[:, t] # DK (2012), eq. 4.63
end

# if decomp_flag
    decomp = zeros(nk, size(A,1), rr, size(data_in_deviations,2)+nk)
    ZRQinv = inv(C * 𝐁 * C')
    for t in 1:size(data_in_deviations,2)
        # : = data_index[t]
        # calculate eta_tm1t
        eta_tm1t = B' * C' * iF[:, :, t] * v[:, t]
        AAA = P[:, :, t] * C' * ZRQinv[:, :] * (C * B .* eta_tm1t')
        # calculate decomposition
        decomp[1, :, :, t+1] = AAA
        for h = 2:nk
            AAA = A * AAA
            decomp[h, :, :, t+h] = AAA
        end
    end
# end

epsilonhat = data_in_deviations - C * ū





# outt = kalman_filter_and_smoother(data_in_deviations,A,𝐁,C,u,P)

# C*outt[1]
# C*outt[3]
# data_in_deviations

using StatsPlots

plot(data_in_deviations[1,:])
plot!((C*ū)[1,:])
plot!((C*u)[1,2:end])


plot(data_in_deviations[2,:])
plot!((C*ū)[2,:])
plot!((C*u)[2,2:end])

plot(ϵ̄[1,:])
plot(ϵ̄[2,:])

ϵ̄

plot(ϵ[1,:])
plot(ϵ[2,:])

ϵ