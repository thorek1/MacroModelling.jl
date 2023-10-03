using MacroModelling, LowLevelParticleFilters, Distributions
import LinearAlgebra as ℒ

include("models/FS2000.jl")

simulations = get_simulation(m)

# (x,u,p,t) = (state, input, parameters, time)
# state_update₁ = function(state::Vector{Float64}, shock::Vector{Float64}) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end

# dynamics(state, input, parameters, time) = m.solution.perturbation.first_order.solution_matrix * [state[m.timings.past_not_future_and_mixed_idx]; input]
dynamics(state, input, parameters, time) = m.solution.perturbation.first_order.state_update(state,input)
measurement(state, input, parameters, time) = m.solution.perturbation.first_order.state_update(state,input)#ℒ.I * state

covariance = get_covariance(m) |> collect
# covariance = sparse(covariance)
# droptol!(covariance,eps())
# covariance = sparse(Symmetric(covariance))
# isposdef(covariance)
ukf = UnscentedKalmanFilter(dynamics, 
                            measurement, 
                            eye(m.timings.nVars), 
                            eye(m.timings.nVars), 
                            MvNormal(diag(covariance)), 
                            nu = m.timings.nExo, 
                            ny = m.timings.nVars)


x,u,y = LowLevelParticleFilters.simulate(ukf,10,MvNormal(ones(m.timings.nExo))) # Simuate trajectory using the model in the filter


using LinearAlgebra
covariance
isposdef((covariance+covariance')/2)
factorize(Symmetric(covariance))




using Statistics, LinearAlgebra, StaticArrays
m = randn(3)
S = randn(3,3)
S = S'S
xs = LowLevelParticleFilters.sigmapoints(m, S)
X = reduce(hcat, xs)
@test vec(mean(X, dims=2)) ≈ m
@test Statistics.cov(X, dims=2) ≈ S

m = [1,2]
S = [3. 1; 1 4]
xs = LowLevelParticleFilters.sigmapoints(m, S)
X = reduce(hcat, xs)
# @test vec(mean(X, dims=2)) ≈ m
# @test cov(X, dims=2) ≈ S




eye(n) = Matrix{Float64}(I,n,n)
mvnormal(d::Int, σ::Real) = MvNormal(LinearAlgebra.Diagonal(fill(float(σ) ^ 2, d)))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, float(σ) ^ 2 * I)

nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = mvnormal(randn(nx),2.0)   # Initial state Distribution
du = mvnormal(2,1) # Control input distribution

# Define random linenar state-space system
Tr = randn(nx,nx)
A = SA[0.99 0.1; 0 0.2]
B = @SMatrix randn(nx,nu)
C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x

# sss = SS(m,derivatives=false) |> collect
# I(m.timings.nVars)[m.timings.past_not_future_and_mixed_idx,:] * [sss...,0,0]

T    = 200 # Number of time steps
kf   = KalmanFilter(A, B, C, 0, eye(nx), eye(ny), d0)
ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(nx), eye(ny), d0; ny, nu)
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter

using SparseArrays, LinearOperators, BenchmarkTools
n = 20
p = 10
x = sprand(p,n^3,.1)
kk = sprand(n,n,.2) 
kkLinOp = kk' |> LinearOperator

@benchmark kron(kron(kron(kkLinOp, kkLinOp), kkLinOp), I(p)) * vec(x)
@benchmark x * kron(kron(kk,kk),kk)

sparse(reshape(kron(kron(kron(kk', kk'), kk'), I(p)) * vec(x),p,n^3))

findnz(sparse(reshape(kron(I(p),kron(kron(kk,kk),kk)) * vec(x),p,n^3)))

x * kron(kron(kk,kk),kk)
findnz(x * kron(kron(kk,kk),kk))
