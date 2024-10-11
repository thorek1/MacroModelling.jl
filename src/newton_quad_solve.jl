using MacroModelling
import LinearAlgebra as ℒ
using SpeedMapping
import ForwardDiff
using DifferentiationInterface
# using SparseConnectivityTracer: TracerSparsityDetector
# using SparseMatrixColorings
pwd()
include("../models/Smets_Wouters_2007.jl")

𝓂 = Smets_Wouters_2007
T = 𝓂.timings

SS_and_pars, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, verbose = verbose) : (𝓂.solution.non_stochastic_steady_state, (eps(), 0))

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)#|> Matrix

expand = @views [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:],
ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]


∇̂₀ =  ℒ.lu(∇₀)
    
A = ∇̂₀ \ ∇₋
B = ∇̂₀ \ ∇₊

C = similar(A)
C̄ = similar(A)

E = similar(C)

sol = speedmapping(zero(A); m! = (C̄, C) -> begin 
                                            ℒ.mul!(E, C, C)
                                            ℒ.mul!(C̄, B, E)
                                            ℒ.axpy!(1, A, C̄)
                                        end,
                                        # C̄ .=  A + B * C^2, 
    tol = eps(), maps_limit = 10000)


X = -sol.minimizer

ℒ.norm(∇₊ * X^2 + ∇₀ * X + ∇₋)
ℒ.norm(A + B * X^2 + X)
# sparse_first_order_backend = AutoSparse(
#     AutoForwardDiff();
#     sparsity_detector=TracerSparsityDetector(),
#     coloring_algorithm=GreedyColoringAlgorithm(),
# )


X = zeros(T.nVars, T.nVars)
for i in 1:100
    jac = DifferentiationInterface.jacobian(X -> ∇₊ * X^2 + ∇₀ * X + ∇₋, AutoForwardDiff(), X) |> sparse

    res = ∇₊ * X^2 + ∇₀ * X + ∇₋

    change = reshape(jac \ vec(res),size(X))

    println(ℒ.norm(change))

    X -= change
    if ℒ.norm(change) < 1e-11
        println("Converged in step $i")
        break 
    end
end


ℒ.norm(X)


X = zeros(T.nVars, T.nVars)
for i in 1:100
    jac = DifferentiationInterface.jacobian(X -> A + B * X^2 + X, AutoForwardDiff(), X) |> sparse

    res = A + B * X^2 + X

    change = reshape(jac \ vec(res),size(X))

    println(ℒ.norm(change))

    X -= change
    if ℒ.norm(change) < 1e-11
        println("Converged in step $i")
        break 
    end
end




X = zeros(T.nVars, T.nVars)

jac = DifferentiationInterface.jacobian(X -> ∇₊ * X^2 + ∇₀ * X + ∇₋, AutoForwardDiff(), X) |> sparse

res = ∇₊ * X^2 + ∇₀ * X + ∇₋

ΔX = reshape(jac \ vec(res),size(X))

∇₊ * ΔX * X + (∇₊ * X + ∇₀) * ΔX - (∇₊ * X^2 + ∇₀ * X + ∇₋) |> ℒ.norm

A * ΔX * B + C * ΔX - D
using MatrixEquations
using BenchmarkTools
# AXB +CXδ = E.


X = zeros(T.nVars, T.nVars)

ΔX = gsylv(∇₊, X, ∇₊ * X + ∇₀, 1, (∇₊ * X^2 + ∇₀ * X + ∇₋))

∇₊ * ΔX * X + (∇₊ * X + ∇₀) * ΔX - (∇₊ * X^2 + ∇₀ * X + ∇₋) |> ℒ.norm


@profview for i in 1:100 calculate_first_order_solution(∇₁, T = T) end
@benchmark calculate_first_order_solution(∇₁, T = T)
@benchmark calculate_linear_time_iteration_solution(∇₁, T = T)
@benchmark calculate_quadratic_iteration_solution(∇₁, T = T)

@benchmark begin
# @profview  begin
X = zeros(T.nVars, T.nVars)
X² = similar(X)
C = similar(X)
E = similar(X)

for i in 1:10
    copy!(C, ∇₀)
    ℒ.mul!(C, ∇₊, X, 1, 1)

    copy!(E, ∇₋)
    ℒ.mul!(E, ∇₀, X, 1, 1)
    ℒ.mul!(X², X, X)
    ℒ.mul!(E, ∇₊, X², 1, 1)

    ΔX = gsylv(∇₊, X, C, 1, E)

    ℒ.axpy!(-1, ΔX, X)

    if ℒ.norm(ΔX) < 1e-11
        println("Converged in step $i")
        break 
    end
end
end


∇₊ * ΔX * X + (∇₊ * X + ∇₀) * ΔX - (∇₊ * X^2 + ∇₀ * X + ∇₋)

import LinearOperators, Krylov

X = zeros(T.nVars, T.nVars)
tmp̄ = zero(X)
tmp̂ = zero(X)
ΔX = zero(X)
A = ∇₊
B = X
C = ∇₊ * X + ∇₀
E = (∇₊ * X^2 + ∇₀ * X + ∇₋)

function sylvester!(sol,𝐱)
    copyto!(ΔX, 𝐱)

    ℒ.mul!(tmp̄, A, ΔX)
    ℒ.mul!(tmp̂, tmp̄, B)

    ℒ.mul!(tmp̂, C, ΔX, 1, 1)

    copyto!(sol, tmp̂)
end

sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), false, false, sylvester!)

# @benchmark begin

Δx, info = Krylov.gmres(sylvester, [vec(E);], rtol = 1e-14, atol = 1e-14)

Δx = reshape(Δx, size(X))


# end

# sylvester!(vec(ΔX),vec(X))

∇₊ * Δx * X + (∇₊ * X + ∇₀) * Δx - (∇₊ * X^2 + ∇₀ * X + ∇₋) |> ℒ.norm

(∇₊ * X + ∇₀) \ ∇₊ * Δx * X + Δx - (∇₊ * X + ∇₀) \ (∇₊ * X^2 + ∇₀ * X + ∇₋) |> ℒ.norm


# A = ∇̂₀ \ ∇₋
# B = ∇̂₀ \ ∇₊

# C = similar(A)
# C̄ = similar(A)

# E = similar(C)

# sol = @suppress begin
#     speedmapping(zero(A); m! = (C̄, C) -> begin 
#                                             ℒ.mul!(E, C, C)
#                                             ℒ.mul!(C̄, B, E)
#                                             ℒ.axpy!(1, A, C̄)
#                                         end,
#                                         # C̄ .=  A + B * C^2, 
#     tol = tol, maps_limit = 10000)
# end

X = zeros(T.nVars, T.nVars)
# tmp̄ = zero(X)
# tmp̂ = zero(X)
import MacroModelling: solve_sylvester_equation, calculate_linear_time_iteration_solution, calculate_quadratic_iteration_solution
# ΔX = zero(X)
A = (∇₊ * X + ∇₀) \ ∇₊
B = X
C = (∇₊ * X + ∇₀) \ (∇₊ * X^2 + ∇₀ * X + ∇₋)

@benchmark begin
Δx,_ = solve_sylvester_equation(-A,B,C, verbose = true)
# Δx *= -1
end

∇₊ * Δx * X + (∇₊ * X + ∇₀) * Δx - (∇₊ * X^2 + ∇₀ * X + ∇₋) |> ℒ.norm

@benchmark begin
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
    Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])

    ℒ.ldiv!(Q, ∇₊)
    ℒ.ldiv!(Q, ∇₀)
    ℒ.ldiv!(Q, ∇₋)
end


n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
n₀₀ = zeros(T.nVars, T.nVars)
n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)

@benchmark begin
    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    # Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
    Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])

    Qinv = Q.Q'

    ℒ.mul!(n₀₊, Qinv, ∇₊)
    ℒ.mul!(n₀₀, Qinv, ∇₀)
    ℒ.mul!(n₀₋, Qinv, ∇₋)
end

    
# @benchmark begin
    @profview  begin
    X = zeros(T.nVars, T.nVars)
    X² = similar(X)
    A = similar(X)
    B = similar(X)
    C = similar(X)
    E = similar(X)
    
    for i in 1:100
        copy!(C, ∇₀)
        ℒ.mul!(C, ∇₊, X, 1, 1)
        Ĉ = ℒ.lu(C)
        ℒ.ldiv!(A, Ĉ, ∇₊)

        copy!(E, ∇₋)
        ℒ.mul!(E, ∇₀, X, 1, 1)
        ℒ.mul!(X², X, X)
        ℒ.mul!(E, ∇₊, X², -1, -1)
        ℒ.ldiv!(Ĉ, E)
    
        # ΔX = gsylv(∇₊, X, C, 1, E)
        ΔX = sylvd(A,X,E)
        # ΔX,_ = solve_sylvester_equation(-A,X,E, sylvester_algorithm = :sylvester, verbose = true)


        ℒ.axpy!(1, ΔX, X)
    
        if ℒ.norm(ΔX) < 1e-11
            # println("Converged in step $i")
            break 
        end
    end
    end

A = (∇₊ * X + ∇₀) \ ∇₊
B = X
C = (∇₊ * X + ∇₀) \ (∇₊ * X^2 + ∇₀ * X + ∇₋)

Δx,_ = solve_sylvester_equation(-A,B,C, verbose = true)




copyto!(ΔX, vec(X))

    ℒ.mul!(tmp̄, A, ΔX)
    ℒ.mul!(tmp̂, tmp̄, B)

    ℒ.mul!(tmp̂, C, ΔX, 1, 1)

    ℒ.axpy!(1, E, tmp̂)
    
    copyto!(sol, tmp̂)

ℒ.mul!(tmp̄, A, ΔX)
ℒ.mul!(sol, tmp̄, B)

ℒ.mul!(sol, C, ΔX, 1, 1)

ℒ.axpy!(1, E, sol)



X = zeros(T.nVars, T.nVars)

𝐀 = ∇₊ / (∇₊ * X^2 + ∇₀ * X + ∇₋)
𝐁 = X
𝐂 = ∇₀ + ∇₊ * X
𝐂¹ = copy(𝐂)
𝐃 = ∇₊ * X^2 + ∇₀ * X + ∇₋

for i in 1:100
    𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐂

    𝐀 = 𝐀^2
    𝐁 = 𝐁^2
    # 𝐃 = 𝐃^2

    # droptol!(𝐀, eps())
    # droptol!(𝐁, eps())

    if i > 10# && i % 2 == 0
        if isapprox(𝐂¹, 𝐂, rtol = 1e-12)
            iters = i
            break 
        end
    end
println(ℒ.norm(𝐂))
    𝐂 = 𝐂¹
end

𝐂¹ = 𝐀 * 𝐂 * 𝐁 + 𝐃 * 𝐂

ΔX = 𝐂¹
