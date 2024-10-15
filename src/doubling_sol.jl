using Revise
using MacroModelling
using BenchmarkTools
import MacroModelling: get_NSSS_and_parameters, calculate_jacobian, calculate_quadratic_iteration_solution, calculate_linear_time_iteration_solution, timings, solve_quadratic_matrix_equation #, calculate_doubling_solution, riccati_forward, calculate_first_order_solution
import LinearAlgebra as ℒ
using SpeedMapping

using TimerOutputs
TimerOutputs.enable_debug_timings(MacroModelling)

# TimerOutputs.enable_debug_timings(Main)


include("../models/RBC_baseline.jl")

sims = simulate(RBC_baseline)

get_loglikelihood(RBC_baseline, sims[[1,6],:,1], RBC_baseline.parameter_values)

import Zygote, FiniteDifferences, ForwardDiff

@benchmark zygdiff = Zygote.gradient(x -> get_loglikelihood(RBC_baseline, sims[[1,6],:,1], x), RBC_baseline.parameter_values)[1]
@benchmark findiff = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x -> get_loglikelihood(RBC_baseline, sims[[1,6],:,1], x), RBC_baseline.parameter_values)[1]
@benchmark fordiff = ForwardDiff.gradient(x -> get_loglikelihood(RBC_baseline, sims[[1,6],:,1], x), RBC_baseline.parameter_values)


# include("../models/Smets_Wouters_2007.jl")


include("../models/NAWM_EAUS_2008.jl")

𝓂 = NAWM_EAUS_2008

parameter_values = 𝓂.parameter_values

SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameter_values, verbose = true)

∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂)# |> Matrix

T = 𝓂.timings

sol = calculate_first_order_solution(∇₁; T = T)

sol2 = calculate_first_order_solution(∇₁; T = T)

sol2 = calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling)

calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :linear_time_iteration, verbose = true)

calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :quadratic_iteration, verbose = true)

@profview for i in 1:10 calculate_first_order_solution(∇₁; T = T) end

@profview for i in 1:10 calculate_first_order_solution(∇₁; T = T) end

@profview for i in 1:100 calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling) end

@profview for i in 1:10 calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :linear_time_iteration) end

@benchmark calculate_first_order_solution(∇₁; T = T)

@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling)

@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :schur)

@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :linear_time_iteration)

@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :quadratic_iteration)


timer = TimerOutput()
for i in 1:100 calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :schur, timer = timer) end
timer

timer = TimerOutput()
for i in 1:100 calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling, timer = timer) end
timer

timer = TimerOutput()
for i in 1:10 calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :linear_time_iteration, timer = timer) end
timer


isapprox(sol[1], sol2[1], rtol = 1e-10)


soll,_ = riccati_forward(∇₁; T = T)
soll *=  expand[2]





dynIndex = T.nPresent_only+1:T.nVars

reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
sort!(comb)

future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)

A₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
A₀ = zeros(T.nVars, T.nVars)
A₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
M = similar(A₀)

∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]    
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]
∇ₑ = copy(∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end])

Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
# Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
Qinv = Q.Q'

ℒ.mul!(A₊, Qinv, ∇₊)
ℒ.mul!(A₀, Qinv, ∇₀)
ℒ.mul!(A₋, Qinv, ∇₋)

Ã₊ = @views A₊[dynIndex,:] * ℒ.I(length(comb))[future_not_past_and_mixed_in_comb,:]
Ã₀ = @views A₀[dynIndex, comb]
Ã₋ = @views A₋[dynIndex,:] * ℒ.I(length(comb))[past_not_future_and_mixed_in_comb,:]




A = Ã₊
B = Ã₀
C = Ã₋

B̂ =  ℒ.lu(B)

C̄ = B̂ \ C
Ā = B̂ \ A

X = similar(Ā)
X̄ = similar(Ā)

X² = similar(X)

sol = speedmapping(zero(A); m! = (X̄, X) -> begin 
                                            ℒ.mul!(X², X, X)
                                            ℒ.mul!(X̄, Ā, X²)
                                            ℒ.axpy!(1, C̄, X̄)
                                        end,
    tol = tol, maps_limit = 50000, σ_min = 1, stabilize = true, orders = [3,2])


X = -sol.minimizer

A * X * X + B * X + C

Ā * X * X + C̄ + X

reached_tol = ℒ.norm(A * X * X + B * X + C)

converged = reached_tol < tol



maxiter = 10000
tol = 1e-8

F = similar(C)
F² = similar(C)
t = similar(C)
ε = similar(C)

verbose = true

for i in 1:maxiter
    # copy!(t₁,B)
    # ℒ.mul!(t₁, A, F, 1, 1)
    ℒ.mul!(T, A, F)
    ℒ.axpby!(-1, B, -1, T)
    T̂ = ℒ.lu!(T, check = false)
    ℒ.ldiv!(F, T̂, C)

    ℒ.mul!(F², F, F)
    ℒ.mul!(ε, A, F²)
    ℒ.mul!(ε, B, F, 1, 1)
    ℒ.axpy!(1, C, ε)
    println("Residual norm: $(ℒ.norm(ε))")
    if ℒ.norm(ε) < tol 
        if verbose println("Converged in $i iterations to residual norm: $(ℒ.norm(ε))") end
        break 
    end
end

sol = speedmapping(zero(A); m! = (F̄, F) -> begin 
            ℒ.mul!(t, A, F)
            ℒ.axpby!(-1, B, -1, t)
            t̂ = ℒ.lu!(t, check = false)
            ℒ.ldiv!(F̄, t̂, C)
        end,
    tol = tol, maps_limit = 1000)

sol.minimizer

B̂ =  ℒ.lu(B)

AA = B̂ \ C
BB = B̂ \ A

X = similar(AA)
X̄ = similar(AA)

X² = similar(X)

sol = speedmapping(zero(A); m! = (X̄, X) -> begin 
                                            ℒ.mul!(X², X, X)
                                            ℒ.mul!(X̄, BB, X²)
                                            ℒ.axpy!(1, AA, X̄)
                                        end,
    tol = 1e-8, maps_limit = 1000)#, σ_min = 0.0, stabilize = false, orders = [3,3,2])






    # see Binder and Pesaran (1997) for more details on this approach
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
        tol = tol, maps_limit = 10000)


    C = -sol.minimizer

sol = solve_quadratic_matrix_equation(Ã₊, Ã₀, Ã₋, 
                                        Val(:doubling), 
                                        T, 
                                        # timer = timer,
                                        verbose = true)

D = sol[length(T.not_mixed_in_past_idx)+1:end,past_not_future_and_mixed_in_comb]
L = ℒ.I(length(past_not_future_and_mixed_in_comb))[:,T.not_mixed_in_past_idx] * sol[1:length(T.not_mixed_in_past_idx),past_not_future_and_mixed_in_comb]

LL - L
Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
A₋ᵤ  = A₋[1:T.nPresent_only,:]

Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

# A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
ℒ.mul!(A₋ᵤ, Ã₀ᵤ, sol[:,past_not_future_and_mixed_in_comb], 1, 1)
ℒ.mul!(nₚ₋, A₊ᵤ, D)
ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)








n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
n₀₀ = zeros(T.nVars, T.nVars)
n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)

∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
# Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
Qinv = Q.Q'

ℒ.mul!(n₀₊, Qinv, ∇₊)
ℒ.mul!(n₀₀, Qinv, ∇₀)
ℒ.mul!(n₀₋, Qinv, ∇₋)
A₊ = n₀₊
A₀ = n₀₀
A₋ = n₀₋

Ã₊  = A₊[dynIndex,:]
Ã₋  = A₋[dynIndex,:]
Ã₀₊ = A₀[dynIndex, T.future_not_past_and_mixed_idx]
@views ℒ.mul!(nₜₚ, A₀[dynIndex, T.past_not_future_idx], ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:])
Ã₀₋ = nₜₚ

Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]

Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
I₋ = ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))

ℒ.rmul!(Ã₋,-1)
ℒ.rmul!(Ã₀₊,-1)
E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))

# this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
schdcmp = ℒ.schur!(D, E)

eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

ℒ.ordschur!(schdcmp, eigenselect)

Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)

Ŝ₁₁ = ℒ.lu!(S₁₁, check = false)

# D      = Z₂₁ / Ẑ₁₁
ℒ.rdiv!(Z₂₁, Ẑ₁₁)
D = Z₂₁

# L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
ℒ.ldiv!(Ŝ₁₁, T₁₁)
ℒ.mul!(n₋₋, Z₁₁, T₁₁)
ℒ.rdiv!(n₋₋, Ẑ₁₁)
LL = n₋₋


sol = vcat(LL[T.not_mixed_in_past_idx,:], D)


# @benchmark calculate_first_order_solution(∇₁; T = T)

# @benchmark calculate_quadratic_iteration_solution(∇₁; T = T)

# @benchmark calculate_linear_time_iteration_solution(∇₁; T = T)

expand = @views [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:],
ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 


nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

A = ∇₊
B = ∇₀
C = ∇₋

# B = ∇₀[:,T.present_only_idx]

B̂ = ℒ.factorize(B)

# Compute initial values X0, Y0, E0, F0
E0 = B̂ \ C
F0 = B̂ \ A
X0 = -E0
Y0 = -F0

# Define the structure-preserving doubling algorithm

timer = TimerOutput()
for i in 1:100 calculate_doubling_solution(X0, Y0, E0, F0, timer = timer) end
timer

X_sol = calculate_doubling_solution2(X0, Y0, E0, F0)

check = A * X_sol^2 + B * X_sol + C;
println("Norm of the residual: $(ℒ.norm(check))")  # Should be close to zero

check = A * soll^2 + B * soll + C;
println("Norm of the residual: $(ℒ.norm(check))")  # Should be close to zero

# soll2 = calculate_linear_time_iteration_solution(∇₁; T = T)
# soll2 * expand[2]
check = A * soll^2 + B * soll + C;
println("Norm of the residual: $(ℒ.norm(check))")  # Should be close to zero

println("Norm of the residual: $(ℒ.norm(check) / ℒ.norm(C))")  # Should be close to zero

check = B \ A * soll^2 + soll + B \ C;

check = B \ A * X_sol^2 + X_sol + B \ C;

println("Norm of the residual: $(ℒ.norm(check) / ℒ.norm(X_sol))")  # Should be close to zero

ℒ.norm(soll-X_sol) / max(ℒ.norm(soll), ℒ.norm(X_sol))

ℒ.norm(soll2 - X_sol) / max(ℒ.norm(soll2), ℒ.norm(X_sol))

# Run the algorithm
@benchmark X_sol = calculate_doubling_solution(X0, Y0, E0, F0)
@profview for i in 1:100 X_sol = calculate_doubling_solution(X0, Y0, E0, F0) end

# Verify that the solution satisfies the original quadratic equation from 1.1
check = A * X_sol^2 + B * X_sol + C
println("Norm of the residual: $(ℒ.norm(check))")  # Should be close to zero





n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
n₀₀ = zeros(T.nVars, T.nVars)
n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)

∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
# Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
Qinv = Q.Q'

ℒ.mul!(n₀₊, Qinv, ∇₊)
ℒ.mul!(n₀₀, Qinv, ∇₀)
ℒ.mul!(n₀₋, Qinv, ∇₋)
A₊ = n₀₊
A₀ = n₀₀
A₋ = n₀₋

dynIndex = T.nPresent_only+1:T.nVars

Ã₊  = A₊[dynIndex,:]
Ã₋  = A₋[dynIndex,:]
Ã₀₊ = A₀[dynIndex, T.future_not_past_and_mixed_idx]
@views ℒ.mul!(nₜₚ, A₀[dynIndex, T.past_not_future_idx], ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:])
Ã₀₋ = nₜₚ


comb = union(T.past_not_future_and_mixed_idx, T.future_not_past_and_mixed_idx)

T.nPast_not_future_and_mixed


∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
# Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
Qinv = Q.Q'

ℒ.mul!(n₀₊, Qinv, ∇₊)
ℒ.mul!(n₀₀, Qinv, ∇₀)
ℒ.mul!(n₀₋, Qinv, ∇₋)
A₊ = n₀₊
A₀ = n₀₀
A₋ = n₀₋



T.future_not_past_and_mixed_idx

Ã₊  = hcat(A₊[dynIndex,:], zeros(T.nVars - T.nPresent_only, length(comb) - T.nFuture_not_past_and_mixed))
Ã₀ = A₀[dynIndex, sort(union(T.future_not_past_and_mixed_idx, T.past_not_future_idx))]
Ã₋  = hcat(A₋[dynIndex,:], zeros(T.nVars - T.nPresent_only, length(comb) - T.nPast_not_future_and_mixed))



comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
sort!(comb)
# T.past_not_future_and_mixed_idx
# ℒ.I(length(comb))[indices_past_not_future_and_mixed_in_comb,:]

indices_past_not_future_and_mixed_in_comb = findall(x -> x in T.past_not_future_and_mixed_idx, comb)


Ã₋ = A₋[dynIndex,:] * ℒ.I(length(comb))[indices_past_not_future_and_mixed_in_comb,:]

indices_future_not_past_and_mixed_in_comb = findall(x -> x in T.future_not_past_and_mixed_idx, comb)

Ã₊  = A₊[dynIndex,:] * ℒ.I(length(comb))[indices_future_not_past_and_mixed_in_comb,:]


# T.nPast_not_future_and_mixed
A = Ã₊
B = Ã₀
C = Ã₋

# B = ∇₀[:,T.present_only_idx]

B̂ = ℒ.factorize(B)

# Compute initial values X0, Y0, E0, F0
E0 = B̂ \ C
F0 = B̂ \ A
X0 = -E0
Y0 = -F0




@benchmark X_sol = calculate_doubling_solution2(X0, Y0, E0, F0)

check = A * X_sol^2 + B * X_sol + C
println("Norm of the residual: $(ℒ.norm(check))")  # Should be close to zero


isapprox(X_sol[:,indices_past_not_future_and_mixed_in_comb], sol[T.dynamic_order,:], rtol = 1e-10)
sparse(X_sol)

X_sol[:, vec(sum(abs, X_sol; dims=1) .> 0)]

Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]

Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
I₋ = ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))

ℒ.rmul!(Ã₋,-1)
ℒ.rmul!(Ã₀₊,-1)
E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))





X0 * Y0
Y0 * X0

function calculate_doubling_solution2(X0, Y0, E0, F0; 
    tol=eps(),
    timer::TimerOutput = TimerOutput(),
    max_iter=100)
    @timeit_debug timer "Setup buffers" begin

    X = copy(X0)
    Y = copy(Y0)
    E = copy(E0) 
    F = copy(F0)

    X_new = similar(X0)
    Y_new = similar(Y0)
    E_new = similar(E0) 
    F_new = similar(F0)
    
    temp1 = similar(Y)  # Temporary for intermediate operations
    # temp2 = similar(Y)  # Temporary for intermediate operations
    n = size(X, 1)
    II = ℒ.I(n)  # Temporary for identity matrix

    Xtol = 1.0
    Ytol = 1.0

    end # timeit_debug
    @timeit_debug timer "Loop" begin

    for i in 1:max_iter
        @timeit_debug timer "Compute EI" begin

        # Compute EI = I - Y * X
        # ℒ.rmul!(temp1, 0)
        # @simd for j in 1:n
        #     @inbounds temp1[j, j] = 1
        # end
        # copy!(temp1, II)
        # ℒ.mul!(temp1, Y, X, -1, 1)
        ℒ.mul!(temp1, Y, X)
        ℒ.axpby!(1, II, -1, temp1)

        end # timeit_debug
        @timeit_debug timer "Invert EI" begin

        fEI = ℒ.lu(temp1, check = false)

        end # timeit_debug
        @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        ℒ.ldiv!(temp1, fEI, E)
        ℒ.mul!(E_new, E, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute FI" begin
            
        # Compute FI = I - X * Y
        # ℒ.rmul!(temp1, 0)
        # @simd for j in 1:n
        #     @inbounds temp1[j, j] = 1
        # end
        copy!(temp1, II)
        ℒ.mul!(temp1, X, Y, -1, 1)
        # ℒ.mul!(temp1, X, Y)
        # ℒ.axpby!(1, II, -1, temp1)

        end # timeit_debug
        @timeit_debug timer "Invert FI" begin

        fFI = ℒ.lu(temp1, check = false)
        
        end # timeit_debug
        @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        ℒ.ldiv!(temp1, fFI, F)
        ℒ.mul!(F_new, F, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(X_new, X, E)
        ℒ.ldiv!(temp1, fFI, X_new)
        ℒ.mul!(X_new, F, temp1)
        if i > 0 Xtol = ℒ.norm(X_new) end
        ℒ.axpy!(1, X, X_new)

        end # timeit_debug
        @timeit_debug timer "Compute Y_new" begin

        # Compute Y_new = Y + E * EI * Y * F
        ℒ.mul!(Y_new, Y, F)
        ℒ.ldiv!(temp1, fEI, Y_new)
        ℒ.mul!(Y_new, E, temp1)
        if i > 0 Ytol = ℒ.norm(Y_new) end
        ℒ.axpy!(1, Y, Y_new)
        
        println("Iter: $i; xtol: $(ℒ.norm(X_new - X)); ytol: $(ℒ.norm(Y_new - Y))")

        # Check for convergence
        if Xtol < tol && Ytol < tol
            println("Converged in $i iterations.")
            break
        end

        end # timeit_debug
        @timeit_debug timer "Copy" begin

        # Update values for the next iteration
        copy!(X, X_new)
        copy!(Y, Y_new)
        copy!(E, E_new)
        copy!(F, F_new)
        end # timeit_debug
    end
    end # timeit_debug

    return X_new  # Converged to solution
end



X_sol = calculate_doubling_solution2(X0, Y0, E0, F0)

check = A * X_sol^2 + B * X_sol + C
println("Norm of the residual: $(ℒ.norm(check)/ℒ.norm(C))")  # Should be close to zero


sparse(X_sol)

∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂)# |> Matrix

T = 𝓂.timings

expand = @views [ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:],
ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

A = ∇₊
B = ∇₀
C = ∇₋

B̂ = ℒ.factorize(B)
# Compute initial values X0, Y0, E0, F0
E0 = B̂ \ C
F0 = B̂ \ A
X0 = -E0
Y0 = -F0




n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
n₀₀ = zeros(T.nVars, T.nVars)
n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)

∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
# Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
Qinv = Q.Q'

ℒ.mul!(n₀₊, Qinv, ∇₊)
ℒ.mul!(n₀₀, Qinv, ∇₀)
ℒ.mul!(n₀₋, Qinv, ∇₋)
A₊ = n₀₊
A₀ = n₀₀
A₋ = n₀₋

dynIndex = T.nPresent_only+1:T.nVars

Ã₊  = A₊[dynIndex,:]
Ã₋  = A₋[dynIndex,:]
Ã₀₊ = A₀[dynIndex, T.future_not_past_and_mixed_idx]
@views ℒ.mul!(nₜₚ, A₀[dynIndex, T.past_not_future_idx], ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:])
Ã₀₋ = nₜₚ

Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]

Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
I₋ = ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))

ℒ.rmul!(Ã₋,-1)
ℒ.rmul!(Ã₀₊,-1)
E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))

# this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
schdcmp = ℒ.schur!(D, E)



    eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

    ℒ.ordschur!(schdcmp, eigenselect)

    Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
    Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)

Ŝ₁₁ = ℒ.lu!(S₁₁, check = false)

# D      = Z₂₁ / Ẑ₁₁
ℒ.rdiv!(Z₂₁, Ẑ₁₁)
D = Z₂₁

# L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
ℒ.ldiv!(Ŝ₁₁, T₁₁)
ℒ.mul!(n₋₋, Z₁₁, T₁₁)
ℒ.rdiv!(n₋₋, Ẑ₁₁)
L = n₋₋

sol = vcat(L[T.not_mixed_in_past_idx,:], D)



function calculate_first_order_solution(∇₁::Matrix{Float64}; 
                                        T::timings, 
                                        quadratic_matrix_equation_solver::Symbol = :doubling,
                                        verbose::Bool = false,
                                        timer::TimerOutput = TimerOutput())::Tuple{Matrix{Float64}, Bool}
    @timeit_debug timer "Calculate 1st order solution" begin
    @timeit_debug timer "Quadratic matrix solution" begin

    n₀₊ = zeros(T.nVars, T.nFuture_not_past_and_mixed)
    n₀₀ = zeros(T.nVars, T.nVars)
    n₀₋ = zeros(T.nVars, T.nPast_not_future_and_mixed)
    n₋₋ = zeros(T.nPast_not_future_and_mixed, T.nPast_not_future_and_mixed)
    nₚ₋ = zeros(T.nPresent_only, T.nPast_not_future_and_mixed)
    nₜₚ = zeros(T.nVars - T.nPresent_only, T.nPast_not_future_and_mixed)

    ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
    ∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
    ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

    Q    = @views ℒ.factorize(∇₀[:,T.present_only_idx])
    # Q    = ℒ.qr!(∇₀[:,T.present_only_idx])
    Qinv = Q.Q'

    ℒ.mul!(n₀₊, Qinv, ∇₊)
    ℒ.mul!(n₀₀, Qinv, ∇₀)
    ℒ.mul!(n₀₋, Qinv, ∇₋)
    A₊ = n₀₊
    A₀ = n₀₀
    A₋ = n₀₋

    dynIndex = T.nPresent_only+1:T.nVars

    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    indices_future_not_past_and_mixed_in_comb = findall(x -> x in T.future_not_past_and_mixed_idx, comb)

    Ã₊  = A₊[dynIndex,:] * ℒ.I(length(comb))[indices_future_not_past_and_mixed_in_comb,:]

    Ã₀ = A₀[dynIndex, comb]

    indices_past_not_future_and_mixed_in_comb = findall(x -> x in T.past_not_future_and_mixed_idx, comb)

    Ã₋ = A₋[dynIndex,:] * ℒ.I(length(comb))[indices_past_not_future_and_mixed_in_comb,:]

    A = Ã₊
    B = Ã₀
    C = Ã₋

    sol = solve_quadratic_matrix_equation(A,B,C, Val(quadratic_matrix_equation_solver), T, verbose = verbose)

    Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
    A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
    Ã₀ᵤ  = A₀[1:T.nPresent_only, T.present_but_not_only_idx]
    A₋ᵤ  = A₋[1:T.nPresent_only,:]

    Ā̂₀ᵤ = ℒ.lu!(Ā₀ᵤ, check = false)

    reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    # A    = vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)
    if T.nPresent_only > 0
        ℒ.mul!(A₋ᵤ, Ã₀ᵤ, sol[:,indices_past_not_future_and_mixed_in_comb], 1, 1)
        ℒ.mul!(nₚ₋, A₊ᵤ, D)
        ℒ.mul!(A₋ᵤ, nₚ₋, L, 1, 1)
        ℒ.ldiv!(Ā̂₀ᵤ, A₋ᵤ)
        ℒ.rmul!(A₋ᵤ,-1)
    end
    A    = vcat(A₋ᵤ, sol[reverse_dynamic_order,indices_past_not_future_and_mixed_in_comb])

    end # timeit_debug
    @timeit_debug timer "Exogenous part solution" begin

    Jm = @view(ℒ.I(T.nVars)[T.past_not_future_and_mixed_idx,:])

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.I(T.nVars)[T.future_not_past_and_mixed_idx,:]
    ∇₀ = copy(∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)])
    ∇ₑ = copy(∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end])
    
    M = similar(∇₀)
    ℒ.mul!(M, A[T.reorder,:], Jm)
    ℒ.mul!(∇₀, ∇₊, M, 1, 1)
    C = ℒ.lu!(∇₀, check = false)
    # C = RF.lu!(∇₊ * A * Jm + ∇₀, check = false)
    
    if !ℒ.issuccess(C)
        return hcat(A[T.reorder,:], zeros(length(T.reorder),T.nExo)), false
    end
    
    ℒ.ldiv!(C, ∇ₑ)
    ℒ.rmul!(∇ₑ, -1)
    # B = -(C \ ∇ₑ) # otherwise Zygote doesnt diff it

    end # timeit_debug
    end # timeit_debug

    return hcat(A[T.reorder,:], ∇ₑ), true
end

AAA,_ = calculate_first_order_solution(∇₁; T = T)
AAAA,_ = calculate_first_order_solution(∇₁; T = T)


calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :schur)[1]
calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling)[1]
calculate_first_order_solution(∇₁; T = T)[1]

@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :schur)
@benchmark calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :doubling)
@benchmark calculate_first_order_solution(∇₁; T = T)
AAsol,_ = riccati_forward(∇₁; T = T)

ℒ.norm(AAAA - AAA)


timer = TimerOutput()
calculate_first_order_solution(∇₁; T = T, quadratic_matrix_equation_solver = :schur, timer = timer)
timer



function solve_quadratic_matrix_equation(A::Matrix{S}, 
                                        B::Matrix{S}, 
                                        C::Matrix{S}, 
                                        ::Val{:schur}, 
                                        T::timings; 
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false) where S
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    indices_future_not_past_and_mixed_in_comb = findall(x -> x in T.future_not_past_and_mixed_idx, comb)
    indices_past_not_future_and_mixed_in_comb = findall(x -> x in T.past_not_future_and_mixed_idx, comb)
    indices_past_not_future_in_comb = findall(x -> x in T.past_not_future_idx, comb)

    Ã₊ = @view A[:,indices_future_not_past_and_mixed_in_comb]
    
    Ã₋ = @view C[:,indices_past_not_future_and_mixed_in_comb]
    
    Ã₀₊ = @view B[:,indices_future_not_past_and_mixed_in_comb]

    Ã₀₋ = @views B[:,indices_past_not_future_in_comb] * ℒ.I(T.nPast_not_future_and_mixed)[T.not_mixed_in_past_idx,:]

    Z₊ = zeros(T.nMixed, T.nFuture_not_past_and_mixed)
    I₊ = ℒ.I(T.nFuture_not_past_and_mixed)[T.mixed_in_future_idx,:]
    
    Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
    I₋ = ℒ.I(T.nPast_not_future_and_mixed)[T.mixed_in_past_idx,:]
    
    D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
    
    ℒ.rmul!(Ã₋,-1)
    ℒ.rmul!(Ã₀₊,-1)
    E = vcat(hcat(Ã₋,Ã₀₊), hcat(Z₋, I₊))
    
    # this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
    schdcmp = ℒ.schur!(D, E)
    
    eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

    ℒ.ordschur!(schdcmp, eigenselect)

    Z₂₁ = schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
    Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
    T₁₁    = schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

    Ẑ₁₁ = ℒ.lu(Z₁₁, check = false)
    
    Ŝ₁₁ = ℒ.lu!(S₁₁, check = false)
    
    # D      = Z₂₁ / Ẑ₁₁
    ℒ.rdiv!(Z₂₁, Ẑ₁₁)
    D = Z₂₁
    
    # L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁
    ℒ.ldiv!(Ŝ₁₁, T₁₁)
    ℒ.mul!(n₋₋, Z₁₁, T₁₁)
    ℒ.rdiv!(n₋₋, Ẑ₁₁)
    L = n₋₋
    
    sol = vcat(L[T.not_mixed_in_past_idx,:], D)

    return sol[T.dynamic_order,:] * ℒ.I(length(comb))[indices_past_not_future_and_mixed_in_comb,:]
end


function solve_quadratic_matrix_equation(A::Matrix{S}, 
                                        B::Matrix{S}, 
                                        C::Matrix{S}, 
                                        ::Val{:doubling}, 
                                        T::timings;
                                        tol::AbstractFloat = eps(),
                                        timer::TimerOutput = TimerOutput(),
                                        verbose::Bool = false,
                                        max_iter::Int = 100) where S
    @timeit_debug timer "Prepare" begin
    B̂ = ℒ.factorize(B)

    # Compute initial values X, Y, E, F
    E = B̂ \ C
    F = B̂ \ A
    X = -E
    Y = -F

    X_new = similar(X)
    Y_new = similar(Y)
    E_new = similar(E) 
    F_new = similar(F)
    
    temp1 = similar(Y)  # Temporary for intermediate operations

    n = size(X, 1)
    II = ℒ.I(n)  # Temporary for identity matrix

    Xtol = 1.0
    Ytol = 1.0

    end # timeit_debug
    @timeit_debug timer "Loop" begin

    for i in 1:max_iter
        @timeit_debug timer "Compute EI" begin

        # Compute EI = I - Y * X
        ℒ.mul!(temp1, Y, X)
        ℒ.axpby!(1, II, -1, temp1)

        end # timeit_debug
        @timeit_debug timer "Invert EI" begin

        fEI = ℒ.lu(temp1, check = false)

        end # timeit_debug
        @timeit_debug timer "Compute E" begin

        # Compute E = E * EI * E
        ℒ.ldiv!(temp1, fEI, E)
        ℒ.mul!(E_new, E, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute FI" begin
            
        # Compute FI = I - X * Y
        copy!(temp1, II)
        ℒ.mul!(temp1, X, Y, -1, 1)

        end # timeit_debug
        @timeit_debug timer "Invert FI" begin

        fFI = ℒ.lu(temp1, check = false)
        
        end # timeit_debug
        @timeit_debug timer "Compute F" begin
        
        # Compute F = F * FI * F
        ℒ.ldiv!(temp1, fFI, F)
        ℒ.mul!(F_new, F, temp1)

        end # timeit_debug
        @timeit_debug timer "Compute X_new" begin
    
        # Compute X_new = X + F * FI * X * E
        ℒ.mul!(X_new, X, E)
        ℒ.ldiv!(temp1, fFI, X_new)
        ℒ.mul!(X_new, F, temp1)
        if i > 5 Xtol = ℒ.norm(X_new) end
        ℒ.axpy!(1, X, X_new)

        end # timeit_debug
        @timeit_debug timer "Compute Y_new" begin

        # Compute Y_new = Y + E * EI * Y * F
        ℒ.mul!(Y_new, Y, F)
        ℒ.ldiv!(temp1, fEI, Y_new)
        ℒ.mul!(Y_new, E, temp1)
        if i > 5 Ytol = ℒ.norm(Y_new) end
        ℒ.axpy!(1, Y, Y_new)
        
        # println("Iter: $i; xtol: $(ℒ.norm(X_new - X)); ytol: $(ℒ.norm(Y_new - Y))")

        # Check for convergence
        if Xtol < tol && Ytol < tol
            if verbose println("Converged in $i iterations.") end
            break
        end

        end # timeit_debug
        @timeit_debug timer "Copy" begin

        # Update values for the next iteration
        copy!(X, X_new)
        copy!(Y, Y_new)
        copy!(E, E_new)
        copy!(F, F_new)
        end # timeit_debug
    end
    end # timeit_debug

    return X_new  # Converged to solution 
end


res1 = solve_quadratic_matrix_equation(A,B,C, Val(:schur), T)

res2 = solve_quadratic_matrix_equation(A,B,C, Val(:doubling), T)

ℒ.norm(A*res1^2 + B*res1 + C) / ℒ.norm(C)
ℒ.norm(A*res2^2 + B*res2 + C) / ℒ.norm(C)

ℒ.norm(A*res1^2 + B*res1 + C)
ℒ.norm(A*res2^2 + B*res2 + C)


@benchmark solve_quadratic_matrix_equation(A,B,C, Val(:schur), T)

@benchmark solve_quadratic_matrix_equation(A,B,C, Val(:doubling), T)

res1 ≈ res2

timer = TimerOutput()
for i in 1:100 solve_quadratic_matrix_equation(A,B,C, Val(:doubling), T, timer = timer) end
timer

sol  =res2