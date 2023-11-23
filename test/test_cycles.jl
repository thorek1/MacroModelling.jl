using MacroModelling
import LinearAlgebra as ℒ
import RecursiveFactorization as RF

@model cycle_prototype begin
    μ[0] * λ[0] = Q[0] * e[1]^φₑ * λ[1]

    Q[0] = (1 + (1 - e[0]) * ϕ * Φ[0])

    # Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - e[ss]))^2 + Φ̄³ * (100 * (e[0] - e[ss]))^3)
    Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - ē))^2 + Φ̄³ * (100 * (e[0] - ē))^3)

    λ[0] = (Y[1] + (1 - δ - γ) / (1 - δ) * X[0] - (1 - δ - ψ) / (1 - δ) * γ * Y[0])^(-ω)

    X[1] = (1 - δ) * X[0] + ψ * Y[1]

    # Y[1] = z[0] * e[0]^α
    Y[1] = e[0]^α

    log(μ[0]) = ρμ * log(μ[-1]) + σμ * ϵμ[x]

    # log(z[0]) = ρz * log(z[-1]) + σz * ϵz[x]
end


@parameters cycle_prototype symbolic = true verbose = true begin
    δ   = 0.05
    α   = 0.67
    ē   = 0.943
    # e[ss] = 0.943 | ē
    e[ss] = 0.943 | Φ̄
    # Φ[ss] = 0.047 | Φ̄
    ω   = 0.2736
    γ   = 0.6259
    ψ   = 0.3905
    φₑ  = 0.046
    ϕ   = 0.9108
    # Φ̄   = 0.047
    Φ̄²  = 1.710280496#0.0018
    Φ̄³  = 186.8311838#0.00066

    # Φ̄²  = 0.0018
    # Φ̄³  = 0.00066

    ρz  = 0#0.6254
    σz  = 0#0.0027

    # ρz  = 0.6254
    # σz  = 0.0027

    ρμ  = 0.0671
    σμ  = 0.00014

    # .7 < e < 1
    # Φ < 1
    # Y < 1
    # X > 7.5
    # Q < .8
    # 1 > Φ > 0
    # 1 > ē > 0.6
    # X > 7.0
    # Y > 0.7
    # Q > 0.7
    # λ > 0.7
    # e > 0.7
end

# ψ   = 0.3905
# ē   = 0.943
# α   = 0.67
# δ   = 0.05

# ψ * ē ^ α / δ


SS(cycle_prototype)
# SS(cycle_prototype, parameters = :Φ̄² => .92)
# include("../models/RBC_baseline.jl")

get_solution(cycle_prototype)

𝓂 = cycle_prototype
verbose = true
parameters = 𝓂.parameter_values
T = 𝓂.timings


SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix
    

∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]


Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
Qinv = Q.Q'

A₊ = Qinv * ∇₊
A₀ = Qinv * ∇₀
A₋ = Qinv * ∇₋

dynIndex = T.nPresent_only+1:T.nVars

Ã₊  = @view A₊[dynIndex,:]
Ã₋  = @view A₋[dynIndex,:]
Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]

Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))
# this is the companion form and by itself the linearisation of the matrix polynomial used in the linear time iteration method. see: https://opus4.kobv.de/opus4-matheon/files/209/240.pdf
schdcmp = ℒ.schur(D,E)


##############
expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

A = [∇₊ zero(∇₊)
     zero(∇₊) ℒ.diagm(fill(1,size(∇₊,1)))]

B = [∇₀ ∇₋
     ℒ.diagm(fill(1,size(∇₊,1))) zero(∇₊) ]


schdcmp = ℒ.schur(A,B)

eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1
ℒ.ordschur!(schdcmp, eigenselect)

eigen(-schdcmp.Z[T.nVars+1:end, 1:T.nVars] \ schdcmp.Z[T.nVars+1:end, T.nVars+1:end])
abs.(eigenvalues)

# check eigenvals
eigenvalues = schdcmp.β ./ schdcmp.α

# inside unit circle
eigenvalue_inside_unit_circle = abs.(eigenvalues) .< 1

# real and > 1
eigenvalue_real_greater_one = isapprox.(imag.(eigenvalues), 0) .&& real.(eigenvalues) .> 1

# infinite
eigenvalue_infinite = abs.(eigenvalues) .> 1e10

eigenvalue_never_include = eigenvalue_infinite .|| eigenvalue_real_greater_one

ny = 𝓂.timings.nPast_not_future_and_mixed

other_eigenvalues = .!(eigenvalue_inside_unit_circle .|| eigenvalue_never_include)

ny - sum(eigenvalue_inside_unit_circle)



ℒ.ordschur!(schdcmp, .!eigenvalue_infinite)

# check eigenvals
eigenvalues = schdcmp.β ./ schdcmp.α

# inside unit circle
eigenvalue_inside_unit_circle = abs.(eigenvalues) .< 1

# real and > 1
eigenvalue_real_greater_one = isapprox.(imag.(eigenvalues), 0) .&& real.(eigenvalues) .> 1

# infinite
eigenvalue_infinite = abs.(eigenvalues) .> 1e10

eigenvalue_never_include = eigenvalue_infinite .|| eigenvalue_real_greater_one

ny = 𝓂.timings.nFuture_not_past_and_mixed

other_eigenvalues = .!(eigenvalue_inside_unit_circle .|| eigenvalue_never_include)

ny - sum(eigenvalue_inside_unit_circle)



ℒ.ordschur!(schdcmp, eigenvalue_inside_unit_circle)



eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1
eigenselect = BitVector([1,1,0,0,1,0])
ℒ.ordschur!(schdcmp, eigenselect)
schdcmp.β ./ schdcmp.α
(schdcmp.S[1:3,1:3]'  * schdcmp.T[1:3,1:3]) |> eigen

# J45

Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]


Ẑ₁₁ = RF.lu(Z₁₁, check = false)

if !ℒ.issuccess(Ẑ₁₁)
    return zeros(T.nVars,T.nPast_not_future_and_mixed), false
end
# end

Ŝ₁₁ = RF.lu(S₁₁, check = false)

if !ℒ.issuccess(Ŝ₁₁)
    return zeros(T.nVars,T.nPast_not_future_and_mixed), false
end

D      = Z₂₁ / Ẑ₁₁
L      = Z₁₁ * (Ŝ₁₁ \ T₁₁) / Ẑ₁₁

sol = @views vcat(L[T.not_mixed_in_past_idx,:], D)

Ā₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_only_idx]
A₊ᵤ  = @view A₊[1:T.nPresent_only,:]
Ã₀ᵤ  = @view A₀[1:T.nPresent_only, T.present_but_not_only_idx]
A₋ᵤ  = @view A₋[1:T.nPresent_only,:]

Ā̂₀ᵤ = RF.lu(Ā₀ᵤ, check = false)

if !ℒ.issuccess(Ā̂₀ᵤ)
    Ā̂₀ᵤ = ℒ.svd(collect(Ā₀ᵤ))
end

A    = @views vcat(-(Ā̂₀ᵤ \ (A₊ᵤ * D * L + Ã₀ᵤ * sol[T.dynamic_order,:] + A₋ᵤ)), sol)

@view(A[T.reorder,:])