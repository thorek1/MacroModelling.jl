using MacroModelling, StatsPlots
# import LinearAlgebra as ℒ
# import RecursiveFactorization as RF
# import MacroModelling: ParameterType, ℳ

include("models/RBC_CME.jl")

@model RBC_habit_invest_adjust begin
	λ²[0] = β * ((1 - δ) * λ²[1] + λ¹[1] * r[1])

	λ¹[0] * W[0] + (-1 + μ) * (1 - L[0])^(-μ) * (C[0] - h * C[-1])^μ * ((1 - L[0])^(1 - μ) * (C[0] - h * C[-1])^μ)^(-η) = 0

	-λ¹[0] + λ²[0] * (1 - 0.5 * φ * (-1 + I[-1]^-1 * I[0])^2 - φ * I[-1]^-1 * I[0] * (-1 + I[-1]^-1 * I[0])) + β * φ * I[0]^-2 * λ²[1] * I[1]^2 * (-1 + I[0]^-1 * I[1]) = 0

	-λ¹[0] - β * μ * h * (1 - L[1])^(1 - μ) * (C[1] - h * C[0])^(-1 + μ) * ((1 - L[1])^(1 - μ) * (C[1] - h * C[0])^μ)^(-η) + μ * (1 - L[0])^(1 - μ) * (C[0] - h * C[-1])^(-1 + μ) * ((1 - L[0])^(1 - μ) * (C[0] - h * C[-1])^μ)^(-η) = 0

	r[0] = α * Z[0] * K[-1]^(-1 + α) * L[0]^(1 - α)

	W[0] = Z[0] * (1 - α) * K[-1]^α * L[0]^(-α)

	Y[0] = Z[0] * K[-1]^α * L[0]^(1 - α)

	C[0] + I[0] = Y[0]

	K[0] = K[-1] * (1 - δ) + I[0] * (1 - φ / 2 * (1 - I[0] / I[-1])^2)

	Z[0] = exp(ϵᶻ[x] + σᶻ * log(Z[-1]))

	U[0] = β * U[1] + (1 - η)^-1 * ((1 - L[0])^(1 - μ) * (C[0] - h * C[-1])^μ)^(1 - η)
end

@parameters RBC_habit_invest_adjust begin
	σᶻ = 0.066
	0.36 * Y[ss] = r[ss] * K[ss] | α
	β = 0.99
	δ = 0.025
	η = 2
	# μ = 0.3
	h = 0.57
	ϕ = 0.95
	φ = 6.771

	# σᵍ
	# ḡ | ḡ = g_y * y[ss]

    # δ = i_y / k_y

    # β = 1 / (α / k_y + (1 - δ))

	μ | L[ss] = 1/3
end

SS(RBC_habit_invest_adjust)

get_eigenvalues(RBC_habit_invest_adjust)

using StatsPlots
plot_irf(RBC_habit_invest_adjust, parameters = :φ => 5, algorithm = :second_order)
plot_solution(RBC_habit_invest_adjust, :I, parameters = :φ => 4.2, algorithm = :second_order)
plot_solution(RBC_habit_invest_adjust, :I, parameters = (:φ => 5., :h => .6), algorithm = :second_order)

plot_irf(RBC_habit_invest_adjust, parameters = (:φ => 5.2, :h => .6), algorithm = :second_order)


get_eigenvalues(RBC_habit_invest_adjust, parameters = (:φ => 100., :h => .9900))



# plot_solution(m, :k, algorithm = :pruned_second_order, σ = 10)
# mn = get_mean(m, derivatives = false)
# SS(m, derivatives = false)
# SSS(m, derivatives = false, algorithm = :pruned_second_order)
mn = get_mean(m, derivatives = false, algorithm = :pruned_second_order)
# plot_solution(m, :k, algorithm = :pruned_second_order, σ = 1, initial_state = :nsss, parameters = :std_eps => .1)
# plot_solution(m, :k, algorithm = :pruned_second_order, σ = 1, initial_state = :sss, parameters = :std_eps => .1)
# plot_solution(m, :k, algorithm = :pruned_second_order, σ = 1, initial_state = :mean, parameters = :std_eps => .1)
plot_solution(m, :k, algorithm = :pruned_second_order, σ = 1, initial_state = collect(mn), parameters = :std_eps => .1)


plot_solution(m, :k, algorithm = :pruned_second_order, σ = 1, initial_state = collect(.9*mn), parameters = :std_eps => .1)

plot_solution(m, :k, algorithm = [:pruned_second_order, :pruned_third_order])#, initial_state = :NSSS)
plot_irf(m, algorithm = :pruned_second_order, shocks = :eps_z)
plot_irf(m, algorithm = :pruned_second_order, shocks = :eps_z, initial_state = :NSSS)
plot_irf(m, algorithm = :pruned_second_order, shocks = :eps_z, initial_state = :mean)

get_irf(m, shocks = :none)
get_irf(m)
get_irf(m, levels = false)


plot_irf(m)
plot_irf(m, algorithm = :second_order, initial_state = :NSSS, parameters = :std_eps => .1)
plot_irf(m, algorithm = :second_order, initial_state = :SSS, parameters = :std_eps => .1)
get_irf(m, shocks = :none)
get_irf(m, algorithm = :second_order)
get_irf(m, algorithm = :second_order, initial_state = :NSSS)
get_irf(m, algorithm = :second_order, initial_state = :SSS)
SS(m)
SSS(m)
1

# 𝑺₁ = RBC.solution.perturbation.first_order.solution_matrix
# T = RBC.timings
# 𝑺₁[:,1:T.nPast_not_future_and_mixed]

# S1 = zeros(T.nVars,T.nVars)

# S1[:,T.past_not_future_and_mixed_idx] = 𝑺₁[:,1:T.nPast_not_future_and_mixed]
# import LinearAlgebra as ℒ
# eigen(S1)


# get_eigenvalues(RBC)
# 𝓂 = RBC

# function get_eigenvalues(𝓂::ℳ;
#                         parameters::ParameterType = nothing,
#                         verbose::Bool = false,
#                         tol::AbstractFloat = eps())
#     solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

#     SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)
        
#     if solution_error > tol
#         @warn "Could not find non-stochastic steady state."
#     end

#     ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

#     T = 𝓂.timings

#     ∇₊ = @view ∇₁[:,1:T.nFuture_not_past_and_mixed]
#     ∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1, T.nVars)]
#     ∇₋ = @view ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1, T.nPast_not_future_and_mixed)]

#     Q    = ℒ.qr(collect(∇₀[:,T.present_only_idx]))
#     Qinv = Q.Q'

#     A₊ = Qinv * ∇₊
#     A₀ = Qinv * ∇₀
#     A₋ = Qinv * ∇₋

#     dynIndex = T.nPresent_only+1:T.nVars

#     Ã₊  = @view A₊[dynIndex,:]
#     Ã₋  = @view A₋[dynIndex,:]
#     Ã₀₊ = @view A₀[dynIndex, T.future_not_past_and_mixed_idx]
#     Ã₀₋ = @views A₀[dynIndex, T.past_not_future_idx] * ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.not_mixed_in_past_idx,:]

#     Z₊ = zeros(T.nMixed,T.nFuture_not_past_and_mixed)
#     I₊ = @view ℒ.diagm(ones(T.nFuture_not_past_and_mixed))[T.mixed_in_future_idx,:]

#     Z₋ = zeros(T.nMixed,T.nPast_not_future_and_mixed)
#     I₋ = @view ℒ.diagm(ones(T.nPast_not_future_and_mixed))[T.mixed_in_past_idx,:]

#     D = vcat(hcat(Ã₀₋, Ã₊), hcat(I₋, Z₊))
#     E = vcat(hcat(-Ã₋,-Ã₀₊), hcat(Z₋, I₊))

#     eigvals = ℒ.eigen(E,D).values
    
#     return KeyedArray(hcat(reim(eigvals)...); Eigenvalue = 1:length(eigs[1]), Parts = [:Real,:Imaginary])
# end

eigs = get_eigenvalues(m)


KeyedArray(hcat(eigs...); Eigenvalue = 1:length(eigs[1]), Parts = [:Real,:Imaginary])

@model reduced_form begin
    K[0] = (1 - δ) * K[-1] + I[-1]
    I[0] = α * K[-1] + G[0]
    G[0] = a * I[-1] ^ 3 + b * I[-1]
end

# irregular limit cycle
@parameters reduced_form begin
    α = .15
    δ = .2
    a = 100
    b = -2
end

# irregular limit cycle
# @parameters reduced_form begin
#     α = .2
#     δ = .2
#     a = -100
#     b = 1.3
# end

# limit cycle
# @parameters reduced_form begin
#     α = .2
#     δ = .2
#     a = 1000
#     b = -.97
# end

SS(reduced_form)
get_solution(reduced_form)
get_solution(reduced_form, algorithm = :second_order)#, parameters = :b => -.0)
get_solution(reduced_form, algorithm = :third_order)#, parameters = :a => .15)

get_parameters(reduced_form, values = true)

plot_irf(reduced_form, initial_state = fill(1e-4, 3), periods = 1000)

plot_irf(reduced_form, initial_state = fill(1e-4, 3), periods = 1000, algorithm = :second_order)

plot_irf(reduced_form, initial_state = fill(1e-4, 3), periods = 1000, algorithm = :third_order)





@model reduced_form_stochastic begin
    K[0] = (1 - δ) * K[-1] + I[-1]
    I[0] = α * K[-1] + G[0]
    G[0] = a * I[-1] ^ 3 + b * I[-1] + σ * ϵ[x]
end

# irregular limit cycle
@parameters reduced_form_stochastic begin
    α = .15
    δ = .2
    σ = .0001
    a = 100
    b = -2
end


SS(reduced_form_stochastic)
get_solution(reduced_form_stochastic)
get_solution(reduced_form_stochastic, algorithm = :second_order)#, parameters = :b => -.0)
get_solution(reduced_form_stochastic, algorithm = :third_order)#, parameters = :a => .15)

get_parameters(reduced_form_stochastic, values = true)

plot_irf(reduced_form_stochastic,  periods = 100)

plot_irf(reduced_form_stochastic,  periods = 1000, algorithm = :second_order)

plot_irf(reduced_form_stochastic, periods = 1000, algorithm = :third_order)
plot_irf(reduced_form_stochastic, algorithm = :third_order)

plot_irf(reduced_form_stochastic, algorithm = :third_order, periods = 100)
plot_irf(reduced_form_stochastic, algorithm = :third_order, periods = 100, negative_shock = true)



plot_irf(reduced_form_stochastic,  periods = 100, algorithm = :linear_time_iteration)

plot_irf(reduced_form_stochastic, algorithm = :pruned_second_order, periods = 100)

plot_irf(reduced_form_stochastic, algorithm = :pruned_third_order, periods = 100)








@model reduced_form_forward_looking_stochastic begin
    K[0] = (1 - δ) * K[-1] + I[0]
    I[0] = α * K[-1] + G[0]
    G[0] = a * I[1] ^ 3 + b * I[1] + σ * ϵ[x]
end

# irregular limit cycle
@parameters reduced_form_forward_looking_stochastic begin
    α = .15
    δ = .02
    σ = .0001
    a = 10
    b = .5
end

SS(reduced_form_forward_looking_stochastic)
get_solution(reduced_form_forward_looking_stochastic)

get_eigenvalues(reduced_form_forward_looking_stochastic)

plot_irf(reduced_form_forward_looking_stochastic, algorithm = :third_order)#, parameters = :b => .5)

get_solution(reduced_form_forward_looking_stochastic, algorithm = :second_order)#, parameters = :b => -.0)
get_solution(reduced_form_forward_looking_stochastic, algorithm = :third_order)#, parameters = :a => .15)

get_parameters(reduced_form_forward_looking_stochastic, values = true)

plot_irf(reduced_form_forward_looking_stochastic,  periods = 100)

plot_irf(reduced_form_forward_looking_stochastic,  periods = 1000, algorithm = :second_order)

plot_irf(reduced_form_forward_looking_stochastic, periods = 1000, algorithm = :third_order)
plot_irf(reduced_form_forward_looking_stochastic, algorithm = :third_order)

plot_irf(reduced_form_forward_looking_stochastic, algorithm = :third_order, periods = 100)
plot_irf(reduced_form_forward_looking_stochastic, algorithm = :third_order, periods = 100, negative_shock = true)



plot_irf(reduced_form_forward_looking_stochastic,  periods = 100, algorithm = :linear_time_iteration)

plot_irf(reduced_form_forward_looking_stochastic, algorithm = :pruned_second_order, periods = 100)

plot_irf(reduced_form_forward_looking_stochastic, algorithm = :pruned_third_order, periods = 100)



function get_eigenvalues(𝓂)
# 𝓂 = reduced_form_forward_looking_stochastic#cycle_prototype
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


    # check eigenvals
    eigenvalues = schdcmp.β ./ schdcmp.α
end





@model larger_forward_looking_stochastic begin
    K[0] = (1 - δ) * K[-1] + I[0]
    I[0] = α * K[-1] + G[0]
    G[0] = a * I[1] ^ 3 + b * I[1] + gg[0]
    gg[0] = ρ * gg[-1] + σ * ϵ[x]
end

# irregular limit cycle
@parameters larger_forward_looking_stochastic begin
    α = .15
    δ = .02
    σ = .0001
    ρ = .1
    a = 10
    b = .5
end


SS(larger_forward_looking_stochastic)
get_solution(larger_forward_looking_stochastic)

get_eigenvalues(larger_forward_looking_stochastic)

plot_irf(larger_forward_looking_stochastic, algorithm = :first_order)

plot_irf(larger_forward_looking_stochastic, algorithm = :second_order)

plot_irf(larger_forward_looking_stochastic, algorithm = :third_order)

plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 100)
plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 1000)
plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 10000)






@model m begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    I[0] = k[0] - (1-delta*z_delta[0])*k[-1]
    A[0]*k[-1]^alpha=c[0]+I[0] + G[0]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
    G[0] = a * I[1] ^ 3 + b * I[1] + g[0]
    g[0] = ρ * g[-1] + σ * ϵ[x]
end


@parameters m verbose = true begin
    σ = .0001
    ρ = .1
    a = 1
    b = .5

    alpha = .157

    beta = .999

    delta = .0226

    Pibar = 1.0008

    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end




SS(m)

get_solution(m)
get_eigenvalues(m)


get_solution(m, parameters = [:b => -.1, :a => 4])

get_eigenvalues(m)


plot_irf(m, algorithm = :first_order)

plot_irf(m, algorithm = :second_order)

plot_irf(m, algorithm = :third_order)




@model larger_more_forward_looking_stochastic begin
    1 / C[0] = β / C[1] * (α * K[0]^(α - 1) + (1 - δ)) 
    K[0] = (1 - δ) * K[-1] + I[0]
    I[0] + C[0] = K[-1]^α + G[1]
    G[0] = a * I[0] ^ 3 + b * I[0] + g[0]
    g[0] = ρ * g[-1] + σ * ϵ[x]
end


# irregular limit cycle
@parameters larger_more_forward_looking_stochastic begin
    α = .25
    β = .95
    δ = .025
    σ = .0001
    ρ = .5
    a = -12
    b = -.5
end


SS(larger_more_forward_looking_stochastic)

get_solution(larger_more_forward_looking_stochastic)
get_eigenvalues(larger_more_forward_looking_stochastic)


get_solution(larger_more_forward_looking_stochastic, parameters = [:b => -10, :a => 40, :β => .96])



plot_irf(larger_more_forward_looking_stochastic, algorithm = :first_order)

plot_irf(larger_more_forward_looking_stochastic, algorithm = :second_order)

plot_irf(larger_more_forward_looking_stochastic, algorithm = :third_order)

plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 100)
plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 1000)
plot_irf(larger_forward_looking_stochastic, algorithm = :third_order, periods = 10000)



𝓂 = reduced_form#cycle_prototype
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

remaining_eigenvalues = ny - sum(eigenvalue_inside_unit_circle)





unique_other_eigenvalues = unique(Float32.([real.(eigenvalues[other_eigenvalues]) abs.(imag.(eigenvalues[other_eigenvalues]))]), dims=1)

number_of_unique_other_eigenvalues = size(unique_other_eigenvalues,1)

eigenvalues


eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1
eigenselect = BitVector([1,0,1,0,0,0])
ℒ.ordschur!(schdcmp, eigenselect)


# check eigenvals
eigenvalues = schdcmp.β ./ schdcmp.α





eigenselect = abs.(schdcmp.β ./ schdcmp.α) .< 1

ℒ.ordschur!(schdcmp, BitVector([0,1]))
# reordering is irrelevant if there are no forward looking variables
Z₂₁ = @view schdcmp.Z[T.nPast_not_future_and_mixed+1:end, 1:T.nPast_not_future_and_mixed]
Z₁₁ = @view schdcmp.Z[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]

S₁₁    = @view schdcmp.S[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]
T₁₁    = @view schdcmp.T[1:T.nPast_not_future_and_mixed, 1:T.nPast_not_future_and_mixed]


Ẑ₁₁ = RF.lu(Z₁₁, check = false)

if !ℒ.issuccess(Ẑ₁₁)
    return zeros(T.nVars,T.nPast_not_future_and_mixed), false
end

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

A = @view(A[T.reorder,:])


Jm = @view(ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:])
    
∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:]
∇₀ = @view ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇ₑ = @view ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

B = -((∇₊ * A * Jm + ∇₀) \ ∇ₑ)

𝐒₁ = hcat(A, B)
𝓂.solution.perturbation.first_order.solution_matrix
∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = eps())

∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)
        
𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 𝓂.solution.perturbation.second_order_auxilliary_matrices, 𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = eps())


𝐒₁ = [𝐒₁[:,1:𝓂.timings.nPast_not_future_and_mixed] zeros(𝓂.timings.nVars) 𝐒₁[:,𝓂.timings.nPast_not_future_and_mixed+1:end]]


state = ones(𝓂.timings.nVars) * 1e-6
shock = zeros(𝓂.timings.nExo)
# state[2] = -state[2] * 2.5
aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
1
shock]

sss = 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6

n_sims = 10000

aug_states = zeros(length(aug_state), n_sims)

for i in 1:n_sims
    aug_state = [sss[𝓂.timings.past_not_future_and_mixed_idx]
                                            1
                                            shock]
    aug_states[:,i] = aug_state
    sss = 𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 + 𝐒₃ * ℒ.kron(ℒ.kron(aug_state,aug_state),aug_state) / 6
end

aug_states'





using StatsPlots

StatsPlots.plot(aug_states[1:2,500:550]')
StatsPlots.plot(aug_states[1:2,1000:1100]')

StatsPlots.plot(aug_states[1:2,:]')




StatsPlots.plot(randn(10000))


using StatsBase

mean(aug_states[1:2,:], dims = 2)
Statistics.std(aug_states[1:2,:], dims = 2)
StatsBase.skewness(aug_states[1,:])
StatsBase.skewness(aug_states[2,:])





@model cycle_prototype begin
    μ[0] * λ[0] = Q[0] * e[1]^φₑ * λ[1]
    # μ[-1] * λ[-1] = Q[-1] * e[0]^φₑ * λ[0]

    Q[0] = (1 + (1 - e[0]) * ϕ * Φ[0])

    # Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - e[ss]))^2 + Φ̄³ * (100 * (e[0] - e[ss]))^3)
    Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - ē))^2 + Φ̄³ * (100 * (e[0] - ē))^3)

    λ[0] = (Y[1] + (1 - δ - γ) / (1 - δ) * X[0] - (1 - δ - ψ) / (1 - δ) * γ * Y[0])^(-ω)

    # X[1] = (1 - δ) * X[0] + ψ * Y[1]
    X[0] = (1 - δ) * X[-1] + ψ * Y[0]

    # Y[1] = z[0] * e[0]^α
    Y[1] = e[0]^α
    # Y[0] = e[-1]^α

    log(μ[0]) = ρμ * log(μ[-1]) + σμ * ϵμ[x]
    # log(μ[1]) = ρμ * log(μ[0]) + σμ * ϵμ[x]

    # log(z[0]) = ρz * log(z[-1]) + σz * ϵz[x]
end



@parameters cycle_prototype symbolic = false verbose = true begin
    δ   = 0.05
    α   = 0.67
    ē   = 0.942955411540974
    # e[ss] = 0.942955411540974 | ē
    e[ss] = ē | Φ̄
    # Φ[ss] = 0.0469853516451966 | Φ̄
    ω   = 0.273610828663897
    γ   = 0.625910502827912
    ψ   = 0.390475455756289
    φₑ  = 0.0460159463504044
    ϕ   = 0.910774708002035

    # Φ̄   = 0.0469853516451966

    Φ̄²  = 1.71028049606731#0.0018
    Φ̄³  = 186.831183827810#0.00066
    # Φ̄²  = 0.0018
    # Φ̄³  = 0.00066

    # Φ̄²  = 0.0018
    # Φ̄³  = 0.00066

    ρz  = 0#0.6254
    σz  = 0#0.0027

    # ρz  = 0.6254
    # σz  = 0.0027

    ρμ  = 0.0671239825332901
    σμ  = 0.000135769197101003

end


SS(cycle_prototype)

get_solution(cycle_prototype)
get_solution(cycle_prototype, algorithm = :linear_time_iteration)

get_solution(cycle_prototype, algorithm = :third_order)





state, converged = third_order_stochastic_steady_state_iterative_solution([sparsevec(𝐒₁); vec(𝐒₂); vec(𝐒₃)]; dims = [size(𝐒₁); size(𝐒₂); size(𝐒₃)], 𝓂 = 𝓂)


elements_per_cluster = zeros(Int, nC)
for j = 1:nC
    nEC[j] = count(==(j), CtoE)
end




using StatsPlots
plot_irf(cycle_prototype)
plot_irf(cycle_prototype, algorithm = :linear_time_iteration)
plot_irf(cycle_prototype, algorithm = :third_order)


@model cycle_prototype begin
    μ[0] * λ[0] = Q[0] * e[1]^φₑ * λ[1]

    Q[0] = (1 + (1 - e[0]) * ϕ * Φ[0])

    # Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - e[ss]))^2 + Φ̄³ * (100 * (e[0] - e[ss]))^3)
    Φ[0] = Φ̄ * exp(Φ̄² * (100 * (e[0] - ē))^2 + Φ̄³ * (100 * (e[0] - ē))^3)

    λ[0] = (Y[1] + (1 - δ - γ) / (1 - δ) * X[0] - (1 - δ - ψ) / (1 - δ) * γ * Y[0])^(-ω)

    # X[1] = (1 - δ) * X[0] + ψ * Y[1]
    X[0] = (1 - δ) * X[-1] + ψ * Y[0]

    # Y[1] = z[0] * e[0]^α
    # Y[1] = e[0]^α
    Y[0] = e[-1]^α

    log(μ[0]) = ρμ * log(μ[-1]) + σμ * ϵμ[x]

    # log(z[0]) = ρz * log(z[-1]) + σz * ϵz[x]
end


@parameters cycle_prototype symbolic = false verbose = true begin
    δ   = 0.05
    α   = 0.67
    ē   = 0.942955411540974
    # e[ss] = 0.942955411540974 | ē
    e[ss] = ē | Φ̄
    # Φ[ss] = 0.0469853516451966 | Φ̄² #Φ̄
    ω   = 0.273610828663897
    γ   = 0.625910502827912
    ψ   = 0.390475455756289
    φₑ  = 0.0460159463504044
    ϕ   = 0.910774708002035
    # Φ̄   = 0.0469853516451966
    Φ̄²  = 1.71028049606731#0.0018
    Φ̄³  = 186.831183827810#0.00066

    # Φ̄²  = 0.0018
    # Φ̄³  = 0.00066

    ρz  = 0#0.6254
    σz  = 0#0.0027

    # ρz  = 0.6254
    # σz  = 0.0027

    ρμ  = 0.0671239825332901
    σμ  = 0.000135769197101003

    # .7 < e < 1
    # Φ < 1
    # Y < 1
    # X > 7.5
    # Q < .8
    # 1 > Φ > 0
    # .943 > ē > 0.942
    # X > 7.0
    # Y > 0.7
    # Q > 0.7
    # λ > 0.7
    # e > 0.7
    # Φ̄ > 0.04
end


SS(cycle_prototype)


# al = in1(2,:);
# del = in1(1,:);
# e_ = in1(3,:);
# gam = in1(5,:);
# om = in1(4,:);
# phie = in1(7,:);
# psi = in1(6,:);
# t2 = e_.^al;
# t3 = 1.0./del;
# t4 = gam-1.0;
# t5 = -om;
# t6 = psi.*t2.*t3;
# t7 = t2+t6;
# t8 = t4.*t7;
# t9 = -t8;
# t10 = t9.^t5;

# argSS = [psi*e_^al*1.0./del;
# e_^al;
# (-(gam-1.0)*(e_^al+psi*e_^al./del))^(-om);
# e_^phie*(-(gam-1.0)*(e_^al+psi*e_^al./del))^(-om);
# (-(gam-1.0)*(e_^al+psi*e_^al./del))];

# Y = e_^al
# X = psi*Y/del

# argSS = [psi*Y/del;
# Y;
# (-(gam-1.0)*(Y+X))^(-om);
# e_^phie*(-(gam-1.0)*(Y+X))^(-om);
# (-(gam-1.0)*(Y+X))];


# 0.942955411540974^0.0460159463504044 * ((1-.625910502827912) * 7.50815) ^(-.273610828663897)


# del     =   0.0500000000000000
# al      =   0.670000000000000
# e_      =   0.942955411540974
# om      =   0.273610828663897
# gam     =   0.625910502827912
# psi     =   0.390475455756289
# phie    =   0.0460159463504044
# phi0    =   0.910774708002035
# Phi0    =   0.0469853516451966
# Phi2    =   1.71028049606731
# Phi3    =   186.831183827810
# rhoz    =   0
# sigz    =   0
# rhomu   =   0.0671239825332901
# sigmu   =   0.000135769197101003



# argSS = [psi*e_^al*1.0./del;
#         e_^al;
#         (-(gam-1.0)*(e_^al+psi*e_^al./del))^(-om);
#         e_^phie*(-(gam-1.0)*(e_^al+psi*e_^al./del))^(-om);
#         (-(gam-1.0)*(e_^al+psi*e_^al./del))];



# SS
# 7.50814776972948  # :X
# 0.961410974626739 # :Y
# 0.729400010068976 # :λ
# 0.727431245459093
# 3.16837297194568

# # ψ   = 0.3905
# # ē   = 0.943
# # α   = 0.67
# # δ   = 0.05

# # ψ * ē ^ α / δ


# # SS(cycle_prototype, parameters = :Φ̄² => .92)
# # include("../models/RBC_baseline.jl")

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
# expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
# ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

# ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
# ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
# ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
# ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

# A = [∇₊ zero(∇₊)
#      zero(∇₊) ℒ.diagm(fill(1,size(∇₊,1)))]

# B = [∇₀ ∇₋
#      ℒ.diagm(fill(1,size(∇₊,1))) zero(∇₊) ]


# schdcmp = ℒ.schur(A,B)

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



ℒ.ordschur!(schdcmp, BitVector([1,0,0,0,1,0]))

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