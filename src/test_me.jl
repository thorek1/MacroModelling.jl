using MacroModelling, MatrixEquations, BenchmarkTools, ThreadedSparseArrays
import MacroModelling: parse_variables_input_to_index, calculate_covariance, solve_matrix_equation_AD, write_functions_mapping!, multiplicate, generateSumVectors, product_moments, solve_matrix_equation_forward, calculate_second_order_moments, determine_efficient_order, calculate_third_order_solution, calculate_quadratic_iteration_solution, calculate_linear_time_iteration_solution
import LinearAlgebra as ℒ
import RecursiveFactorization as RF
import SpeedMapping: speedmapping





include("../test/models/SW03.jl")
# m = SW07



include("../test/models/GNSS_2010.jl")
m = GNSS_2010

m = RBC_baseline
m = green_premium_recalib



𝓂 = m
write_functions_mapping!(𝓂, 3)
parameters = m.parameter_values
verbose = true
silent = false
T = m.timings
tol =eps()
M₂ = 𝓂.solution.perturbation.second_order_auxilliary_matrices;
M₃ = 𝓂.solution.perturbation.third_order_auxilliary_matrices;





SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

@benchmark sol_mat, converged = calculate_quadratic_iteration_solution(∇₁; T = 𝓂.timings, tol = eps(Float32))
@benchmark sol_mat = calculate_linear_time_iteration_solution(∇₁; T = 𝓂.timings, tol = eps(Float32))
@benchmark sol_mat, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)



expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

∇̂₀ =  RF.lu(∇₀)

A = ∇̂₀ \ ∇₋# |> sparse |> ThreadedSparseMatrixCSC
B = ∇̂₀ \ ∇₊# |> sparse |> ThreadedSparseMatrixCSC

C = zero(∇₋)
C̄ = zero(∇₋)

@benchmark sol = speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = eps(Float32), maps_limit = 10000)

sol.minimizer


iter = 1
change = 1
𝐂  = zero(A) * eps()
𝐂¹ = one(A) * eps()
while change > eps(Float32) && iter < 10000
    𝐂¹ = A + B * 𝐂^2
    if !(𝐂¹ isa DenseMatrix)
        droptol!(𝐂¹, eps())
    end
    if iter > 500
        change = maximum(abs, 𝐂¹ - 𝐂)
    end
    𝐂 = 𝐂¹
    iter += 1
end














Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

nᵉ = 𝓂.timings.nExo

nˢ = 𝓂.timings.nPast_not_future_and_mixed

iˢ = 𝓂.timings.past_not_future_and_mixed_idx

Σᶻ₁ = Σʸ₁[iˢ, iˢ]

# precalc second order
## mean
I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

## covariance
E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

quadrup = multiplicate(nᵉ, 4)

comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

for j = 1:size(comb⁴,1)
    E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
end

e⁴ = quadrup * E_e⁴

# second order
∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

# first order
s_to_y₁ = 𝐒₁[:, 1:nˢ]
e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]

s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


# second order
s_s_to_y₂ = 𝐒₂[:, kron_s_s]
e_e_to_y₂ = 𝐒₂[:, kron_e_e]
v_v_to_y₂ = 𝐒₂[:, kron_v_v]
s_e_to_y₂ = 𝐒₂[:, kron_s_e]

s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

# # Set up in pruned state transition matrices
ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
            zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
            zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
            zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
            zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

ŝv₂ = [ zeros(nˢ) 
        vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec(ℒ.I(nᵉ))
        e_to_s₁_by_e_to_s₁ * vec(ℒ.I(nᵉ))]

yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec(ℒ.I(nᵉ))) / 2

## Mean
μˢ⁺₂ = (ℒ.I - ŝ_to_ŝ₂) \ ŝv₂
Δμˢ₂ = vec((ℒ.I - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec(ℒ.I(nᵉ))) / 2))
μʸ₂  = SS_and_pars[1:𝓂.timings.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂


# Covariance
Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
        zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
        zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

r1,c1,v1 = findnz(sparse(ŝ_to_ŝ₂))

coordinates = Tuple{Vector{Int}, Vector{Int}}[]
push!(coordinates,(r1,c1))

dimensions = Tuple{Int, Int}[]
push!(dimensions,size(ŝ_to_ŝ₂))
push!(dimensions,size(C))

values = vcat(v1, vec(collect(-C)))



Σᶻ₂, info = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
using BenchmarkTools
@benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
@benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :lyapunov)
@benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :gmres)
@benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :bicgstab)
@benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :speedmapping)






observables = [:gdT]
dependencies_tol = 1e-8


write_functions_mapping!(𝓂, 3)

Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(parameters, 𝓂, verbose = verbose)

∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                            𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                            𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

orders = determine_efficient_order(𝐒₁, 𝓂.timings, observables, tol = dependencies_tol)

nᵉ = 𝓂.timings.nExo

# precalc second order
## covariance
E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

quadrup = multiplicate(nᵉ, 4)

comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

for j = 1:size(comb⁴,1)
    E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
end

e⁴ = quadrup * E_e⁴

# precalc third order
sextup = multiplicate(nᵉ, 6)
E_e⁶ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)

comb⁶   = reduce(vcat, generateSumVectors(nᵉ, 6))

comb⁶ = comb⁶ isa Int64 ? reshape([comb⁶],1,1) : comb⁶

for j = 1:size(comb⁶,1)
    E_e⁶[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁶[j,:])
end

e⁶ = sextup * E_e⁶

Σʸ₃ = zeros(size(Σʸ₂))


# Threads.@threads for ords in orders 
# for ords in orders 
ords = orders[1]
    variance_observable, dependencies_all_vars = ords

    sort!(variance_observable)

    sort!(dependencies_all_vars)

    dependencies = intersect(𝓂.timings.past_not_future_and_mixed, dependencies_all_vars)

    obs_in_y = indexin(variance_observable, 𝓂.timings.var)

    dependencies_in_states_idx = indexin(dependencies, 𝓂.timings.past_not_future_and_mixed)

    dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))

    nˢ = length(dependencies)

    iˢ = dependencies_in_var_idx

    Σ̂ᶻ₁ = Σʸ₁[iˢ, iˢ]

    dependencies_extended_idx = vcat(dependencies_in_states_idx, 
            dependencies_in_states_idx .+ 𝓂.timings.nPast_not_future_and_mixed, 
            findall(ℒ.kron(𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),), 𝓂.timings.past_not_future_and_mixed .∈ (intersect(𝓂.timings.past_not_future_and_mixed,dependencies),))) .+ 2*𝓂.timings.nPast_not_future_and_mixed)
    
    Σ̂ᶻ₂ = Σᶻ₂[dependencies_extended_idx, dependencies_extended_idx]
    
    Δ̂μˢ₂ = Δμˢ₂[dependencies_in_states_idx]

    s_in_s⁺ = BitVector(vcat(𝓂.timings.past_not_future_and_mixed .∈ (dependencies,), zeros(Bool, nᵉ + 1)))
    e_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed + 1), ones(Bool, nᵉ)))
    v_in_s⁺ = BitVector(vcat(zeros(Bool, 𝓂.timings.nPast_not_future_and_mixed), 1, zeros(Bool, nᵉ)))

    # precalc second order
    ## mean
    I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

    e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
    e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
    ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
    s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

    # first order
    s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
    e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
    e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]

    # second order
    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

    s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
    s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

    s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
    e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
    v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
    s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

    s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
    e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
    s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

    # third order
    kron_s_v = ℒ.kron(s_in_s⁺, v_in_s⁺)
    kron_e_v = ℒ.kron(e_in_s⁺, v_in_s⁺)

    s_s_s_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
    s_s_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
    s_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
    e_e_e_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
    s_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
    e_v_v_to_y₃ = 𝐒₃[obs_in_y,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

    s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
    s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
    s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
    e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
    s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
    e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]

    # Set up pruned state transition matrices
    ŝ_to_ŝ₃ = [  s_to_s₁                zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                        zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                        zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                        s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                        ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                        zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]

    ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                    ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                    zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]

    ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]

    ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]

    μˢ₃δμˢ₁ = reshape((ℒ.I - s_to_s₁_by_s_to_s₁) \ vec( 
                                (s_s_to_s₂  * reshape(ss_s * vec(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂'),nˢ^2, nˢ) +
                                s_s_s_to_s₃ * reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ^3, nˢ) / 6 +
                                s_e_e_to_s₃ * ℒ.kron(Σ̂ᶻ₁, vec(ℒ.I(nᵉ))) / 2 +
                                s_v_v_to_s₃ * Σ̂ᶻ₁ / 2) * s_to_s₁' +
                                (s_e_to_s₂  * ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ)) +
                                e_e_e_to_s₃ * reshape(e⁴, nᵉ^3, nᵉ) / 6 +
                                s_s_e_to_s₃ * ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ)) / 2 +
                                e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                ), nˢ, nˢ)

    Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σ̂ᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    reshape(e⁴, nᵉ, nᵉ^3)
            spzeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σ̂ᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            ℒ.kron(Δ̂μˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δ̂μˢ₂ * Δ̂μˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δ̂μˢ₂, reshape(e⁴, nᵉ, nᵉ^3))
            ℒ.kron(vec(Σ̂ᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σ̂ᶻ₁) * Δ̂μˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σ̂ᶻ₁), reshape(e⁴, nᵉ, nᵉ^3))
            spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σ̂ᶻ₁, reshape(e⁴, nᵉ^2, nᵉ^2))    spzeros(nˢ*nᵉ^2,nᵉ^3)
            reshape(e⁴, nᵉ^3, nᵉ)  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δ̂μˢ₂', reshape(e⁴, nᵉ^3, nᵉ))     ℒ.kron(vec(Σ̂ᶻ₁)', reshape(e⁴, nᵉ^3, nᵉ))  spzeros(nᵉ^3, nˢ*nᵉ^2)     reshape(e⁶, nᵉ^3, nᵉ^3)]


    Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
            ℒ.kron(Σ̂ᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(reshape(ss_s * vec(Σ̂ᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δ̂μˢ₂ * vec(Σ̂ᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(reshape(Σ̂ᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σ̂ᶻ₁) * vec(Σ̂ᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
            spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
    
    droptol!(ŝ_to_ŝ₃, eps())
    droptol!(ê_to_ŝ₃, eps())
    droptol!(Eᴸᶻ, eps())
    droptol!(Γ₃, eps())
    
    A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'
    droptol!(A, eps())

    C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'
    droptol!(C, eps())

    r1,c1,v1 = findnz(ŝ_to_ŝ₃)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    
    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(ŝ_to_ŝ₃))
    push!(dimensions,size(C))
    
    values = vcat(v1, vec(collect(-C)))

    # Σᶻ₃, info = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :doubling)
    Σᶻ₃, info = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)


    @benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    @benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :lyapunov)
    @benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :gmres)
    @benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :bicgstab)
    @benchmark solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :speedmapping)
    
    
    @benchmark lyapd(collect(ŝ_to_ŝ₃),collect(C))





SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

𝑺₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)



    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n  = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];
    
    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];


    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = - ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

    X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
    droptol!(X,tol)

    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    B = spinv * ∇₁₊
    droptol!(B,tol)

    C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
    droptol!(C,tol)


    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))

    𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :sylvester, sparse_output = true);

    𝐒₂ *= M₂.𝐔₂

    using MatrixEquations, BenchmarkTools

    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :iterative, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :bicgstab, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :sylvester, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :speedmapping, sparse_output = true)

    droptol!(𝐒₂,eps())
    length(𝐒₂.nzval) / length(𝐒₂)
    length(B.nzval) / length(B)
    length(C.nzval) / length(C)
    length(X.nzval) / length(X)
    


    @profview 𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :iterative, sparse_output = true)

    @profview 𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :sylvester, sparse_output = true)

    @profview 𝐒₂, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)




    write_functions_mapping!(𝓂, 3)

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)
            


    # Indices and number of variables
    i₊ = T.future_not_past_and_mixed_idx;
    i₋ = T.past_not_future_and_mixed_idx;

    n₋ = T.nPast_not_future_and_mixed
    n₊ = T.nFuture_not_past_and_mixed
    nₑ = T.nExo;
    n = T.nVars
    nₑ₋ = n₋ + 1 + nₑ

    # 1st order solution
    𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
    droptol!(𝐒₁,tol)

    𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

    ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                                𝐒₁
                                spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

    𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                    zeros(n₋ + n + nₑ, nₑ₋)];

    ∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]


    ∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

    spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
    droptol!(spinv,tol)

    B = spinv * ∇₁₊
    droptol!(B,tol)

    ⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎 = @views [(𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + 𝐒₁ * [𝐒₂[i₋,:] ; zeros(nₑ + 1, nₑ₋^2)])[i₊,:]
            𝐒₂
            zeros(n₋ + nₑ, nₑ₋^2)];
        
    𝐒₂₊╱𝟎 = @views [𝐒₂[i₊,:] 
            zeros(n₋ + n + nₑ, nₑ₋^2)];

    aux = M₃.𝐒𝐏 * ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋



    using Kronecker
    aux ⊗ 2

    @benchmark 𝐗₃ = -∇₃ * ℒ.kron(ℒ.kron(aux, aux), aux)
    𝐗₃ |> collect
    𝐗₃ = -∇₃ * aux ⊗ 3


    @benchmark 𝐗₃ = reshape(- ((aux' ⊗ 3) ⊗ ℒ.I(size(∇₃,1))) * vec(∇₃),size(∇₃,1),size(aux,2)^3) |> sparse

    tmpkron = ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔)
    out = - ∇₃ * tmpkron - ∇₃ * M₃.𝐏₁ₗ̂ * tmpkron * M₃.𝐏₁ᵣ̃ - ∇₃ * M₃.𝐏₂ₗ̂ * tmpkron * M₃.𝐏₂ᵣ̃
    𝐗₃ += out
    
    tmp𝐗₃ = -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋,⎸𝐒₂k𝐒₁₋╱𝟏ₑ➕𝐒₁𝐒₂₋⎹╱𝐒₂╱𝟎)

    tmpkron1 = -∇₂ *  ℒ.kron(𝐒₁₊╱𝟎,𝐒₂₊╱𝟎)
    tmpkron2 = ℒ.kron(M₂.𝛔,𝐒₁₋╱𝟏ₑ)
    out2 = tmpkron1 * tmpkron2 +  tmpkron1 * M₃.𝐏₁ₗ * tmpkron2 * M₃.𝐏₁ᵣ
    
    𝐗₃ += (tmp𝐗₃ + out2 + -∇₂ * ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, 𝐒₂₊╱𝟎 * M₂.𝛔)) * M₃.𝐏# |> findnz
    
    𝐗₃ += @views -∇₁₊ * 𝐒₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, [𝐒₂[i₋,:] ; zeros(size(𝐒₁)[2] - n₋, nₑ₋^2)]) * M₃.𝐏
    droptol!(𝐗₃,tol)
    
    X = spinv * 𝐗₃ * M₃.𝐂₃
    droptol!(X,tol)
    
    tmpkron = ℒ.kron(𝐒₁₋╱𝟏ₑ,M₂.𝛔)
    
    C = M₃.𝐔₃ * tmpkron + M₃.𝐔₃ * M₃.𝐏₁ₗ̄ * tmpkron * M₃.𝐏₁ᵣ̃ + M₃.𝐔₃ * M₃.𝐏₂ₗ̄ * tmpkron * M₃.𝐏₂ᵣ̃
    C += M₃.𝐔₃ * ℒ.kron(𝐒₁₋╱𝟏ₑ,ℒ.kron(𝐒₁₋╱𝟏ₑ,𝐒₁₋╱𝟏ₑ))
    C *= M₃.𝐂₃
    droptol!(C,tol)

    r1,c1,v1 = findnz(B)
    r2,c2,v2 = findnz(C)
    r3,c3,v3 = findnz(X)

    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    push!(coordinates,(r1,c1))
    push!(coordinates,(r2,c2))
    push!(coordinates,(r3,c3))
    
    values = vcat(v1, v2, v3)

    dimensions = Tuple{Int, Int}[]
    push!(dimensions,size(B))
    push!(dimensions,size(C))
    push!(dimensions,size(X))
    

    𝐒₃, solved = solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true);



    length(𝐒₃.nzval) / length(𝐒₃)
    length(B.nzval) / length(B)
    length(C.nzval) / length(C)
    length(X.nzval) / length(X)

    # 0.028028557464041336 -> gmres
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :gmres, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :bicgstab, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :sylvester, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :iterative, sparse_output = true)
    @benchmark solve_matrix_equation_forward(values, coords = coordinates, dims = dimensions, solver = :speedmapping, sparse_output = true)


    





@model firm_investment_problem begin
    K[0] = (1 - δ) * K[-1] + I[0]
    Z[0] = (1 - ρ) * μ + ρ * Z[-1] 
    I[1]  = ((ρ + δ - Z[0])/(1 - δ))  + ((1 + ρ)/(1 - δ)) * I[0]
end

@parameters firm_investment_problem begin
    ρ = 0.05
    δ = 0.10
    μ = .17
    σ = .2
end

SSS(GNSS_2010)
m = GNSS_2010

include("../test/models/FS2000.jl")

SSS(m)

get_covariance(m)



using MatrixEquations, BenchmarkTools

@benchmark sylvd(collect(-A),collect(B),-C)

@benchmark begin 
iter = 1
change = 1
𝐂  = C
𝐂¹ = C
# println(A)
# println(B)
# println(C)
while change > eps(Float32) && iter < 10000
    𝐂¹ = A * 𝐂 * B - C
    if !(A isa DenseMatrix)
        droptol!(𝐂¹, eps())
    end
    if iter > 500
        change = maximum(abs, 𝐂¹ - 𝐂)
    end
    𝐂 = 𝐂¹
    iter += 1
end
solved = change < eps(Float32)
end

m= firm_investment_problem
𝓂 = firm_investment_problem
parameters = m.parameter_values
algorithm = :first_order
verbose = true
silent = false
variables = :all_including_auxilliary
# parameter_derivatives = :all

SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) 

T = m.timings

expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 


∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1] # derivatives wrt variables with timing in the future
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)] |>collect # derivatives wrt variables with timing in the present
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2] # derivatives wrt variables with timing in the past

-solution = ∇₊/∇₀ * solution * solution + ∇₋/∇₀


δ=0.1; ρ=0.05; z= ρ + δ +.02; β=1/(1+ρ); # parameters

A= √β*[1.0 0.0; 0.0  (1.0 - δ)]
B= √β*[0.0; 1.0]
Q=-1*[0.0 z/2; z/2 0.0]
R=-1*[-0.5;;]; 
S=-1*[-1/2; 0.0]
P, CLSEIG, F = ared(A,B,R,Q,S)

ared(zero(collect(∇₀)), ∇₊/∇₀, zero(collect(∇₀)), zero(collect(∇₀)), ∇₋/∇₀)

out = ared(∇₊/∇₀, ∇₋/∇₀, ℒ.diagm(zeros(T.nVars)), ℒ.diagm(zeros(T.nVars)), zero(collect(∇₀)))
out = ared(∇₊/∇₀, ∇₋/∇₀, ℒ.diagm(zeros(T.nVars)), ℒ.diagm(zeros(T.nVars)), zero(collect(∇₀)))
out[1]
ared(∇₊, ∇₋, zero(collect(∇₀)), zero(collect(∇₀)), ∇₀)

m.solution.perturbation.first_order.solution_matrix[:,1:end-T.nExo] * expand[2]


sol, solved = calculate_first_order_solution(Matrix(∇₁); T = 𝓂.timings)

# covar_raw, solved_cov = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))

A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]


C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed+1:end]

CC = C * C'


coordinates = Tuple{Vector{Int}, Vector{Int}}[]

dimensions = Tuple{Int, Int}[]
push!(dimensions,size(A))
push!(dimensions,size(CC))

values = vcat(vec(A), vec(collect(-CC)))


using BenchmarkTools
@benchmark lyapd(A,CC)
@benchmark covar_raw, _ = solve_sylvester_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)







tol = eps()


Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

nᵉ = 𝓂.timings.nExo

nˢ = 𝓂.timings.nPast_not_future_and_mixed

iˢ = 𝓂.timings.past_not_future_and_mixed_idx

Σᶻ₁ = Σʸ₁[iˢ, iˢ]

# precalc second order
## mean
I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

## covariance
E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

quadrup = multiplicate(nᵉ, 4)

comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

for j = 1:size(comb⁴,1)
    E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
end

e⁴ = quadrup * E_e⁴

# second order
∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))
e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

# first order
s_to_y₁ = 𝐒₁[:, 1:nˢ]
e_to_y₁ = 𝐒₁[:, (nˢ + 1):end]

s_to_s₁ = 𝐒₁[iˢ, 1:nˢ]
e_to_s₁ = 𝐒₁[iˢ, (nˢ + 1):end]


# second order
s_s_to_y₂ = 𝐒₂[:, kron_s_s]
e_e_to_y₂ = 𝐒₂[:, kron_e_e]
v_v_to_y₂ = 𝐒₂[:, kron_v_v]
s_e_to_y₂ = 𝐒₂[:, kron_s_e]

s_s_to_s₂ = 𝐒₂[iˢ, kron_s_s] |> collect
e_e_to_s₂ = 𝐒₂[iˢ, kron_e_e]
v_v_to_s₂ = 𝐒₂[iˢ, kron_v_v] |> collect
s_e_to_s₂ = 𝐒₂[iˢ, kron_s_e]

s_to_s₁_by_s_to_s₁ = ℒ.kron(s_to_s₁, s_to_s₁) |> collect
e_to_s₁_by_e_to_s₁ = ℒ.kron(e_to_s₁, e_to_s₁)
s_to_s₁_by_e_to_s₁ = ℒ.kron(s_to_s₁, e_to_s₁)

# # Set up in pruned state transition matrices
ŝ_to_ŝ₂ = [ s_to_s₁             zeros(nˢ, nˢ + nˢ^2)
            zeros(nˢ, nˢ)       s_to_s₁             s_s_to_s₂ / 2
            zeros(nˢ^2, 2*nˢ)   s_to_s₁_by_s_to_s₁                  ]

ê_to_ŝ₂ = [ e_to_s₁         zeros(nˢ, nᵉ^2 + nᵉ * nˢ)
            zeros(nˢ,nᵉ)    e_e_to_s₂ / 2       s_e_to_s₂
            zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁]

ŝ_to_y₂ = [s_to_y₁  s_to_y₁         s_s_to_y₂ / 2]

ê_to_y₂ = [e_to_y₁  e_e_to_y₂ / 2   s_e_to_y₂]

ŝv₂ = [ zeros(nˢ) 
        vec(v_v_to_s₂) / 2 + e_e_to_s₂ / 2 * vec(ℒ.I(nᵉ))
        e_to_s₁_by_e_to_s₁ * vec(ℒ.I(nᵉ))]

yv₂ = (vec(v_v_to_y₂) + e_e_to_y₂ * vec(ℒ.I(nᵉ))) / 2

## Mean
μˢ⁺₂ = (ℒ.I - ŝ_to_ŝ₂) \ ŝv₂
Δμˢ₂ = vec((ℒ.I - s_to_s₁) \ (s_s_to_s₂ * vec(Σᶻ₁) / 2 + (v_v_to_s₂ + e_e_to_s₂ * vec(ℒ.I(nᵉ))) / 2))
μʸ₂  = SS_and_pars[1:𝓂.timings.nVars] + ŝ_to_y₂ * μˢ⁺₂ + yv₂

# if !covariance
#     return μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂
# end

# Covariance
Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
        zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
        zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

r1,c1,v1 = findnz(sparse(ŝ_to_ŝ₂))

coordinates = Tuple{Vector{Int}, Vector{Int}}[]
push!(coordinates,(r1,c1))

dimensions = Tuple{Int, Int}[]
push!(dimensions,size(ŝ_to_ŝ₂))
push!(dimensions,size(C))

values = vcat(v1, vec(collect(-C)))

# Σᶻ₂, info = solve_sylvester_equation_forward(values, coords = coordinates, dims = dimensions, solver = :doubling)
@benchmark Σᶻ₂, info = solve_sylvester_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)


@benchmark lyapd(ŝ_to_ŝ₂,C)















solve!(𝓂, parameters = parameters, algorithm = algorithm, verbose = verbose, silent = silent)

# write_parameters_input!(𝓂,parameters, verbose = verbose)

var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

# parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

# if parameter_derivatives == :all
#     length_par = length(𝓂.parameters)
#     param_idx = 1:length_par
# elseif isa(parameter_derivatives,Symbol)
#     @assert parameter_derivatives ∈ 𝓂.parameters string(parameter_derivatives) * " is not part of the free model parameters."

#     param_idx = indexin([parameter_derivatives], 𝓂.parameters)
#     length_par = 1
# elseif length(parameter_derivatives) > 1
#     for p in vec(collect(parameter_derivatives))
#         @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
#     end
#     param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
#     length_par = length(parameter_derivatives)
# end

NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())


covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)


# if length_par * length(NSSS) > 200 || (!variance && !standard_deviation && !non_stochastic_steady_state && !mean)
#     derivatives = false
# end

# if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state || mean)
#     derivatives = true
# end


axis1 = 𝓂.var

if any(x -> contains(string(x), "◖"), axis1)
    axis1_decomposed = decompose_name.(axis1)
    axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
end

axis2 = 𝓂.timings.exo

if any(x -> contains(string(x), "◖"), axis2)
    axis2_decomposed = decompose_name.(axis2)
    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
end


if covariance
    if algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
        if mean
            var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
        end
    elseif algorithm == :pruned_third_order
        covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
        if mean
            var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
        end
    else
        covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
    end
end


