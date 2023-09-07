using MacroModelling
import MacroModelling: ℳ, calculate_covariance, multiplicate, generateSumVectors, product_moments, calculate_second_order_covariance_AD, determine_efficient_order, calculate_third_order_moments, calculate_second_order_moments
import LinearAlgebra as ℒ
include("models/FS2000.jl")

corr(m,algorithm = :pruned_third_order)
corr(m,algorithm = :pruned_second_order)
corr(m)


𝓂 = m
parameter_values = m.parameter_values
parameters = m.parameters
algorithm = :pruned_third_order
verbose = true

function get_statistics(𝓂, 
    parameter_values::Vector{T}; 
    parameters::Vector{Symbol} = Symbol[], 
    non_stochastic_steady_state::Vector{Symbol} = Symbol[],
    mean::Vector{Symbol} = Symbol[],
    standard_deviation::Vector{Symbol} = Symbol[],
    variance::Vector{Symbol} = Symbol[],
    covariance::Vector{Symbol} = Symbol[],
    autocorrelation::Vector{Symbol} = Symbol[],
    autocorrelation_periods::U = 1:5,
    algorithm::Symbol = :first_order,
    verbose::Bool = false) where {U,T}


    @assert algorithm ∈ [:first_order,:linear_time_iteration,:quadratic_iteration,:pruned_second_order,:pruned_third_order] "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = indexin(non_stochastic_steady_state, 𝓂.var)

    mean_var_idx = indexin(mean, 𝓂.var)

    std_var_idx = indexin(standard_deviation, 𝓂.var)

    var_var_idx = indexin(variance, 𝓂.var)

    covar_var_idx = indexin(covariance, 𝓂.var)

    autocorr_var_idx = indexin(autocorrelation, 𝓂.var)

    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.parameters, parameters), 𝓂.parameters)]

    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.parameters, parameters), 𝓂.parameters), indexin(parameters, 𝓂.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]

    if algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]))
        algorithm = :pruned_second_order
    end

    solve!(𝓂, algorithm = algorithm, verbose = verbose)

    if algorithm == :pruned_third_order

        if !(autocorrelation == Symbol[])
            autocorrelation = Symbol[]
        end

        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[])
            covar_dcmp, state_μ, SS_and_pars = calculate_third_order_moments(all_parameters, union(variance,covariance,standard_deviation), 𝓂, verbose = verbose)
        end

    elseif algorithm == :pruned_second_order

        if !(autocorrelation == Symbol[])
            autocorrelation = Symbol[]
        end

        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[])
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(all_parameters, 𝓂, verbose = verbose)
        else
            state_μ, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(all_parameters, 𝓂, verbose = verbose, covariance = false)
        end

    else
        covar_dcmp, sol, _, SS_and_pars = calculate_covariance(all_parameters, 𝓂, verbose = verbose)
    end

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]

    if !(variance == Symbol[])
        varrs = convert(Vector{Real},ℒ.diag(covar_dcmp))
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(varrs)
        end
    elseif !(autocorrelation == Symbol[])
        A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

        autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in autocorrelation_periods])
    else
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(abs.(convert(Vector{Real},ℒ.diag(covar_dcmp))))
        end
    end

    ret = []
    if !(non_stochastic_steady_state == Symbol[])
        push!(ret,SS[SS_var_idx])
    end
    if !(mean == Symbol[])
        if algorithm ∉ [:pruned_second_order,:pruned_third_order]
            push!(ret,SS[mean_var_idx])
        else
            push!(ret,state_μ[mean_var_idx])
        end
    end
    if !(standard_deviation == Symbol[])
        push!(ret,st_dev[std_var_idx])
    end
    if !(variance == Symbol[])
        push!(ret,varrs[var_var_idx])
    end
    if !(covariance == Symbol[])
        covar_dcmp_sp = sparse(ℒ.triu(covar_dcmp))

        droptol!(covar_dcmp_sp,eps(Float64))

        push!(ret,covar_dcmp_sp[covar_var_idx,covar_var_idx])
    end
    if !(autocorrelation == Symbol[]) 
        push!(ret,autocorr[autocorr_var_idx,:] )
    end

    return ret
end

get_statistics(m,m.parameter_values,parameters = m.parameters, mean = [:c,:k])

get_statistics(m,m.parameter_values,parameters = m.parameters, mean = [:c,:k], algorithm = :pruned_second_order)
get_statistics(m,m.parameter_values,parameters = m.parameters, mean = [:c,:k], algorithm = :pruned_third_order)

get_statistics(m,m.parameter_values,parameters = m.parameters, mean = [:c,:k], standard_deviation = [:y,:log_gp_obs], algorithm = :pruned_second_order)



using ForwardDiff

ForwardDiff.jacobian(x->get_statistics(m,x,parameters = m.parameters, mean = [:c,:k], standard_deviation = [:y,:log_gp_obs], algorithm = :pruned_third_order)[2],m.parameter_values)


get_std(m, algorithm = :pruned_third_order, derivatives = false)
get_std(m, algorithm = :pruned_third_order)

get_std(m)
get_statistics(m, m.parameter_values, parameters = [m.parameters[1]], standard_deviation = [m.var[5]])


import Optim, LineSearches
sol = Optim.optimize(x -> sum(abs2, get_statistics(m, x, parameters = [m.parameters[1]], standard_deviation = [m.var[5]])[1] - [.09]),
    [0], [1], [.16], 
    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 2))); autodiff = :forward)

sol.minimizer

get_std(m, algorithm = :pruned_second_order, derivatives = false)
get_std(m, algorithm = :pruned_second_order)
get_std(m, algorithm = :pruned_third_order, derivatives = false)
get_std(m, algorithm = :pruned_third_order)
@time get_std(m, algorithm = :pruned_third_order)

get_variance_decomposition(m)

get_std(m, algorithm = :pruned_second_order)

@profview get_std(m, algorithm = :pruned_third_order)

using SparseArrays
y= sprand(100,.1)

map(x->x^2,A)
map(eachindex(IndexCartesian(), y)) do i
    y[i]^2
end

using BenchmarkTools
@benchmark get_std(m)
@benchmark get_std(m, algorithm = :pruned_third_order)
# iterative solve:          400ms
# iterative solve and ID for 1st cov:   420ms
# direct solve:             22s
@benchmark get_std(m, algorithm = :pruned_third_order, derivatives = false)
@benchmark get_covariance(m, algorithm = :pruned_third_order, derivatives = false)
@benchmark get_covariance(m, algorithm = :pruned_second_order, derivatives = false)
@profview for i in 1:100 get_covariance(m, algorithm = :pruned_third_order) end



@benchmark get_covariance(m, algorithm = :pruned_second_order)

@benchmark get_std(m, algorithm = :pruned_third_order, parameter_derivatives = :alp)
@benchmark get_std(m, algorithm = :pruned_third_order)
@benchmark get_std(m, algorithm = :pruned_second_order, derivatives = false)

get_var(m, algorithm = :pruned_third_order, derivatives = false)


get_irf(m, algorithm = :pruned_third_order)

get_std(m, algorithm = :pruned_second_order, derivatives = false)
get_std(m, algorithm = :pruned_third_order, derivatives = false)
# get_covariance(m, algorithm = :pruned_third_order)

using ForwardDiff, LinearOperators, Krylov
import LinearAlgebra as ℒ

parameters = m.parameter_values
tol::Float64 = eps()
dependencies_tol::Float64 = 1e-15
verbose = true
𝓂 = m

m.var
order = determine_efficient_order(m,[:log_gp_obs,:log_gy_obs,:n,:l])

out = calculate_third_order_moments(m.parameter_values,:full_covar,m)
using LinearAlgebra
out[1]|>diag.|>sqrt




observables = [:log_gp_obs,:log_gy_obs,:n,:l]

# function calculate_third_order_covariances(parameters::Vector{<: Real}, 
#     observables::Vector{Symbol},
#     𝓂::ℳ; 
#     verbose::Bool = false, 
#     tol::AbstractFloat = eps())
    Σʸ₂, Σᶻ₂, μʸ₂, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
    
    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    orders = determine_efficient_order(∇₁, 𝓂.timings, observables)

    nᵉ = 𝓂.timings.nExo

    s⁺ = vcat(𝓂.timings.past_not_future_and_mixed, :Volatility, 𝓂.timings.exo)

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

    Σʸ₃ = zero(Σʸ₂)

    ords = orders[1]
    # for ords in orders 
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
        # precalc second order
        ## mean
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

        # second order
        s_in_s⁺ = s⁺ .∈ (dependencies,)
        e_in_s⁺ = s⁺ .∈ (𝓂.timings.exo,)
        v_in_s⁺ = s⁺ .∈ ([:Volatility],)

        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

        # first order
        s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
        e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
        
        s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
        e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]


        # second order
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

        A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'

        C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'

        # if size(initial_guess³) == (0,0)
        #     initial_guess³ = collect(C)
        # end

        if length(C) < 1e7
            function sylvester!(sol,𝐱)
                𝐗 = reshape(𝐱, size(C))
                sol .= vec(ŝ_to_ŝ₃ * 𝐗 * ŝ_to_ŝ₃' - 𝐗)
                return sol
            end

            sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

            Σ̂ᶻ₃, info = Krylov.gmres(sylvester, sparsevec(collect(-C)), atol = eps())

            if !info.solved
                Σ̂ᶻ₃, info = Krylov.bicgstab(sylvester, sparsevec(collect(-C)), atol = eps())
            end

            Σᶻ₃ = reshape(Σ̂ᶻ₃, size(C))
        else
            soll = speedmapping(collect(C); m! = (Σᶻ₃, Σ̂ᶻ₃) -> Σᶻ₃ .= ŝ_to_ŝ₃ * Σ̂ᶻ₃ * ŝ_to_ŝ₃' + C, 
            # time_limit = 200, 
            stabilize = true)
            
            Σᶻ₃ = soll.minimizer

            if !soll.converged
                return Inf
            end
        end
        Σʸ₃tmp = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃'

        for obs in variance_observable
            Σʸ₃[indexin([obs], 𝓂.timings.var), indexin(variance_observable, 𝓂.timings.var)] = Σʸ₃tmp[indexin([obs], variance_observable), :]
        end
    # end

    return Σʸ₃, μʸ₂
# end


using LinearOperators, Krylov

out = calculate_third_order_covariances(m.parameter_values,[:log_gp_obs,:log_gy_obs],m)

out[1]


calculate_third_order_moments(m.parameter_values, m.var => m.var, m)

𝓂 = m
dependencies = [:n,:y,:k,:m]
dependencies_in_states_idx = indexin(intersect(𝓂.timings.past_not_future_and_mixed,dependencies),𝓂.timings.past_not_future_and_mixed)


s⁺ = vcat(𝓂.timings.past_not_future_and_mixed, :Volatility, 𝓂.timings.exo)

ℒ.kron(s⁺ .∈ (𝓂.timings.past_not_future_and_mixed,), s⁺ .∈ (𝓂.timings.past_not_future_and_mixed,))






variance_observable, dependencies = order[1]
sort!(dependencies)
obs_in_y = indexin(variance_observable, 𝓂.timings.var)

Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))


T=𝓂.timings
SS_and_pars, solution_error = m.SS_solve_func(m.parameter_values, m, true)
    
∇₁ = calculate_jacobian(m.parameter_values, SS_and_pars, m) |> collect

expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
            ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * sparse(expand[2])
∇ₑ = ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

incidence = sparse(abs.(∇₊) + abs.(∇₀) + abs.(∇₋))
# droptol!(incidence,eps())

using BlockTriangularForm

Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence))
R̂ = []
for i in 1:n_blocks
    [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
end
push!(R̂,1)

vars = hcat(P, R̂)'
eqs = hcat(Q, R̂)'


dependency_matrix = incidence[vars[1,:],eqs[1,:]] .!= 0


function warshall_algorithm!(R)
    n, m = size(R)
    
    for k in 1:n
        for i in 1:n
            for j in 1:n
                R[i, j] = R[i, j] || (R[i, k] && R[k, j])
            end
        end
    end
    return R
end

warshall_algorithm!(dependency_matrix)

dependency_matrix |> collect

sum(dependency_matrix,dims=2)


m.timings.var[eqs[1,:]]


observabls = [:R, :n, :log_gy_obs, :log_gp_obs]

# sort(observabls, order = m.timings.var[eqs[1,:]])
indexin(observabls,m.timings.var[eqs[1,:]])

permut = sortperm(indexin(observabls, m.timings.var[eqs[1,:]]))

observabls = observabls[permut]

calc_cov = Vector{Symbol}[]
already_done = Set{Symbol}()
for obs in observabls
    dependencies = m.timings.var[eqs[1,:]][findall(dependency_matrix[indexin([obs], m.timings.var[eqs[1,:]])[1],:])]
    tbsolved_for = setdiff(intersect(observabls, dependencies),already_done)
    if length(tbsolved_for) > 0
        push!(calc_cov, tbsolved_for)
    end
    push!(already_done,intersect(observabls, dependencies)...)
end



function warshall_algorithm!(R)
    n, m = size(R)
    
    for k in 1:n
        for i in 1:n
            for j in 1:n
                R[i, j] = R[i, j] || (R[i, k] && R[k, j])
            end
        end
    end
    return R
end



function determine_efficient_order(𝓂::ℳ, observables::Vector{Symbol}; verbose::Bool = false)
    SS_and_pars, solution_error = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose)
    
    ∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> collect
    
    expand = [  spdiagm(ones(𝓂.timings.nVars))[𝓂.timings.future_not_past_and_mixed_idx,:],
                spdiagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]] 
    
    ∇₊ = ∇₁[:,1:𝓂.timings.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = ∇₁[:,𝓂.timings.nFuture_not_past_and_mixed .+ range(1,𝓂.timings.nVars)]
    ∇₋ = ∇₁[:,𝓂.timings.nFuture_not_past_and_mixed + 𝓂.timings.nVars .+ range(1,𝓂.timings.nPast_not_future_and_mixed)] * expand[2]

    incidence = abs.(∇₊) + abs.(∇₀) + abs.(∇₋)

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence))
    R̂ = []
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
    end
    push!(R̂,1)
    
    vars = hcat(P, R̂)'
    eqs  = hcat(Q, R̂)'
    
    dependency_matrix = incidence[vars[1,:], eqs[1,:]] .!= 0
    
    warshall_algorithm!(dependency_matrix)

    permut = sortperm(indexin(observables, 𝓂.timings.var[eqs[1,:]]))
    
    solve_order = Vector{Symbol}[]
    already_solved_for = Set{Symbol}()
    corresponding_dependencies = Vector{Symbol}[]

    for obs in observables[permut]
        dependencies = 𝓂.timings.var[eqs[1,:]][findall(dependency_matrix[indexin([obs], 𝓂.timings.var[eqs[1,:]])[1],:])]
        to_be_solved_for = setdiff(intersect(observables, dependencies), already_solved_for)
        if length(to_be_solved_for) > 0
            push!(solve_order, to_be_solved_for)
            push!(corresponding_dependencies, dependencies)
        end
        push!(already_solved_for, intersect(observables, dependencies)...)
    end

    return solve_order .=> corresponding_dependencies
end




function calculate_third_order_moments(parameters::Vector{<: Real}, 
    variance_observables_and_dependencies::Pair{Vector{Symbol}, Vector{Symbol}},
    𝓂::ℳ; 
    verbose::Bool = false, 
    tol::AbstractFloat = eps())

    nᵉ = 𝓂.timings.nExo

    variance_observable, dependencies = variance_observables_and_dependencies

    obs_in_y = indexin([variance_observable], 𝓂.timings.var)

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)

    dependencies_in_states_idx = indexin(dependencies,𝓂.timings.past_not_future_and_mixed)
    dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))

    nˢ = length(dependencies)

    iˢ = dependencies_in_var_idx

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


    # precalc third order
    sextup = multiplicate(nᵉ, 6)
    E_e⁶ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)

    comb⁶   = reduce(vcat, generateSumVectors(nᵉ, 6))

    comb⁶ = comb⁶ isa Int64 ? reshape([comb⁶],1,1) : comb⁶

    for j = 1:size(comb⁶,1)
        E_e⁶[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁶[j,:])
    end

    e⁶ = sextup * E_e⁶

    e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
    e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
    ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
    s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))

    # second order
    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    s⁺ = vcat(𝓂.timings.past_not_future_and_mixed, :Volatility, 𝓂.timings.exo)

    s_in_s⁺ = s⁺ .∈ (dependencies,)
    e_in_s⁺ = s⁺ .∈ (𝓂.timings.exo,)
    v_in_s⁺ = s⁺ .∈ ([:Volatility],)

    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)

    # first order
    s_to_y₁ = 𝐒₁[obs_in_y,:][:,dependencies_in_states_idx]
    e_to_y₁ = 𝐒₁[obs_in_y,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
    e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]


    # second order
    s_s_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_e_e]
    v_v_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_v_v]
    s_e_to_y₂ = 𝐒₂[obs_in_y,:][:, kron_s_e]

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
    μʸ₂  = SS_and_pars[obs_in_y] + ŝ_to_y₂ * μˢ⁺₂ + yv₂


    # Covariance
    Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
            zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

    C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    Σᶻ₂, info = calculate_second_order_covariance_AD([vec(ŝ_to_ŝ₂); vec(C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])

    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

    # third order
    kron_s_v = ℒ.kron(s_in_s⁺, v_in_s⁺)
    kron_e_v = ℒ.kron(e_in_s⁺, v_in_s⁺)

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

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
                                (s_s_to_s₂  * reshape(ss_s * vec(Σᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σᶻ₁) * Δμˢ₂'),nˢ^2, nˢ) +
                                s_s_s_to_s₃ * reshape(Σᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', nˢ^3, nˢ) / 6 +
                                s_e_e_to_s₃ * ℒ.kron(Σᶻ₁, vec(ℒ.I(nᵉ))) / 2 +
                                s_v_v_to_s₃ * Σᶻ₁ / 2) * s_to_s₁' +
                                (s_e_to_s₂  * ℒ.kron(Δμˢ₂,ℒ.I(nᵉ)) +
                                e_e_e_to_s₃ * reshape(e⁴, nᵉ^3, nᵉ) / 6 +
                                s_s_e_to_s₃ * ℒ.kron(vec(Σᶻ₁), ℒ.I(nᵉ)) / 2 +
                                e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                ), nˢ, nˢ)


    Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δμˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    reshape(e⁴, nᵉ, nᵉ^3)
            spzeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            ℒ.kron(Δμˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δμˢ₂ * Δμˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δμˢ₂ * vec(Σᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δμˢ₂, reshape(e⁴, nᵉ, nᵉ^3))
            ℒ.kron(vec(Σᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σᶻ₁) * Δμˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σᶻ₁), reshape(e⁴, nᵉ, nᵉ^3))
            spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σᶻ₁, reshape(e⁴, nᵉ^2, nᵉ^2))    spzeros(nˢ*nᵉ^2,nᵉ^3)
            reshape(e⁴, nᵉ^3, nᵉ)  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δμˢ₂', reshape(e⁴, nᵉ^3, nᵉ))     ℒ.kron(vec(Σᶻ₁)', reshape(e⁴, nᵉ^3, nᵉ))  spzeros(nᵉ^3, nˢ*nᵉ^2)     reshape(e⁶, nᵉ^3, nᵉ^3)]


    Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
            ℒ.kron(Σᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(reshape(ss_s * vec(Σᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δμˢ₂ * vec(Σᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(reshape(Σᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
            spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]

    A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'

    C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'

    # if size(initial_guess³) == (0,0)
    #     initial_guess³ = collect(C)
    # end

    if length(C) < 1e7
        function sylvester!(sol,𝐱)
            𝐗 = reshape(𝐱, size(C))
            sol .= vec(ŝ_to_ŝ₃ * 𝐗 * ŝ_to_ŝ₃' - 𝐗)
            return sol
        end

        sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

        Σ̂ᶻ₃, info = Krylov.gmres(sylvester, sparsevec(collect(-C)), atol = eps())

        if !info.solved
            Σ̂ᶻ₃, info = Krylov.bicgstab(sylvester, sparsevec(collect(-C)), atol = eps())
        end

        Σᶻ₃ = reshape(Σ̂ᶻ₃, size(C))
    else
        soll = speedmapping(collect(C); m! = (Σᶻ₃, Σ̂ᶻ₃) -> Σᶻ₃ .= ŝ_to_ŝ₃ * Σ̂ᶻ₃ * ŝ_to_ŝ₃' + C, 
        # time_limit = 200, 
        stabilize = true)
        
        Σᶻ₃ = soll.minimizer

        if !soll.converged
            return Inf
        end
    end

    Σʸ₃ = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃'

    return Σʸ₃, μʸ₂
end

m.var
order = determine_efficient_order(m,[:log_gp_obs,:log_gy_obs])
calculate_third_order_moments(m.parameter_values,order[1],m)


eff = determine_efficient_order(m,[:R,:n,:gp_obs])
eff[2][2]

dependencies = m.timings.var[eqs[1,:]][findall(dependency_matrix[indexin(observabls, m.timings.var[eqs[1,:]])[3],:])]

intersect(observabls, dependencies)
[setdiff!(observabls,[i]) for i in dependencies]


import RecursiveFactorization as RF
# ∇₀nzs = findnz(∇₀)
# ∇₀₁ = sparse(∇₀nzs[1],∇₀nzs[2],10 .+rand(length(∇₀nzs[2])),size(∇₀,1),size(∇₀,2)) |> collect

# ∇₊nzs = findnz(∇₊)
# ∇₊₁ = sparse(∇₊nzs[1],∇₊nzs[2],10 .+rand(length(∇₊nzs[2])),size(∇₊,1),size(∇₊,2))

# ∇₋nzs = findnz(∇₋)
# ∇₋₁ = sparse(∇₋nzs[1],∇₋nzs[2],10 .+rand(length(∇₋nzs[2])),size(∇₋,1),size(∇₋,2))

∇̂₀ =  RF.lu(∇₀)

# droptol!(∇̂₀)

A = sparse(∇̂₀ \ ∇₋)
B = sparse(∇̂₀ \ ∇₊)
droptol!(A, 1e-15)
droptol!(B, 1e-15)
A = collect(A)
B = collect(B)

C = similar(A)
C̄ = similar(A)
using SpeedMapping
sol = speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = tol, maps_limit = 10000)

C = -sol.minimizer
C = sparse(C)
droptol!(C,1e-15)
C = collect(C)

Cnzs = findnz(sparse(C))
c = sparse(Cnzs[1],Cnzs[2],1,size(C,1),size(C,2))

(c * c') |> collect

get_solution(m)

nzs = findnz(∇₁)

sparse(nzs[1],nzs[2],1,size(∇₁,1),size(∇₁,2))
findnz(∇₁)[2]

variance_observable = :y



function calculate_third_order_moments(parameters::Vector{<: Real}, 
    variance_observable::Symbol,
    𝓂::ℳ; 
    verbose::Bool = false, 
    tol::AbstractFloat = eps(),
    dependencies_tol::AbstractFloat = 1e-15)

    nᵉ = 𝓂.timings.nExo
    n̂ˢ = 𝓂.timings.nPast_not_future_and_mixed

    if variance_observable == :all
        obs_in_var_idx = 1:𝓂.timings.nVars
    else
        obs_in_var_idx = indexin([variance_observable], 𝓂.timings.var)
    end

    Σʸ₁, 𝐒₁, ∇₁, SS_and_pars = calculate_covariance(parameters, 𝓂, verbose = verbose)


    dependencies_in_states_bitvector = vec(sum(abs, 𝐒₁[obs_in_var_idx,1:n̂ˢ], dims=1) .> dependencies_tol) .> 0

    while dependencies_in_states_bitvector .| vec(abs.(dependencies_in_states_bitvector' * 𝐒₁[indexin(𝓂.timings.past_not_future_and_mixed, 𝓂.timings.var),1:n̂ˢ]) .> dependencies_tol) != dependencies_in_states_bitvector
        dependencies_in_states_bitvector = dependencies_in_states_bitvector .| vec(abs.(dependencies_in_states_bitvector' * 𝐒₁[indexin(𝓂.timings.past_not_future_and_mixed, 𝓂.timings.var),1:n̂ˢ]) .> dependencies_tol)
    end

    dependencies = 𝓂.timings.past_not_future_and_mixed[dependencies_in_states_bitvector]
    # println(length(dependencies))
    dependencies_in_states_idx = indexin(dependencies,𝓂.timings.past_not_future_and_mixed)
    dependencies_in_var_idx = Int.(indexin(dependencies, 𝓂.timings.var))
    

    nˢ = length(dependencies)

    iˢ = dependencies_in_var_idx

    Σᶻ₁ = Σʸ₁[iˢ, iˢ]

    #precalc second order
    # mean
    I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)

    #covariance
    E_e⁴ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)

    quadrup = multiplicate(nᵉ, 4)

    comb⁴ = reduce(vcat, generateSumVectors(nᵉ, 4))

    comb⁴ = comb⁴ isa Int64 ? reshape([comb⁴],1,1) : comb⁴

    for j = 1:size(comb⁴,1)
        E_e⁴[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁴[j,:])
    end

    e⁴ = quadrup * E_e⁴


    #precalc third order
    sextup = multiplicate(nᵉ, 6)
    E_e⁶ = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)

    comb⁶   = reduce(vcat, generateSumVectors(nᵉ, 6))

    comb⁶ = comb⁶ isa Int64 ? reshape([comb⁶],1,1) : comb⁶

    for j = 1:size(comb⁶,1)
        E_e⁶[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb⁶[j,:])
    end

    e⁶ = sextup * E_e⁶


    e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ*nˢ)), nˢ*nᵉ^2, nˢ*nᵉ^2))
    e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ*nˢ^2, nᵉ*nˢ^2))
    ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
    s_s  = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))









    ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)

    𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    s⁺ = vcat(𝓂.timings.past_not_future_and_mixed, :Volatility, 𝓂.timings.exo)

    s_in_s⁺ = s⁺ .∈ (dependencies,)
    e_in_s⁺ = s⁺ .∈ (𝓂.timings.exo,)
    v_in_s⁺ = s⁺ .∈ ([:Volatility],)

    kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
    kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
    kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
    kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)
    # first order
    s_to_y₁ = 𝐒₁[obs_in_var_idx,:][:,dependencies_in_states_idx]
    e_to_y₁ = 𝐒₁[obs_in_var_idx,:][:, (𝓂.timings.nPast_not_future_and_mixed + 1):end]
    
    s_to_s₁ = 𝐒₁[iˢ, dependencies_in_states_idx]
    e_to_s₁ = 𝐒₁[iˢ, (𝓂.timings.nPast_not_future_and_mixed + 1):end]


    # second order
    s_s_to_y₂ = 𝐒₂[obs_in_var_idx,:][:, kron_s_s]
    e_e_to_y₂ = 𝐒₂[obs_in_var_idx,:][:, kron_e_e]
    v_v_to_y₂ = 𝐒₂[obs_in_var_idx,:][:, kron_v_v]
    s_e_to_y₂ = 𝐒₂[obs_in_var_idx,:][:, kron_s_e]

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
    Δμʸ₂ = ŝ_to_y₂ * μˢ⁺₂ + yv₂
    μʸ₂  = SS_and_pars[obs_in_var_idx] + ŝ_to_y₂ * μˢ⁺₂ + yv₂


    # Covariance

    Γ₂ = [ ℒ.I(nᵉ)             zeros(nᵉ, nᵉ^2 + nᵉ * nˢ)
            zeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     zeros(nᵉ^2, nᵉ * nˢ)
            zeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))]

    C = ê_to_ŝ₂ * Γ₂ * ê_to_ŝ₂'

    Σᶻ₂, info = calculate_second_order_covariance_AD([vec(ŝ_to_ŝ₂); vec(C)], dims = [size(ŝ_to_ŝ₂) ;size(C)])

    Σʸ₂ = ŝ_to_y₂ * Σᶻ₂ * ŝ_to_y₂' + ê_to_y₂ * Γ₂ * ê_to_y₂'

    # return Σʸ₂, mean_of_variables, Σʸ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂



    # third order

    kron_s_v = ℒ.kron(s_in_s⁺, v_in_s⁺)
    kron_e_v = ℒ.kron(e_in_s⁺, v_in_s⁺)

    ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)

    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
                                                𝓂.solution.perturbation.second_order_auxilliary_matrices, 
                                                𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

    s_s_s_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_s_s, s_in_s⁺)]
    s_s_e_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_s_s, e_in_s⁺)]
    s_e_e_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_s_e, e_in_s⁺)]
    e_e_e_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_e_e, e_in_s⁺)]
    s_v_v_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_s_v, v_in_s⁺)]
    e_v_v_to_y₃ = 𝐒₃[obs_in_var_idx,:][:, ℒ.kron(kron_e_v, v_in_s⁺)]

    s_s_s_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, s_in_s⁺)]
    s_s_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_s, e_in_s⁺)]
    s_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_e, e_in_s⁺)]
    e_e_e_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_e, e_in_s⁺)]
    s_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_s_v, v_in_s⁺)]
    e_v_v_to_s₃ = 𝐒₃[iˢ, ℒ.kron(kron_e_v, v_in_s⁺)]


    # # Set up in pruned state transition matrices
    ŝ_to_ŝ₃ = [  s_to_s₁                      zeros(nˢ, 2*nˢ + 2*nˢ^2 + nˢ^3)
                                        zeros(nˢ, nˢ) s_to_s₁   s_s_to_s₂ / 2   zeros(nˢ, nˢ + nˢ^2 + nˢ^3)
                                        zeros(nˢ^2, 2 * nˢ)               s_to_s₁_by_s_to_s₁  zeros(nˢ^2, nˢ + nˢ^2 + nˢ^3)
                                        s_v_v_to_s₃ / 2    zeros(nˢ, nˢ + nˢ^2)      s_to_s₁       s_s_to_s₂    s_s_s_to_s₃ / 6
                                        ℒ.kron(s_to_s₁,v_v_to_s₂ / 2)    zeros(nˢ^2, 2*nˢ + nˢ^2)     s_to_s₁_by_s_to_s₁  ℒ.kron(s_to_s₁,s_s_to_s₂ / 2)    
                                        zeros(nˢ^3, 3*nˢ + 2*nˢ^2)   ℒ.kron(s_to_s₁,s_to_s₁_by_s_to_s₁)]
    # checked

    ê_to_ŝ₃ = [ e_to_s₁   zeros(nˢ,nᵉ^2 + 2*nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    zeros(nˢ,nᵉ)  e_e_to_s₂ / 2   s_e_to_s₂   zeros(nˢ,nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    zeros(nˢ^2,nᵉ)  e_to_s₁_by_e_to_s₁  I_plus_s_s * s_to_s₁_by_e_to_s₁  zeros(nˢ^2, nᵉ * nˢ + nᵉ * nˢ^2 + nᵉ^2 * nˢ + nᵉ^3)
                                    e_v_v_to_s₃ / 2    zeros(nˢ,nᵉ^2 + nᵉ * nˢ)  s_e_to_s₂    s_s_e_to_s₃ / 2    s_e_e_to_s₃ / 2    e_e_e_to_s₃ / 6
                                    ℒ.kron(e_to_s₁, v_v_to_s₂ / 2)    zeros(nˢ^2, nᵉ^2 + nᵉ * nˢ)      s_s * s_to_s₁_by_e_to_s₁    ℒ.kron(s_to_s₁, s_e_to_s₂) + s_s * ℒ.kron(s_s_to_s₂ / 2, e_to_s₁)  ℒ.kron(s_to_s₁, e_e_to_s₂ / 2) + s_s * ℒ.kron(s_e_to_s₂, e_to_s₁)  ℒ.kron(e_to_s₁, e_e_to_s₂ / 2)
                                    zeros(nˢ^3, nᵉ + nᵉ^2 + 2*nᵉ * nˢ) ℒ.kron(s_to_s₁_by_s_to_s₁,e_to_s₁) + ℒ.kron(s_to_s₁, s_s * s_to_s₁_by_e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_s_to_s₁) * e_ss   ℒ.kron(s_to_s₁_by_e_to_s₁,e_to_s₁) + ℒ.kron(e_to_s₁,s_to_s₁_by_e_to_s₁) * e_es + ℒ.kron(e_to_s₁, s_s * s_to_s₁_by_e_to_s₁) * e_es  ℒ.kron(e_to_s₁,e_to_s₁_by_e_to_s₁)]
    #checked

    ŝ_to_y₃ = [s_to_y₁ + s_v_v_to_y₃ / 2  s_to_y₁  s_s_to_y₂ / 2   s_to_y₁    s_s_to_y₂     s_s_s_to_y₃ / 6]
    #checked

    ê_to_y₃ = [e_to_y₁ + e_v_v_to_y₃ / 2  e_e_to_y₂ / 2  s_e_to_y₂   s_e_to_y₂     s_s_e_to_y₃ / 2    s_e_e_to_y₃ / 2    e_e_e_to_y₃ / 6]
    #checked

    μˢ₃δμˢ₁ = reshape((ℒ.I - s_to_s₁_by_s_to_s₁) \ vec( 
                                (s_s_to_s₂  * reshape(ss_s * vec(Σᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σᶻ₁) * Δμˢ₂'),nˢ^2, nˢ) +
                                s_s_s_to_s₃ * reshape(Σᶻ₂[2 * nˢ + 1 : end , 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', nˢ^3, nˢ) / 6 +
                                s_e_e_to_s₃ * ℒ.kron(Σᶻ₁, vec(ℒ.I(nᵉ))) / 2 +
                                s_v_v_to_s₃ * Σᶻ₁ / 2) * s_to_s₁' +
                                (s_e_to_s₂  * ℒ.kron(Δμˢ₂,ℒ.I(nᵉ)) +
                                e_e_e_to_s₃ * reshape(e⁴, nᵉ^3, nᵉ) / 6 +
                                s_s_e_to_s₃ * ℒ.kron(vec(Σᶻ₁), ℒ.I(nᵉ)) / 2 +
                                e_v_v_to_s₃ * ℒ.I(nᵉ) / 2) * e_to_s₁'
                                ), nˢ, nˢ)
    #checked


    Γ₃ = [ ℒ.I(nᵉ)             spzeros(nᵉ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δμˢ₂', ℒ.I(nᵉ))  ℒ.kron(vec(Σᶻ₁)', ℒ.I(nᵉ)) spzeros(nᵉ, nˢ * nᵉ^2)    reshape(e⁴, nᵉ, nᵉ^3)
            spzeros(nᵉ^2, nᵉ)    reshape(e⁴, nᵉ^2, nᵉ^2) - vec(ℒ.I(nᵉ)) * vec(ℒ.I(nᵉ))'     spzeros(nᵉ^2, 2*nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            spzeros(nˢ * nᵉ, nᵉ + nᵉ^2)    ℒ.kron(Σᶻ₁, ℒ.I(nᵉ))   spzeros(nˢ * nᵉ, nˢ*nᵉ + nˢ^2*nᵉ + nˢ*nᵉ^2 + nᵉ^3)
            ℒ.kron(Δμˢ₂,ℒ.I(nᵉ))    spzeros(nᵉ * nˢ, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σᶻ₂[nˢ + 1:2*nˢ,nˢ + 1:2*nˢ] + Δμˢ₂ * Δμˢ₂',ℒ.I(nᵉ)) ℒ.kron(Σᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δμˢ₂ * vec(Σᶻ₁)',ℒ.I(nᵉ))   spzeros(nᵉ * nˢ, nˢ * nᵉ^2) ℒ.kron(Δμˢ₂, reshape(e⁴, nᵉ, nᵉ^3))
            ℒ.kron(vec(Σᶻ₁), ℒ.I(nᵉ))  spzeros(nᵉ * nˢ^2, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Σᶻ₂[2 * nˢ + 1 : end, nˢ + 1:2*nˢ] + vec(Σᶻ₁) * Δμˢ₂', ℒ.I(nᵉ))  ℒ.kron(Σᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', ℒ.I(nᵉ))   spzeros(nᵉ * nˢ^2, nˢ * nᵉ^2)  ℒ.kron(vec(Σᶻ₁), reshape(e⁴, nᵉ, nᵉ^3))
            spzeros(nˢ*nᵉ^2, nᵉ + nᵉ^2 + 2*nᵉ * nˢ + nˢ^2*nᵉ)   ℒ.kron(Σᶻ₁, reshape(e⁴, nᵉ^2, nᵉ^2))    spzeros(nˢ*nᵉ^2,nᵉ^3)
            reshape(e⁴, nᵉ^3, nᵉ)  spzeros(nᵉ^3, nᵉ^2 + nᵉ * nˢ)    ℒ.kron(Δμˢ₂', reshape(e⁴, nᵉ^3, nᵉ))     ℒ.kron(vec(Σᶻ₁)', reshape(e⁴, nᵉ^3, nᵉ))  spzeros(nᵉ^3, nˢ*nᵉ^2)     reshape(e⁶, nᵉ^3, nᵉ^3)]
    #checked


    Eᴸᶻ = [ spzeros(nᵉ + nᵉ^2 + 2*nᵉ*nˢ + nᵉ*nˢ^2, 3*nˢ + 2*nˢ^2 +nˢ^3)
            ℒ.kron(Σᶻ₁,vec(ℒ.I(nᵉ)))   zeros(nˢ*nᵉ^2, nˢ + nˢ^2)  ℒ.kron(μˢ₃δμˢ₁',vec(ℒ.I(nᵉ)))    ℒ.kron(reshape(ss_s * vec(Σᶻ₂[nˢ + 1:2*nˢ,2 * nˢ + 1 : end] + Δμˢ₂ * vec(Σᶻ₁)'), nˢ, nˢ^2), vec(ℒ.I(nᵉ)))  ℒ.kron(reshape(Σᶻ₂[2 * nˢ + 1 : end, 2 * nˢ + 1 : end] + vec(Σᶻ₁) * vec(Σᶻ₁)', nˢ, nˢ^3), vec(ℒ.I(nᵉ)))
            spzeros(nᵉ^3, 3*nˢ + 2*nˢ^2 +nˢ^3)]
    # checked

    A = ê_to_ŝ₃ * Eᴸᶻ * ŝ_to_ŝ₃'

    C = ê_to_ŝ₃ * Γ₃ * ê_to_ŝ₃' + A + A'

    # if size(initial_guess³) == (0,0)
    #     initial_guess³ = collect(C)
    # end

    if length(C) < 1e7
        # println("Using Krylov")
        function sylvester!(sol,𝐱)
            𝐗 = reshape(𝐱, size(C))
            sol .= vec(ŝ_to_ŝ₃ * 𝐗 * ŝ_to_ŝ₃' - 𝐗)
            return sol
        end

        sylvester = LinearOperators.LinearOperator(Float64, length(C), length(C), true, true, sylvester!)

        Σ̂ᶻ₃, info = Krylov.gmres(sylvester, sparsevec(collect(-C)), atol = eps())

        if !info.solved
            Σ̂ᶻ₃, info = Krylov.bicgstab(sylvester, sparsevec(collect(-C)), atol = eps())
        end

        Σᶻ₃ = reshape(Σ̂ᶻ₃, size(C))
    else
        # println("Using Iteration")
        soll = speedmapping(collect(C); m! = (Σᶻ₃, Σ̂ᶻ₃) -> Σᶻ₃ .= ŝ_to_ŝ₃ * Σ̂ᶻ₃ * ŝ_to_ŝ₃' + C, 
        # time_limit = 200, 
        stabilize = true)
        
        # println(soll.maps)
        Σᶻ₃ = soll.minimizer

        if !soll.converged
            return Inf
        end
    end

    Σʸ₃ = ŝ_to_y₃ * Σᶻ₃ * ŝ_to_y₃' + ê_to_y₃ * Γ₃ * ê_to_y₃'
end



out = calculate_third_order_moments(m.parameter_values, :all, m)


calculate_third_order_moments(m.parameter_values, :y, m)
calculate_third_order_moments(m.parameter_values, :n, m)

out[obs_in_var_idx,:]
obs_in_var_idx = indexin([:y], m.timings.var)
m.var

using BenchmarkTools

@benchmark calculate_third_order_moments(m.parameter_values, m)
@profview for i in 1:100 calculate_third_order_moments(m.parameter_values, m) end
