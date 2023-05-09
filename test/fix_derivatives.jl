using MacroModelling
import MacroModelling: timings, ℳ
using ForwardDiff, FiniteDifferences, Zygote
import Optim, LineSearches
import ChainRulesCore: @ignore_derivatives
using ImplicitDifferentiation
import LinearAlgebra as ℒ
using LinearMaps
import IterativeSolvers as ℐ

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



data = simulate(RBC_CME)[:,:,1]
observables = [:c,:k]
calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = RBC_CME.parameter_values)



reverse_grad = Zygote.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))[1]


# @test isapprox(420.25039827148197,calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables),rtol = 1e-5)

forw_grad = ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))



function calculate_covariance_forward(𝑺₁::AbstractMatrix{<: Real}; T::timings, subset_indices::Vector{Int64})
    A = @views 𝑺₁[subset_indices,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[indexin(T.past_not_future_and_mixed_idx,subset_indices),:]
    C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]
    
    CC = C * C'

    lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))
    
    # reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))
    return reshape(ℐ.gmres(lm, vec(-CC)), size(CC)), true
end




function calculate_covariance_conditions(𝑺₁::AbstractMatrix{<: Real}, covar::AbstractMatrix{<: Real}, z::Bool; T::timings, subset_indices::Vector{Int64}) where S
    A = @views 𝑺₁[subset_indices,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,subset_indices)),:]
    C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]
    
    A * covar * A' + C * C' .- covar
end


calculate_covariance_AD = ImplicitFunction(calculate_covariance_forward, calculate_covariance_conditions)

# calculate_covariance_AD(sol; T, subset_indices) = ImplicitFunction(sol->calculate_covariance_forward(sol, T=T, subset_indices = subset_indices), (x,y)->calculate_covariance_conditions(x,y,T=T, subset_indices = subset_indices))
# calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

function calc_kalman_filter_loglikelihood(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose::Bool = false, tol::AbstractFloat = eps())
    @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    @ignore_derivatives sort!(observables)

    @ignore_derivatives solve!(𝓂, verbose = verbose)

    if isnothing(parameters)
        parameters = 𝓂.parameter_values
    else
        ub = @ignore_derivatives fill(1e12+rand(),length(𝓂.parameters) + length(𝓂.➕_vars))
        lb = @ignore_derivatives -ub

        for (i,v) in enumerate(𝓂.bounded_vars)
            if v ∈ 𝓂.parameters
                @ignore_derivatives lb[i] = 𝓂.lower_bounds[i]
                @ignore_derivatives ub[i] = 𝓂.upper_bounds[i]
            end
        end

        if min(max(parameters,lb),ub) != parameters 
            return -Inf
        end
    end

    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    if !solved
        return -Inf
    end

    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

    A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    # println(sol)
    P, _ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])
# println(P)
    # P = calculate_covariance_forward(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])
    # println(P)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
    u = zeros(length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u
    
    loglik = 0.0

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
end

calc_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = RBC_CME.parameter_values)


reverse_grad = Zygote.gradient(x->calc_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))[1]



reverse_grad = Zygote.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))[1]


# @test isapprox(420.25039827148197,calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables),rtol = 1e-5)

forw_grad = ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))

fin_grad = FiniteDifferences.grad(central_fdm(4,1),x->calc_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),RBC_CME.parameter_values)[1]

@test isapprox(forw_grad,fin_grad, rtol = 1e-6)
@test isapprox(forw_grad,reverse_grad, rtol = eps(Float32))




get_solution(RBC_CME)

get_irf(RBC_CME; parameters = RBC_CME.parameter_values)

forw_grad = ForwardDiff.gradient(x->get_irf(RBC_CME, x)[4,1,2],Float64.(RBC_CME.parameter_values))

fin_grad = FiniteDifferences.grad(central_fdm(2,1),x->get_irf(RBC_CME, x)[4,1,2],RBC_CME.parameter_values)[1]

@test isapprox(forw_grad,fin_grad,rtol = 1e-5)
