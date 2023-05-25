using MacroModelling
import Turing
import Turing: NUTS, sample, logpdf
# import Optim, LineSearches, Plots
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL: logjoint
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives, ignore_derivatives

cd("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/test")
include("models/FS2000.jl")

FS2000 = m

# load data
dat = CSV.read("data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)

# declare parameters 
alp     = 0.356
bet     = 0.993
gam     = 0.0085
mst     = 1.0002
rho     = 0.129
psi     = 0.65
del     = 0.01
z_e_a   = 0.035449
z_e_m   = 0.008862
parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]
# filter data with parameters
filtered_errors = MacroModelling.get_estimated_shocks(FS2000, data; parameters= parameters) # filtered_states = get_estimated_variables(FS2000, data; parameters= parameters)

# Define DSGE Turing model
Turing.@model function FS2000_filter_free_loglikelihood_function(data, model, observables)
   
    alp     ~ Beta(0.356, 0.02, μσ = true)
    #bet     ~ Beta(0.993, 0.002, μσ = true)
    #gam     ~ Normal(0.0085, 0.003)
    #mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    #psi     ~ Beta(0.65, 0.05, μσ = true)
    #del     ~ Beta(0.01, 0.005, μσ = true)
    #z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    #z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    
    #alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    #rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862

    # Log likehood function inputs -
    # I did not manage to delegate the sampling to another function - Would it be possible to call it in with an include() command? 
    shock_distribution = Turing.Normal()
    algorithm = :first_order 
    parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]
    verbose::Bool = false
    tol::AbstractFloat = eps()
    filter = :filter_free

# BEGINNING OF OBJECTIVE FUNCTION 

     # draw intial conditions
     x0 ~ Turing.filldist(shock_distribution,m.timings.nPast_not_future_and_mixed) # Initial conditions  

     # draw errors
     ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2)) #Shocks  
 
     # reshape errors to vector
     ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))
 
     # Checks
     @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
     @assert length(observables) <= m.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."
 
     @ignore_derivatives sort!(observables)
     @ignore_derivatives solve!(m, verbose = verbose)
 
     if isnothing(parameters)
         parameters = m.parameter_values
     else
         ub = @ignore_derivatives fill(1e12 + rand(), length(m.parameters) + length(m.➕_vars))
         lb = @ignore_derivatives - ub
 
         for (i,v) in enumerate(m.bounded_vars)
             if v ∈ m.parameters
                 @ignore_derivatives lb[i] = m.lower_bounds[i]
                 @ignore_derivatives ub[i] = m.upper_bounds[i]
             end
         end
 
         if min(max(parameters,lb),ub) != parameters 
             return -Inf
         end
     end
 
     SS_and_pars, solution_error = m.SS_solve_func(parameters, m, verbose)
 
     if solution_error > tol || isnan(solution_error)
         return -Inf
     end
 
     NSSS_labels = @ignore_derivatives [sort(union(m.exo_present,m.var))...,m.calibration_equations_parameters...]
 
     obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)
 
     data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]
 
     observables_and_states = @ignore_derivatives sort(union(m.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(m.aux,m.var,m.exo_present)))))
 
     # solve DSGE with parameters
     solution = get_solution(m, parameters, algorithm = algorithm)
 
     # store solution 
     if algorithm == :first_order
         𝐒₁ = solution[2]
     else
         𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
     end
 
     # Thore: we can probably skip this because it is computationally expensive and should drop out in sampling ... Mátyás: We cannot as the initial condition bias for the erros is important
     
     # Option 1 - no initial condition sampling - biased errors but faster - no need to compute LR covariance matrix
         # x0 = zeros(m.timings.nPast_not_future_and_mixed)
     
     # Option 2 - initial condition is sampled - unbiased errors - slow as LR covariance is needed.
         calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = m.timings.past_not_future_and_mixed_idx)    
         long_run_covariance = calculate_covariance_(solution[2])
         initial_conditions =long_run_covariance  * x0 # x0 
 
     # Declare states
     state = zeros(typeof(ϵ_draw[1]), m.timings.nVars, size(data, 2) )
 
     # propagate the state space
     if algorithm == :first_order
         
         aug_state = [initial_conditions
         ϵ[:,1]]
         state[:,1] .=  𝐒₁ * aug_state
         
         for t in 2:size(data, 2)
             aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                         ϵ[:,t]]
             state[:,t] .=  𝐒₁ * aug_state 
         end
     elseif algorithm == :second_order
         
         aug_state = [initial_conditions
         1 
         ϵ[:,1]]
         state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
 
         for t in 2:size(data, 2)
             aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                         1 
                         ϵ[:,t]]
             state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
         end
 
     elseif algorithm == :pruned_second_order
 
         aug_state = [initial_conditions
         1 
         ϵ[:,1]]
 
         state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
 
         for t in 2:size(data, 2)
             aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                         1 
                         ϵ[:,t]]
             state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
         end
     end
 
     # define data in deviations form SS
     data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]
     
     # compute observation predictions - without ME
     state_deviations = data_in_deviations - state[obs_indices,:]
     # make_sure_state_equals_observable = sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(1e-4*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)]) *10^2
     make_sure_state_equals_observable = sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(1e-8*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)])
     # make_sure_state_equals_observable = -sum(abs2,state_deviations) * 1e30
# END OF OBJECTIVE FUNCTION 
    
    Turing.@addlogprob! make_sure_state_equals_observable#calculate_filterfree_loglikelihood(model, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_filterfree = FS2000_filter_free_loglikelihood_function(data, FS2000, observables)

n_samples = 1000

samps = sample(FS2000_filterfree, NUTS(), n_samples, progress = true)#, init_params = sol)

symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(samps[:, [Symbol("ϵ_draw[$a]") for a in 1:m.timings.nExo*size(data,2)], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = Turing.describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_mean = reshape(ϵ_mean, m.timings.nExo, Integer(size(ϵ_mean,1)/m.timings.nExo))
ϵ_std = tmp[1][:, 3]
ϵ_std = reshape(ϵ_std,  m.timings.nExo, Integer(size(ϵ_std,1)/m.timings.nExo))

sum(abs,ϵ_mean[1,end-20:end]-collect(filtered_errors[1,end-20:end]))<10^-4
