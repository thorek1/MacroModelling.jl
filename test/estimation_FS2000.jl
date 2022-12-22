import Turing
import Turing: Normal, Beta, Gamma, InverseGamma, Uniform, Poisson, truncated, NUTS, HMC, HMCDA, MH, PG, SMC, IS, sample, mean, var, @logprob_str, logpdf
using MacroModelling, OptimizationNLopt, OptimizationOptimisers
using CSV, DataFrames, AxisKeys

# load and solve model once
include("models/FS2000.jl")
# m.SS_optimizer = NLopt.LD_TNEWTON_PRECOND_RESTART
# m.SS_optimizer = NLopt.LN_NEWUOA_BOUND # nope
# m.SS_optimizer = NLopt.GD_STOGO# nope
# m.SS_optimizer = NLopt.LN_COBYLA# nope
# m.SS_optimizer = NLopt.LN_COBYLA# nope
# m.SS_optimizer = NLopt.LN_PRAXIS
# m.SS_optimizer = NLopt.LN_NELDERMEAD
# m.SS_optimizer = NLopt.LD_MMA #nope
# m.SS_optimizer = NLopt.LD_SLSQP
# m.SS_optimizer = NLopt.LD_VAR1
# m.SS_optimizer = Optimisers.ADAM
# m.SS_optimizer = Optimisers.AdaMax
# solve!(m,symbolic_SS = true)
get_solution(m)
get_SS(m)

# m.NSSS_solver_cache
# # get_SS(m,parameters = [0.44728295202065166, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])
# length(m.NSSS_solver_cache) < 1
# # m.SS_solve_func
# parameters = [0.4035,0.9909, 0.0046, 1.0143, 0.8455, 0.6890, 0.0017, 0.0136, 0.0033]


# findmin([sum(abs2,pars[end] - parameters) for pars in m.NSSS_solver_cache])[2]
# findmin([sum(abs2,pars[end] ./ parameters .- 1) for pars in m.NSSS_solver_cache])[2]

# get_SS(m,parameters = [0.6750005457453657, 0.7705742051621937, -0.13003647764699267, 0.6057594085497515, 0.7171103532068533, 0.7901279425902789, 0.5380666025781062, 0.2961217015642633, 4.264173335281647])
# get_SS(m,parameters = [0.6695526157125993, 0.7782097900770949, 1.399954432139673, 0.9777895536920106, 0.7173213885987454, 0.7899294918080639, 0.5385159934129353, 0.29591747435564625, 4.253028889230297])
# # get_SS(m,parameters = ([0.6585547093545611, 0.797796110027983, -27.99086509624363, 1.7260870685765246, 0.7176543116605183, 0.7893164495740828, 0.5286306119368696, 0.2900532604080456, 4.158761080384637] .+ [0.6695526157125993, 0.7782097900770949, 1.399954432139673, 0.9777895536920106, 0.7173213885987454, 0.7899294918080639, 0.5385159934129353, 0.29591747435564625, 4.253028889230297])/2)
# get_SS(m,parameters = [0.6585547093545611, 0.797796110027983, -27.99086509624363, 1.7260870685765246, 0.7176543116605183, 0.7893164495740828, 0.5286306119368696, 0.2900532604080456, 4.158761080384637])

# old = [0.6695526157125993, 0.7782097900770949, 1.399954432139673, 0.9777895536920106, 0.7173213885987454, 0.7899294918080639, 0.5385159934129353, 0.29591747435564625, 4.253028889230297]
# new = [0.6585547093545611, 0.797796110027983, -27.99086509624363, 1.7260870685765246, 0.7176543116605183, 0.7893164495740828, 0.5286306119368696, 0.2900532604080456, 4.158761080384637]
# x = .918
# get_SS(m,parameters = x * old + (1-x) * new)







# get_SS(m,parameters = [0.17276964952137486, 0.7920896296538025, -0.8033846461311837, 0.8677000303674002, 0.16328753269059, 0.8373335368867071, 0.7368146848615, 1.0262811493098323, 0.17124795333976325])

# old = [0.17276964952137486, 0.7920896296538025, -0.8033846461311837, 0.8677000303674002, 0.16328753269059, 0.8373335368867071, 0.7368146848615, 1.0262811493098323, 0.17124795333976325]
# new = [0.42765909391665374, 0.9907036907663795, 2.1176282461553012e76, 1.1960317010498268e10, 0.16585006558379037, 0.8188150453606395, 0.15798111883923274, 0.39220085985173697, 0.0752297106516448]
# x = .999
# get_SS(m,parameters = x * old + (1-x) * new)


# get_SS(m,parameters = [0.4506243383700943, 0.8339225002267898, 0.1800531394330218, 1.051725550641863, 0.7507896555638963, 0.195768694997821, 0.22133536264548384, 2.821842569292302, 0.19527805437581103])

# ([0.6585547093545611, 0.797796110027983, -27.99086509624363, 1.7260870685765246, 0.7176543116605183, 0.7893164495740828, 0.5286306119368696, 0.2900532604080456, 4.158761080384637] .+ [0.6695526157125993, 0.7782097900770949, 1.399954432139673, 0.9777895536920106, 0.7173213885987454, 0.7899294918080639, 0.5385159934129353, 0.29591747435564625, 4.253028889230297])/2



parameters = deepcopy(m.parameter_values)

# load data
dat = CSV.read("test/FS2000_data.csv",DataFrame)
# data = KeyedArray(Array(dat)',Variable = Symbol.(names(dat)),Time = 1:size(dat)[1])
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
# observables = Symbol.(names(dat))#[:dinve, :dc]
observables = sort(Symbol.("log_".*names(dat)))#[:dinve, :dc]

data = data(observables,:)

# calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
# calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [0.4035, 0.9909, 0.0046, 1.0143, 0.8455, 0.6891, 0.0017, 0.0136, 0.0033])

# functions to map mean and standard deviations to distribution parameters
function beta_map(μ, σ) 
    α = ((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2
    β = α * (1 / μ - 1)
    return α, β
end

function inv_gamma_map(μ, σ)
    α = (μ / σ) ^ 2 + 2
    β = μ * ((μ / σ) ^ 2 + 1)
    return α, β
end

function gamma_map(μ, σ)
    k = μ^2/σ^2 
    θ = σ^2 / μ
    return k, θ
end

# test functions
# Gamma(gamma_map(.5,.2)...)|>mean
# Gamma(gamma_map(.5,.2)...)|>var|>sqrt

# InverseGamma(inv_gamma_map(.1,2)...)|>mean
# InverseGamma(inv_gamma_map(.1,2)...)|>var|>sqrt

# Beta(beta_map(0.356, 0.02)...)|>mean
# Beta(beta_map(0.356, 0.02)...)|>var|>sqrt

# Normal(0.0085, 0.003)|>mean
# Normal(0.0085, 0.003)|>var|>sqrt


# calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

# @profview calculate_kalman_filter_loglikelihood(m, data(observables,1:10), observables; parameters = parameters)
# @time calculate_kalman_filter_loglikelihood(m, data(observables,1:10), observables; parameters = parameters)
# using ForwardDiff
#find mode
function calculate_posterior_loglikelihood(parameters, u)
    # println(ForwardDiff.value.(parameters))
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
    log_lik = 0
    log_lik -= calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    log_lik -= logpdf(Beta(beta_map(0.356, 0.02)...),alp)
    log_lik -= logpdf(Beta(beta_map(0.993, 0.002)...),bet)
    log_lik -= logpdf(Normal(0.0085, 0.003),gam)
    log_lik -= logpdf(Normal(1.0002, 0.007),mst)
    log_lik -= logpdf(Beta(beta_map(0.129, 0.223)...),rho)
    log_lik -= logpdf(Beta(beta_map(0.65, 0.05)...),psi)
    log_lik -= logpdf(Beta(beta_map(0.01, 0.005)...),del)
    log_lik -= logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),z_e_a)
    log_lik -= logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),z_e_m)
    return log_lik
end

# using BenchmarkTools
# @benchmark find_mode(parameters, [])
# @profview find_mode(parameters, [])
# find_mode([0.4035,0.9909, 0.0046, 1.0143, 0.8455, 0.6890, 0.0017, 0.0136, 0.0033],[])
using OptimizationNLopt
using OptimizationOptimisers

# parameters = [0.6750005457453657, 0.7705742051621937, -0.13003647764699267, 0.6057594085497515, 0.7171103532068533, 0.7901279425902789, 0.5380666025781062, 0.2961217015642633, 4.264173335281647]
# parameters = [0.4035,0.9909, 0.0046, 1.0143, 0.8455, 0.6890, 0.0017, 0.0136, 0.0033]

# lbs = [.0001,.5,eps(),.5,.0001,.0001,.0001,.000001,.000001]
# ubs = [.6,.999,.5,1.5,.95,.999,.1,1,1]


lbs = [eps(), eps(), -1e12, -1e12, eps(), eps(), eps(), eps(), eps()]
ubs = [1-eps(), 1-eps(), 1e12, 1e12, 1-eps(), 1-eps(), 1-eps(), 1e12, 1e12]

f = OptimizationFunction(calculate_posterior_loglikelihood, Optimization.AutoForwardDiff())

prob = OptimizationProblem(f, Float64[parameters...], [])#, lb = lbs, ub = ubs)
sol = solve(prob, Optimisers.ADAM(), maxiters = 1000, progress = true)
sol.minimum
# m.NSSS_solver_cache

prob = OptimizationProblem(f, sol.u, [], lb = lbs, ub = ubs)
sol_new = solve(prob, NLopt.LD_LBFGS(), maxiters = 100000)
sol_new.minimum


# sol_new = solve(prob, NLopt.G_MLSL_LDS(), local_method = NLopt.LN_BOBYQA(), population = length(ubs), local_maxtime = 120, maxtime = 120, progress = true)

# prob = OptimizationProblem(f, Float64[parameters...], [])#, lb = lbs, ub = ubs)
# sol_new = solve(prob, Optimisers.ADAM(), maxiters = 1000, progress = true)

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.LN_BOBYQA(), maxiters = 100000, maxtime = 100, local_maxtime = 100)
# sol.minimum

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.LN_COBYLA(), maxiters = 100000, maxtime = 100, local_maxtime = 100)
# sol.minimum

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 100000, maxtime = 100, local_maxtime = 100)
# sol.minimum

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.LD_TNEWTON_PRECOND_RESTART(), maxiters = 100000, maxtime = 100, local_maxtime = 100)

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.GD_STOGO_RAND(), maxiters = 100000, maxtime = 10, local_maxtime = 10)

# prob = OptimizationProblem(f, Float64[parameters...], [], lb = lbs, ub = ubs)
# sol = solve(prob, NLopt.GN_AGS(), maxiters = 100000, maxtime = 10, local_maxtime = 10)

# prob = OptimizationProblem(f, Float64[parameters...], [])#, lb = fill(eps(),9), ub = [1,1,1,2,1,1,1,10,10])
# sol = solve(prob, Optimisers.AdaMax(), maxiters = 10000)


# lbs = [.2,.5,eps(),.5,.0001,.0001,.0001,.000001,.000001]
# ubs = [.5,.999,.5,1.5,.95,.999,.1,1,1]

# using ForwardDiff
# define priors and likelihood
Turing.@model function kalman(data, m, observables)
    # parameters = deepcopy(m.parameter_values)
    alp     ~ truncated(Beta(beta_map(0.356, 0.02)...), .0001, .5)
    bet     ~ truncated(Beta(beta_map(0.993, 0.002)...), .5, .999)
    gam     ~ truncated(Normal(0.0085, 0.003),eps(),.5)
    mst     ~ truncated(Normal(1.0002, 0.007), .5, 1.5)
    rho     ~ truncated(Beta(beta_map(0.129, 0.223)...), .0001, .999)
    psi     ~ truncated(Beta(beta_map(0.65, 0.05)...), .0001, .999)
    del     ~ truncated(Beta(beta_map(0.01, 0.005)...), .0001, .1)
    z_e_a   ~ truncated(InverseGamma(inv_gamma_map(0.035449, Inf)...), eps(), 1.5)
    z_e_m   ~ truncated(InverseGamma(inv_gamma_map(0.008862, Inf)...), eps(), 1.5)
    # z_e_a   ~ Uniform(eps(),2)
    # z_e_m   ~ Uniform(eps(),2)

    parameters[indexin([:alp],m.parameters)] .= alp
    parameters[indexin([:bet],m.parameters)] .= bet
    parameters[indexin([:rho],m.parameters)] .= rho
    parameters[indexin([:psi],m.parameters)] .= psi
    parameters[indexin([:del],m.parameters)] .= del
    parameters[indexin([:gam],m.parameters)] .= gam
    parameters[indexin([:mst],m.parameters)] .= mst
    parameters[indexin([:z_e_a],m.parameters)] .= z_e_a
    parameters[indexin([:z_e_m],m.parameters)] .= z_e_m

    # println(ForwardDiff.value.(parameters))
    data_log_lik = calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    # println(ForwardDiff.value.(data_log_lik))
    # data likelihood
    Turing.@addlogprob! data_log_lik

    Turing.@addlogprob! logpdf(Beta(beta_map(0.356, 0.02)...),alp)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.993, 0.002)...),bet)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.129, 0.223)...),rho)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.65, 0.05)...),psi)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.01, 0.005)...),del)
    Turing.@addlogprob! logpdf(Normal(0.0085, 0.003),gam)
    Turing.@addlogprob! logpdf(Normal(1.0002, 0.007),mst)
    Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),z_e_a)
    Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),z_e_m)
    # # system prior on IRFs
    # Turing.@addlogprob! loglikelihood(Normal(.03, .01), get_irf(parameters,m)[6,4,2])

    # # system prior on SS and stdev
    # moments = get_variances(RBC_estim, parameters)
    # Turing.@addlogprob! loglikelihood(Normal(.6, .05), moments[1][1])
    # Turing.@addlogprob! loglikelihood(Normal(.014, .001), moments[2][5])
end

# get_SS(m,parameters = [.495,.99765,.0015,1.0043,.59,.509,.0075,.0789,.013])
# m.solution.non_stochastic_steady_state

# logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),0.035449)
# logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),0.008862)
# # ldens(idx) = log(2) - gammaln(.5*nu(idx)) - .5*nu(idx).*(log(2)-log(s(idx))) - (nu(idx)+1).*log(x(idx)) - .5*s(idx)./(x(idx).*x(idx)) ;
# # nu = 2
# logpdf(Normal(1.0002, 0.007),1.0002)
# logpdf(Normal(0.0085, 0.003),0.0085)


# logpdf(Beta(beta_map(0.356, 0.02)...),  0.356)
# logpdf(Beta(beta_map(0.993, 0.002)...), 0.993)
# logpdf(Beta(beta_map(0.129, 0.223)...), 0.129)
# logpdf(Beta(beta_map(0.65, 0.05)...),   0.65)
# logpdf(Beta(beta_map(0.01, 0.005)...),  0.01)



turing_model = kalman(data, m, observables) # passing observables from before 

# logprob"data = data, observables = observables, m = m | model = turing_model,  alp = 0.356, bet = 0.993, gam = 0.0085, mst = 1.0002, rho = 0.129, psi = 0.65, del = 0.01, z_e_a = 0.035449, z_e_m = 0.008862"
# pars = [0.5999999999999941, 0.999, 0.00022410786148670898, 1.5, 0.2918845841974984, 0.0023597321974304225, 0.0036122338038623494, 9.667230108300879e-16, 6.186816831182828e-7]
# get_SS(m, parameters = pars)
# x = .04
# get_SS(m, parameters = pars * x + m.NSSS_solver_cache[100][2] * (1 - x))
# findmin([sum(abs2,i[2]- pars) for i in m.NSSS_solver_cache])
# sample
n_samples = 100
# chain_NUTS = sample(turing_model, NUTS(1000, .65, max_depth = 10, Δ_max = 400.0, init_ϵ = .02), n_samples; θ = sol.u, progress = true)
chain_NUTS = sample(turing_model, NUTS(), n_samples; progress = true)
# chain_HMC = sample(turing_model, HMC(.02,10), n_samples; θ = sol.u, progress = true)
chain_HMCDA = sample(turing_model, HMCDA(2000, 0.65, .02), n_samples; θ = sol.u, progress = true)
# chain_PG = sample(turing_model, PG(20), n_samples; θ = sol.u, progress = true)
chain_MH = sample(turing_model, MH(), Int(1e5); θ = sol.u, progress = true)
# chain_MH = sample(turing_model, SMC(), Int(1e5); θ = sol.u, progress = true)






m.SS_init_guess
get_SS(m,parameters = [0.4462257879893407, 0.9988469595418614, 0.0037864220598411366, 1.008999298796174, 0.6052810770277287, 0.5360474465184352, 0.0015706409192591743, 0.07484474682965438, 0.012156383477978388])

get_SS(m,parameters = [0.447388317851787, 0.9988884614609232, 0.002725486533226825, 1.0084286121820523, 0.6187187429894266, 0.5315973807951158, 0.0011616665715356112, 0.07715423484471479, 0.012839096692139697])

get_SS(m,parameters = [0.33, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])

get_SS(m,parameters = [0.44728295202065166, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])
