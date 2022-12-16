import Turing
import Turing: Normal, Beta, Gamma, InverseGamma, Uniform, truncated, NUTS, HMC, HMCDA, MH, PG, SMC, IS, sample, mean, var
using MacroModelling, OptimizationNLopt
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
# solve!(m,symbolic_SS = true)
get_solution(m)
get_SS(m)
# get_SS(m,parameters = [0.44728295202065166, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])



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

calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [0.4035, 0.9909, 0.0046, 1.0143, 0.8455, 0.6891, 0.0017, 0.0136, 0.0033])

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

using ForwardDiff
# define priors and likelihood
Turing.@model function kalman(data, m, observables)
    # parameters = deepcopy(m.parameter_values)
    # alp     ~ Beta(beta_map(0.356, 0.02)...)
    # bet     ~ Beta(beta_map(0.993, 0.002)...)
    # rho     ~ Beta(beta_map(0.129, 0.223)...)
    # psi     ~ Beta(beta_map(0.65, 0.05)...)
    # del     ~ Beta(beta_map(0.01, 0.005)...)
    # gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    z_e_a   ~ InverseGamma(inv_gamma_map(0.035449, Inf)...)
    z_e_m   ~ InverseGamma(inv_gamma_map(0.008862, Inf)...)
    # z_e_a   ~ Uniform(eps(),2)
    # z_e_m   ~ Uniform(eps(),2)

    # parameters[indexin([:alp],m.parameters)] .= alp
    # parameters[indexin([:bet],m.parameters)] .= bet
    # parameters[indexin([:rho],m.parameters)] .= rho
    # parameters[indexin([:psi],m.parameters)] .= psi
    # parameters[indexin([:del],m.parameters)] .= del
    # parameters[indexin([:gam],m.parameters)] .= gam
    parameters[indexin([:mst],m.parameters)] .= mst
    parameters[indexin([:z_e_a],m.parameters)] .= z_e_a
    parameters[indexin([:z_e_m],m.parameters)] .= z_e_m

    # println(ForwardDiff.value.(parameters))
    data_log_lik = calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    # println(ForwardDiff.value.(data_log_lik))
    # data likelihood
    Turing.@addlogprob! data_log_lik
    # # system prior on IRFs
    # Turing.@addlogprob! loglikelihood(Normal(.03, .01), get_irf(parameters,m)[6,4,2])

    # # system prior on SS and stdev
    # moments = get_variances(RBC_estim, parameters)
    # Turing.@addlogprob! loglikelihood(Normal(.6, .05), moments[1][1])
    # Turing.@addlogprob! loglikelihood(Normal(.014, .001), moments[2][5])
end

# get_SS(m,parameters = [.495,.99765,.0015,1.0043,.59,.509,.0075,.0789,.013])
# m.solution.non_stochastic_steady_state

turing_model = kalman(data, m, observables) # passing observables from before 

# sample
n_samples = 1000
chain_1_marginal = sample(turing_model, NUTS(), n_samples; progress = true)
chain_1_marginal = sample(turing_model, HMC(.01,10), n_samples; progress = true)
chain_HMCDA = sample(turing_model, HMCDA(200, 0.65, 0.1), n_samples; progress = true)
chain_PG = sample(turing_model, PG(20), n_samples; progress = true)
chain_MH = sample(turing_model, MH(), Int(1e5); progress = true)
chain_MH = sample(turing_model, SMC(), Int(1e5); progress = true)






m.SS_init_guess
get_SS(m,parameters = [0.4462257879893407, 0.9988469595418614, 0.0037864220598411366, 1.008999298796174, 0.6052810770277287, 0.5360474465184352, 0.0015706409192591743, 0.07484474682965438, 0.012156383477978388])

get_SS(m,parameters = [0.447388317851787, 0.9988884614609232, 0.002725486533226825, 1.0084286121820523, 0.6187187429894266, 0.5315973807951158, 0.0011616665715356112, 0.07715423484471479, 0.012839096692139697])

get_SS(m,parameters = [0.33, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])

get_SS(m,parameters = [0.44728295202065166, 0.9988884445965371, 0.0028058657322557485, 1.0075373840302122, 0.6142408307950538, 0.5318308372997724, 0.0011670521404654407, 0.07535090815216333, 0.012834752100689741])
