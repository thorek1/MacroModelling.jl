import Turing
import Turing: Normal, Beta, Gamma, InverseGamma, truncated, NUTS, sample, mean, var, MCMCDistributed
using MacroModelling
using CSV, DataFrames, AxisKeys

# load and solve model once
include("models/SW07.jl")
get_solution(m)
parameters = deepcopy(m.parameter_values)

# load data
dat = CSV.read("test/SW07_data.csv",DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.(names(dat)),Time = 1:size(dat)[1])

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

# Beta(beta_map(.5,.2)...)|>mean
# Beta(beta_map(.5,.2)...)|>var|>sqrt

get_SS(m)
using ForwardDiff
# define priors and likelihood
Turing.@model function kalman(data, m, observables)
    z_ea    ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)
    z_eb    ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.025,   5)
    z_eg    ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)
    z_eqs   ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)
    z_em    ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)
    z_epinf ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)
    z_ew    ~ truncated(InverseGamma(inv_gamma_map(0.1, 2)...), 0.01,    3)

    parameters[indexin([:z_ea],m.parameters)] .=    z_ea
    parameters[indexin([:z_eb],m.parameters)] .=    z_eb
    parameters[indexin([:z_eg],m.parameters)] .=    z_eg
    parameters[indexin([:z_eqs],m.parameters)] .=   z_eqs
    parameters[indexin([:z_em],m.parameters)] .=    z_em
    parameters[indexin([:z_epinf],m.parameters)] .= z_epinf
    parameters[indexin([:z_ew],m.parameters)] .=    z_ew


    constepinf    ~ truncated(Gamma(gamma_map(0.625, 0.1)...), 0.1,  2.0)
    constebeta    ~ truncated(Gamma(gamma_map(0.25,  0.1)...), 0.01, 2.0)

    parameters[indexin([:constepinf],m.parameters)] .= constepinf
    parameters[indexin([:constebeta],m.parameters)] .= constebeta


    crhoa   ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhob   ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhog   ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhoqs  ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhoms  ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhopinf~ truncated(Beta(beta_map(0.5,  0.2)...), 0.01,   .9999)
    crhow   ~ truncated(Beta(beta_map(0.5,  0.2)...), 0.001,  .9999)
    cmap    ~ truncated(Beta(beta_map(0.5,  0.2)...),  0.01,   .9999)
    cmaw    ~ truncated(Beta(beta_map(0.5,  0.2)...),  0.01,   .9999)
    chabb   ~ truncated(Beta(beta_map(0.7,  0.1)...),  0.001,  0.99)
    cprobw  ~ truncated(Beta(beta_map(0.5,  0.1)...),  0.3,    0.95)
    cprobp  ~ truncated(Beta(beta_map(0.5,  0.1)...), 0.5,    0.95)
    cindw   ~ truncated(Beta(beta_map(0.5,  0.15)...), 0.01,   0.99)
    cindp   ~ truncated(Beta(beta_map(0.5,  0.15)...), 0.01,   0.99)
    czcap   ~ truncated(Beta(beta_map(0.5,  0.15)...), 0.01,   1)
    crr     ~ truncated(Beta(beta_map(0.75, 0.1)...), 0.5,    0.975)

    parameters[indexin([:crhoa],m.parameters)] .= crhoa
    parameters[indexin([:crhob],m.parameters)] .= crhob
    parameters[indexin([:crhog],m.parameters)] .= crhog
    parameters[indexin([:crhoqs],m.parameters)] .= crhoqs
    parameters[indexin([:crhoms],m.parameters)] .= crhoms
    parameters[indexin([:crhopinf],m.parameters)] .= crhopinf
    parameters[indexin([:crhow],m.parameters)] .= crhow
    parameters[indexin([:cmap],m.parameters)] .= cmap
    parameters[indexin([:cmaw],m.parameters)] .= cmaw
    parameters[indexin([:chabb],m.parameters)] .= chabb
    parameters[indexin([:cprobw],m.parameters)] .= cprobw
    parameters[indexin([:cprobp],m.parameters)] .= cprobp
    parameters[indexin([:cindw],m.parameters)] .= cindw
    parameters[indexin([:cindp],m.parameters)] .= cindp
    parameters[indexin([:czcap],m.parameters)] .= czcap
    parameters[indexin([:crr],m.parameters)] .= crr


    csigl    ~ truncated(Normal(2.0     ,0.75), (0.25) ,(10))
    csadjcost~ truncated(Normal(4.0     ,1.5),  (2.0)  ,(15))
    csigma   ~ truncated(Normal(1.50    ,0.375),(0.25) ,(3))
    cfc      ~ truncated(Normal(1.25    ,0.125),(1.0)  ,(3))
    crpi     ~ truncated(Normal(1.5     ,0.25), (1.0)  ,(3))
    cry      ~ truncated(Normal(0.125   ,0.05), (0.001),(0.5))
    crdy     ~ truncated(Normal(0.125   ,0.05), (0.001),(0.5))
    constelab~ truncated(Normal(0.0     ,2.0),  (-10.0),(10.0))
    ctrend   ~ truncated(Normal(0.4     ,0.10), (0.1)  ,(0.8))
    cgy      ~ truncated(Normal(0.5     ,0.25), (0.01) ,(2.0))
    calfa    ~ truncated(Normal(0.3     ,0.05), (0.01) ,(1.0))

    parameters[indexin([:csigl],m.parameters)] .= csigl
    parameters[indexin([:csadjcost],m.parameters)] .= csadjcost
    parameters[indexin([:csigma],m.parameters)] .= csigma
    parameters[indexin([:cfc],m.parameters)] .= cfc
    parameters[indexin([:crpi],m.parameters)] .= crpi
    parameters[indexin([:cry],m.parameters)] .= cry
    parameters[indexin([:crdy],m.parameters)] .= crdy
    parameters[indexin([:constelab],m.parameters)] .= constelab
    parameters[indexin([:ctrend],m.parameters)] .= ctrend
    parameters[indexin([:cgy],m.parameters)] .= cgy
    parameters[indexin([:calfa],m.parameters)] .= calfa
    
    
    # data likelihood
    # Turing.@addlogprob! begin
    #     try 
    #         calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    #     catch e
    #         return -Inf
    #     end
    # end

    # println(ForwardDiff.value.(parameters))
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

    
    # # system prior on IRFs
    # Turing.@addlogprob! loglikelihood(Normal(.03, .01), get_irf(parameters,m)[6,4,2])

    # # system prior on SS and stdev
    # moments = get_variances(RBC_estim, parameters)
    # Turing.@addlogprob! loglikelihood(Normal(.6, .05), moments[1][1])
    # Turing.@addlogprob! loglikelihood(Normal(.014, .001), moments[2][5])
end

observables = Symbol.(names(dat))#[:dinve, :dc]


using BenchmarkTools
calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = m.parameter_values)
@benchmark calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = m.parameter_values)

@profview calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = m.parameter_values)

using ForwardDiff
ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x), Float64[m.parameter_values...])
@benchmark ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x), Float64[m.parameter_values...]) samples = 10 seconds = 100

@profview ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = x), Float64[m.parameter_values...])



turing_model = kalman(data, m, observables) # passing observables from before 

# sample
n_samples = 1000
n_adapts = 250
δ = 0.65

chain_1_marginal = sample(turing_model, NUTS(), n_samples; progress = true)
# chain_1_marginal = sample(turing_model, NUTS(), MCMCDistributed(), n_samples, 2; progress = true) # multi-threaded sampling 
