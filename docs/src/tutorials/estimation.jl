using MacroModelling
using Random
Random.seed!(30)
@model FS2000 begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])

    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]

    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0

    W[0] = l[0] / n[0]

    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0

    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]

    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0

    c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]

    P[0] * c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    e[0] = exp(z_e_a  *  e_a[x])

    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))

    gy_obs[0] = dA[0] * y[0] / y[-1]

    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]

    log_gy_obs[0] = log(gy_obs[0])

    log_gp_obs[0] = log(gp_obs[0])
end

@parameters FS2000 begin  
    alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862
end

using CSV, DataFrames, AxisKeys

# load data
dat = CSV.read("./test/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))#[:dinve, :dc]

data = data(observables,:)


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
    
import Turing
import Turing: Normal, Beta, InverseGamma, Uniform, NUTS, sample, logpdf

Turing.@model function kalman(data, m, observables)

    # if DynamicPPL.leafcontext(__context__) === DefaultContext()
        alp     ~ Beta(beta_map(0.356, 0.02)...)
        bet     ~ Beta(beta_map(0.993, 0.002)...)
        gam     ~ Normal(0.0085, 0.003)#, eps(), .1)
        mst     ~ Normal(1.0002, 0.007)
        rho     ~ Beta(beta_map(0.129, 0.223)...)
        psi     ~ Beta(beta_map(0.65, 0.05)...)
        del     ~ Beta(beta_map(0.01, 0.005)...)
        z_e_a   ~ InverseGamma(inv_gamma_map(0.035449, Inf)...)
        z_e_m   ~ InverseGamma(inv_gamma_map(0.008862, Inf)...)
    # end
    # alp     ~ Uniform(0, 1)
    # bet     ~ Uniform(0, 1)
    # gam     ~ Uniform(-1, 1)
    # mst     ~ Uniform(0, 2)
    # rho     ~ Uniform(0, 1)
    # psi     ~ Uniform(0, 1)
    # del     ~ Uniform(0, 1)
    # z_e_a   ~ Uniform(0, 1)
    # z_e_m   ~ Uniform(0, 1)
    
    # if DynamicPPL.leafcontext(__context__) === LikelihoodContext()
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    # end
    # Turing.@addlogprob! logpdf(Beta(beta_map(0.356, 0.02)...),alp)
    # Turing.@addlogprob! logpdf(Beta(beta_map(0.993, 0.002)...),bet)
    # Turing.@addlogprob! logpdf(Beta(beta_map(0.129, 0.223)...),rho)
    # Turing.@addlogprob! logpdf(Beta(beta_map(0.65, 0.05)...),psi)
    # Turing.@addlogprob! logpdf(Beta(beta_map(0.01, 0.005)...),del)
    # Turing.@addlogprob! logpdf(Normal(0.0085, 0.003),gam)
    # Turing.@addlogprob! logpdf(Normal(1.0002, 0.007),mst)
    # Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),z_e_a)
    # Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),z_e_m)
end

using DynamicPPL
model = kalman(data, FS2000, observables) | (alp = parameters[1], 
                                             bet = parameters[2], 
                                             gam = parameters[3], 
                                             mst = parameters[4], 
                                             rho = parameters[5], 
                                             psi = parameters[6], 
                                             del = parameters[7], 
                                             z_e_a = parameters[8], 
                                             z_e_m = parameters[9])
logjoint(model,[])
logprior(model,[])
loglikelihood(model,[])
x = string(:alp)
@varname(var"string(:alp)")
using ComponentArrays
get_log_probability_all(ComponentArray(FS2000.parameter_values,Axis(FS2000.parameters)), FS2000, data, observables)
# pars = [0.2275748664844605, 0.9979002291586807, -2130.4765345983997, 306.64891752510715, 0.8040205104339292, 0.7531446779265871, 0.1023822401672587, 0.5046824367348219, 2.854558196537693]
# pars = [0.16567032505797824, 0.2542259201085809, -0.9339496397117015, 1.2051479057232557, 0.7840464832066878, 0.5852821752759191, 0.5992002161458438, 2.831618660156801, 1.1469946639099047]
# pars = [0.4123085111118648, 0.9999921989053889, 1046.4705369500402, -40.5722449868309, 0.7741402851964618, 0.5860029149568086, 0.13639983741497175, 1.2892130991770796, 0.4039048831341653]
# ForwardDiff.gradient(x-> calculate_kalman_filter_loglikelihood(FS2000, data(observables), observables; parameters = x), pars)
# calculate_kalman_filter_loglikelihood(FS2000, data(observables), observables; parameters = pars)
# FS2000.NSSS_solver_cache
turing_model = kalman(data, FS2000, observables)
n_samples = 1000
chain_NUTS  = sample(turing_model, NUTS(), n_samples; progress = true)


parameters = FS2000.parameter_values

logjoint(turing_model,(alp =    parameters[1], 
                       bet =    parameters[2], 
                       gam =    parameters[3], 
                       mst =    parameters[4], 
                       rho =    parameters[5], 
                       psi =    parameters[6], 
                       del =    parameters[7], 
                       z_e_a =  parameters[8], 
                       z_e_m =  parameters[9]))

logprior(turing_model,(alp =    parameters[1], 
                       bet =    parameters[2], 
                       gam =    parameters[3], 
                       mst =    parameters[4], 
                       rho =    parameters[5], 
                       psi =    parameters[6], 
                       del =    parameters[7], 
                       z_e_a =  parameters[8], 
                       z_e_m =  parameters[9]))
                       loglikelihood(turing_model,pars)
loglikelihood(turing_model,(alp =   parameters[1], 
                            bet =   parameters[2], 
                            gam =   parameters[3], 
                            mst =   parameters[4], 
                            rho =   parameters[5], 
                            psi =   parameters[6], 
                            del =   parameters[7], 
                            z_e_a = parameters[8], 
                            z_e_m = parameters[9]))

chain_subset = get(chain_NUTS,FS2000.parameters)
chain_subset[FS2000.parameters]

lll = [logjoint(turing_model, ComponentArray(reduce(hcat, get(chain_NUTS, FS2000.parameters)[FS2000.parameters])[s,:], Axis(FS2000.parameters))) for s in 1:length(chain_NUTS)]


lll = [logjoint(turing_model,(alp =    parameters[1], 
                                    bet =    parameters[2], 
                                    gam =    parameters[3], 
                                    mst =    parameters[4], 
                                    rho =    parameters[5], 
                                    psi =    parameters[6], 
                                    del =    parameters[7], 
                                    z_e_a =  parameters[8], 
                                    z_e_m =  parameters[9])) 
                                    for parameters in [reduce(hcat,get(chain_NUTS,FS2000.parameters)[FS2000.parameters])[s,:] 
                                        for s in 1:length(chain_NUTS)]]

sum(lll)/1000
maximum(lll)
minimum(lll)
get(chain_NUTS,:lp)
chain_NUTS
lll = pointwise_loglikelihoods(turing_model, get(chain_NUTS,FS2000.parameters))
pointwise_loglikelihoods
# chain_NUTS

using StatsPlots
StatsPlots.plot(chain_NUTS)
savefig("FS2000_chain_NUTS.png")

using ComponentArrays
using MCMCChains
parameter_mean = mean(chain_NUTS)
pars = ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1]))

function get_log_probability(par1, par2, pars_syms, orig_pars, m, data, observables)
    orig_pars[pars_syms] = [par1, par2]
    (; alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m) = orig_pars

    logprob  = 0
    logprob += calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = collect(orig_pars))
    logprob += logpdf(Beta(beta_map(0.356, 0.02)...), alp)
    logprob += logpdf(Beta(beta_map(0.993, 0.002)...), bet)
    logprob += logpdf(Normal(0.0085, 0.003), gam)
    logprob += logpdf(Normal(1.0002, 0.007), mst)
    logprob += logpdf(Beta(beta_map(0.129, 0.223)...), rho)
    logprob += logpdf(Beta(beta_map(0.65, 0.05)...), psi)
    logprob += logpdf(Beta(beta_map(0.01, 0.005)...), del)
    logprob += logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...), z_e_a)
    logprob += logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...), z_e_m)
end



function get_log_probability_all(orig_pars, m, data, observables)
    (; alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m) = orig_pars

    logprob  = 0
    logprob += calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = collect(orig_pars))
    logprob += logpdf(Beta(beta_map(0.356, 0.02)...), alp)
    logprob += logpdf(Beta(beta_map(0.993, 0.002)...), bet)
    logprob += logpdf(Normal(0.0085, 0.003), gam)
    logprob += logpdf(Normal(1.0002, 0.007), mst)
    logprob += logpdf(Beta(beta_map(0.129, 0.223)...), rho)
    logprob += logpdf(Beta(beta_map(0.65, 0.05)...), psi)
    logprob += logpdf(Beta(beta_map(0.01, 0.005)...), del)
    logprob += logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...), z_e_a)
    logprob += logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...), z_e_m)
end


par_samples = reduce(hcat,[chain_NUTS[s] for s in parameter_mean.nt[1]])
ll = [get_log_probability_all(ComponentArray(par_samples[s,:],Axis(parameter_mean.nt[1])),FS2000,data,observables) for s in 1:n_samples]

get_log_probability_all(ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1])),FS2000,data,observables)

using Plots
granularity = 32

par1 = :gam
par2 = :psi
par_range1 = collect(range(minimum(chain_NUTS[par1]), stop = maximum(chain_NUTS[par1]), length = granularity))
par_range2 = collect(range(minimum(chain_NUTS[par2]), stop = maximum(chain_NUTS[par2]), length = granularity))

p = surface(par_range1, par_range2, 
            (x,y) -> get_log_probability(x,y, [par1,par2], pars, FS2000, data, observables),
            camera=(30, 65),
            colorbar=false,
            color=:inferno)


scatter3d!(vec(collect(chain_NUTS[par1])),
           vec(collect(chain_NUTS[par2])),
           ll,
        #    vec(collect(chain_NUTS[:lp])),
            mc =:viridis, 
            marker_z=collect(1:length(chain_NUTS)), 
            msw=0,
            legend=false, 
            colorbar=false, 
            xlabel = string(par1),
            ylabel = string(par2),
            zlabel = "Log probability",
            alpha=0.5)

p
savefig("FS2000_posterior_surface.png")