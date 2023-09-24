using MacroModelling
import Turing
import Turing: Normal, Beta, Gamma, InverseGamma, NUTS, sample, logpdf, Truncated
using Random, CSV, DataFrames, Zygote, AxisKeys, ComponentArrays, Optimization, OptimizationNLopt, OptimizationOptimisers, MCMCChains
import DynamicPPL: logjoint
import ChainRulesCore: @ignore_derivatives, ignore_derivatives

@model SW07 begin
    a[0] = calfa * rkf[0] + (1 - calfa) * (wf[0])

    zcapf[0] = (1 / (czcap / (1 - czcap))) * rkf[0]

    rkf[0] = wf[0] + labf[0] - kf[0]

    kf[0] = kpf[-1] + zcapf[0]

    invef[0] = (1 / (1 + cbetabar * cgamma)) * (invef[-1] + cbetabar * cgamma * invef[1] + (1 / (cgamma ^ 2 * csadjcost)) * pkf[0]) + qs[0]

    pkf[0] =  - rrf[0] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[0] + (crk / (crk + (1 - ctou))) * rkf[1] + ((1 - ctou) / (crk + (1 - ctou))) * pkf[1]

    cf[0] = (chabb / cgamma) / (1 + chabb / cgamma) * cf[-1] + (1 / (1 + chabb / cgamma)) * cf[1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (labf[0] - labf[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (rrf[0]) + b[0]

    yf[0] = ccy * cf[0] + ciy * invef[0] + g[0] + crkky * zcapf[0]

    yf[0] = cfc * (calfa * kf[0] + (1 - calfa) * labf[0] + a[0])

    wf[0] = csigl * labf[0]	 + (1 / (1 - chabb / cgamma)) * cf[0] - (chabb / cgamma) / (1 - chabb / cgamma) * cf[-1]

    kpf[0] = (1 - cikbar) * kpf[-1] + (cikbar) * invef[0] + (cikbar) * (cgamma ^ 2 * csadjcost) * qs[0]

    mc[0] = calfa * rk[0] + (1 - calfa) * (w[0]) - a[0]

    zcap[0] = (1 / (czcap / (1 - czcap))) * rk[0]

    rk[0] = w[0] + lab[0] - k[0]

    k[0] = kp[-1] + zcap[0]

    inve[0] = (1 / (1 + cbetabar * cgamma)) * (inve[-1] + cbetabar * cgamma * inve[1] + (1 / (cgamma ^ 2 * csadjcost)) * pk[0]) + qs[0]

    pk[0] =  - r[0] + pinf[1] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[0] + (crk / (crk + (1 - ctou))) * rk[1] + ((1 - ctou) / (crk + (1 - ctou))) * pk[1]

    c[0] = (chabb / cgamma) / (1 + chabb / cgamma) * c[-1] + (1 / (1 + chabb / cgamma)) * c[1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (lab[0] - lab[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (r[0] - pinf[1]) + b[0]

    y[0] = ccy * c[0] + ciy * inve[0] + g[0] + crkky * zcap[0]

    y[0] = cfc * (calfa * k[0] + (1 - calfa) * lab[0] + a[0])

    pinf[0] = (1 / (1 + cbetabar * cgamma * cindp)) * (cbetabar * cgamma * pinf[1] + cindp * pinf[-1] + ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1) * (mc[0])) + spinf[0]

    w[0] = (1 / (1 + cbetabar * cgamma)) * w[-1] + (cbetabar * cgamma / (1 + cbetabar * cgamma)) * w[1] + (cindw / (1 + cbetabar * cgamma)) * pinf[-1] - (1 + cbetabar * cgamma * cindw) / (1 + cbetabar * cgamma) * pinf[0] + (cbetabar * cgamma) / (1 + cbetabar * cgamma) * pinf[1] + (1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)) * (csigl * lab[0] + (1 / (1 - chabb / cgamma)) * c[0] - ((chabb / cgamma) / (1 - chabb / cgamma)) * c[-1] - w[0]) + sw[0]

    r[0] = crpi * (1 - crr) * pinf[0] + cry * (1 - crr) * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]

    a[0] = crhoa * a[-1] + z_ea * ea[x]

    b[0] = crhob * b[-1] + z_eb * eb[x]

    g[0] = crhog * g[-1] + z_eg * eg[x] + cgy * z_ea * ea[x]

    qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]

    ms[0] = crhoms * ms[-1] + z_em * em[x]

    spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

    epinfma[0] = z_epinf * epinf[x]

    sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

    ewma[0] = z_ew * ew[x]

    kp[0] = (1 - cikbar) * kp[-1] + cikbar * inve[0] + cikbar * cgamma ^ 2 * csadjcost * qs[0]

    dy[0] = y[0] - y[-1] + ctrend

    dc[0] = c[0] - c[-1] + ctrend

    dinve[0] = inve[0] - inve[-1] + ctrend

    dw[0] = w[0] - w[-1] + ctrend

    pinfobs[0] = (pinf[0]) + constepinf

    robs[0] = (r[0]) + conster

    labobs[0] = lab[0] + constelab

end


@parameters SW07 begin  
    ctou=.025
    clandaw=1.5
    cg=0.18
    curvp=10
    curvw=10
    
    calfa=.24
    # cgamma=1.004
    # cbeta=.9995
    csigma=1.5
    # cpie=1.005
    cfc=1.5
    cgy=0.51
    
    csadjcost= 6.0144
    chabb=    0.6361    
    cprobw=   0.8087
    csigl=    1.9423
    cprobp=   0.6
    cindw=    0.3243
    cindp=    0.47
    czcap=    0.2696
    crpi=     1.488
    crr=      0.8762
    cry=      0.0593
    crdy=     0.2347
    
    crhoa=    0.9977
    crhob=    0.5799
    crhog=    0.9957
    crhols=   0.9928
    crhoqs=   0.7165
    crhoas=1 
    crhoms=0
    crhopinf=0
    crhow=0
    cmap = 0
    cmaw  = 0
    
    clandap=cfc
    cbetabar=cbeta*cgamma^(-csigma)
    cr=cpie/(cbeta*cgamma^(-csigma))
    crk=(cbeta^(-1))*(cgamma^csigma) - (1-ctou)
    cw = (calfa^calfa*(1-calfa)^(1-calfa)/(clandap*crk^calfa))^(1/(1-calfa))
    cikbar=(1-(1-ctou)/cgamma)
    cik=(1-(1-ctou)/cgamma)*cgamma
    clk=((1-calfa)/calfa)*(crk/cw)
    cky=cfc*(clk)^(calfa-1)
    ciy=cik*cky
    ccy=1-cg-cik*cky
    crkky=crk*cky
    cwhlc=(1/clandaw)*(1-calfa)/calfa*crk*cky/ccy
    cwly=1-crk*cky
    
    conster=(cr-1)*100
    # ctrend=(cgamma-1)*100
    ctrend=(1.004-1)*100
    # constepinf=(cpie-1)*100
    constepinf=(1.005-1)*100

    cpie=1+constepinf/100
    cgamma=1+ctrend/100 

    cbeta=1/(1+constebeta/100)
    constebeta = 100 / .9995 - 100

    constelab=0

    z_ea = 0.4618
    z_eb = 1.8513
    z_eg = 0.6090
    z_eqs = 0.6017
    z_em = 0.2397
    z_epinf = 0.1455
    z_ew = 0.2089
end

# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]

# Subsample from 1966Q1 - 2004Q4
# subset observables in data
data = data(observables,75:230)

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




Turing.@model function SW07_loglikelihood_function(data, m, observables,fixed_parameters)
    z_ea    ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    z_eb    ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.025,5)
    z_eg    ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    z_eqs   ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    z_em    ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    z_epinf ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    z_ew    ~   Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3)
    crhoa   ~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhob   ~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhog   ~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhoqs  ~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhoms  ~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhopinf~   Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999)
    crhow   ~   Truncated(Beta(beta_map(0.5,0.20)...),.001,.9999)
    cmap    ~   Truncated(Beta(beta_map(0.5,0.2)...),0.01,.9999)
    cmaw    ~   Truncated(Beta(beta_map(0.5,0.2)...),0.01,.9999)
    csadjcost~  Truncated(Normal(4,1.5),2,15)
    csigma  ~   Truncated(Normal(1.50,0.375),0.25,3)
    chabb   ~   Truncated(Beta(beta_map(0.7,0.1)...),0.001,0.99)
    cprobw  ~   Truncated(Beta(beta_map(0.5,0.1)...),0.3,0.95)
    csigl   ~   Truncated(Normal(2,0.75),0.25,10)
    cprobp  ~   Truncated(Beta(beta_map(0.5,0.10)...),0.5,0.95)
    cindw   ~   Truncated(Beta(beta_map(0.5,0.15)...),0.01,0.99)
    cindp   ~   Truncated(Beta(beta_map(0.5,0.15)...),0.01,0.99)
    czcap   ~   Truncated(Beta(beta_map(0.5,0.15)...),0.01,1)
    cfc     ~   Truncated(Normal(1.25,0.125),1.0,3)
    crpi    ~   Truncated(Normal(1.5,0.25),1.0,3)
    crr     ~   Truncated(Beta(beta_map(0.75,0.10)...),0.5,0.975)
    cry     ~   Truncated(Normal(0.125,0.05),0.001,0.5)
    crdy    ~   Truncated(Normal(0.125,0.05),0.001,0.5)
    constepinf~ Truncated(Gamma(gamma_map(0.625,0.1)...),0.1,2.0)
    constebeta~ Truncated(Gamma(gamma_map(0.25,0.1)...),0.01,2.0)
    constelab ~ Truncated(Normal(0.0,2.0),-10.0,10.0)
    ctrend  ~   Truncated(Normal(0.4,0.10),0.1,0.8)
    cgy     ~   Truncated(Normal(0.5,0.25),0.01,2.0)
    calfa   ~   Truncated(Normal(0.3,0.05),0.01,1.0)

    ctou, clandaw, cg, curvp, curvw, crhols, crhoas = fixed_parameters

    parameters_combined = [ctou,clandaw,cg,curvp,curvw,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhols,crhoqs,crhoas,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta]

    kalman_prob = calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters_combined)

    # println(kalman_prob)
    
    Turing.@addlogprob! kalman_prob 
end


SW07.parameter_values[indexin([:crhoms, :crhopinf, :crhow, :cmap, :cmaw],SW07.parameters)] .= 0.02

fixed_parameters = SW07.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters)]

SW07_loglikelihood = SW07_loglikelihood_function(data, SW07, observables, fixed_parameters)



n_samples = 1000

# Turing.setadbackend(:zygote)
samps = sample(SW07_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)


serialize("chain-file.jls", samps)




# pars = ComponentArray(Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...],Axis(SW07.parameters[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]))


# Mode calculation

function calculate_posterior_loglikelihoods(parameters, u)
    ctou, clandaw, cg, curvp, curvw, crhols, crhoas = @ignore_derivatives SW07.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters)]

    calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhoqs,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta = parameters

    parameters_combined = [ctou,clandaw,cg,curvp,curvw,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhols,crhoqs,crhoas,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta]

    log_lik = 0
    log_lik -= calculate_kalman_filter_loglikelihoods(SW07, data(observables), observables; parameters = parameters_combined)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_ea)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.025,5),z_eb)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_eg)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_eqs)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_em)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_epinf)
    log_lik -= logpdf(Truncated(InverseGamma(inv_gamma_map(0.1,2)...),0.01,3),z_ew)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhoa)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhob)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhog)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhoqs)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhoms)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.01,.9999),crhopinf)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.20)...),.001,.9999),crhow)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.2)...),0.01,.9999),cmap)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.2)...),0.01,.9999),cmaw)
    log_lik -= logpdf(Truncated(Normal(4,1.5),2,15),csadjcost)
    log_lik -= logpdf(Truncated(Normal(1.50,0.375),0.25,3),csigma)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.7,0.1)...),0.001,0.99),chabb)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.1)...),0.05,0.95),cprobw)
    # log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.1)...),0.3,0.95),cprobw)
    log_lik -= logpdf(Truncated(Normal(2,0.75),0.25,10),csigl)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.10)...),0.05,0.95),cprobp)
    # log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.10)...),0.5,0.95),cprobp)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.15)...),0.01,0.99),cindw)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.15)...),0.01,0.99),cindp)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.5,0.15)...),0.01,1),czcap)
    log_lik -= logpdf(Truncated(Normal(1.25,0.125),1.0,3),cfc)
    log_lik -= logpdf(Truncated(Normal(1.5,0.25),1.0,3),crpi)
    log_lik -= logpdf(Truncated(Beta(beta_map(0.75,0.10)...),0.05,0.975),crr)
    # log_lik -= logpdf(Truncated(Beta(beta_map(0.75,0.10)...),0.5,0.975),crr)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5),cry)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5),crdy)
    log_lik -= logpdf(Truncated(Gamma(gamma_map(0.625,0.1)...),0.1,2.0),constepinf)
    log_lik -= logpdf(Truncated(Gamma(gamma_map(0.25,0.1)...),0.01,2.0),constebeta)
    log_lik -= logpdf(Truncated(Normal(0.0,2.0),-10.0,10.0),constelab)
    log_lik -= logpdf(Truncated(Normal(0.4,0.10),0.1,0.8),ctrend)
    log_lik -= logpdf(Truncated(Normal(0.5,0.25),0.01,2.0),cgy)
    log_lik -= logpdf(Truncated(Normal(0.3,0.05),0.01,1.0),calfa)

    return log_lik
end

SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]

SW07.parameter_values[indexin([:cprobw,:cprobp,:crr],SW07.parameters)]

SW07.parameter_values[indexin([:crhoms, :crhopinf, :crhow, :cmap, :cmaw],SW07.parameters)] .= 0.02

calculate_posterior_loglikelihood(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[])


𝓂 = SW07
verbose = true
parameters = nothing
tol = eps()
import LinearAlgebra as ℒ
using ImplicitDifferentiation
import MacroModelling: ℳ 

calculate_covariance_AD(sol; T, subset_indices) = ImplicitFunction(sol->calculate_covariance_forward(sol, T=T, subset_indices = subset_indices), (x,y)->calculate_covariance_conditions(x,y,T=T, subset_indices = subset_indices))
# calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])



function calculate_kalman_filter_loglikelihoods(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose::Bool = false, tol::Float64 = eps())
    @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    @ignore_derivatives sort!(observables)

    # @ignore_derivatives solve!(𝓂, verbose = verbose)

    if isnothing(parameters)
        parameters = 𝓂.parameter_values
    else
        ub = @ignore_derivatives fill(1e12+rand(),length(𝓂.parameters))
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

    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, true, verbose)
    
    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

    sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

    A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior

    calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

    P = calculate_covariance_(sol)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
    u = zeros(length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u

    loglik = 0.0

    v = similar(z)
    F = C * C'
    K = similar(C')

    for t in 1:size(data)[2]
        v .= data_in_deviations[:,t] - z

        F .= C * P * C'

        # F = (F + F') / 2

        # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
        # K = P * C' * ℒ.pinv(F)

        # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
        Fdet = ℒ.det(F)

        if Fdet < eps() return -Inf end

        loglik += log(Fdet) + v' / F  * v
        
        K .= P * C' / F

        P .= A * (P - K * C * P) * A' + 𝐁

        u .= A * (u + K * v)
        
        z .= C * u 
    end

    return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment
end




function calculate_kalman_filter_loglikelihoods(𝓂::ℳ, data::AbstractArray{Float64}, observables::Vector{Symbol}; parameters = nothing, verbose::Bool = false, tol::Float64 = eps())
    @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    @ignore_derivatives sort!(observables)

    # @ignore_derivatives solve!(𝓂, verbose = verbose)

    if isnothing(parameters)
        parameters = 𝓂.parameter_values
    else
        ub = @ignore_derivatives fill(1e12+rand(),length(𝓂.parameters))
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

    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, true, verbose)
    
    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

    sol = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))))

    A = @views sol[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = @views sol[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior

    calculate_covariance_ = calculate_covariance_AD(sol, T = 𝓂.timings, subset_indices = Int64[observables_and_states...])

    P = calculate_covariance_(sol)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))
    u = zeros(length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u

    loglik = 0.0

    for t in 1:size(data)[2]
        v = data_in_deviations[:,t] - z

        F = P * C * C'

        # F = (F + F') / 2

        # loglik += log(max(eps(),ℒ.det(F))) + v' * ℒ.pinv(F) * v
        # K = P * C' * ℒ.pinv(F)

        # loglik += log(max(eps(),ℒ.det(F))) + v' / F  * v
        Fdet = ℒ.det(F)

        if Fdet < eps() return -Inf end

        loglik += log(Fdet) + v' / F  * v
        
        F = RecursiveFactorization.lu!(F)
        
        K = P * C' / F

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)
        
        z = C * u 
    end

    return -(loglik + length(data) * log(2 * 3.141592653589793)) / 2 # otherwise conflicts with model parameters assignment
end


using BenchmarkTools, Zygote, ForwardDiff
using TriangularSolve, RecursiveFactorization, Octavian

F = C * P * C'
F = RecursiveFactorization.lu!(F)
C' / F
TriangularSolve.rdiv!(AA, collect(C'), UpperTriangular(F.U))
BB = similar(AA)
TriangularSolve.ldiv!(AA, LowerTriangular(F.L), AA)
TriangularSolve.rdiv!(BB, AA, (F.L))
C' / F.U / F.L
F.U / C' * F.L

F.L \ C' / F.U
ForwardDiff.jacobian(P-> begin F = C * P * C'
F = RecursiveFactorization.lu!(F)
C' / F
end, P)

@benchmark begin F = C * P * C'
    # F = RecursiveFactorization.lu!(F)
    C' / F
    end


N = 100
A = rand(N,N); B = rand(N,N); C = similar(A);
TriangularSolve.rdiv!(C, A, UpperTriangular(B))* UpperTriangular(B)
A

A / UpperTriangular(B)

C

F
det
using LinearAlgebra, Octavian
@benchmark F .= RecursiveFactorization.lu!(F)
@benchmark FS = lu!(F)
TriangularSolve.rdiv!(AA,C',F)
AA = similar(C')


Fdet = ℒ.det(F)

if Fdet < eps() return -Inf end

loglik += log(Fdet) + v' / F  * v

K = P * C' / F

P = A * (P - K * C * P) * A' + 𝐁

u = A * (u + K * v)

z = C * u 


@benchmark calculate_posterior_loglikelihoods(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[])

@benchmark calculate_posterior_loglikelihood(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[])

ForwardDiff.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])

@benchmark ForwardDiff.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])

using Profile
@profile for i in 1:3 ForwardDiff.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...]) end


import ForwardDiff
@profview ForwardDiff.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])


import ProfileView, ForwardDiff
ProfileView.@profview ForwardDiff.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])


Zygote.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])[1]

# logpdf(Truncated(Beta(beta_map(0.75,0.10)...),0.5,0.975),.51)
# Zygote.gradient(x->logpdf(Truncated(Beta(beta_map(0.75,0.10)...),0.5,0.975),x),.51)

# Zygote.gradient(x->logpdf(Truncated(Beta(beta_map(0.5,0.1)...),0.01,0.95),x),.51)


@benchmark Zygote.gradient(x->calculate_posterior_loglikelihood(x,[]),Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...])[1]


f = OptimizationFunction(calculate_posterior_loglikelihood, Optimization.AutoZygote())

prob = OptimizationProblem(f, Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...], []);
sol = solve(prob, Optimisers.Adam(), maxiters = 1000)
sol.minimum

lbs = [0.01    ,0.025   ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.01    ,0.001   ,0.01    ,0.01    ,2.0 ,0.25    ,0.001   ,0.05    ,0.25    ,0.05    ,0.01    ,0.01    ,0.01    ,1.0 ,1.0 ,0.05    ,0.001   ,0.001   ,0.1 ,0.01    ,0.0 ,0.1 ,0.01    ,0.01]

ubs = [3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 15.0, 3.0, 0.99, 0.95, 10.0, 0.95, 0.99, 0.99, 1.0, 3.0, 3.0, 0.975, 0.5, 0.5, 2.0, 2.0, 10.0, 0.8, 2.0, 1.0]

sort_idx = sortperm(indexin([:z_ea,:z_eb,:z_eg,:z_eqs,:z_em,:z_epinf,:z_ew,:crhoa,:crhob,:crhog,:crhoqs,:crhoms,:crhopinf,:crhow,:cmap,:cmaw,:csadjcost,:csigma,:chabb,:cprobw,:csigl,:cprobp,:cindw,:cindp,:czcap,:cfc,:crpi,:crr,:cry,:crdy,:constepinf,:constebeta,:constelab,:ctrend,:cgy,:calfa],SW07.parameters))

prob = OptimizationProblem(f, Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...], [], lb = lbs[sort_idx], ub = ubs[sort_idx]);
sol = solve(prob, NLopt.LD_LBFGS(), maxtime = 100)
sol.minimum

sol = solve(prob, NLopt.LN_SBPLX(), maxtime = 10)
sol.minimum

# using MAT

# vars = matread("/Users/thorekockerols/Downloads/sw07/usmodel_mode.mat")


# calculate_posterior_loglikelihood(vec(vars["xparam1"]),[])
# ([a-z_]+),([\d\.\s-]+),([\d\.\s-]+),([\d\.\s-]+),([a-z_]+),([\d\.\s-]+),([\d\.\s-]+);
# logpdf(Truncated($5($5_map($6,$7)...),$3,$4),$1)

