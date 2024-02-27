using MacroModelling
import Turing, Pigeons, DynamicPPL
import Turing: NUTS, sample, logpdf, Truncated#, Normal, Beta, Gamma, InverseGamma,
using Random, CSV, DataFrames, Zygote, AxisKeys, MCMCChains
# using ComponentArrays, Optimization, OptimizationNLopt, OptimizationOptimisers
import DynamicPPL: logjoint
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
# ]add CSV, DataFrames, Zygote, AxisKeys, MCMCChains, Turing, DynamicPPL

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


Turing.@model function SW07_loglikelihood_function(data, m, observables,fixed_parameters)
    # truncation failed with Zygote for beta distribution

    # z_ea    ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)
    z_eb    ~   InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true)
    # z_eg    ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)
    # z_eqs   ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)
    # z_em    ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)
    # z_epinf ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)
    # z_ew    ~   InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true)

    zs      ~ Turing.filldist(InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true), 6)

    z_ea, z_eg, z_eqs, z_em, z_epinf, z_ew = zs

    # crhoa   ~   Beta(0.5, 0.2, μσ = true)
    # crhob   ~   Beta(0.5, 0.2, μσ = true)
    # crhog   ~   Beta(0.5, 0.2, μσ = true)
    # crhoqs  ~   Beta(0.5, 0.2, μσ = true)
    # crhoms  ~   Beta(0.5, 0.2, μσ = true)
    # crhopinf~   Beta(0.5, 0.2, μσ = true)
    # crhow   ~   Beta(0.5, 0.2, μσ = true)

    rhos    ~ Turing.filldist(Beta(0.5, 0.2, μσ = true), 7)

    crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow = rhos

    cmap    ~   Beta(0.5, 0.2, μσ = true)
    cmaw    ~   Beta(0.5, 0.2, μσ = true)

    crpi    ~   Normal(1.5, 0.25, 1.0, 3.0)
    crr     ~   Beta(0.75, 0.10, μσ = true) #truncation causes trouble here
    cry     ~   Normal(0.125, 0.05, 0.001, 0.5)
    crdy    ~   Normal(0.125, 0.05, 0.001, 0.5)

    csadjcost~  Normal(4.0, 1.5, 2.0, 15.0)
    csigma  ~   Normal(1.50, 0.375, 0.25, 3.0)
    chabb   ~   Beta(0.7, 0.1, μσ = true)
    cprobw  ~   Beta(0.5, 0.1, μσ = true)
    csigl   ~   Normal(2.0, 0.75, 0.25, 10.0)
    cprobp  ~   Beta(0.5, 0.10, μσ = true)
    
    # cindw   ~   Beta(0.5, 0.15, μσ = true)
    # cindp   ~   Beta(0.5, 0.15, μσ = true)
    # czcap   ~   Beta(0.5, 0.15, μσ = true)

    indcap    ~ Turing.filldist(Beta(0.5, 0.15, μσ = true), 3)

    cindw, cindp, czcap = indcap

    cfc     ~   Normal(1.25, 0.125, 1.0, 3.0)

    constepinf~ Gamma(0.625, 0.1, 0.1, 2.0, μσ = true)
    constebeta~ Gamma(0.25, 0.1, 0.01, 2.0, μσ = true)
    constelab ~ Normal(0.0, 2.0, -10.0, 10.0)
    ctrend  ~   Normal(0.4, 0.10, 0.1, 0.8)
    cgy     ~   Normal(0.5, 0.25, 0.01, 2.0)
    calfa   ~   Normal(0.3, 0.05, 0.01, 1.0)
    
    ctou, clandaw, cg, curvp, curvw, crhols, crhoas = fixed_parameters

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        parameters_combined = [ctou,clandaw,cg,curvp,curvw,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhols,crhoqs,crhoas,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta]

        kalman_prob = get_loglikelihood(m, data(observables), parameters_combined)

        Turing.@addlogprob! kalman_prob 
    end
end


SW07.parameter_values[indexin([:crhoms, :crhopinf, :crhow, :cmap, :cmaw],SW07.parameters)] .= 0.02

fixed_parameters = SW07.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw, :crhols, :crhoas], SW07.parameters)]

SW07_loglikelihood = SW07_loglikelihood_function(data, SW07, observables, fixed_parameters)


n_samples = 1000

# Turing.setadbackend(:zygote) # deprecated
samps = Turing.sample(SW07_loglikelihood, NUTS(adtype = Turing.AutoZygote()), n_samples, progress = true)#, init_params = sol)


# generate a Pigeons log potential
sw07_lp = Pigeons.TuringLogPotential(SW07_loglikelihood)

using BenchmarkTools
# find a feasible starting point
# @benchmark Pigeons.pigeons(target = sw07_lp, n_rounds = 1, n_chains = 1);
Pigeons.pigeons(target = sw07_lp, n_rounds = 1, n_chains = 1)
@profview Pigeons.pigeons(target = sw07_lp, n_rounds = 1, n_chains = 1)



serialize("chain-file.jls", samps)




# pars = ComponentArray(Float64[SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]...],Axis(SW07.parameters[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]))


# Mode calculation

function calculate_posterior_loglikelihoods(parameters, u)
    ctou, clandaw, cg, curvp, curvw, crhols, crhoas = @ignore_derivatives SW07.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters)]

    calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhoqs,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta = parameters

    parameters_combined = [ctou,clandaw,cg,curvp,curvw,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhols,crhoqs,crhoas,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta]

    log_lik = 0
    log_lik -= get_loglikelihood(SW07, data(observables), parameters_combined, filter = :kalman)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_ea)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.025,5), z_eb)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_eg)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_eqs)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_em)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_epinf)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_ew)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoa)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhob)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhog)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoqs)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoms)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhopinf)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.001,.9999), crhow)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.2, μσ = true),0.01,.9999), cmap)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.2, μσ = true),0.01,.9999), cmaw)
    log_lik -= logpdf(Truncated(Normal(4,1.5),2,15), csadjcost)
    log_lik -= logpdf(Truncated(Normal(1.50,0.375),0.25,3), csigma)
    log_lik -= logpdf(Truncated(Beta(0.7, 0.1, μσ = true),0.001,0.99), chabb)
    log_lik -= logpdf(Beta(0.5, 0.1, μσ = true), cprobw)
    # log_lik -= logpdf(Truncated(Beta(0.5, 0.1, μσ = true),0.3,0.95), cprobw)
    log_lik -= logpdf(Truncated(Normal(2,0.75),0.25,10), csigl)
    log_lik -= logpdf(Beta(0.5, 0.10, μσ = true), cprobp)
    # log_lik -= logpdf(Truncated(Beta(0.5, 0.10, μσ = true),0.5,0.95), cprobp)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,0.99), cindw)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,0.99), cindp)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,1), czcap)
    log_lik -= logpdf(Truncated(Normal(1.25,0.125),1.0,3), cfc)
    log_lik -= logpdf(Truncated(Normal(1.5,0.25),1.0,3), crpi)
    # log_lik -= logpdf(Truncated(Beta(0.75, 0.10, μσ = true),0.5,0.975), crr)
    log_lik -= logpdf(Beta(0.75, 0.10, μσ = true), crr)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5), cry)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5), crdy)
    log_lik -= logpdf(Truncated(Gamma(0.625,0.1, μσ = true),0.1,2.0), constepinf)
    log_lik -= logpdf(Truncated(Gamma(0.25,0.1, μσ = true),0.01,2.0), constebeta)
    log_lik -= logpdf(Truncated(Normal(0.0,2.0),-10.0,10.0), constelab)
    log_lik -= logpdf(Truncated(Normal(0.4,0.10),0.1,0.8), ctrend)
    log_lik -= logpdf(Truncated(Normal(0.5,0.25),0.01,2.0), cgy)
    log_lik -= logpdf(Truncated(Normal(0.3,0.05),0.01,1.0), calfa)

    return log_lik
end


SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))]

SW07.parameter_values[indexin([:cprobw,:cprobp,:crr],SW07.parameters)]

SW07.parameter_values[indexin([:crhoms, :crhopinf, :crhow, :cmap, :cmaw],SW07.parameters)] .= 0.02

calculate_posterior_loglikelihoods(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[])


using ForwardDiff, BenchmarkTools#, FiniteDifferences



@benchmark calculate_posterior_loglikelihoods(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[])

@profview for i in 1:100 calculate_posterior_loglikelihoods(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))],[]) end


forw_grad = ForwardDiff.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])

reverse_grad = Zygote.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])[1]

fin_grad = FiniteDifferences.grad(central_fdm(4,1),x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])[1]




@benchmark ForwardDiff.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])

@benchmark Zygote.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))] .* (1 + randn()*1e-6))[1]

# BenchmarkTools.Trial: 59 samples with 1 evaluation.
#  Range (min … max):  79.393 ms … 117.609 ms  ┊ GC (min … max): 0.00% … 26.87%
#  Time  (median):     81.066 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   84.742 ms ±   8.168 ms  ┊ GC (mean ± σ):  2.83% ±  6.44%

#   █ ▂                                                           
#   ███▇▅▃▁▁▅▅▃▁▁▃▃▁▁▃▁▃▃▁▃▁▁▁▃▁▁▃▃▁▁▁▃▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
#   79.4 ms         Histogram: frequency by time          115 ms <

#  Memory estimate: 93.94 MiB, allocs estimate: 217016. 
@benchmark FiniteDifferences.grad(central_fdm(4,1),x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])[1]



@profview ForwardDiff.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])

@profview for i in 1:10 Zygote.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))])[1] end


@profview for i in 1:10 
    Zygote.gradient(x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))] .* (1 +randn()*1e-6))[1] 
end



@profview for i in 1:10 FiniteDifferences.grad(central_fdm(4,1),x -> calculate_posterior_loglikelihoods(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters))])[1] end


include("../models/RBC_baseline.jl")

include("../models/FS2000.jl")

𝓂 = SW07
𝓂 = FS2000
# 𝓂 = RBC_baseline
verbose = true
parameters = nothing
tol = eps()
import LinearAlgebra as ℒ
using ImplicitDifferentiation
import MacroModelling: ℳ, get_and_check_observables, calculate_jacobian, calculate_first_order_solution, calculate_kalman_filter_loglikelihood, solve_matrix_equation_AD
import RecursiveFactorization as RF
import SpeedMapping: speedmapping

parameter_values = 𝓂.parameter_values
algorithm = :first_order
filter = :kalman
warmup_iterations = 0
tol = 1e-16
T = 𝓂.timings


function update_loglikelihood!(loglik::S, P::Matrix{S}, u::Vector{S}, z::Vector{S}, C::Matrix{T}, A::Matrix{S}, 𝐁::Matrix{S}, data_point::Vector{S}) where {S,T}
    v = data_point - z
    F = C * P * C'

    F̄ = ℒ.lu(F, check = false)

    if !ℒ.issuccess(F̄)
        return -Inf, P, u, z
    end

    Fdet = ℒ.det(F̄)

    # Early return if determinant is too small, indicating numerical instability.
    if Fdet < eps(S)
        return -Inf, P, u, z
    end

    invF = inv(F̄)
    loglik_increment = log(Fdet) + v' * invF * v
    K = P * C' * invF
    P = A * (P - K * C * P) * A' + 𝐁
    u = A * (u + K * v)
    z = C * u

    return loglik + loglik_increment, P, u, z
end



function calculate_kalman_filter_ll(𝓂::ℳ, observables::Vector{Symbol}, 𝐒₁::Matrix{S}, data_in_deviations::Matrix{S})::S where S
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present))))

    calculate_kalman_filter_ll(𝓂, obs_idx, 𝐒₁, data_in_deviations)
end

function calculate_kalman_filter_ll(𝓂::ℳ, observables::Vector{String}, 𝐒₁::Matrix{S}, data_in_deviations::Matrix{S})::S where S
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present))))

    calculate_kalman_filter_ll(𝓂, obs_idx, 𝐒₁, data_in_deviations)
end

function calculate_kalman_filter_ll(𝓂::ℳ, observables_index::Vector{Int}, 𝐒₁::Matrix{S}, data_in_deviations::Matrix{S})::S where S
    observables_and_states = @ignore_derivatives sort(union(𝓂.timings.past_not_future_and_mixed_idx,observables_index))

    A = 𝐒₁[observables_and_states,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(𝓂.timings.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = 𝐒₁[observables_and_states,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index),observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    
    dimensions = [size(A),size(𝐁)]
    
    values = vcat(vec(A), vec(collect(-𝐁)))

    P, _ = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)
    # P = reshape((ℒ.I - ℒ.kron(A, A)) \ reshape(𝐁, prod(size(A)), 1), size(A))

    u = zeros(S, length(observables_and_states))
    # u = SS_and_pars[sort(union(𝓂.timings.past_not_future_and_mixed,observables))] |> collect
    z = C * u

    loglik = S(0)
    for t in 1:size(data_in_deviations, 2)
        loglik, P, u, z = update_loglikelihood!(loglik, P, u, z, C, A, 𝐁, data_in_deviations[:, t])
        if loglik == -Inf
            break
        end
    end

    return -(loglik + length(data_in_deviations) * log(2 * 3.141592653589793)) / 2 
end



function get_ll(𝓂, 
    data::KeyedArray{Float64}, 
    parameter_values::Vector{S}; 
    algorithm::Symbol = :first_order, 
    filter::Symbol = :kalman, 
    warmup_iterations::Int = 0, 
    tol::AbstractFloat = 1e-12, 
    verbose::Bool = false)::S where S
    
    # checks to avoid errors further down the line and inform the user
    @assert filter ∈ [:kalman, :inversion] "Currently only the kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    if algorithm ∈ [:second_order,:pruned_second_order,:third_order,:pruned_third_order]
        filter = :inversion
    end

    observables = @ignore_derivatives get_and_check_observables(𝓂, data)

    @ignore_derivatives solve!(𝓂, verbose = verbose, algorithm = algorithm)

    # keep the parameters within bounds
    for (k,v) in 𝓂.bounds
        if k ∈ 𝓂.parameters
            if min(max(parameter_values[indexin([k], 𝓂.parameters)][1], v[1]), v[2]) != parameter_values[indexin([k], 𝓂.parameters)][1]
                return -Inf
            end
        end
    end

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

    state = zeros(𝓂.timings.nVars)

    ∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂) |> Matrix

    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    # 𝐒₁, solved = calculate_quadratic_iteration_solution_AD(∇₁; T = 𝓂.timings)
    
    if !solved return -Inf end

    state_update = function(state::Vector{T}, shock::Vector{S}) where {T,S} 
        aug_state = [state[𝓂.timings.past_not_future_and_mixed_idx]
                    shock]
        return 𝐒₁ * aug_state # you need a return statement for forwarddiff to work
    end

    # prepare data
    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data) .- SS_and_pars[obs_indices]

    loglikelihood = calculate_kalman_filter_ll(𝓂, observables, 𝐒₁, data_in_deviations)

    return loglikelihood
end




function calculate_posterior_ll(parameters, u)
    ctou, clandaw, cg, curvp, curvw, crhols, crhoas = @ignore_derivatives SW07.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters)]

    calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhoqs,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta = parameters

    parameters_combined = [ctou,clandaw,cg,curvp,curvw,calfa,csigma,cfc,cgy,csadjcost,chabb,cprobw,csigl,cprobp,cindw,cindp,czcap,crpi,crr,cry,crdy,crhoa,crhob,crhog,crhols,crhoqs,crhoas,crhoms,crhopinf,crhow,cmap,cmaw,constelab,z_ea,z_eb,z_eg,z_eqs,z_em,z_epinf,z_ew,ctrend,constepinf,constebeta]

    log_lik = 0
    log_lik -= get_ll(SW07, data(observables), parameters_combined, filter = :kalman)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_ea)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.025,5), z_eb)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_eg)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_eqs)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_em)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_epinf)
    log_lik -= logpdf(Truncated(InverseGamma(0.1, 2.0, μσ = true),0.01,3), z_ew)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoa)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhob)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhog)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoqs)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhoms)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.01,.9999), crhopinf)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.20, μσ = true),.001,.9999), crhow)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.2, μσ = true),0.01,.9999), cmap)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.2, μσ = true),0.01,.9999), cmaw)
    log_lik -= logpdf(Truncated(Normal(4,1.5),2,15), csadjcost)
    log_lik -= logpdf(Truncated(Normal(1.50,0.375),0.25,3), csigma)
    log_lik -= logpdf(Truncated(Beta(0.7, 0.1, μσ = true),0.001,0.99), chabb)
    log_lik -= logpdf(Beta(0.5, 0.1, μσ = true), cprobw)
    # log_lik -= logpdf(Truncated(Beta(0.5, 0.1, μσ = true),0.3,0.95), cprobw)
    log_lik -= logpdf(Truncated(Normal(2,0.75),0.25,10), csigl)
    log_lik -= logpdf(Beta(0.5, 0.10, μσ = true), cprobp)
    # log_lik -= logpdf(Truncated(Beta(0.5, 0.10, μσ = true),0.5,0.95), cprobp)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,0.99), cindw)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,0.99), cindp)
    log_lik -= logpdf(Truncated(Beta(0.5, 0.15, μσ = true),0.01,1), czcap)
    log_lik -= logpdf(Truncated(Normal(1.25,0.125),1.0,3), cfc)
    log_lik -= logpdf(Truncated(Normal(1.5,0.25),1.0,3), crpi)
    # log_lik -= logpdf(Truncated(Beta(0.75, 0.10, μσ = true),0.5,0.975), crr)
    log_lik -= logpdf(Beta(0.75, 0.10, μσ = true), crr)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5), cry)
    log_lik -= logpdf(Truncated(Normal(0.125,0.05),0.001,0.5), crdy)
    log_lik -= logpdf(Truncated(Gamma(0.625,0.1, μσ = true),0.1,2.0), constepinf)
    log_lik -= logpdf(Truncated(Gamma(0.25,0.1, μσ = true),0.01,2.0), constebeta)
    log_lik -= logpdf(Truncated(Normal(0.0,2.0),-10.0,10.0), constelab)
    log_lik -= logpdf(Truncated(Normal(0.4,0.10),0.1,0.8), ctrend)
    log_lik -= logpdf(Truncated(Normal(0.5,0.25),0.01,2.0), cgy)
    log_lik -= logpdf(Truncated(Normal(0.3,0.05),0.01,1.0), calfa)

    return log_lik
end


calculate_posterior_ll(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))], [])


calculate_posterior_ll(SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))] .* (1 +randn()*1e-6), [])


Zygote.gradient(x -> calculate_posterior_ll(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))] .* (1 +randn()*1e-6))[1] 

@profview for i in 1:100
    Zygote.gradient(x -> calculate_posterior_ll(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))] .* (1 +randn()*1e-6))[1] 
end

@benchmark Zygote.gradient(x -> calculate_posterior_ll(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))] .* (1 +randn()*1e-6))[1] 

@benchmark Zygote.gradient(x -> calculate_posterior_ll(x,[]), SW07.parameter_values[setdiff(1:length(SW07.parameters),indexin([:ctou,:clandaw,:cg,:curvp,:curvw,:crhols,:crhoas],SW07.parameters ))])[1] 












solve!(𝓂, verbose = verbose, algorithm = algorithm)

SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)

∇₁ = calculate_jacobian(parameter_values, SS_and_pars, 𝓂) |> Matrix

using BenchmarkTools
import SparseArrays: spdiagm
import ThreadedSparseArrays
[spdiagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], spdiagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 



# Assuming `size(datmp∇₁, 1)` gives the number of rows and `length(∇₁)` gives the number of columns
num_rows = size(datmp∇₁, 1)
num_cols = length(∇₁)

# Create an empty sparse matrix with the given dimensions
spd∇₁a = spzeros(num_rows, num_cols)


# @profview for i in 1:50 begin
𝐒₁, solved = MacroModelling.riccati_forward(∇₁;T = T, explosive = false)

sp𝐒₁ = sparse(𝐒₁) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
sp∇₁ = sparse(∇₁) |> ThreadedSparseArrays.ThreadedSparseMatrixCSC

droptol!(sp𝐒₁, 10*eps())
droptol!(sp∇₁, 10*eps())

# expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 
expand = [
    spdiagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC, 
    spdiagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:] |> ThreadedSparseArrays.ThreadedSparseMatrixCSC
] 

A = sp∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
B = sp∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

sol_buf = sp𝐒₁ * expand[2]
sol_buf2 = sol_buf * sol_buf

spd𝐒₁a = (ℒ.kron(expand[2] * sp𝐒₁, A') + 
        ℒ.kron(expand[2] * expand[2]', sol_buf' * A' + B'))
        
droptol!(spd𝐒₁a, 10*eps())

d𝐒₁a = spd𝐒₁a' |> collect

# Initialize empty spd∇₁a
spd∇₁a = spzeros(length(sp𝐒₁), length(∇₁))

# Directly allocate dA, dB, dC into spd∇₁a
# Note: You need to calculate the column indices where each matrix starts and ends
# This is conceptual; actual implementation would depend on how you can obtain or compute these indices
dA_cols = 1:(T.nFuture_not_past_and_mixed * size(𝐒₁,1))
dB_cols = dA_cols[end] .+ (1 : size(𝐒₁, 1)^2)
dC_cols = dB_cols[end] .+ (1 : length(sp𝐒₁))
18^2
spd∇₁a[:,dA_cols] = ℒ.kron(expand[1] * sol_buf2 * expand[2]' , ℒ.I(size(𝐒₁, 1)))'
spd∇₁a[:,dB_cols] = ℒ.kron(sp𝐒₁, ℒ.I(size(𝐒₁, 1)))' 
spd∇₁a[:,dC_cols] = ℒ.I(length(𝐒₁))

tmp = -(d𝐒₁a \ spd∇₁a)'
# end
# end

rand(3160)
b = rand(800)
x = -(d𝐒₁a \ spd∇₁a)' * b
-d𝐒₁a' * b
spd∇₁a' * x
-d𝐒₁a' \ spd∇₁a' 

spd∇₁a' * rand(800)
spd𝐒₁a = sparse(d𝐒₁a)
d∇₁a = collect(spd∇₁a)

@benchmark spd𝐒₁a \ d∇₁a
@benchmark d𝐒₁a \ d∇₁a
@benchmark d𝐒₁a \ spd∇₁a


inv(d𝐒₁a)

using Krylov


d𝐒₁a_dense = Matrix(d𝐒₁a)  # Ensure it's a dense matrix if it's not already
d∇₁a_dense = Matrix(d∇₁a)


function sylvester!(sol,𝐱)
    𝐗 = reshape(𝐱, size(d∇₁a))
    sol .= vec(d𝐒₁a * 𝐗)
    return sol
end

sylvester = LinearOperators.LinearOperator(Float64, length(d∇₁a), length(d∇₁a), true, true, sylvester!)

𝐂, info = Krylov.gmres(sylvester, [vec(d∇₁a);], itmax = 10)

Krylov.block_gmres(collect(d𝐒₁a), d∇₁a)
# Assuming d𝐒₁a and d∇₁a are already computed as per your code
# Convert matrices to appropriate types if necessary. Krylov methods work with dense matrices.

# Define a function for the matrix-vector multiplication
A_mul_B!(y, A, x) = mul!(y, A, x)

# Setup a vector for the result of GMRES
b = vec(d∇₁a_dense)  # Make sure it's a vector
x = zeros(size(b))   # Initial guess can be a vector of zeros
using LinearOperators

# Assuming the rest of your code has defined d𝐒₁a and d∇₁a

# Define the linear operator based on d𝐒₁a
A_op = LinearOperator(Float64, size(d𝐒₁a, 1), size(d𝐒₁a, 2), false, false,
                      v -> d𝐒₁a * v, 
                      v -> d𝐒₁a' * v, 
                      v -> d𝐒₁a' * v)
# Apply GMRES
# You may need to adjust the tolerance and max iterations based on your problem's specifics
sol, history = Krylov.gmres(A_op, b)

# Since the original operation was -(d𝐒₁a \ d∇₁a)', we need to reshape and transpose the solution vector
tmp = reshape(sol, size(d∇₁a_dense)...)'

# 1st order perturbation solution solver
expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]

∇̂₀ =  RF.lu(∇₀)

A = ∇̂₀ \ ∇₋
B = ∇̂₀ \ ∇₊

C = copy(A)
C̄ = similar(A)

maxiter = 10000  # Maximum number of iterations

error = one(tol) + tol
iter = 0
while error > tol && iter <= maxiter
    C̄ = copy(C)  # Store the current C̄ before updating it

    # Update C̄ based on the given formula
    C = A + B * C^2

    # Check for convergence
    error = maximum(abs, C - C̄)

    iter += 1
end

D = -(∇₊ * -C + ∇₀) \ ∇ₑ

return hcat(-C[:, T.past_not_future_and_mixed_idx], D), error <= tol
s1 = hcat(-C[:, T.past_not_future_and_mixed_idx], D)
𝐒₁, solved = MacroModelling.riccati_forward(∇₁; T = 𝓂.timings)

maximum(abs, 𝐒₁ + C[:, T.past_not_future_and_mixed_idx])


@benchmark begin
    A = ∇̂₀ \ ∇₋
    B = ∇̂₀ \ ∇₊

    C = similar(A)
    C̄ = similar(A)

    sol = speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = eps(), maps_limit = 10000)
end

sol.minimizer

C = -sol.minimizer
# maximum(abs,C + (A + B * C^2))

tol=eps()
@benchmark begin
    A = ∇̂₀ \ ∇₋
    B = ∇̂₀ \ ∇₊

    C = copy(A)
    C̄ = similar(A)

    maxiter = 10000  # Maximum number of iterations

    error = one(tol) + tol
    iter = 0
    while error > tol && iter <= maxiter
        C̄ = copy(C)  # Store the current C̄ before updating it

        # Update C̄ based on the given formula
        C = A + B * C^2

        # Check for convergence
        error = maximum(abs, C - C̄)

        iter += 1
    end
end



@benchmark begin
    A = sparse(∇̂₀ \ ∇₋)
    B = sparse(∇̂₀ \ ∇₊)

    droptol!(A, 1e-15)
    droptol!(B, 1e-15)

    C = copy(collect(A))
    C̄ = similar(C)

    maxiter = 10000  # Maximum number of iterations

    error = one(tol) + tol
    iter = 0
    while error > tol && iter <= maxiter
        C̄ = copy(C)  # Store the current C̄ before updating it

        # Update C̄ based on the given formula
        C = A + B * C^2

        # droptol!(C, 1e-15)

        # Check for convergence
        error = maximum(abs, C - C̄)

        iter += 1
    end
end


D = -(∇₊ * -C + ∇₀) \ ∇ₑ

return hcat(-C[:, T.past_not_future_and_mixed_idx], D), error < tol

s1 =hcat(-C[:, T.past_not_future_and_mixed_idx], D)

𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    




𝐒₁, solved = MacroModelling.riccati_forward(∇₁; T = 𝓂.timings)


function riccati_conditions(∇₁::AbstractMatrix{M}, sol_d::AbstractMatrix{N}, solved::Bool; T, explosive::Bool = false) where {M,N}
    expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    colA = ℒ.diagm(ones(size(∇₁,2)))[:,1:T.nFuture_not_past_and_mixed]
    colB = ℒ.diagm(ones(size(∇₁,2)))[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    colC = ℒ.diagm(ones(size(∇₁,2)))[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)]

    A = ∇₁ * colA * expand[1]
    B = ∇₁ * colB
    C = ∇₁ * colC * expand[2]

    sol_buf = sol_d * expand[2]

    sol_buf2 = sol_buf * sol_buf

    err1 = (A * sol_buf2 + B * sol_buf + C) * expand[2]'
    # err1 = (B * sol_buf + C) * expand[2]'

    # err1 = A * sol_buf2  # + B * sol_buf + C

    return err1 # [:,T.past_not_future_and_mixed_idx]
end

riccati_conditions(∇₁, 𝐒₁, solved, T = 𝓂.timings)

d𝐒₁f = ForwardDiff.jacobian(x -> riccati_conditions(∇₁, x, solved, T = 𝓂.timings), 𝐒₁) |> sparse
d𝐒₁z = Zygote.jacobian(x -> riccati_conditions(∇₁, x, solved, T = 𝓂.timings), 𝐒₁)[1] |> sparse
collect(d𝐒₁z)


sol_d = 𝐒₁
expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

A = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
B = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
# C = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

sol_buf = sol_d * expand[2]

sol_buf2 = sol_buf * sol_buf

# err1 = A * sol_buf2 + B * sol_buf + C


d𝐒₁a = (kron(expand[2] * sol_d, A') + 
        kron(expand[2] * expand[2]', sol_buf' * A' + B'))' |> collect


sum(abs, d𝐒₁a - d𝐒₁f)


using LinearAlgebra

d∇₁f = ForwardDiff.jacobian(x -> riccati_conditions(x, 𝐒₁, solved, T = 𝓂.timings), ∇₁) |> sparse

dA = kron(expand[1] * sol_buf2 * expand[2]' , I(size(𝐒₁,1)))'
dB = kron(sol_d, I(size(𝐒₁,1)))' 
dC = I(length(sol_d))

datmp∇₁ = hcat(dA,dB,dC)

d∇₁a = hcat(datmp∇₁, zeros(size(datmp∇₁, 1), length(∇₁) - size(datmp∇₁, 2))) |> sparse

sum(abs, d∇₁a - d∇₁f)


collect(d∇₁a)\d𝐒₁a |> sparse
collect(d∇₁f)\d𝐒₁f |> sparse

collect(d𝐒₁a)\d∇₁a |> sparse
collect(d𝐒₁f)\d∇₁f |> sparse





import MacroModelling: timings, riccati_forward


function calculate_quadratic_iteration_solution(∇₁::AbstractMatrix{Float64}; T::timings, tol::AbstractFloat = 1e-12)
    # see Binder and Pesaran (1997) for more details on this approach
    expand = @views [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:],
            ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    ∇₊ = @views ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    ∇₀ = @views ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    ∇₋ = @views ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]
    ∇ₑ = @views ∇₁[:,(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1):end]
    
    ∇̂₀ =  RF.lu(∇₀)
    
    A = ∇̂₀ \ ∇₋
    B = ∇̂₀ \ ∇₊

    C = similar(A)
    C̄ = similar(A)

    sol = speedmapping(zero(A); m! = (C̄, C) -> C̄ .=  A + B * C^2, tol = tol, maps_limit = 100000)

    C = -sol.minimizer

    # D = -(∇₊ * C + ∇₀) \ ∇ₑ

    # @views hcat(C[:,T.past_not_future_and_mixed_idx],D), sol.converged
    C[:,T.past_not_future_and_mixed_idx], sol.converged
end

S1, solved = calculate_quadratic_iteration_solution(∇₁,T = 𝓂.timings)



using ChainRulesCore, LinearAlgebra

function ChainRulesCore.rrule(::typeof(calculate_quadratic_iteration_solution), ∇₁::AbstractMatrix{Float64}; T::timings, tol::AbstractFloat = eps())
    # Forward pass to compute the output and intermediate values needed for the backward pass
    𝐒₁, solved = calculate_quadratic_iteration_solution(∇₁, T=T, tol=tol)

    expand = [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    A = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
    B = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

    sol_buf = 𝐒₁ * expand[2]
    sol_buf2 = sol_buf * sol_buf
    
    d𝐒₁a = (kron(expand[2] * 𝐒₁, A') + 
            kron(expand[2] * expand[2]', sol_buf' * A' + B'))' |> collect

    dA = kron(expand[1] * sol_buf2 * expand[2]' , I(size(𝐒₁,1)))'
    dB = kron(𝐒₁, I(size(𝐒₁,1)))' 
    dC = I(length(𝐒₁))
            
    datmp∇₁ = hcat(dA,dB,dC)
            
    d∇₁a = hcat(datmp∇₁, zeros(size(datmp∇₁, 1), length(∇₁) - size(datmp∇₁, 2))) |> collect
    tmp = -(d𝐒₁a \ d∇₁a)'

    function calculate_quadratic_iteration_solution_pullback(Δ𝐒₁)
        # Backward pass to compute the derivatives with respect to inputs
        # This would involve computing the derivatives for each operation in reverse order
        # and applying chain rule to propagate through the function
        return NoTangent(), reshape(tmp * vec(Δ𝐒₁[1]), size(∇₁)) # Return NoTangent() for non-Array inputs or if there's no derivative w.r.t. them
        # return NoTangent(), (reshape(-d𝐒₁a \ d∇₁a * vec(Δ𝐒₁) , size(∇₁))) # Return NoTangent() for non-Array inputs or if there's no derivative w.r.t. them
    end

    return (𝐒₁, solved), calculate_quadratic_iteration_solution_pullback
end


𝐒₁, solved = calculate_quadratic_iteration_solution(∇₁, T=T, tol=tol)

expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

A = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand[1]
B = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]

sol_buf = 𝐒₁ * expand[2]
sol_buf2 = sol_buf * sol_buf

d𝐒₁a = (kron(expand[2] * 𝐒₁, A') + 
        kron(expand[2] * expand[2]', sol_buf' * A' + B'))' |> collect

dA = kron(expand[1] * sol_buf2 * expand[2]' , I(size(𝐒₁,1)))'
dB = kron(𝐒₁, I(size(𝐒₁,1)))' 
dC = I(length(𝐒₁))
        
datmp∇₁ = hcat(dA,dB,dC)
        
d∇₁a = hcat(datmp∇₁, zeros(size(datmp∇₁, 1), length(∇₁) - size(datmp∇₁, 2))) |> sparse



fid = FiniteDifferences.jacobian(central_fdm(3,1), x->calculate_quadratic_iteration_solution(x, T = 𝓂.timings)[1], ∇₁)[1]
red = Zygote.jacobian(x->calculate_quadratic_iteration_solution(x, T = 𝓂.timings)[1], ∇₁)[1]

@profview for i in 1:10 red = Zygote.jacobian(x->calculate_quadratic_iteration_solution(x, T = 𝓂.timings)[1], ∇₁)[1] end

dy = vec([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0])

-(d∇₁a \ d𝐒₁a) * dy
-(d𝐒₁a \ d∇₁a)' * dy

fid[27,:]
isapprox(-(d𝐒₁a \ d∇₁a), fid, rtol = 1e-7)

findmax(abs,(d𝐒₁a \ d∇₁a) + fid)
(d𝐒₁a \ d∇₁a)[21,126]
fid[21,126]

-(d𝐒₁a \ d∇₁a)[:,18]
fid[1,:]





d𝐒₁a' \ d∇₁a

using FiniteDifferences, Zygote

@profview for i in 1:100 Zygote.jacobian(x->begin
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(x, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    aa = calculate_jacobian(x, SS_and_pars, 𝓂) |> Matrix
    
    calculate_quadratic_iteration_solution(aa, T = 𝓂.timings)
    end,parameter_values)[1] end

fid = FiniteDifferences.jacobian(central_fdm(3,1),
    x->begin
    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(x, 𝓂, verbose, false, 𝓂.solver_parameters)

    aa = calculate_jacobian(x, SS_and_pars, 𝓂) |> Matrix

    calculate_quadratic_iteration_solution(aa, T = 𝓂.timings)
    end, parameter_values)[1]



d∇₁a * vec(∇₁)
d∇₁a \ d𝐒₁a * vec(𝐒₁)
# droptol!(d∇₁,1e-14)
@benchmark d∇₁\d𝐒₁
@benchmark d𝐒₁\d∇₁

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

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
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

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
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

