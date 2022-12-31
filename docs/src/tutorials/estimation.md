# Estimate a simple model - Schorfheide (2000)
This tutorial is intended to show the workflow to estimate a model using the NUTS sampler. The tutorial works with a benchmark model for estimation and can therefore be compared to results from other software packages (e.g. [dynare](https://archives.dynare.org/documentation/examples.html)).

## Define the model
The first step is always to name the model and write down the equations. For the Schorfheide (2000) model this would go as follows:
```@setup tutorial_2
ENV["GKSwstype"] = "100"
using Random
Random.seed!(3)
```
```@repl tutorial_2
using MacroModelling

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
```
First, we load the package and then use the [`@model`](@ref) macro to define our model. The first argument after [`@model`](@ref) is the model name and will be the name of the object in the global environment containing all information regarding the model. The second argument to the macro are the equations, which we write down between `begin` and `end`. Equations can contain an equality sign or the expression is assumed to equal 0. Equations cannot span multiple lines and the timing of endogenous variables are expressed in the squared brackets following the variable name (e.g. `[-1]` for the past period). Exogenous variables (shocks) are followed by a keyword in squared brackets indicating them being exogenous (in this case [x]). In this example there are also variables in the non stochastic steady state denoted by `[ss]`. Note that names can leverage julia's unicode capabilities (alpha can be written as α).

## Define the parameters
Next we need to add the parameters of the model. The macro [`@parameters`](@ref) takes care of this:
```@repl tutorial_2
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
```
The block defining the parameters above only describes the simple parameter definitions the same way you assign values (e.g. `alp = .356`). 

Note that we have to write one parameter definition per line.

## Load data, declare observables, and write moments mapping to distribution parameters
Given the equations and parameters, we only need the data and define the observables to be able to estimate the model. 
Frist, we load in the data using the from a CSV file and convert it to a `KeyedArray`. Furthermore, we log transform the data provided in levels, and define the observables of the model. Last but not least we select only those variables in the data which are declared observables in the model.
```@repl tutorial_2
using CSV, DataFrames, AxisKeys

# load data
dat = CSV.read("../assets/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

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
```

## Define bayesian model including priors
Next we are defining the priors over the parameters:
```@repl tutorial_2
import Turing
import Turing: Normal, Beta, InverseGamma, NUTS, sample, logpdf

Turing.@model function kalman(data, m, observables)
    alp     ~ Beta(beta_map(0.356, 0.02)...)
    bet     ~ Beta(beta_map(0.993, 0.002)...)
    gam     ~ Normal(0.0085, 0.003)#, eps(), .1)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(beta_map(0.129, 0.223)...)
    psi     ~ Beta(beta_map(0.65, 0.05)...)
    del     ~ Beta(beta_map(0.01, 0.005)...)
    z_e_a   ~ InverseGamma(inv_gamma_map(0.035449, Inf)...)
    z_e_m   ~ InverseGamma(inv_gamma_map(0.008862, Inf)...)

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    Turing.@addlogprob! logpdf(Beta(beta_map(0.356, 0.02)...),alp)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.993, 0.002)...),bet)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.129, 0.223)...),rho)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.65, 0.05)...),psi)
    Turing.@addlogprob! logpdf(Beta(beta_map(0.01, 0.005)...),del)
    Turing.@addlogprob! logpdf(Normal(0.0085, 0.003),gam)
    Turing.@addlogprob! logpdf(Normal(1.0002, 0.007),mst)
    Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.035449, Inf)...),z_e_a)
    Turing.@addlogprob! logpdf(InverseGamma(inv_gamma_map(0.008862, Inf)...),z_e_m)
end
```

## Sample from posterior
Having set up the prior loglikelihood given the data we use the NUTS sampler to retrieve the posterior distribution:
```@repl tutorial_2
turing_model = kalman(data, FS2000, observables)
n_samples = 1000
chain_NUTS  = sample(turing_model, NUTS(), n_samples)
```
