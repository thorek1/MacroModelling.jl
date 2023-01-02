# Estimate a simple model - Schorfheide (2000)
This tutorial is intended to show the workflow to estimate a model using the No-U-Turn sampler (NUTS). The tutorial works with a benchmark model for estimation and can therefore be compared to results from other software packages (e.g. [dynare](https://archives.dynare.org/documentation/examples.html)).

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
First, we load the package and then use the [`@model`](@ref) macro to define our model. The first argument after [`@model`](@ref) is the model name and will be the name of the object in the global environment containing all information regarding the model. The second argument to the macro are the equations, which we write down between `begin` and `end`. Equations can contain an equality sign or the expression is assumed to equal 0. Equations cannot span multiple lines and the timing of endogenous variables are expressed in the squared brackets following the variable name (e.g. `[-1]` for the past period). Exogenous variables (shocks) are followed by a keyword in squared brackets indicating them being exogenous (in this case `[x]`). Note that names can leverage julia's unicode capabilities (e.g. alpha can be written as α).

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
First, we load in the data from a CSV file (using the CSV and DataFrames packages) and convert it to a `KeyedArray` (using the AxisKeys package). Furthermore, we log transform the data provided in levels, and define the observables of the model. Last but not least we select only those variables in the data which are declared observables in the model.

```@repl tutorial_2
using CSV, DataFrames, AxisKeys

# load data
dat = CSV.read("../assets/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)
```

In order to make the prior definitions more intuitive for users we define helper functions translating the mean and standard deviation of the beta, inverse gamma, and gamma distributions to the respective distribution parameters (e.g. α and β for the beta distribution).

```@repl tutorial_2
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

## Define bayesian model
Next we define the parameter priors using the Turing package. The `@model` macro of the Turing package allows us to define the prior distributions over the parameters and combine it with the loglikelihood of the model and parameters given the data with the help of the `calculate_kalman_filter_loglikelihood` function. Inside the macro we first define the priors using distributions of the Distributions package (reexported by Turing) and the previously defined helper functions. See the documentation of the Turing package for more details. Next, we define the loglikelihood and add it to the posterior loglikelihood with the help of the `@addlogprob!` macro.
```@repl tutorial_2
import Turing
import Turing: Normal, Beta, InverseGamma, NUTS, sample, logpdf

Turing.@model function FS2000_loglikelihood_function(data, m, observables)
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
end
```

## Sample from posterior: No-U-Turn Sampler (NUTS) 
We use the NUTS sampler to retrieve the posterior distribution of the parameters. This sampler uses the gradient of the posterior loglikelihood with respect to the model parameters to navigate the parameter space. The NUTS sampler is considered robust, fast, and user-friendly (auto-tuning of hyper-parameters).

First we define the loglikelihood model with the specific data, observables, and model. Next, we draw 1000 samples from the model:
```@repl tutorial_2
FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables)
n_samples = 1000
chain_NUTS  = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = false)
```


### Inspect posterior
In order to understand the posterior distribution and the sequence of sample we are plot them:
```@repl tutorial_2
using StatsPlots
StatsPlots.plot(chain_NUTS);
```
![NUTS chain](../assets/FS2000_chain_NUTS.png)


Next, we are plotting the posterior loglikelihood along two parameters dimensions, with the other parameters ket at the posterior mean, and add the samples to the visualisation. This visualisation allows us to understand the curvature of the posterior and puts the samples in context.

```@repl tutorial_2
using ComponentArrays, MCMCChains, DynamicPPL, Plots

parameter_mean = mean(chain_NUTS)
pars = ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1]))

logjoint(FS2000_loglikelihood, pars)

function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
    orig_pars[pars_syms] = [par1, par2]
    logjoint(model, orig_pars)
end

granularity = 32;

par1 = :del;
par2 = :gam;
par_range1 = collect(range(minimum(chain_NUTS[par1]), stop = maximum(chain_NUTS[par1]), length = granularity));
par_range2 = collect(range(minimum(chain_NUTS[par2]), stop = maximum(chain_NUTS[par2]), length = granularity));

p = surface(par_range1, par_range2, 
            (x,y) -> calculate_log_probability(x, y, [par1, par2], pars, FS2000_loglikelihood),
            camera=(30, 65),
            colorbar=false,
            color=:inferno);


joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray(reduce(hcat, get(chain_NUTS, FS2000.parameters)[FS2000.parameters])[s,:], Axis(FS2000.parameters))) for s in 1:length(chain_NUTS)]

scatter3d!(vec(collect(chain_NUTS[par1])),
           vec(collect(chain_NUTS[par2])),
           joint_loglikelihood,
            mc = :viridis, 
            marker_z = collect(1:length(chain_NUTS)), 
            msw = 0,
            legend = false, 
            colorbar = false, 
            xlabel = string(par1),
            ylabel = string(par2),
            zlabel = "Log probability",
            alpha = 0.5);

p
```
![Posterior surface](../assets/FS2000_posterior_surface.png)


## Find posterior mode
Other than the mean and median of the posterior distribution we can also calculate the mode. To this end we will use optimisation routines from the Optimization, OptimizationNLopt, and OptimizationOptimisers packages.

First, we define the posterior loglikelihood function, similar to how we defined it for the Turing model macro.

```@repl tutorial_2
function calculate_posterior_loglikelihood(parameters, u)
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
    log_lik = 0
    log_lik -= calculate_kalman_filter_loglikelihood(FS2000, data(observables), observables; parameters = parameters)
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
```

Next, we set up the optimsation problem and first use the ADAM global optimiser for 1000 iterations in order to avoid local optima and then fine tune with L-BFGS.
```@repl tutorial_2
using Optimization, OptimizationNLopt, OptimizationOptimisers

f = OptimizationFunction(calculate_posterior_loglikelihood, Optimization.AutoForwardDiff())

prob = OptimizationProblem(f, collect(pars), []);
sol = solve(prob, Optimisers.ADAM(), maxiters = 1000)
sol.minimum

lbs = fill(-1e12, length(FS2000.parameters));
ubs = fill(1e12, length(FS2000.parameters));

bounds_index_in_pars = indexin(intersect(FS2000.bounded_vars,FS2000.parameters),FS2000.parameters);
bounds_index_in_bounds = indexin(intersect(FS2000.bounded_vars,FS2000.parameters),FS2000.bounded_vars);

lbs[bounds_index_in_pars] = max.(-1e12,FS2000.lower_bounds[bounds_index_in_bounds]);
ubs[bounds_index_in_pars] = min.(1e12,FS2000.upper_bounds[bounds_index_in_bounds]);

prob = OptimizationProblem(f, min.(max.(sol.u,lbs),ubs), [], lb = lbs, ub = ubs);
sol = solve(prob, NLopt.LD_LBFGS())
sol.minimum
```