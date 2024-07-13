# Estimate a simple model - Schorfheide (2000)

This tutorial is intended to show the workflow to estimate a model using the No-U-Turn sampler (NUTS). The tutorial works with a benchmark model for estimation and can therefore be compared to results from other software packages (e.g. [dynare](https://archives.dynare.org/documentation/examples.html)).

## Define the model

The first step is always to name the model and write down the equations. For the [schorfheide2000](@citet) model this would go as follows:

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

First, we load the package and then use the [`@model`](@ref) macro to define our model. The first argument after [`@model`](@ref) is the model name and will be the name of the object in the global environment containing all information regarding the model. The second argument to the macro are the equations, which we write down between `begin` and `end`. Equations can contain an equality sign or the expression is assumed to equal 0. Equations cannot span multiple lines (unless you wrap the expression in brackets) and the timing of endogenous variables are expressed in the square brackets following the variable name (e.g. `[-1]` for the past period). Exogenous variables (shocks) are followed by a keyword in square brackets indicating them being exogenous (in this case `[x]`). Note that names can leverage julia's unicode capabilities (e.g. alpha can be written as α).

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

## Load data

Given the equations and parameters, we only need the entries in the data which correspond to the observables in the model (need to have the exact same name) to estimate the model.
First, we load in the data from a CSV file (using the CSV and DataFrames packages) and convert it to a `KeyedArray` (using the AxisKeys package). Furthermore, we log transform the data provided in levels, and last but not least we select only those variables in the data which are observables in the model.

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

## Define bayesian model

Next we define the parameter priors using the Turing package. The `@model` macro of the Turing package allows us to define the prior distributions over the parameters and combine it with the (Kalman filter) loglikelihood of the model and parameters given the data with the help of the `get_loglikelihood` function. We define the prior distributions in an array and pass it on to the `arraydist` function inside the `@model` macro from the Turing package. It is also possible to define the prior distributions inside the macro but especially for reverse mode auto differentiation the `arraydist` function is substantially faster. When defining the prior distributions we can rely n the distribution implemented in the Distributions package. Note that the `μσ` parameter allows us to hand over the moments (`μ` and `σ`) of the distribution as parameters in case of the non-normal distributions (Gamma, Beta, InverseGamma), and we can also define upper and lower bounds truncating the distribution as third and fourth arguments to the distribution functions. Last but not least, we define the loglikelihood and add it to the posterior loglikelihood with the help of the `@addlogprob!` macro.

```@repl tutorial_2
import Zygote
import DynamicPPL
import Turing
import Turing: NUTS, sample, logpdf, AutoZygote

prior_distributions = [
    Beta(0.356, 0.02, μσ = true),           # alp
    Beta(0.993, 0.002, μσ = true),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, μσ = true),          # rho
    Beta(0.65, 0.05, μσ = true),            # psi
    Beta(0.01, 0.005, μσ = true),           # del
    InverseGamma(0.035449, Inf, μσ = true), # z_e_a
    InverseGamma(0.008862, Inf, μσ = true)  # z_e_m
]

Turing.@model function FS2000_loglikelihood_function(data, model)
    parameters ~ Turing.arraydist(prior_distributions)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(model, data, parameters)
    end
end
```

## Sample from posterior: No-U-Turn Sampler (NUTS)

We use the NUTS sampler to retrieve the posterior distribution of the parameters. This sampler uses the gradient of the posterior loglikelihood with respect to the model parameters to navigate the parameter space. The NUTS sampler is considered robust, fast, and user-friendly (auto-tuning of hyper-parameters).

First we define the loglikelihood model with the specific data, and model. Next, we draw 1000 samples from the model:

```@repl tutorial_2
FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000);

n_samples = 1000

chain_NUTS  = sample(FS2000_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = false);
```

### Inspect posterior

In order to understand the posterior distribution and the sequence of sample we are plot them:

```@repl tutorial_2; setup = :(chain_NUTS = read("../assets/chain_FS2000.jls", Chains))
using StatsPlots
plot(chain_NUTS);
```

![NUTS chain](../assets/FS2000_chain_NUTS.png)

Next, we are plotting the posterior loglikelihood along two parameters dimensions, with the other parameters ket at the posterior mean, and add the samples to the visualisation. This visualisation allows us to understand the curvature of the posterior and puts the samples in context.

```@repl tutorial_2
using ComponentArrays, MCMCChains
import DynamicPPL: logjoint

parameter_mean = mean(chain_NUTS)

pars = ComponentArray([parameter_mean.nt[2]], Axis(:parameters));

logjoint(FS2000_loglikelihood, pars)

function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
    orig_pars[1][pars_syms] = [par1, par2]
    logjoint(model, orig_pars)
end

granularity = 32;

par1 = :del;
par2 = :gam;

paridx1 = indexin([par1], FS2000.parameters)[1];
paridx2 = indexin([par2], FS2000.parameters)[1];

par_range1 = collect(range(minimum(chain_NUTS[Symbol("parameters[$paridx1]")]), stop = maximum(chain_NUTS[Symbol("parameters[$paridx1]")]), length = granularity));
par_range2 = collect(range(minimum(chain_NUTS[Symbol("parameters[$paridx2]")]), stop = maximum(chain_NUTS[Symbol("parameters[$paridx2]")]), length = granularity));

p = surface(par_range1, par_range2, 
            (x,y) -> calculate_log_probability(x, y, [paridx1, paridx2], pars, FS2000_loglikelihood),
            camera=(30, 65),
            colorbar=false,
            color=:inferno);

joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray([reduce(hcat, get(chain_NUTS, :parameters)[1])[s,:]], Axis(:parameters))) for s in 1:length(chain_NUTS)];

scatter3d!(vec(collect(chain_NUTS[Symbol("parameters[$paridx1]")])),
            vec(collect(chain_NUTS[Symbol("parameters[$paridx2]")])),
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

Other than the mean and median of the posterior distribution we can also calculate the mode as follows:

```@repl tutorial_2
modeFS2000 = Turing.maximum_a_posteriori(FS2000_loglikelihood, 
                                        adtype = AutoZygote(), 
                                        initial_params = FS2000.parameter_values)
```

## Model estimates given the data and the model solution

Having found the parameters at the posterior mode we can retrieve model estimates of the shocks which explain the data used to estimate it. This can be done with the `get_estimated_shocks` function:

```@repl tutorial_2
get_estimated_shocks(FS2000, data, parameters = collect(modeFS2000.values))
```

As the first argument we pass the model, followed by the data (in levels), and then we pass the parameters at the posterior mode. The model is solved with this parameterisation and the shocks are calculated using the Kalman smoother.

We estimated the model on two variables but our model allows us to look at all variables given the data. Looking at the estimated variables can be done using the `get_estimated_variables` function:

```@repl tutorial_2
get_estimated_variables(FS2000, data)
```

Since we already solved the model with the parameters at the posterior mode we do not need to do so again. The function returns a KeyedArray with the values of the variables in levels at each point in time.

Another useful tool is a historical shock decomposition. It allows us to understand the contribution of the shocks for each variable. This can be done using the `get_shock_decomposition` function:

```@repl tutorial_2
get_shock_decomposition(FS2000, data)
```

We get a 3-dimensional array with variables, shocks, and time periods as dimensions. The shocks dimension also includes the initial value as a residual between the actual value and what was explained by the shocks. This computation also relies on the Kalman smoother.

Last but not least, we can also plot the model estimates and the shock decomposition. The model estimates plot, using `plot_model_estimates`:

```@repl tutorial_2
plot_model_estimates(FS2000, data)
```

![Model estimates](../assets/estimation__m__2.png)

shows the variables of the model (blue), the estimated shocks (in the last panel), and the data (red) used to estimate the model.

The shock decomposition can be plotted using `plot_shock_decomposition`:

```@repl tutorial_2
plot_shock_decomposition(FS2000, data)
```

![Shock decomposition](../assets/estimation_shock_decomp__m__2.png)

and it shows the contribution of the shocks and the contribution of the initial value to the deviations of the variables.
