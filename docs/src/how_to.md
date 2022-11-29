## How to estimate a model 
<!-- τ * (1 - ν) / (ν * (π[ss])^2 * ϕ) -->

<!-- @model HSestimation begin
    y[0] = y[1] + g[0] - g[-1] - (R[0] - π[1] - z[1]) / τ
    π[0] = π[1] + κ * (y[0] - g[0])
    R[0] = ρᴿ * R[-1] + (1 - ρᴿ) * ψ₁ * π[0] + (1 - ρᴿ) * ψ₂ * (y[0] - g[0]) + ϵᴿ[x]
    g[0] = ρᵍ * g[-1] + ϵᵍ[x]
    z[0] = ρᶻ * z[-1] + ϵᶻ[x]
    YGR[0] = γᵍ + 100 * (y[0] - y[-1] + z[0])
    INFL[0] = πᵃ + 400 * π[0]
    INT[0] = πᵃ + rᵃ + 4 * γᵍ + 400 * R[0]
end -->
```@repl
using MacroModelling
@model HSestimation begin
    y[0] = y[1] + g[0] - g[-1] - (R[0] - π[0] - z[1]) / τ
    π[-1] = π[0] + κ * (y[0] - g[0])
    R[0] = ρᴿ * R[-1] + (1 - ρᴿ) * ψ₁ * π[-1] + (1 - ρᴿ) * ψ₂ * (y[0] - g[0]) + σᴿ * ϵᴿ[x]
    g[0] = ρᵍ * g[-1] + σᵍ * ϵᵍ[x]
    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
    YGR[0] = γᵍ + 100 * (y[0] - y[-1] + z[0])
    INFL[0] = πᵃ + 400 * π[0]
    INT[0] = πᵃ + rᵃ + 4 * γᵍ + 400 * R[0]
end

@parameters HSestimation begin
    rᵃ  = .5
    γᵍ  = .6
    κ   = .85
    πᵃ  = 3.2
    ρᴿ  = .75
    ρᵍ  = .75
    ρᶻ  = .75
    τ   = 2.5
    ψ₁  = 1.9
    ψ₂  = .6
    σᴿ  = .1
    σᵍ  = .1
    σᶻ  = .1
end
```





```@repl
get_solution(HSestimation)
using Turing

Turing.@model function HSestimation_kalman(data, m, observables)
    rᵃ  ~   Gamma(.5,.5)
    γᵍ  ~   Normal(.4,.2)
    κ   ~   Uniform(0,1)
    πᵃ  ~   Gamma(7,2)
    ρᴿ  ~   Uniform(0,1)
    ρᵍ  ~   Uniform(0,1)
    ρᶻ  ~   Uniform(0,1)
    τ   ~   Gamma(2,.5)
    ψ₁  ~   Gamma(1.5,.25)
    ψ₂  ~   Gamma(.5,.25)
    σᴿ  ~   InverseGamma(.4, 4)
    σᵍ  ~   InverseGamma(1, 4)
    σᶻ  ~   InverseGamma(.5, 4)
    
    parameters = [rᵃ, γᵍ, κ, πᵃ, ρᴿ, ρᵍ, ρᶻ, σᴿ, σᵍ, σᶻ, τ, ψ₁, ψ₂]
    
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data, observables; parameters = parameters)
end
using DelimitedFiles
data = readdlm("docs/src/assets/us.csv", ',', Float64)' |>collect
observables = [:YGR, :INFL, :INT]

turing_model = HSestimation_kalman(data, HSestimation, observables) # passing observables from before 

n_samples = 50
n_adapts = 250
δ = 0.5
chain_1_marginal = sample(turing_model, NUTS(n_adapts, δ), n_samples; progress = true)

```


## Use calibration equations

Next we need to add the parameters of the model. The macro `@parameters <name of the model>` takes care of this:
```julia
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
```
No need for line endings. If you want to define a parameter as a function of another parameter you can do this:
```julia
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    beta1 = 1
    beta2 = .95
    β | β = beta2/beta1
end
```
Note that the parser takes parameters assigned to a numerical value first and then solves for the parameters defined by relationships: `β | ...`. This means also the following will work:
```julia
@parameters RBC begin
    β | β = beta2/beta1
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    beta1 = 1
    beta2 = .95
end
```
More interestingly one can use (non-stochastic) steady state values in the relationships:
```julia
@parameters RBC begin
    β = .95
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α | k[ss] / (4 * q[ss]) = 1.5
end
```

## Higher order perturbation solutions




## Interactive plotting
