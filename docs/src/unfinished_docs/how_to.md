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


## How to estimate a model 


## Interactive plotting
