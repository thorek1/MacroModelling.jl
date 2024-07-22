
## DSL

MacroModelling parses models written using a user-friendly syntax:
```julia
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
```
The most important rule is that variables are followed by the timing in square brackets for endogenous variables, e.g. `Y[0]`, exogenous variables are marked by certain keywords (see below), e.g. `ϵ[x]`, and parameters need no further syntax, e.g. `α`.

A model written with this syntax allows the parser to identify, endogenous and exogenous variables and their timing as well as parameters.

Note that variables in the present (period *t* or *0*) have to be denoted as such: `[0]`. The parser also takes care of creating auxilliary variables in case the model contains leads or lags of the variables larger than 1:
```julia
@model RBC_lead_lag begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * (eps_z[x-8] + eps_z[x-4] + eps_z[x+4] + eps_z_s[x])
    c̄⁻[0] = (c[0] + c[-1] + c[-2] + c[-3]) / 4
    c̄⁺[0] = (c[0] + c[1] + c[2] + c[3]) / 4
end
```

The parser recognises a variable as exogenous if the timing bracket contains one of the keyword/letters (case insensitive): *x, ex, exo, exogenous*. 

Valid declarations of exogenous variables: `ϵ[x], ϵ[Exo], ϵ[exOgenous]`. 

Invalid declarations: `ϵ[xo], ϵ[exogenously], ϵ[main shock x]`

Endogenous and exogenous variables can be in lead or lag, e.g.: 
the following describe a lead of 1 period: `Y[1], Y[+1], Y[+ 1], eps[x+1], eps[Exo + 1]`
and the same goes for lags and periods > 1: `k[-2], c[+12], eps[x-4]

Invalid declarations: `Y[t-1], Y[t], Y[whatever], eps[x+t+1]`

Equations must be within one line and the `=` sign is optional.

The parser recognises all functions in julia including those from [StatsFuns.jl](https://github.com/JuliaStats/StatsFuns.jl). Note that the syntax for distributions is the same as in MATLAB, e.g. `normcdf`. For those familiar with R the following also work: `pnorm`, `dnorm`, `qnorm`, and it also recognises: `norminvcdf` and `norminv`.

Given these rules it is straightforward to write down a model. Once declared using the `@model <name of the model>` macro, the package creates an object containing all necessary information regarding the equations of the model.

## Lead / lags and auxilliary variables

