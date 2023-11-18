# Programmatic model writing

Programmatic model writing is a powerful tool to write complex models using concise code. More specifically, the `@model` and `@parameters` macros allow for the use of indexed variables and for-loops.

## Model block

### for loops for time indices

In practice this means that you no longer need to write this:

```julia
Y_annual[0] = Y[0] + Y[-1] + Y[-2] + Y[-3]
```

but instead you can write this:

```julia
Y_annual[0] = for lag in -3:0 Y[lag] end
```

In the background the package expands the `for` loop and adds up the elements for the different values of `lag`.

In case you don't want the elements to be added up but multiply the items you can do so:

```julia
R_annual[0] = for operator = :*, lag in -3:0 R[lag] end
```

### for loops for variables / parameter specific indices

Another use-case are models with repetitive equations such as multi-sector or multi-country models.

For example, defining the production function for two countries (home country `H` and foreign country `F`) would  look as follows without the use of programmatic features:

```julia
y_H[0] = A_H[0] * k_H[-1]^alpha_H
y_F[0] = A_F[0] * k_F[-1]^alpha_F
```

and this can be written more conveniently using loops:

```julia
for co in [H, F] y{co}[0] = A{co}[0] * k{co}[-1]^alpha{co} end
```

Note that the package internally writes out the for loop and creates two equations; one each for country `H` and `F`. The variables and parameters are indexed using the curly braces `{}`. These can also be used outside loops. When using more than one index it is important to make sure the indices are in the right order.

### Example model block

Putting these these elements together we can write the multi-country model equations of the Backus, Kehoe and Kydland (1992) model like this:

```@setup howto_loops
ENV["GKSwstype"] = "100"
using Random
Random.seed!(30)
```

```@repl howto_loops
using MacroModelling
@model Backus_Kehoe_Kydland_1992 begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1 - theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1 / nu{co})

        K{co}[0] = (1 - delta{co}) * K{co}[-1] + S{co}[0]

        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end

        A{co}[0] = (1 - eta{co}) * A{co}[-1] + N{co}[0]

        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1 - alpha{co}) * eta{co} * A{co}[-1]

        U{co}[0] = (C{co}[0]^mu{co} * L{co}[0]^(1 - mu{co}))^gamma{co}

        psi{co} * mu{co} / C{co}[0] * U{co}[0] = LGM[0]

        psi{co} * (1 - mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1 - theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1 - theta{co}))^(-nu{co}) * Y{co}[0]^(1 + nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1 - delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1 - theta{co})) ^ (-nu{co}) * Y{co}[+4]^(1 + nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1 + sigma{co} * Z{co}[0]^(-nu{co} - 1) * Y{co}[+1]^(1 + nu{co}))

        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1])) / Y{co}[0]
    end

    (LAMBDA{H}[0] - 1) = rho{H}{H} * (LAMBDA{H}[-1] - 1) + rho{H}{F} * (LAMBDA{F}[-1] - 1) + Z_E{H} * E{H}[x]

    (LAMBDA{F}[0] - 1) = rho{F}{F} * (LAMBDA{F}[-1] - 1) + rho{F}{H} * (LAMBDA{H}[-1] - 1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end
end
```

## Parameter block

Having defined parameters and variables with indices in the model block we can also declare parameter values, including by means of calibration equations, in the parameter block.

In the above example we defined the production function fro countries `H` and `F`. Implicitly we have two parameters `alpha` and we can define their value individually by setting

```julia
alpha{H} = 0.3
alpha{F} = 0.3
```

or jointly by writing

```julia
alpha = 0.3
```

By not using the index, the package understands that there are two parameters with this name and different indices and will set both accordingly.

This logic extends to calibration equations. We can write:

```julia
y{H}[ss] = 1 | alpha{H}
y{F}[ss] = 1 | alpha{F}
```

to find the value of `alpha` that corresponds to `y` being equal to 1 in the non-stochastic steady state. Alternatively we can not use indices and the package understands that we refer to both indices:

```julia
y[ss] = 1 | alpha
```

Making use of the indices we could also target a level of `y` for country `H` with `alpha` for country `H` and target ratio of the two `y`s with the `alpha` for country `F`:

```julia
y{H}[ss] = 1 | alpha{H}
y{H}[ss] / y{F}[ss] = y_ratio | alpha{F}
y_ratio =  0.9
```

### Example parameter block

Making use of this and continuing the example of the Backus, Kehoe and Kydland (1992) model we can define the parameters as follows:

```@repl howto_loops
@parameters Backus_Kehoe_Kydland_1992 begin
    F_H_ratio = .9
    K{F}[ss] / K{H}[ss] = F_H_ratio | beta{F}
    K{H}[ss] = 11 | beta{H}

    beta    =    0.99
    mu      =    0.34
    gamma   =    -1.0
    alpha   =    1
    eta     =    0.5
    theta   =    0.36
    nu      =    3
    sigma   =    0.01
    delta   =    0.025
    phi     =    1/4
    psi     =    0.5

    Z_E = 0.00852
    
    rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end
```
