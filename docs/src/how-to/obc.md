# Occasionally Binding Constraints

Occasionally binding constraints are a form of nonlinearity frequently used to model effects like the zero lower bound on interest rates, or borrowing constraints. Perturbation method are not able to capture them as they are local approximations. Nonetheless, there are ways to combine the speed of perturbation solutions and the flexibility of occasionally binding constraints. `MacroModelling.jl` provides a convenient way to write down the constraints and automatically enforces them with shocks. More specifically, the constraints are enforced for each periods unconditional forecast (default forecast horizon of 40 periods) by constraint equation specific anticipated shocks, while minimising the shock size.

## Writing a model with occasionally binding constraints

Let us start with a consumption model containing a borrowing constraint (see [@citet cuba2019likelihood] for details). The output is exogenously given, households can only borrow up to a fraction output and decide between saving and consumption. The first order conditions of the model are:

```math
C_t + RB_{t-1} = Y_t + B_t
\ln Y_t = \rho \ln Y_{t-1} + \sigma \varepsilon_t
C_t^{-\gamma} = \beta \mathbb{E}_t (C_{t+1}^{-\gamma}) + \lambda_t
\lambda_t (B_t - mY_t) = 0
```

in order to write this model down we need to express the Karush-Kuhn-Tucker condition (last equation) using a max (or min) operator, so that it becomes:

```math
\max(B_t - mY_t, -\lambda_t) = 0
```

We can write this model containing an occasionally binding constraint in a very convenient way:

```julia
@model borrowing_constraint begin
    C[0] = Y[0] + B[0] - R * B[-1]

    log(Y[0]) = ρ * log(Y[-1]) + σ * ε[x]

    C[0]^(-γ) = β * R * C[1]^(-γ) + λ[0]

    max(B[0] - m * Y[0], -λ[0]) = 0
end
```

In the background the system of equations is augmented by a series of news shocks added to the equation containing the constraint (max/min operator). This explains the large number of auxilliary variables and shocks.

Next we define the parameters as usual:

```julia
@parameters borrowing_constraint begin
    R = 1.05
    β = 0.945
    ρ = 0.9
    σ = 0.05
    m = 1
    γ = 1
end
```

## Working with the model

For the non stochastic steady state (NSSS) to exist the constraint has to be binding (`B[0] = m * Y[0]`). This implies a wedge in the Euler equation (`λ > 0`).

We can check this by getting the NSSS:

```julia
SS(borrowing_constraint)
```

A common task is to plot impulse response function for positive and negative shocks. This should allow us to understand the role of the constraint.

First, we need to import the StatsPlots package and then we can plot the positive shock.

```julia
import StatsPlots
plot_irf(borrowing_constraint)
```

![Positive_shock](../assets/borrowing_constraint__ε_pos.png)

We can see that the constraint is no longer binding in the first five periods because `Y` and `B` do not increase by the same amount. They should move by the same amount in the case of a negative shocks:

```julia
import StatsPlots
plot_irf(borrowing_constraint, negative_shock = true)
```

![Negative_shock](../assets/borrowing_constraint__ε_neg.png)

and indeed in this case they move by the same amount. The difference between a positive and negative shock demonstrates the influence of the occasionally binding constraint.

Another common exercise is to plot the impulse response functions from a series of shocks. Let's assume in period 10 there is a positive shocks and in period 30 a negative one. Let's view the results for 50 more periods. We can do this as follows:

```julia
shcks = zeros(1,30)
shcks[10] =  .6
shcks[30] = -.6

sks = KeyedArray(shcks;  Shocks = [:ε], Periods = 1:30)

plot_irf(borrowing_constraint, shocks = sks, periods = 50)
```

![Simulation](../assets/borrowing_constraint__obc.png)

In this case the difference between the shocks and the impact of the constraint become quite obvious. Let's compare this with a version of the model that ignores the occasionally binding constraint. In order to plot the impulse response functions without dynamically enforcing the constraint we can simply write:

```julia
plot_irf(borrowing_constraint, shocks = sks, periods = 50, ignore_obc = true)
```

![Simulation](../assets/borrowing_constraint__no_obc.png)

Another interesting statistic is model moments. As there are no theoretical moments we have to rely on simulated data.

```julia
sims = get_irf(borrowing_constraint, periods = 10000, shocks = :simulate, levels = true)
```

Let's look at the mean and standard deviation of borrowing:

```julia
import Statistics
Statistics.mean(sims(:B,:,:))
```

and

```julia
Statistics.std(sims(:B,:,:))
```

Compare this to the theoretical mean of the model without the occasionally binding constraint:

```julia
get_mean(borrowing_constraint)
```

and the theoretical standard deviation:

```julia
get_std(borrowing_constraint)
```

The mean of borrowing is lower in the model with occasionally binding constraints compared to the model without and the standard deviation is higher.

## Bibliography

```@bibliography
```
