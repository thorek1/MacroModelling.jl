# Occasionally Binding Constraints

Occasionally binding constraints are a form of nonlinearity frequently used to model effects like the zero lower bound on interest rates, or borrowing constraints. Perturbation methods are not able to capture them as they are local approximations. Nonetheless, there are ways to combine the speed of perturbation solutions and the flexibility of occasionally binding constraints. `MacroModelling.jl` provides a convenient way to write down the constraints and automatically enforces binding constraints with shocks. More specifically, the binding constraints are enforced for each periods unconditional forecast (default forecast horizon of 40 periods) by constraint equation specific anticipated shocks, while minimising the shock size.

## Writing a model with occasionally binding constraints

Let us take the Gali 2015 Chapter 3 model containing a Taylor rule and implement an effective lower bound on interest rates. The Taylor rule in the model: `R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0])` needs to be modified so that `R[0]` never goes below an effective lower bound `R̄`. Wecan do this using the `max` operator: `R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))`

The model definition after the change of the Taylor rule looks like this:

```julia
@model Gali_2015_chapter_3_obc begin
    W_real[0] = C[0] ^ σ * N[0] ^ φ

    [0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

    R[0] = 1 / Q[0]

    Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

    R[0] = Pi[1] * realinterest[0]

    R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

    C[0] = Y[0]

    log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]

    log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]

    nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]

    MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

    1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)

    S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]

    Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

    x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

    x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]

    log_y[0] = log(Y[0])

    log_W_real[0] = log(W_real[0])

    log_N[0] = log(N[0])

    pi_ann[0] = 4 * log(Pi[0])

    i_ann[0] = 4 * log(R[0])

    r_real_ann[0] = 4 * log(realinterest[0])

    M_real[0] = Y[0] / R[0] ^ η

end
```

In the background the system of equations is augmented by a series of anticipated shocks added to the equation containing the constraint (max/min operator). This explains the large number of auxilliary variables and shocks.

Next we define the parameters including the new parameter defining the effective lower bound (which we set to 1, which implements a zero lower bound):

```julia
@parameters Gali_2015_chapter_3_obc begin
    R̄ = 1.0

    σ = 1

    φ = 5

    ϕᵖⁱ = 1.5

    ϕʸ = 0.125

    θ = 0.75

    ρ_ν = 0.5

    ρ_z = 0.5

    ρ_a = 0.9

    β = 0.99

    η = 3.77

    α = 0.25

    ϵ = 9

    τ = 0

    std_a = .01

    std_z = .05

    std_nu = .0025

end
```

Let's check out the non stochastic steady state (NSSS):

```julia
SS(Gali_2015_chapter_3_obc)
```

There are a few things to note here. First, we get the NSSS values of the auxilliary variables related to the occasionally binding constraint. Second, the NSSS value of `R` is 1, and thereby the effective lower bound is binding in the NSSS. While this is a viable NSSS it is not a viable approximation point for perturbation. We can only find a perturbation solution if the effective lower bound is not binding in NSSS. Calling `get_solution` reveals that he cannot find a stable solution:

```julia
get_solution(Gali_2015_chapter_3_obc)
```

In order to get the other viable NSSS we have to restrict the values of R to be larger than the effective lower bound. We can do this by adding a constraint on the variable in the `@parameter` section. Let us redefine the model:

```julia
@model Gali_2015_chapter_3_obc begin
    W_real[0] = C[0] ^ σ * N[0] ^ φ

    Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

    R[0] = 1 / Q[0]

    Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

    R[0] = Pi[1] * realinterest[0]

    R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

    C[0] = Y[0]

    log(A[0]) = ρ_a * log(A[-1]) + std_a * eps_a[x]

    log(Z[0]) = ρ_z * log(Z[-1]) - std_z * eps_z[x]

    nu[0] = ρ_ν * nu[-1] + std_nu * eps_nu[x]

    MC[0] = W_real[0] / (S[0] * Y[0] * (1 - α) / N[0])

    1 = θ * Pi[0] ^ (ϵ - 1) + (1 - θ) * Pi_star[0] ^ (1 - ϵ)

    S[0] = (1 - θ) * Pi_star[0] ^ (( - ϵ) / (1 - α)) + θ * Pi[0] ^ (ϵ / (1 - α)) * S[-1]

    Pi_star[0] ^ (1 + ϵ * α / (1 - α)) = ϵ * x_aux_1[0] / x_aux_2[0] * (1 - τ) / (ϵ - 1)

    x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ + α * ϵ / (1 - α)) * x_aux_1[1]

    x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-σ) + β * θ * Pi[1] ^ (ϵ - 1) * x_aux_2[1]

    log_y[0] = log(Y[0])

    log_W_real[0] = log(W_real[0])

    log_N[0] = log(N[0])

    pi_ann[0] = 4 * log(Pi[0])

    i_ann[0] = 4 * log(R[0])

    r_real_ann[0] = 4 * log(realinterest[0])

    M_real[0] = Y[0] / R[0] ^ η

end

@parameters Gali_2015_chapter_3_obc begin
    R̄ = 1.0

    σ = 1

    φ = 5

    ϕᵖⁱ = 1.5

    ϕʸ = 0.125

    θ = 0.75

    ρ_ν = 0.5

    ρ_z = 0.5

    ρ_a = 0.9

    β = 0.99

    η = 3.77

    α = 0.25

    ϵ = 9

    τ = 0

    std_a = .01

    std_z = .05

    std_nu = .0025

    R > 1.000001
end
```

and check the NSSS once more:

```julia
SS(Gali_2015_chapter_3_obc)
```

Now we get `R > R̄`, so that the constraint is not binding in the NSSS and we can work with a stable first order solution:

```julia
get_solution(Gali_2015_chapter_3_obc)
```

Having defined the system with an occasionally binding constraint we can simply simulate the model by calling:

```julia
import StatsPlots
plot_simulations(Gali_2015_chapter_3_obc)
```

In the background an optimisation problem is set up to find the smallest shocks in magnitude which enforce the equation containing the occasionally binding constraint over the unconditional forecast horizon (default 40 periods) at each period of the simulation. The plots show multiple spells of a binding effective lower bound and many other variables are skewed as a result of the nonlinearity. It can happen that it is not possible to find a combination of shocks which enforce the occasionally binding constraint equation. In this case one solution can be to make the horizon larger over which the algorithm tries to enforce the equation. You can do this by setting the parameter at the beginning of the `@model` section: `@model Gali_2015_chapter_3_obc max_obc_horizon = 60 begin ... end`

We can change the bound by simply changing the parameter:

```julia
plot_simulations(Gali_2015_chapter_3_obc, parameters = :R̄ => 0.99)
```

Now, the effect of the effective lower bound becomes les important as it binds less often.

If you want to ignore the occasionally binding constraint you can simply call:

```julia
plot_simulations(Gali_2015_chapter_3_obc, ignore_obc = true)
```

and you get the simulation based on the first order solution approximated around the NSSS, which is the same as the one for the model without the modified Taylor rule.

We can plot the impulse response functions for the `eps_z` shock, while setting the parameter of the occasionally binding constraint back to 1, as follows:

```julia
plot_irf(Gali_2015_chapter_3_obc, parameters = :R̄ => 1.0)
```

As you can see `R` remains above the effective lower bound in the first period.

Next, let us simulate the model using a series of shocks. E.g. three positive shocks to `eps_z` in periods 5, 10, and 15 in decreasing magnitude:

```julia
shcks = zeros(1,15)
shcks[5] =  3.0
shcks[10] = 2.0
shcks[15] = 1.0

sks = KeyedArray(shcks;  Shocks = [:eps_z], Periods = 1:15)

plot_irf(Gali_2015_chapter_3_obc, shocks = sks, periods = 10)
```

The effective lower bound is binding after all three shocks but the length of the constraint being binding varies with the shock size and is completely endogenous.

Last but not least, we can get the simulated moments of the model (theoretical moments are not available):

```julia
sims = get_irf(Gali_2015_chapter_3_obc, periods = 1000, shocks = :simulate, levels = true)
```

Let's look at the mean and standard deviation of borrowing:

```julia
import Statistics
Statistics.mean(sims(:Y,:,:))
```

and

```julia
Statistics.std(sims(:Y,:,:))
```

Compare this to the theoretical mean of the model without the occasionally binding constraint:

```julia
get_mean(borrowing_constraint)
```

and the theoretical standard deviation:

```julia
get_std(borrowing_constraint)
```

The mean of output is lower in the model with effective lower bound compared to the model without and the standard deviation is higher.





Let us start with a consumption model containing a borrowing constraint (see [@citet cuba2019likelihood] for details). Output is exogenously given, households can only borrow up to a fraction output and decide between saving and consumption. The first order conditions of the model are:

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

In the background the system of equations is augmented by a series of anticipated shocks added to the equation containing the constraint (max/min operator). This explains the large number of auxilliary variables and shocks.

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

We can see that the constraint is no longer binding in the first five periods because `Y` and `B` do not increase by the same amount. They should move by the same amount in the case of a negative shock:

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

Another interesting statistic is model moments. As there are no theoretical moments we have to rely on simulated data:

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
