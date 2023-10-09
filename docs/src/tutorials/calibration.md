# Calibration / method of moments - Gali (2015)

This tutorial is intended to show the workflow to calibrate a model using the method of moments. The tutorial is based on a standard model of monetary policy and will showcase the the use of gradient based optimisers and 2nd and 3rd order pruned solutions.

## Define the model

The first step is always to name the model and write down the equations. For the Gali (2015) model (chapter 3 of the book) this would go as follows:

```@setup tutorial_2
ENV["GKSwstype"] = "100"
using Random
Random.seed!(30)
```

```@repl tutorial_3
using MacroModelling

@model Gali_2015 begin
    W_real[0] = C[0] ^ σ * N[0] ^ φ

    Q[0] = β * (C[1] / C[0]) ^ (-σ) * Z[1] / Z[0] / Pi[1]

    R[0] = 1 / Q[0]

    Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)

    R[0] = Pi[1] * realinterest[0]

    R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0])

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

First, we load the package and then use the [`@model`](@ref) macro to define our model. The first argument after [`@model`](@ref) is the model name and will be the name of the object in the global environment containing all information regarding the model. The second argument to the macro are the equations, which we write down between `begin` and `end`. Equations can contain an equality sign or the expression is assumed to equal 0. Equations cannot span multiple lines (unless you wrap the expression in brackets) and the timing of endogenous variables are expressed in the squared brackets following the variable name (e.g. `[-1]` for the past period). Exogenous variables (shocks) are followed by a keyword in squared brackets indicating them being exogenous (in this case `[x]`). Note that names can leverage julia's unicode capabilities (e.g. alpha can be written as α).

## Define the parameters

Next we need to add the parameters of the model. The macro [`@parameters`](@ref) takes care of this:

```@repl tutorial_3
@parameters Gali_2015 begin
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

The block defining the parameters above only describes the simple parameter definitions the same way you assign values (e.g. `α = .25`).

Note that we have to write one parameter definition per line.

## Linear solution

### Inspect model moments

Given the equations and parameters, we have everything to we need for the package to generate the theoretical model moments. You can retrieve the mean of the linearised model as follows:

```@repl tutorial_3
get_mean(Gali_2015)
```

and the standard deviation like this:

```@repl tutorial_3
get_standard_deviation(Gali_2015)
```

You could also simply use: `std` or `get_std` to the same effect.

Another interesting output is the autocorrelation of the model variables:

```@repl tutorial_3
get_autocorrelation(Gali_2015)
```

or the covariance:

```@repl tutorial_3
get_covariance(Gali_2015)
```

### Parameter sensitivities

Before embarking on calibrating the model it is useful to get familiar with the impact of parameter changes on model moments. `MacroModelling.jl` provides the partial derivatives of the model moments with respect to the model parameters. The model we are working with is of a medium size and by default derivatives are automatically shown as long as the calculation does not take too long (too many derivatives need to be taken). In this case they are not shown but it is possible to show them by explicitly defining the parameter for which to take the partial derivatives for:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = :σ)
```

or for multiple parameters:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :α, :β, :ϕᵖⁱ, :φ])
```

We can do the same for standard deviation or variance, and all parameters:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = get_parameters(Gali_2015))
```

```@repl tutorial_3
get_variance(Gali_2015, parameter_derivatives = get_parameters(Gali_2015))
```

You can use this information to calibrate certain values to your targets. For example, let's say we want to have higher real wages (`:W_real`), and lower inflation volatility. Since there are too many variables and parameters for them to be shown here, let's print only a subset of them:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi])
```

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi])
```

Looking at the sensitivity table we see that lowering the production function parameter `:α` will increase real wages, but at the same time it will increase inflation volatility. We could compensate that effect by decreasing the standard deviation of the total factor productivity shock `:std_a`.

### Method of moments

Instead of doing this by hand we can also set a target and have an optimiser find the corresponding parameter values. In order to do that we need to define targets, and set up an optimisation problem.

Our targets are:

- Mean of `W_real = 0.7`
- Standard deviation of `Pi = 0.01`

For the optimisation problem we use the L-BFGS algorithm implemented in `Optim.jl`. This optimisation algorithm is very efficient and gradient based. Note that all model outputs are differentiable with respect to the parameters using automatic and implicit differentiation.

The package provides functions specialised for the use with gradient based code (e.g. gradient-based optimisers or samplers). For model statistics we can use `get_statistics` to get the mean of real wages and the standard deviation of inflation like this:

```@repl tutorial_3
get_statistics(Gali_2015, Gali_2015.parameter_values, parameters = Gali_2015.parameters, mean = [:W_real], standard_deviation = [:Pi])
```

First we pass on the model object, followed by the parameter values and the parameter names the values correspond to. Then we define the outputs we want: for the mean we want real wages and for the standard deviation we want inflation. We can also get outputs for variance, covariance, or autocorrelation the same way as for the mean and standard deviation.

Next, let's define a function measuring how close we are to our target for given values of `:α` and `:std_a`:

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics...) - targets)
end
```

Now let's test the function with the current parameter values. In case we forgot the parameter values we can also look them up like this:

```@repl tutorial_3
get_parameters(Gali_2015, values = true)
```

with this we can test the distance function:

```@repl tutorial_3
distance_to_target([0.25, 0.01])
```

Next we can pass it on to an optimiser and find the parameters corresponding to the best fit like this:

```@repl tutorial_3
using Optim, LineSearches
sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,1], 
                        [0.25, 0.01], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))
```

The first argument to the optimisation call is the function we defined previously, followed by lower and upper bounds, the starting values, and finally the algorithm. For the algorithm we have to add `Fminbox` because we have bounds (optional) and we set the specific line search method to speed up convergence (recommended but optional).

The output shows that we could almost perfectly match the target and the values of the parameters found by the optimiser are:

```@repl tutorial_3
sol.minimizer
```

slightly lower for both parameters (in line with what we understood from the sensitivities).

You can combine the method of moments with estimation by simply adding the distance to the target to the posterior loglikelihood.

## Nonlinear solutions

So far we used the linearised solution of the model. The package also provides nonlinear solutions and can calculate the theoretical model moments for pruned second and third order perturbation solutions. This can be of interest because nonlinear solutions capture volatility effects (at second order) and asymmetries (at third order). Furthermore, the moments of the data are often non-gaussian while linear solutions with gaussian noise can only generate gaussian distributions of model variables. Nonetheless, already pruned second order solutions produce non-gaussian skewness and kurtosis with gaussian noise.

From a user perspective little changes other than specifying that the solution algorithm is `:pruned_second_order` or `:pruned_third_order`.

For example we can get the mean for the pruned second order solution:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_second_order)
```

Note that the mean of real wages is lower, while inflation is higher. We can see the effect of volatility with the partial derivatives for the shock standard deviations being non-zero. Larger shocks sizes drive down the mean of real wages while they increase inflation.

The mean of the variables does not change if we use pruned third order perturbation by construction but the standard deviation does. Let's look at the standard deviations for the pruned second order solution first:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_second_order)
```

for both inflation and real wages the volatility is higher and the standard deviation of the total factor productivity shock `std_a` has a much larger impact on the standard deviation of real wages compared to the linear solution.

At third order we get the following results:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_third_order)
```

standard deviations of inflation is more than two times as high and for real wages it is also substantially higher. Furthermore, standard deviations of shocks matter even more for the volatility of the endogenous variables.

These results make it clear that capturing the nonlinear interactions by using nonlinear solutions has important implications for the model moments and by extension the model dynamics.

### Method of moments for nonlinear solutions

Matching the theoretical moments of the nonlinear model solution to the data is no more complicated for the user than in the linear solution case (see above).

We need to define the target value and function and let an optimiser find the parameters minimising the distance to the target.

Keeping the targets:

- Mean of `W_real = 0.7`
- Standard deviation of `Pi = 0.01`

we need to define the target function and specify that we use a nonlinear solution algorithm (e.g. pruned third order):

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, algorithm = :pruned_third_order, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics...) - targets)
end
```

and then we can use the same code to optimise as in the linear solution case:

```@repl tutorial_3
sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,1], 
                        [0.25, 0.01], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))
```

the calculations take substantially longer and we don't get as close to our target as for the linear solution case. The parameter values minimising the distance are:

```@repl tutorial_3
sol.minimizer
```

lower than for the linear solution case and the theoretical moments given these parameter are:

```@repl tutorial_3
get_statistics(Gali_2015, sol.minimizer, algorithm = :pruned_third_order, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
```

The solution does not match the standard deviation of inflation very well.

Potentially the partial derivatives change a lot for small changes in parameters and even though the partial derivatives for standard deviation of inflation were large wrt `std_a` they might be small for value returned from the optimisation. We can check this with:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_third_order, parameters = [:α, :std_a] .=> sol.minimizer)
```

and indeed it seems also the second derivative is large since the first derivative changed significantly.

Another parameter we can try is `σ`. It has a positive impact on the mean of real wages and a negative impact on standard deviation of inflation.

We need to redefine our target function and optimise it. Note that the previous call made a permanent change of parameters (as do all calls where parameters are explicitly set) and now `std_a` is set to 2.91e-9 and no longer 0.01.

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, algorithm = :pruned_third_order, parameters = [:α, :σ], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics...) - targets)
end

sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,3], 
                        [0.25, 1], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))

sol.minimizer
```

Given the new value for `std_a` and optimising over `σ` allows us to match the target exactly.
