# Calibration / method of moments - Gali (2015)

This tutorial is intended to show the workflow to calibrate a model using the method of moments. The tutorial is based on a standard model of monetary policy and will showcase the use of gradient based optimisers and 2nd and 3rd order pruned solutions.

## Define the model

The first step is always to name the model and write down the equations. For the [gali2015; Chapter 3](@citet) this would go as follows:

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

First, the package is loaded and then the [`@model`](@ref) macro is used to define the model. The first argument after [`@model`](@ref) is the model name and will be the name of the object in the global environment containing all information regarding the model. The second argument to the macro are the equations, which are written down between `begin` and `end`. Equations can contain an equality sign or the expression is assumed to equal 0. Equations cannot span multiple lines (unless the expression is wrapped in brackets) and the timing of endogenous variables are expressed in the square brackets following the variable name (e.g. `[-1]` for the past period). Exogenous variables (shocks) are followed by a keyword in square brackets indicating them being exogenous (in this case `[x]`). Note that names can leverage julia's unicode capabilities (e.g. alpha can be written as α).

## Define the parameters

Next the parameters of the model need to be added. The macro [`@parameters`](@ref) takes care of this:

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

The block defining the parameters above only describes the simple parameter definitions the same way values are assigned (e.g. `α = .25`).

Note that one parameter definition per line is required.

## Linear solution

### Inspect model moments

Given the equations and parameters, everything is available for the package to generate the theoretical model moments. The mean of the linearised model can be retrieved as follows:

```@repl tutorial_3
get_mean(Gali_2015)
```

and the standard deviation like this:

```@repl tutorial_3
get_standard_deviation(Gali_2015)
```

Alternatively, `std` or `get_std` can be used to achieve the same effect.

Another interesting output is the autocorrelation of the model variables:

```@repl tutorial_3
get_autocorrelation(Gali_2015)
```

or the covariance:

```@repl tutorial_3
get_covariance(Gali_2015)
```

### Parameter sensitivities

Before calibrating the model, examine how parameter changes affect model moments. MacroModelling.jl provides partial derivatives of model moments with respect to model parameters. This model is medium-sized, and derivatives are shown automatically. In this example, the sensitivity of the mean of all variables with respect to the production function parameter `σ` can be obtained like this:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = :σ)
```

or for multiple parameters:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :α, :β, :ϕᵖⁱ, :φ])
```

The same can be done for standard deviation or variance, and all parameters:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = get_parameters(Gali_2015))
```

```@repl tutorial_3
get_variance(Gali_2015, parameter_derivatives = get_parameters(Gali_2015))
```

This information can be used to calibrate certain values to targets. For example, assuming higher real wages (`:W_real`), and lower inflation volatility are desired. Since there are too many variables and parameters to be shown here, only a subset of them is printed:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi])
```

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi])
```

Looking at the sensitivity table it can be seen that lowering the production function parameter `:α` will increase real wages, but at the same time it will increase inflation volatility. This effect could be compensated by decreasing the standard deviation of the total factor productivity shock `:std_a`.

### Method of moments

Instead of doing this by hand a target can also be set and an optimiser can find the corresponding parameter values. In order to do that targets need to be defined, and an optimisation problem needs to be set up.

The targets are:

- Mean of `W_real = 0.7`
- Standard deviation of `Pi = 0.01`

For the optimisation problem the L-BFGS algorithm implemented in `Optim.jl` is used. This optimisation algorithm is very efficient and gradient based. Note that all model outputs are differentiable with respect to the parameters using automatic and implicit differentiation.

The package provides functions specialised for the use with gradient based code (e.g. gradient-based optimisers or samplers). For model statistics `get_statistics` can be used to get the mean of real wages and the standard deviation of inflation like this:

```@repl tutorial_3
get_statistics(Gali_2015, Gali_2015.parameter_values, parameters = Gali_2015.parameters, mean = [:W_real], standard_deviation = [:Pi])
```

First the model object is passed on, followed by the parameter values and the parameter names the values correspond to. Then the desired outputs are defined: for the mean real wages are wanted and for the standard deviation inflation is wanted. Outputs for variance, covariance, or autocorrelation can also be obtained the same way as for the mean and standard deviation.

Next, a function measuring how close the model is to the target for given values of `:α` and `:std_a` can be defined:

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics[:mean], model_statistics[:standard_deviation]) - targets)
end
```

Now the function can be tested with the current parameter values. In case the parameter values are not known they can also be looked up like this:

```@repl tutorial_3
get_parameters(Gali_2015, values = true)
```

this allows testing the distance function:

```@repl tutorial_3
distance_to_target([0.25, 0.01])
```

Next pass it on to an optimiser and find the parameters corresponding to the best fit like this:

```@repl tutorial_3
using Optim, LineSearches
sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,1], 
                        [0.25, 0.01], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))
```

The first argument to the optimisation call is the function defined previously, followed by lower and upper bounds, the starting values, and finally the algorithm. For the algorithm `Fminbox` has to be added because bounds are present (optional) and the specific line search method is set to speed up convergence (recommended but optional).

The output shows that the optimisation almost perfectly matches the target and the values of the parameters found by the optimiser are:

```@repl tutorial_3
sol.minimizer
```

slightly lower for both parameters (in line with the previous insights from the sensitivities).

One can combine the method of moments with estimation by simply adding the distance to the target to the posterior loglikelihood.

## Nonlinear solutions

Up to this point the linearised solution of the model was used. The package also provides nonlinear solutions and can calculate the theoretical model moments for pruned second and third order perturbation solutions. This can be of interest because nonlinear solutions capture volatility effects (at second order) and asymmetries (at third order). Furthermore, the moments of the data are often non-gaussian while linear solutions with gaussian noise can only generate gaussian distributions of model variables. Nonetheless, already pruned second order solutions produce non-gaussian skewness and kurtosis with gaussian noise.

From a user perspective little changes other than specifying that the solution algorithm is `:pruned_second_order` or `:pruned_third_order`.

For example the mean for the pruned second order solution can be obtained as follows:

```@repl tutorial_3
get_mean(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_second_order)
```

Note that the mean of real wages is lower, while inflation is higher. The effect of volatility can be seen with the partial derivatives for the shock standard deviations being non-zero. Larger shocks sizes drive down the mean of real wages while they increase inflation.

The mean of the variables does not change if pruned third order perturbation is used by construction but the standard deviation does. Consider the standard deviations for the pruned second order solution first:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_second_order)
```

for both inflation and real wages the volatility is higher and the standard deviation of the total factor productivity shock `std_a` has a much larger impact on the standard deviation of real wages compared to the linear solution.

At third order the results are:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_third_order)
```

standard deviations of inflation is more than two times as high and for real wages it is also substantially higher. Furthermore, standard deviations of shocks matter even more for the volatility of the endogenous variables.

These results make it clear that capturing the nonlinear interactions by using nonlinear solutions has important implications for the model moments and by extension the model dynamics.

### Method of moments for nonlinear solutions

Matching the theoretical moments of the nonlinear model solution to the data is no more complicated for the user than in the linear solution case (see above).

Define the target value and function and let an optimiser find the parameters minimising the distance to the target.

Keeping the targets:

- Mean of `W_real = 0.7`
- Standard deviation of `Pi = 0.01`

the target function needs to specify that a nonlinear solution algorithm is used (e.g. pruned third order):

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, algorithm = :pruned_third_order, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics[:mean], model_statistics[:standard_deviation]) - targets)
end
```

and then use the same code to optimise as in the linear solution case:

```@repl tutorial_3
sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,1], 
                        [0.25, 0.01], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))
```

the calculations take substantially longer and the solution does not get as close to the target as for the linear solution case. The parameter values minimising the distance are:

```@repl tutorial_3
sol.minimizer
```

lower than for the linear solution case and the theoretical moments given these parameter are:

```@repl tutorial_3
get_statistics(Gali_2015, sol.minimizer, algorithm = :pruned_third_order, parameters = [:α, :std_a], mean = [:W_real], standard_deviation = [:Pi])
```

The solution does not match the standard deviation of inflation very well.

Potentially the partial derivatives change a lot for small changes in parameters and even though the partial derivatives for standard deviation of inflation were large wrt `std_a` they might be small for values returned from the optimisation. This can be checked with:

```@repl tutorial_3
get_std(Gali_2015, parameter_derivatives = [:σ, :std_a, :α], variables = [:W_real,:Pi], algorithm = :pruned_third_order, parameters = [:α, :std_a] .=> sol.minimizer)
```

and indeed it seems also the second derivative is large since the first derivative changed significantly.

Another parameter to try is `σ`. It has a positive impact on the mean of real wages and a negative impact on standard deviation of inflation.

The target function needs to be redefined and optimised. Note that the previous call made a permanent change of parameters (as do all calls where parameters are explicitly set) and now `std_a` is set to 2.91e-9 and no longer 0.01.

```@repl tutorial_3
function distance_to_target(parameter_value_inputs)
    model_statistics = get_statistics(Gali_2015, parameter_value_inputs, algorithm = :pruned_third_order, parameters = [:α, :σ], mean = [:W_real], standard_deviation = [:Pi])
    targets = [0.7, 0.01]
    return sum(abs2, vcat(model_statistics[:mean], model_statistics[:standard_deviation]) - targets)
end

sol = Optim.optimize(distance_to_target,
                        [0,0], 
                        [1,3], 
                        [0.25, 1], 
                        Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))))

sol.minimizer
```

Given the new value for `std_a` and optimising over `σ` allows matching the target exactly.
