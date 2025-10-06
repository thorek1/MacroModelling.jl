# Plotting

MacroModelling.jl integrates a comprehensive plotting toolkit based on [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl). The plotting API is exported together with the modelling macros, so once you define a model you can immediately visualise impulse responses, simulations, conditional forecasts, model estimates, variance decompositions, and policy functions. All plotting functions live in the `StatsPlotsExt` extension, which is loaded automatically when StatsPlots is imported or used.

## Setup

Load the packages once per session:

```julia
using MacroModelling
import StatsPlots
```

Load a model for the examples below:

```julia
@model Gali_2015_chapter_3_nonlinear begin
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


@parameters Gali_2015_chapter_3_nonlinear begin
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

## Impulse response functions (IRF)

A call to `plot_irf` computes IRFs for **every exogenous shock** and **every endogenous variable**, using the model's default solution method (first-order perturbation) and a **one-standard-deviation positive** shock.

```julia
plot_irf(Gali_2015_chapter_3_nonlinear)
```

![RBC IRF](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The plot shows every endogenous variable affected by each exogenous shock and annotates the title with the model name, shock identifier, sign of the impulse (positive by default), and the page indicator (e.g. `(1/3)`). Each subplot overlays the steady state as a horizontal reference line (non‑stochastic for first‑order solutions, stochastic otherwise) and, when the variable is strictly positive, adds a secondary axis with percentage deviations.

### Algorithm 

You can plot IRFs for different solution algorithms. The `algorithm` keyword argument accepts: `:first_order` (default), `:second_order`, `:pruned_second_order`, `:third_order`, or `:pruned_third_order`. Here we use a second-order perturbation solution:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order)
```

![RBC IRF second order](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_second_order.png)

The most notable difference is that at second order we observe dynamics for S, which is constant at first order (under certainty equivalence). Furthermore, the steady state levels changed due to the stochastic steady state incorporating precautionary behaviour (see horizontal lines).

We can compare the two solution methods side by side by plotting them on the same graph using the `plot_irf!` function:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :second_order)
```

![RBC IRF comparison](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_first_and_second_order.png)

In the plots we now see both solution methods overlaid. The first-order solution is shown in blue, the second-order solution in orange, as indicated in the legend below the plot. Note that the steady state levels can be different for the two solution methods. For variables where the relevant steady state (non-stochastic steady state for first order and stochastic steady state for higher order) is the same (e.g. A) we see the level on the left axis and percentage deviations on the right axis. For variables where the steady state differs between the two solution methods (e.g. C) we only see absolute level deviations (abs. Δ) on the left axis. Furthermore, the relevant steady state level is mentioned in a table below the plot for reference.

We can add more solution methods to the same plot:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, algorithm = :pruned_third_order)
```

![RBC IRF multiple orders](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_multiple_orders.png)

Note that the pruned third-order solution includes the effect of time varying risk and flips the sign for the reaction of MC and N. The additional solution is added to the plot as another colored line and another entry in the legend and a new entry in the table below highlighting the relevant steady states.

### Initial state 

The initial state defines the starting point for the IRF. The `initial_state` keyword argument accepts a vector of initial values for all model variables. For pruned solution algorithms, it can also accept multiple state vectors (`Vector{Vector{Float64}}`). In this case the initial state must be given in deviations from the non-stochastic steady state. In all other cases the initial state must be given in levels.

The initial state needs to contain all variables of the model as well as any leads or lags if present. One way to get the correct ordering and number of variables is to call `get_irf(𝓂, shocks = :none, variables = :all, periods = 1)` which returns a `KeyedArray` with all variables in the correct order.

```julia
init_state = get_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, variables = :all, periods = 1, levels = true)
```

Only state variables will have an impact on the IRF. You can check which variables are state variables using:

```julia
get_state_variables(Gali_2015_chapter_3_nonlinear)
```

Now let's modify the initial state and set nu to 0.1:

```julia
init_state(:nu,:,:) .= 0.1
```

Now we can input the modified initial state into the `plot_irf` function as a vector:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, initial_state = vec(init_state))
```

![RBC IRF with initial state](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_init_state.png)

You can see the difference in the IRF compared to the IRF starting from the non-stochastic steady state. By setting nu to a higher level we essentially mix the effect of a shock to nu with a shock to A. Since here we are working with the linear solution we can disentangle the two effects by stacking the two components:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :none, initial_state = vec(init_state))
plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_type = :stack)
```

![RBC IRF stacked](assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1.png)

Note how the two components are shown with a label attached to it that is explained in the table below. The blue line refers to the first input: without a shock and a non-zero initial state and the red line corresponds to the second input which starts from the relevant steady state and shocks eps_a. Both components add up to the solid line.

### Shocks

The `shocks` keyword argument specifies which shocks to plot. Inputs can be:
- A shock name as a `Symbol` or `String` (e.g. `:eps_a` or `"eps_a"`)
- A `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`
- `:all_excluding_obc` (default) - all shocks except those related to occasionally binding constraints
- `:all` - all shocks including OBC-related ones
- `:simulate` - triggers random draws of all shocks
- `:none` - can be used with `initial_state` for deterministic simulations
- A `Matrix{Float64}` or `KeyedArray{Float64}` representing a series of shocks

We can call individual shocks by name:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a)
```

![RBC IRF single shock](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

Or multiple shocks at once:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = [:eps_a, :eps_z])
```

![RBC IRF eps_a](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__3.png)
![RBC IRF eps_z](assets/irf__Gali_2015_chapter_3_nonlinear__eps_z__3.png)

The `:simulate` option triggers random draws of all shocks:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :simulate)
```

![RBC IRF simulation](assets/irf__Gali_2015_chapter_3_nonlinear__simulation__1.png)

We can also compare shocks by overlaying them:

```julia
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1])
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s)
end
```

![RBC IRF multiple shocks](assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1_linear.png)

A series of shocks can be passed as a `KeyedArray`:

```julia
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
n_periods = 3
shock_keyedarray = KeyedArray(zeros(length(shocks), n_periods), Shocks = shocks, Periods = 1:n_periods)
shock_keyedarray("eps_a",[1]) .= 1
shock_keyedarray("eps_z",[2]) .= -1/2
shock_keyedarray("eps_nu",[3]) .= 1/3

plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shock_keyedarray)
```

![RBC IRF shock matrix](assets/irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2.png)

### Periods

The `periods` keyword argument (default: 40) sets the number of periods to plot. In case a matrix of shocks was provided, `periods` defines how many periods after the series of shocks the output continues.

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, periods = 10, shocks = :eps_a)
```

![RBC IRF 10 periods](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_10_periods.png)

### Shock size

The `shock_size` keyword argument (default: 1.0) sets the size of shocks in standard deviations. A negative value will flip the sign of the shock:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, shock_size = -2)
```

![RBC IRF shock size](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_shock_size.png)

### Negative shock

The `negative_shock` keyword argument (default: false) calculates IRFs for a negative shock:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, negative_shock = true)
```

![RBC IRF negative shock](assets/irf__Gali_2015_chapter_3_nonlinear__eps_z__1_neg_shock.png)

### Variables

The `variables` keyword argument (default: `:all_excluding_obc`) specifies which variables to plot. Inputs can be:
- A variable name as a `Symbol` or `String` (e.g. `:Y` or `"Y"`)
- A `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`
- `:all_excluding_auxiliary_and_obc` - all variables except auxiliary and OBC-related ones
- `:all_excluding_obc` - all variables except OBC-related ones
- `:all` - all variables including auxiliary and OBC-related ones

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi])
```

![RBC IRF variable selection](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_var_select.png)

### Parameters

The `parameters` keyword argument allows you to change parameter values. Acceptable inputs are:
- A `Vector` of parameter values (in alphabetical order)
- A `Vector` or `Tuple` of `Pair`s of parameter `Symbol`/`String` and value

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a)
```

![RBC IRF beta 0.95](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_beta_0_95.png)

We can compare different parameter values:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.99, shocks = :eps_a)
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = :β => 0.95, shocks = :eps_a)
```

![RBC IRF compare beta](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_compare_beta.png)

Multiple parameters can be changed at once:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.97, :τ => 0.5), shocks = :eps_a)
```

![RBC IRF beta and tau](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_beta_tau.png)

### Ignore OBC

The `ignore_obc` keyword argument (default: false) allows you to ignore occasionally binding constraints:

```julia
@model Gali_2015_chapter_3_obc begin
    # ... (model with OBC)
    R[0] = max(R̄ , 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))
    # ...
end

plot_irf(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3, ignore_obc = true)
```

![RBC IRF ignore OBC](assets/irf__Gali_2015_chapter_3_obc__eps_z__1_ignore_obc.png)

### Generalised IRF

The `generalised_irf` keyword argument (default: false) calculates generalised IRFs (GIRFs) instead of standard IRFs. GIRFs are useful for models with non-linearities and/or state-dependent dynamics:

```julia
plot_irf(Gali_2015_chapter_3_obc, generalised_irf = true, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
plot_irf!(Gali_2015_chapter_3_obc, shocks = :eps_z, variables = [:Y,:R,:Pi,:C], shock_size = 3)
```

![RBC IRF GIRF](assets/irf__Gali_2015_chapter_3_obc__eps_z__1_girf.png)

You can adjust the number of draws and warmup iterations using `generalised_irf_draws` (default: 100) and `generalised_irf_warmup_iterations` (default: 50):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, generalised_irf = true, shocks = :eps_a, algorithm = :pruned_second_order)
plot_irf!(Gali_2015_chapter_3_nonlinear, generalised_irf = true, generalised_irf_draws = 1000, shocks = :eps_a, algorithm = :pruned_second_order)
```

![RBC IRF GIRF draws](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_girf_1000_draws.png)

### Label

The `label` keyword argument provides custom labels for plots when using `plot_irf!`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.99, :τ => 0.0), shocks = :eps_a, label = "Std. params")
plot_irf!(Gali_2015_chapter_3_nonlinear, parameters = (:β => 0.95, :τ => 0.5), shocks = :eps_a, label = "Alt. params")
```

![RBC IRF custom labels](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__2_custom_labels.png)

### Plot attributes

The `plot_attributes` keyword argument (default: `Dict()`) allows you to pass plot attributes to customize the appearance:

```julia
ec_color_palette = ["#FFD724", "#353B73", "#2F9AFB", "#B8AAA2", "#E75118", "#6DC7A9", "#F09874", "#907800"]

shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = shocks[1])
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear, shocks = s, plot_attributes = Dict(:palette => ec_color_palette), plot_type = :stack)
end
```

![RBC IRF custom colors](assets/irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__2_ec_colors.png)

You can also change other attributes such as font family:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, shocks = :eps_a, plot_attributes = Dict(:fontfamily => "computer modern"))
```

![RBC IRF custom font](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_cm_font.png)

### Plots per page

The `plots_per_page` keyword argument (default: 9) controls how many subplots to show per page:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A], shocks = :eps_a, plots_per_page = 2)
```

![RBC IRF plots per page](assets/irf__Gali_2015_chapter_3_nonlinear__eps_a__1_9_vars_2_per_page.png)

### Show and save plots

The `show_plots` keyword argument (default: true) controls whether plots are displayed. The `save_plots` keyword argument (default: false) enables saving plots to disk. Use `save_plots_path` (default: current working directory), `save_plots_format` (default: `:pdf`), and `save_plots_name` to control output:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear, 
         save_plots = true, 
         save_plots_format = :png, 
         save_plots_path = "./plots", 
         save_plots_name = :impulse_response)
```

Acceptable formats are those supported by the Plots.jl package (see [GR output formats](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats)).

### Advanced options

Additional keyword arguments for advanced users:

- `verbose` (default: false) - print information about solution
- `tol` (default: `Tolerances()`) - set custom tolerances for numerical solvers
- `quadratic_matrix_equation_algorithm` (default: `:schur`) - algorithm for quadratic matrix equation (`:schur` or `:doubling`)
- `sylvester_algorithm` - algorithm for Sylvester equations (`:doubling`, `:bartels_stewart`, `:bicgstab`, `:dqgmres`, `:gmres`)
- `lyapunov_algorithm` (default: `:doubling`) - algorithm for Lyapunov equation
- `max_elements_per_legend_row` (default: 4) - number of columns in legend
- `extra_legend_space` (default: 0.0) - space between plots and legend

## Other plotting functions

### Simulations

`plot_simulations` and `plot_simulation` are wrappers for `plot_irf` with `shocks = :simulate`:

```julia
plot_simulations(Gali_2015_chapter_3_nonlinear, periods = 100)
```

### Solution

`plot_solution` visualizes the policy functions for a given state variable:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, variables = [:Y, :C])
```

### Conditional forecasts

`plot_conditional_forecast` shows forecasts conditional on observed data:

```julia
plot_conditional_forecast(model, data, conditions)
```

### Variance decompositions

`plot_conditional_variance_decomposition` and `plot_forecast_error_variance_decomposition` (alias: `plot_fevd`) show variance decompositions:

```julia
plot_fevd(model, periods = 20)
```

### Model estimates

`plot_model_estimates` shows estimated variables and shocks given data:

```julia
plot_model_estimates(model, data)
```

`plot_shock_decomposition` is a wrapper for `plot_model_estimates` with `shock_decomposition = true`.

## Backend selection

You can switch between plotting backends:

```julia
gr_backend()        # Use GR backend (default)
plotlyjs_backend()  # Use PlotlyJS backend for interactive plots
```
