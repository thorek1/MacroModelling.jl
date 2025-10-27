# Model Estimates Visualization

The `plot_model_estimates` function visualizes various estimation outputs including filtered and smoothed state estimates, shock decompositions, and historical decompositions of the data.

## Basic Usage

```julia
using MacroModelling

@model Gali_2015_chapter_3_nonlinear begin
    # Households
    C[0] = (1 - β) * ((W_real[0] * N[0] + Div[0]) - T[0])
    
    W_real[0] = χ * N[0]^φ * C[0]^σ_c
    
    # Firms
    MC[0] = W_real[0] / Z[0]
    
    # Monetary policy
    R[0] = R[-1]^ρ_R * (steady_state(R) * (Pi[0] / steady_state(Pi))^ϕ_π * (Y[0] / steady_state(Y))^ϕ_y)^(1 - ρ_R) * exp(nu[0])
    
    # Market clearing
    Y[0] = C[0]
    
    # Production
    Y[0] = A[0] * N[0]
    
    # Price setting
    1 = θ * Pi[0]^(ϵ - 1) + (1 - θ) * (Pi_star[0])^(1 - ϵ)
    
    # Exogenous processes
    log(A[0]) = ρ_a * log(A[-1]) + eps_a[x]
    log(Z[0]) = ρ_z * log(Z[-1]) + eps_z[x]
end

@parameters Gali_2015_chapter_3_nonlinear begin
    σ_c = 1
    φ = 5
    ϵ = 9
    θ = 0.75
    β = 0.99
    ρ_R = 0.8
    ϕ_π = 1.5
    ϕ_y = 0.125
    ρ_a = 0.9
    ρ_z = 0.9
    χ = 1
end

# Simulate some data for estimation
data = simulate(Gali_2015_chapter_3_nonlinear)

# Plot filtered estimates
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data)
```

This creates plots showing the estimated states, comparing them with observed data where available.

## Arguments

### Data Argument

The `data` argument (required, type: `Matrix` or `KeyedArray`) provides the observed data used for filtering and estimation. The data should match the observables specified in the model.

```julia
# Using a matrix of observations
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data_matrix)

# Using KeyedArray with variable names
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data_keyed)
```

### Estimation Type

The `type` argument (default: `:filter`, type: `Symbol`) specifies what to plot. Options include:
- `:filter` - Filtered state estimates (forward pass)
- `:smoother` - Smoothed state estimates (forward-backward pass)
- `:shock_decomposition` - Contribution of each shock to variables over time
- `:historical_decomposition` - Historical contribution of shocks to deviations from steady state

```julia
# Plot smoothed estimates
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data, type = :smoother)

# Show shock decomposition
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data, 
    type = :shock_decomposition)

# Display historical decomposition
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data, 
    type = :historical_decomposition)
```

### Variables Selection

The `variables` argument (default: all variables, type: `Symbol` or `Vector{Symbol}`) specifies which variables to plot.

```julia
# Plot specific variables
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    variables = [:Y, :Pi, :R])
```

### Shocks Selection (for decomposition plots)

The `shocks` argument (default: all shocks, type: `Symbol` or `Vector{Symbol}`) determines which shocks to include in shock decomposition plots.

```julia
# Show only selected shock contributions
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    type = :shock_decomposition,
    shocks = [:eps_a, :nu])
```

### Solution Algorithm

The `algorithm` argument (default: `:first_order`, type: `Symbol`) specifies the solution method for the Kalman filter.

```julia
# Use second-order approximation for filtering
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    algorithm = :second_order)
```

### Alternative Parameters

The `parameters` argument (default: model parameters, type: `Vector{Real}`) allows comparing estimates under different parameter values.

```julia
# Compare filtered estimates with different parameters
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    parameters = alternative_params)
```

### Confidence Intervals

The `show_ci` argument (default: `true`, type: `Bool`) determines whether to display confidence intervals around the estimates.

The `ci_level` argument (default: `0.95`, type: `Real`) specifies the confidence level for the intervals.

```julia
# Show 90% confidence intervals
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    show_ci = true,
    ci_level = 0.90)

# Hide confidence intervals
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    show_ci = false)
```

### Include Observables

The `show_data` argument (default: `true`, type: `Bool`) determines whether to overlay observed data on filtered/smoothed estimates.

```julia
# Hide observed data points
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    show_data = false)
```

### Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) accepts standard Plots.jl attributes for customization.

```julia
# Customize appearance
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    plot_attributes = Dict(
        :size => (1000, 800),
        :linewidth => 2,
        :legend => :bottomright,
        :title => "Filtered State Estimates"
    ))
```

### Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps names to more readable labels.

```julia
# Use descriptive labels
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    type = :shock_decomposition,
    rename_dictionary = Dict(
        :Y => "Output",
        :Pi => "Inflation Rate",
        :eps_a => "Technology Shock",
        :nu => "Monetary Policy Shock"
    ))
```

### Plot Labels

The `label` argument (default: `""`, type: `String`) provides a label for the plot series, useful when overlaying multiple estimates.

```julia
# Compare estimates from different specifications
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    parameters = params1,
    label = "Baseline")
plot_model_estimates!(Gali_2015_chapter_3_nonlinear, data,
    parameters = params2,
    label = "Alternative")
```

### Saving Plots

The `save_plots` argument (default: `false`, type: `Bool`) determines whether to save the plots.

Related saving arguments:
- `save_plots_name` (default: `"estimates"`, type: `Union{String, Symbol}`)
- `save_plots_format` (default: `:pdf`, type: `Symbol`)
- `save_plots_path` (default: `"."`, type: `String`)

```julia
# Save estimation plots
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    type = :smoother,
    save_plots = true,
    save_plots_name = "smoothed_estimates",
    save_plots_format = :png,
    save_plots_path = "estimation_output")
```

### Prior and Posterior Comparison

The `show_prior` argument (default: `false`, type: `Bool`) determines whether to show prior distributions alongside posterior estimates.

```julia
# Compare prior and posterior
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data,
    type = :parameter_estimates,
    show_prior = true)
```

## Estimation Types Explained

### Filtering
Shows the real-time estimate of states based on information available up to each point in time. Useful for understanding what was known at each period.

### Smoothing
Provides the best estimate of states using all available information (full sample). Generally more precise than filtered estimates.

### Shock Decomposition
Shows how much of each variable's movement is attributable to specific shocks over time. Helps identify the key drivers of economic fluctuations.

### Historical Decomposition
Similar to shock decomposition but shows contributions to deviations from steady state, making it easier to interpret the economic significance of shocks.

## Comparison with plot_model_estimates!

The `plot_model_estimates!` function overlays new estimates on existing plots:

```julia
# Compare estimates from different sample periods
plot_model_estimates(Gali_2015_chapter_3_nonlinear, data_full,
    label = "Full Sample")
plot_model_estimates!(Gali_2015_chapter_3_nonlinear, data_subsample,
    label = "Subsample")
```

## Diagnostic Checking

When evaluating model fit, consider:
- **Filtered vs Smoothed**: Large differences suggest information from future observations matters
- **Confidence Intervals**: Width indicates estimation uncertainty
- **Shock Decomposition**: Which shocks explain most variation
- **Observed vs Estimated**: How well the model tracks actual data
