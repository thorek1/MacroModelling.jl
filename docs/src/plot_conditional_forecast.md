# Conditional Forecasting

Conditional forecasting allows generating model projections conditional on specified future paths for certain variables or shocks. The `plot_conditional_forecast` function visualizes these conditional forecasts, showing how the model evolves given constraints on particular variables.

## Basic Usage

```julia
using MacroModelling

@model Gali_2015_chapter_3_nonlinear begin
    # ... (model equations)
end

@parameters Gali_2015_chapter_3_nonlinear begin
    # ... (parameter values)
end

plot_conditional_forecast(Gali_2015_chapter_3_nonlinear, 
                         conditional_variance = [:y, :c],
                         periods = 20)
```

This creates conditional forecast plots showing the projected paths of all endogenous variables conditional on specified constraints.

## Arguments

### Conditional Variables

The `conditional_variance` argument (default: empty vector, type: `Vector{Symbol}`) specifies which variables have constrained future paths. When empty, creates an unconditional forecast.

```julia
# Conditional forecast with output and consumption constrained
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         periods = 20)
```

### Forecast Horizon

The `periods` argument (default: `40`, type: `Int`) determines the number of periods to forecast forward.

```julia
# 10-period ahead forecast
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         periods = 10)
```

### Variable Selection

The `variables` argument (default: all variables, type: `Union{Symbol, Vector{Symbol}, String, Vector{String}}`) selects which variables to display in the forecast plots.

```julia
# Show only output, consumption, and inflation
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         variables = [:y, :c, :Pi],
                         periods = 20)
```

### Shock Selection

The `shocks` argument (default: all shocks, type: `Union{Symbol, Vector{Symbol}, String, Vector{String}}`) selects which shocks to include in the forecast.

```julia
# Forecast conditioning only on productivity shocks
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         shocks = [:eps_a],
                         periods = 20)
```

### Alternative Parameters

The `parameters` argument (default: calibrated values, type: `Union{Vector{<:Real}, ParameterType}`) allows forecasting with alternative parameter values.

```julia
# Conditional forecast with different discount factor
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         parameters = :select_parameters,
                         periods = 20)
```

### Historical Data

The `data` argument (default: none, type: `KeyedArray{Float64}`) provides historical data for the forecast starting point.

```julia
# Conditional forecast starting from observed data
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         data = historical_data,
                         conditional_variance = [:y],
                         periods = 20)
```

### Initial Conditions

The `initial_state` argument (default: steady state, type: `Union{Vector{Float64}, Nothing}`) sets the starting point for the forecast.

```julia
# Forecast from specific initial conditions
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         initial_state = custom_state,
                         conditional_variance = [:y],
                         periods = 20)
```

### Confidence Intervals

The `confidence_bands` argument (default: `[0.9]`, type: `Vector{Float64}`) specifies confidence interval levels to display.

```julia
# Show 68% and 95% confidence bands
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         confidence_bands = [0.68, 0.95],
                         periods = 20)
```

### Algorithm Selection

The `algorithm` argument (default: `:first_order`, type: `Symbol`) chooses the solution algorithm for the forecast.

The `algorithm` argument accepts `:first_order`, `:second_order`, or `:third_order` for perturbation-based solutions, or `:linear_time_iteration` and `:riccati` for linear solutions. Higher-order approximations capture nonlinear dynamics but increase computational cost.

```julia
# Second-order conditional forecast
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         algorithm = :second_order,
                         periods = 20)
```

### Plot Display and Saving

The `show_plots` argument (default: `true`, type: `Bool`) controls whether plots are displayed immediately.

The `save_plots` argument (default: `false`, type: `Bool`) enables automatic saving of generated plots.

The `save_plots_name` argument (default: `"conditional_forecast"`, type: `Union{String, Symbol}`) sets the prefix for saved plot filenames. The full filename follows the pattern: `{name}__ModelName__variable__page.{format}`.

The `save_plots_format` argument (default: `:pdf`, type: `Symbol`) specifies the output format for saved plots (`:pdf`, `:png`, `:svg`, etc.).

The `save_plots_path` argument (default: `"."`, type: `String`) sets the directory where plots are saved.

```julia
# Save conditional forecasts as PNG files
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         save_plots = true,
                         save_plots_format = :png,
                         save_plots_name = "forecast_2024",
                         save_plots_path = "./forecasts",
                         periods = 20)
```

### Layout Options

The `plots_per_page` argument (default: `6`, type: `Int`) controls how many variable forecasts appear on each plot page.

```julia
# Show 9 forecasts per page
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         plots_per_page = 9,
                         periods = 20)
```

### Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps internal variable/shock names to display names.

```julia
# Rename variables for clearer presentation
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         rename_dictionary = Dict(:y => "Output", 
                                                 :c => "Consumption",
                                                 :Pi => "Inflation"),
                         periods = 20)
```

### Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) allows customization of plot appearance using Plots.jl attributes.

```julia
# Customize plot appearance
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         plot_attributes = Dict(:linewidth => 2,
                                               :gridstyle => :dash,
                                               :framestyle => :box),
                         periods = 20)
```

### Legend Options

The `max_elements_per_legend_row` argument (default: `3`, type: `Int`) controls legend layout by setting maximum items per row.

The `extra_legend_space` argument (default: `0.1`, type: `Float64`) adds vertical spacing for multi-row legends.

```julia
# Adjust legend layout
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y, :c],
                         max_elements_per_legend_row = 4,
                         extra_legend_space = 0.15,
                         periods = 20)
```

### Numerical Options

The `tol` argument (default: `Tolerances()`, type: `Tolerances`) sets numerical tolerance levels for the solution algorithm.

The `verbose` argument (default: `false`, type: `Bool`) enables detailed output during computation.

```julia
# Enable verbose output with custom tolerances
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         verbose = true,
                         tol = Tolerances(1e-12),
                         periods = 20)
```

## Comparing Conditional Forecasts

The `plot_conditional_forecast!` function overlays multiple conditional forecasts on existing plots, useful for comparing different conditioning assumptions.

```julia
# Compare forecasts with different conditioning variables
plot_conditional_forecast(Gali_2015_chapter_3_nonlinear,
                         conditional_variance = [:y],
                         periods = 20,
                         label = "Output constrained")

plot_conditional_forecast!(Gali_2015_chapter_3_nonlinear,
                          conditional_variance = [:c],
                          periods = 20,
                          label = "Consumption constrained")
```

This comparison shows how different conditioning assumptions lead to different forecast paths for all variables in the model.
