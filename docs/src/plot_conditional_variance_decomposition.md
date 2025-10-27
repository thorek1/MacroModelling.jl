# Conditional Variance Decomposition

The `plot_conditional_variance_decomposition` function visualizes the forecast error variance decomposition (FEVD), showing how much of the variance in forecast errors for each variable can be attributed to different shocks over various forecast horizons.

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

plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear)
```

This creates variance decomposition plots showing the contribution of each shock to the forecast error variance of each variable across different forecast horizons.

## Arguments

### Periods Argument

The `periods` argument (default: `40`, type: `Int`) specifies the number of forecast horizons to include in the variance decomposition. This determines how far into the future the decomposition extends.

```julia
# Show variance decomposition up to 20 periods ahead
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, periods = 20)
```

### Variables Selection

The `variables` argument (default: `model.var`, type: `Symbol` or `Vector{Symbol}`) specifies which variables to include in the variance decomposition. Variables can be selected by passing their symbols.

```julia
# Decomposition for output and inflation only
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    variables = [:Y, :Pi])
```

### Shocks Selection

The `shocks` argument (default: `model.exo`, type: `Symbol` or `Vector{Symbol}`) determines which shocks to include in the decomposition. This is useful for focusing on specific sources of variation.

```julia
# Show only technology shock contributions
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    shocks = :eps_a)

# Compare multiple shock contributions
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    shocks = [:eps_a, :nu])
```

### Solution Algorithm

The `algorithm` argument (default: `:first_order`, type: `Symbol`) specifies the solution method. Options include `:first_order`, `:second_order`, and `:third_order`.

```julia
# Use second-order approximation
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    algorithm = :second_order)
```

### Alternative Parameters

The `parameters` argument (default: model parameters, type: `Vector{Real}` or `Matrix{Real}`) allows plotting variance decompositions under different parameter values.

```julia
# Compare decompositions with different monetary policy responses
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    parameters = :ϕ_π => [1.5, 2.0])
```

### Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) accepts standard Plots.jl attributes for customizing the appearance.

```julia
# Customize plot appearance
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear,
    plot_attributes = Dict(
        :size => (800, 600),
        :linewidth => 2,
        :title => "Forecast Error Variance Decomposition"
    ))
```

### Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps variable and shock names to more readable labels in the plots.

```julia
# Use more descriptive names
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear,
    rename_dictionary = Dict(
        :Y => "Output",
        :Pi => "Inflation",
        :eps_a => "Technology Shock",
        :nu => "Monetary Policy Shock"
    ))
```

### Saving Plots

The `save_plots` argument (default: `false`, type: `Bool`) determines whether to save the generated plots to disk.

Related arguments for saving:
- `save_plots_name` (default: `"fevd"`, type: `Union{String, Symbol}`) - filename prefix
- `save_plots_format` (default: `:pdf`, type: `Symbol`) - output format (`:pdf`, `:png`, `:svg`, etc.)
- `save_plots_path` (default: `"."`, type: `String`) - directory for saved files

```julia
# Save variance decomposition plots
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear,
    save_plots = true,
    save_plots_name = "my_fevd",
    save_plots_format = :png,
    save_plots_path = "output/plots")
```

Files are saved with the format: `{name}__ModelName__variable__horizon.{format}`

### Occasionally Binding Constraints

The `ignore_obc` argument (default: `false`, type: `Bool`) determines whether to ignore occasionally binding constraints when computing the variance decomposition.

```julia
# Compare decompositions with and without OBC
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear)
plot_conditional_variance_decomposition!(Gali_2015_chapter_3_nonlinear, 
    ignore_obc = true)
```

## Understanding the Decomposition

The conditional variance decomposition shows:
- **Short horizons**: Which shocks immediately affect each variable
- **Medium horizons**: How shock impacts propagate through the economy
- **Long horizons**: Which shocks dominate long-run fluctuations

The sum of all shock contributions at each horizon equals 100% of the forecast error variance for each variable.

## Comparison with plot_conditional_variance_decomposition!

The `plot_conditional_variance_decomposition!` function overlays new decompositions on existing plots, useful for comparing across specifications:

```julia
# Compare different policy rules
plot_conditional_variance_decomposition(Gali_2015_chapter_3_nonlinear, 
    parameters = :ϕ_π => 1.5)
plot_conditional_variance_decomposition!(Gali_2015_chapter_3_nonlinear, 
    parameters = :ϕ_π => 2.5)
```
