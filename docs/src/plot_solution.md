# Policy Functions

The `plot_solution` function visualizes policy functions by plotting the relationship between a state variable and endogenous variables. This shows how variables respond to changes in a state variable around the steady state, revealing the model's decision rules.

## Basic Usage

Calling `plot_solution` requires specifying a state variable. By default, it plots **all endogenous variables** as functions of the specified state over a range of ±2 standard deviations:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A)
```

The function plots each endogenous variable against the state variable `A`. Each subplot shows how the variable changes as `A` varies within the specified range. The steady state is indicated by horizontal and vertical reference lines.

## Function Arguments

### State Variable (Required)

The `state` argument (type: `Union{Symbol, String}`) specifies which state variable to vary. This must be a state variable from the model (variables with lagged values).

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A)  # Using Symbol
plot_solution(Gali_2015_chapter_3_nonlinear, "A") # Using String
```

### Variables to Plot

The `variables` argument (default: `:all_excluding_obc`, type: `Union{Symbol, String, Vector{Symbol}, Vector{String}}`) determines which endogenous variables to display.

Available options: `:all`, `:all_excluding_obc`, `:all_excluding_aux`, `:all_excluding_aux_and_obc`, or specify variables explicitly using symbols, strings, or vectors.

Select specific variables:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi])
```

Plot all variables including auxiliary variables:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = :all)
```

### Solution Algorithm

The `algorithm` argument (default: `:first_order`, type: `Symbol`) specifies which algorithm to solve for the dynamics of the model. Available algorithms: `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order)
```

At higher orders, policy functions become nonlinear, showing how the response varies across different states.

### State Variable Range

The `σ` argument (default: `2`, type: `Union{Int64, Float64}`) specifies the range of the state variable as a multiple of its standard deviation. The state variable varies from `-σ * std(state)` to `+σ * std(state)`.

Plot over a wider range (±3 standard deviations):

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 3)
```

Plot over a narrower range (±1 standard deviation):

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 1)
```

### Alternative Parameters

The `parameters` argument (default: `nothing`, type: `Union{Nothing, Vector{Float64}, Vector{Int64}}`) allows plotting with different parameter values without modifying the model.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = [1, 5, 1.5, 0.125, 0.75, 0.5, 0.5, 0.9, 0.99, 3.77, 0.25, 9, 0.5, 0.01, 0.05, 0.0025])
```

The parameter vector must match the model's parameter order and length.

### Occasionally Binding Constraints

The `ignore_obc` argument (default: `false`, type: `Bool`) determines whether to ignore occasionally binding constraints when solving the model.

```julia
plot_solution(model_with_obc, :state,
    ignore_obc = true)
```

### Plot Labels

The `label` argument (default: `""`, type: `Union{Real, String, Symbol}`) adds custom labels to the plot legend. This is useful when comparing multiple solutions using `plot_solution!` to overlay plots:

```julia
# Plot first-order solution
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    label = "First Order")

# Add second-order solution
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    algorithm = :second_order,
    label = "Second Order")

# Add third-order solution
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C, :Pi],
    algorithm = :third_order,
    label = "Third Order")
```

This allows direct comparison of how policy functions differ across solution methods, revealing the importance of nonlinearities in the model.

### Display Control

The `show_plots` argument (default: `true`, type: `Bool`) controls whether plots are displayed in the plotting pane.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    show_plots = false)  # Generate plots without displaying
```

### Saving Plots

The `save_plots` argument (default: `false`, type: `Bool`) determines whether to save plots to disk.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true)
```

#### Save Plot Format

The `save_plots_format` argument (default: `:pdf`, type: `Symbol`) specifies the file format for saved plots. Common formats: `:pdf`, `:png`, `:svg`.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_format = :png)
```

#### Save Plot Name

The `save_plots_name` argument (default: `"solution"`, type: `Union{String, Symbol}`) specifies the prefix for saved plot filenames. The filename format is: `prefix__ModelName__state__page.format`.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_name = "policy_A")
# Creates: policy_A__Gali_2015_chapter_3_nonlinear__A__1.pdf
```

#### Save Plot Path

The `save_plots_path` argument (default: `"."`, type: `String`) specifies the directory where plots are saved.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    save_plots = true,
    save_plots_path = "plots/policy_functions")
```

### Plots Per Page

The `plots_per_page` argument (default: `6`, type: `Int`) controls how many subplots appear on each page. Useful for managing large numbers of variables.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plots_per_page = 9)  # 3x3 grid
```

### Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) allows renaming variables and shocks in plot labels for clearer display.

Basic renaming for readable labels:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    rename_dictionary = Dict(:Y => "Output", :C => "Consumption", :Pi => "Inflation"))
```

This feature is particularly useful when comparing models with different variable naming conventions. For example, when overlaying policy functions from FS2000 (which uses lowercase `c` for consumption) and Gali_2015_chapter_3_nonlinear (which uses uppercase `C`):

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:C, :Y],
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output"))

plot_solution!(FS2000, :e_a,
    variables = [:c, :y],
    rename_dictionary = Dict(:c => "Consumption", :y => "Output"))
```

The `rename_dictionary` accepts flexible type combinations for keys and values. The following are all equivalent:

```julia
# Symbol keys, String values
rename_dictionary = Dict(:Y => "Output")

# String keys, String values
rename_dictionary = Dict("Y" => "Output")

# Symbol keys, Symbol values
rename_dictionary = Dict(:Y => :Output)

# String keys, Symbol values
rename_dictionary = Dict("Y" => :Output)
```

For models with special characters in variable names (like the Backus_Kehoe_Kydland_1992 model which uses symbols like `Symbol("C{H}")`):

```julia
plot_solution(Backus_Kehoe_Kydland_1992, :K,
    rename_dictionary = Dict(
        Symbol("C{H}") => "Home Consumption",
        Symbol("C{F}") => "Foreign Consumption"))
```

The renaming applies to all plot elements: legends, axis labels, and tables.

### Custom Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) allows passing additional styling attributes to the plotting backend.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    plot_attributes = Dict(
        :linewidth => 3,
        :linestyle => :dash,
        :color => :red))
```

### Verbosity

The `verbose` argument (default: `true`, type: `Bool`) controls whether to print progress messages during computation.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    verbose = false)  # Suppress output
```

### Numerical Tolerance

The `tol` argument (default: `Tolerances()`, type: `Tolerances`) specifies numerical tolerance settings for the solver.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    tol = Tolerances(tol = 1e-12))
```

### Quadratic Matrix Equation Solver

The `quadratic_matrix_equation_algorithm` argument (default: `:bicgstab`, type: `Symbol`) specifies which algorithm to use for solving quadratic matrix equations in higher-order solutions.

Available algorithms: `:bicgstab`, `:gmres`, `:dqgmres`.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order,
    quadratic_matrix_equation_algorithm = :gmres)
```

### Sylvester Equation Solver

The `sylvester_algorithm` argument (default: depends on model size, type: `Union{Symbol, Vector{Symbol}, Tuple{Symbol, Vararg{Symbol}}}`) specifies which algorithm to use for solving Sylvester equations.

Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:dqgmres`.

For second-order solutions, specify a single algorithm:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order,
    sylvester_algorithm = :bartels_stewart)
```

For third-order solutions, different algorithms can be specified for the second- and third-order Sylvester equations using a `Tuple`:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :third_order,
    sylvester_algorithm = (:doubling, :bicgstab))
```

### Lyapunov Equation Solver

The `lyapunov_algorithm` argument (default: `:doubling`, type: `Symbol`) specifies which algorithm to use for solving Lyapunov equations.

Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:dqgmres`.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    lyapunov_algorithm = :bartels_stewart)
```
