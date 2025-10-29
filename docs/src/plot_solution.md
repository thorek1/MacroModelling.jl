# Policy Functions

The `plot_solution` function visualizes policy functions by plotting the relationship between a state variable and endogenous variables. This shows how variables respond to changes in a state variable around the steady state, revealing the model's decision rules.

## Basic Usage

First, define and load a model:

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
# Plot first-order policy function
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    algorithm = :first_order)

# Overlay second-order to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
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
# Plot with default parameters
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    label = "Default β=0.99")

# Overlay with different discount factor to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    parameters = :β => 0.95)
```

The parameter vector must match the model's parameter order and length. This example demonstrates how the policy functions change with different parameter values.

### Occasionally Binding Constraints

The `ignore_obc` argument (default: `false`, type: `Bool`) determines whether to ignore occasionally binding constraints when solving the model. This can be used to compare how policy functions differ with and without OBC:

```julia
# Plot policy function with OBC
plot_solution(model_with_obc, :state,
    variables = [:Y, :C],
    parameters = :β => 0.99,
    label = "With OBC")

# Add policy function without OBC for comparison
plot_solution!(model_with_obc, :state,
    variables = [:Y, :C],
    ignore_obc = true)
```

### Plot Labels

The `label` argument (default: `""`, type: `Union{Real, String, Symbol}`) adds custom labels to the plot legend. This is useful when comparing multiple solutions using `plot_solution!` to overlay plots:

```julia
# Compare policy functions with different settings
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    algorithm = :second_order,
    ignore_obc = false,
    label = "2nd Order with OBC")

# Add solution without OBC
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    algorithm = :second_order,
    ignore_obc = true,
    label = "2nd Order without OBC")

# Add different parameter setting
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :C],
    algorithm = :second_order,
    parameters = :β => 0.95,
    label = "2nd Order with OBC and β=0.95")
```

This demonstrates comparing policy functions across multiple dimensions: solution algorithms, occasionally binding constraints, and parameter values, revealing how different model specifications affect the dynamics.

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
@model FS2000 begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])
    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
    W[0] = l[0] / n[0]
    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
    c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
    P[0] * c[0] = m[0]
    m[0] - 1 + d[0] = l[0]
    e[0] = exp(z_e_a  *  e_a[x])
    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
    gy_obs[0] = dA[0] * y[0] / y[-1]
    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
    log_gy_obs[0] = log(gy_obs[0])
    log_gp_obs[0] = log(gp_obs[0])
end

@parameters FS2000 begin
    alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862
end
```

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :k,
    variables = [:C, :Y],
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output"))

plot_solution!(FS2000, :k,
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
        "C{H}" => "Home Consumption",
        "C{F}" => "Foreign Consumption"))
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
