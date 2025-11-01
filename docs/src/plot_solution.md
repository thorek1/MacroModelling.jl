# Policy Functions

The `plot_solution` function visualizes the solution of the model (mapping of past states to present variables) around the relevant steady state (e.g. higher order perturbation algorithms are centred around the stochastic steady state).

The relevant steady state is plotted along with the mapping from the chosen past state to one present variable per plot. All other (non-chosen) states remain in the relevant steady state.

In the case of pruned higher order solutions there are as many (latent) state vectors as the perturbation order. The first and third order baseline state vectors are the non-stochastic steady state and the second order baseline state vector is the stochastic steady state. Deviations for the chosen state are only added to the first order baseline state. The plot shows the mapping from `σ` standard deviations (first order) added to the first order non-stochastic steady state and the present variables. Note that there is no unique mapping from the "pruned" states and the "actual" reported state. Hence, the plots shown are just one realisation of infinitely many possible mappings.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

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

Calling `plot_solution` requires specifying a state variable. By default, it plots **all endogenous variables**, that do vary for different values of the specified state, as functions of the specified state over a range of ±2 standard deviations:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A)
```

The function plots each endogenous variable in period `t` against the state variable `A` in `t-1`. Each subplot shows how the variable changes on the y-axis as `A` varies within the specified range over the x-axis. The relevant steady state is indicated by a circle of the same color as the line. The title of each subplot indicates the variable name and the title of the overall plot indicates the model name, and page number (if multiple pages are needed). The legend below the plots indicate the solution algorithm used and the nature of the steady state (stochastic or non-stochastic).

## Function Arguments

### State Variable (Required)

The `state` argument (type: `Union{Symbol, String}`) specifies which state variable to vary. This must be a state variable from the model (variables with lagged values). If a state variable is provided that is not part of the model's state vector, an error is raised and the valid state variables are listed.

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A)  # Using Symbol
plot_solution(Gali_2015_chapter_3_nonlinear, "A") # Using String
```

### Variables to Plot

The `variables` argument (default: `:all_excluding_obc`) specifies for which variables to show results. Variable names can be specified as either a `Symbol` or `String` (e.g. `:y` or `"y"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` includes all variables except auxiliary variables and those related to occasionally binding constraints (OBC). `:all_excluding_obc` includes all variables except those related to occasionally binding constraints. `:all` includes all variables.

Specific variables can be selected to plot. The following example selects only output (`Y`) and inflation (`Pi`) using a `Vector` of `Symbol`s:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y, :Pi])
```

![Gali 2015 solution - selected variables (Y, Pi)](../assets/.png)

The plot now displays only the two selected variables (sorted alphabetically), with two subplots for each shock.
The same can be done using a `Tuple`:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = (:Y, :Pi))
```

a `Matrix`:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = [:Y :Pi])
```

or providing the variable names as `String`s:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = ["Y", "Pi"])
```

or a single variable as a `Symbol`:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = :Y)
```

or as a `String`:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = "Y")
```

Then there are some predefined options:

`:all_excluding_auxiliary_and_obc` (default) plots all variables except auxiliary variables and those used to enforce occasionally binding constraints (OBC).

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = :all_excluding_auxiliary_and_obc)
```

`:all_excluding_obc` plots all variables except those used to enforce occasionally binding constraints (OBC).

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    variables = :all_excluding_obc)
```

To see auxiliary variables, use a model that defines them. The FS2000 model can be used:

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

Since both `c` and `P` appear in t+2, they generate auxiliary variables in the model. Plotting the IRF for all variables excluding OBC-related ones reveals the auxiliary variables:

```julia
plot_solution(FS2000, :k,
	variables = :all_excluding_obc)
```

![FS2000 solution - e_a shock with auxiliary variables](../assets/.png)

Both `c` and `P` appear twice: once as the variable itself and once as an auxiliary variable with the `ᴸ⁽¹⁾` superscript, representing the value of the variable in t+1 as expected in t.

`:all` plots all variables including auxiliary variables and those used to enforce occasionally binding constraints (OBC).

Use the `Gali_2015_chapter_3` model with an effective lower bound (note the use of the `max` function in the Taylor rule):

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
    R > 1.0001
end
```

Plotting the IRF for all variables including OBC-related ones reveals the OBC-related auxiliary variables:

```julia
plot_solution(Gali_2015_chapter_3_obc, :A,
    variables = :all)
```

![Gali 2015 OBC solution - eps_z shock with OBC variables](../assets/.png)

The OBC-related variables appear in the last subplot.
Note that with the `eps_z` shock, the interest rate `R` hits the effective lower bound in period 1:

![Gali 2015 OBC solution - eps_z shock hitting lower bound](../assets/.png)

The effective lower bound is enforced using shocks to the equation containing the `max` statement. See the documentation for details on constructing occasionally binding constraints. For this specific model, examine the equations the parser generated to enforce the OBC:

```julia
get_equations(Gali_2015_chapter_3_obc)
# 68-element Vector{String}:
#  "W_real[0] = C[0] ^ σ * N[0] ^ φ"
#  "Q[0] = ((β * (C[1] / C[0]) ^ -σ * Z[1]) / Z[0]) / Pi[1]"
#  "R[0] = 1 / Q[0]"
#  "Y[0] = A[0] * (N[0] / S[0]) ^ (1 - α)"
#  "R[0] = Pi[1] * realinterest[0]"
#  "χᵒᵇᶜ⁺ꜝ¹ꜝˡ[0] = R̄ - R[0]"
#  ⋮
#  "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾[0] = ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁵⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽⁴⁾[x]"
#  "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾[0] = ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁶⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽³⁾[x]"
#  "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾[0] = ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁷⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽²⁾[x]"
#  "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾[0] = ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁸⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽¹⁾[x]"
#  "ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻⁴⁰⁾[0] = ϵᵒᵇᶜ⁺ꜝ¹ꜝᴸ⁽⁻³⁹⁾[-1] + activeᵒᵇᶜshocks * ϵᵒᵇᶜ⁺ꜝ¹ꜝ⁽⁰⁾[x]"
```

### Solution Algorithm

The `algorithm` argument (default: `:first_order`, type: `Symbol`) specifies which algorithm to solve for the dynamics of the model. Available algorithms: `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`.

You can compare different solution algorithms by overlaying plots using `plot_solution!`. The following example first plots the first-order solution and then overlays the second-order solution for comparison:

```julia
# Plot first-order policy function
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :first_order)

# Overlay second-order to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :second_order)
```

![Gali 2015 solution - multiple solution methods](../assets/compare_orders_solution__Gali_2015_chapter_3_nonlinear__2.png)

The plot now features both policy functions overlaid. The first-order solution is shown in blue, the second-order solution is shown in orange, as indicated in the legend below the plot. The lines correspond to the policy functions at different orders and the circles indicate the relevant steady state for each solution method. Higher order solutions may have different steady states due to the inclusion of risk effects (see e.g. `W_real`) and their policy functions may differ due to non-linearities captured at higher orders (see e.g. `S` which has only higher order dynamics).

Additional solution methods can be added to the same plot:

```julia
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    algorithm = :pruned_third_order)
```

![Gali 2015 IRF - eps_a shock (multiple orders)](../assets/multiple_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Note that the pruned third-order solution incorporates time-varying risk and reverses the sign of the response for `MC` and `N`. The additional solution appears as another colored line with corresponding entries in the legend.

### State Variable Range

The `σ` argument (default: `2`, type: `Union{Int64, Float64}`) specifies the range of the state variable as a multiple of its standard deviation. The state variable varies from `-σ * std(state)` to `+σ * std(state)`.

Plot over a wider range (±5 standard deviations) uing the pruned third-order solution:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    σ = 5,
    algorithm = :pruned_third_order)
```

![Gali 2015 IRF - eps_a shock (multiple orders)](../assets/multiple_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

This expands the x-axis range, showing how the policy functions behave further from the steady state.

### Parameter Values

When no parameters are provided, the solution uses the previously defined parameter values. Parameters can be provided as a `Vector` of values, or as a `Vector` or `Tuple` of `Pair`s mapping parameter `Symbol`s or `String`s to values. The solution is recalculated when new parameter values differ from the previous ones.

Start by changing the discount factor `β` to 0.95:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.95)
```

![Gali 2015 solution - different parameter values](../assets/different_parameters_solution__Gali_2015_chapter_3_nonlinear__1.png)

The steady states and dynamics changed as a result of changing the discount factor. To better visualize the differences between `β = 0.99` and `β = 0.95`, the two policy functions can be overlaid (compared). Since parameter changes are permanent, first reset `β = 0.99` before overlaying the IRF with `β = 0.95` on top of it:

```julia
# Plot with default parameters
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.99)

# Overlay with different discount factor to compare
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    parameters = :β => 0.95)
```

![Gali 2015 IRF - eps_a shock comparing β values](../assets/compare_beta_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The legend below the plot indicates which color corresponds to each `β` value. Note that both the steady states and dynamics differ across the two `β` values.

Multiple parameters can also be changed simultaneously to compare the results to previous plots. This example changes `β` to 0.97 and `τ` to 0.5 using a `Tuple` of `Pair`s and define the variables with `Symbol`s:

```julia
plot_solution!(Gali_2015_chapter_3_nonlinear, :A,
    parameters = (:β => 0.97, :τ => 0.5))
```

![Gali 2015 IRF - eps_a shock with multiple parameter changes](../assets/multi_params_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

Since the plot function calls now differ in multiple input arguments, the legend indicates which color corresponds to each input combination, with the table showing relevant input combinations.

A `Vector` of `Pair`s can also be used:

```julia
plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = [:β => 0.98, :τ => 0.25])
```

Alternatively, use a `Vector` of parameter values in the order they were defined in the model. To obtain them:

```julia
params = get_parameters(Gali_2015_chapter_3_nonlinear, values = true)
# 16-element Vector{Pair{String, Float64}}:
#       "σ" => 1.0
#       "φ" => 5.0
#     "ϕᵖⁱ" => 1.5
#      "ϕʸ" => 0.125
#       "θ" => 0.75
#     "ρ_ν" => 0.5
#     "ρ_z" => 0.5
#     "ρ_a" => 0.9
#       "β" => 0.95
#       "η" => 3.77
#       "α" => 0.25
#       "ϵ" => 9.0
#       "τ" => 0.5
#   "std_a" => 0.01
#   "std_z" => 0.05
#  "std_nu" => 0.0025

param_vals = [p[2] for p in params]
# 16-element Vector{Float64}:
#  1.0
#  5.0
#  1.5
#  0.125
#  0.75
#  0.5
#  0.5
#  0.9
#  0.95
#  3.77
#  0.25
#  9.0
#  0.5
#  0.01
#  0.05
#  0.0025

plot_solution(Gali_2015_chapter_3_nonlinear, :A,
    parameters = param_vals)
```

### Ignoring Occasionally Binding Constraints

The `ignore_obc` argument (default: `false`, type: `Bool`), when `true`, ignores occasionally binding constraints (OBC) even if they are part of the model. This is useful for comparing dynamics with and without OBC.
For models with defined OBC, use the `ignore_obc` argument to ignore them. The following example compares the policy functions of the `Gali_2015_chapter_3_obc` model with and without OBC. First, examine the policy function with OBC enforced. Since OBC is enforced by default, call:

```julia
plot_solution(Gali_2015_chapter_3_obc, :A)
```

![Gali 2015 OBC IRF - eps_z shock with OBC](../assets/obc_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

Then overlay the policy function ignoring the OBC:

```julia
plot_solution!(Gali_2015_chapter_3_obc, :A,
    ignore_obc = true)
```

![Gali 2015 OBC IRF - eps_z shock comparing with and without OBC](../assets/compare_obc_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

The legend indicates which color corresponds to each `ignore_obc` value. The difference between the two can be noticed at the effective lower bound for `R`. For values of `A` where the effective lower bound is reached the shocks enforcing the lower bound act on the economy and the policy function changes for most other variables as well.

### Plot Labels

The `label` argument (default: `""`, type: `Union{Real, String, Symbol}`) adds custom labels to the plot legend. This is useful when comparing multiple solutions using `plot_solution!` to overlay plots:

```julia
# Compare policy functions with different settings
plot_solution(Gali_2015_chapter_3_obc, :A,
    variables = [:Y, :C, :R],
    algorithm = :second_order,
    # ignore_obc = false,
    parameters = :β => 0.99,
    # label = "2nd Order with OBC"
    )

# Add solution without OBC
plot_solution!(Gali_2015_chapter_3_obc, :A,
    # variables = [:Y, :C],
    algorithm = :second_order,
    ignore_obc = true,
    # label = "2nd Order without OBC"
    )

# Add different parameter setting
plot_solution!(Gali_2015_chapter_3_obc, :A,
    variables = [:Y, :C, :R],
    algorithm = :second_order,
    parameters = :β => 0.95,
    # label = "2nd Order with OBC and β=0.95"
    )
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
        :linestyle => :dash))
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
