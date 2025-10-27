# Impulse Response Functions (IRF)

Calling `plot_irf` computes IRFs for **every exogenous shock** and **every endogenous variable** by default, using the model's default solution method (first-order perturbation) and a **one-standard-deviation positive** shock.

```julia
plot_irf(Gali_2015_chapter_3_nonlinear)
```

![Gali 2015 IRF - eps_a shock](../assets/default_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The plot displays every endogenous variable affected by each exogenous shock. The title includes the model name, shock identifier, sign of the impulse (positive by default), and the page indicator (e.g., `(1/3)`). Each subplot shows the steady state as a horizontal reference line (non-stochastic for first-order solutions, stochastic for higher orders). For strictly positive variables, a secondary axis displays percentage deviations.

IRFs can be plotted using different solution algorithms. The following example uses a second-order perturbation solution:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :second_order)
```

![Gali 2015 IRF - eps_a shock (second order)](../assets/second_order_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The most notable difference is that at second order, dynamics are observed for `S`, which remains constant at first order (under certainty equivalence). Additionally, the steady state levels change because the stochastic steady state incorporates precautionary behavior (see horizontal lines).

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock (first order)](../assets/first_order_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

To compare the two solution methods side by side, use `plot_irf!` to add to an existing plot:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :second_order)
```

![Gali 2015 IRF - eps_a shock (first vs second order)](../assets/compare_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The plots now show both solution methods overlaid. The first-order solution is shown in blue, the second-order solution in orange, as indicated in the legend below the plot. Note that the steady state levels can be different for the two solution methods. For variables where the relevant steady state is the same for both methods (e.g., `A`), the level appears on the left axis and percentage deviations on the right axis. For variables where the steady state differs between methods (e.g., `C`), only absolute level deviations (`abs. Δ`) appear on the left axis. The relevant steady state levels are shown in a table below the plot for reference (rounded to help identify differences).

Additional solution methods can be added to the same plot:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :pruned_third_order)
```

![Gali 2015 IRF - eps_a shock (multiple orders)](../assets/multiple_orders_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Note that the pruned third-order solution incorporates time-varying risk and reverses the sign of the response for `MC` and `N`. The additional solution appears as another colored line with corresponding entries in both the legend and the steady state table below.

### Initial State

The `initial_state` argument (default: `[0.0]`, type: `Union{Vector{Vector{Float64}},Vector{Float64}}`) defines the starting point for the model. For pruned solution algorithms, the initial state can be provided as multiple state vectors (Vector{Vector{Float64}}). In this case, the initial state must be specified in deviations from the non-stochastic steady state. For all other cases, specify the initial state in levels. For pruned solution algorithms with a `Vector{Float64}` initial state, only the first-order initial state vector is affected. The state includes all variables as well as exogenous variables in leads or lags if present. `get_irf(𝓂, shocks = :none, variables = :all, periods = 1)` returns a `KeyedArray` with all variables. The `KeyedArray` type is provided by the AxisKeys package.

The initial state defines the starting point for the IRF and must contain all model variables, including any leads or lags. To obtain the correct ordering and number of variables, call `get_irf(𝓂, shocks = :none, variables = :all, periods = 1)`, which returns a `KeyedArray` with all variables in the correct order. The `KeyedArray` type is provided by the AxisKeys package. For example:

```julia
init_state = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true)
```

Only state variables will have an impact on the IRF. To check which variables are state variables:

```julia
get_state_variables(Gali_2015_chapter_3_nonlinear)
```

Now modify the initial state and set `nu` to 0.1:

```julia
init_state(:nu,:,:) .= 0.1
```

The modified initial state can now be input into the `plot_irf` function as a vector:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state))
```

![Gali 2015 IRF - eps_a shock with custom initial state](../assets/custom_init_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Note that the example also defines the shock `eps_a` to show how the model reacts to a shock to `A`. For more details on the `shocks` argument see the corresponding section.
This shows the difference in the IRF compared to starting from the non-stochastic steady state. Setting `nu` to a higher level effectively combines the effects of shocks to both `nu` and `A`. Since this uses a linear solution, these effects can be separated by stacking the components. Start with the IRF from the initial state as defined above:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    initial_state = vec(init_state))
```

![Gali 2015 IRF - no shock with initial state](../assets/no_shock_init_irf__Gali_2015_chapter_3_nonlinear__no_shock__1.png)

then stack the IRF from a shock to A on top of it:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    plot_type = :stack)
```

![Gali 2015 IRF - stacked initial state and eps_a shock](../assets/stacked_init_irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1.png)

The two components are shown with labels explained in the table below. The blue line represents the first input (no shock, non-zero initial state), while the red line corresponds to the second input (starting from the steady state with an `eps_a` shock). Both components add up to the solid line that is the same as in the case of combining the `eps_a` shock with the initial state.

The same approach works for higher order solutions. Start with the second order solution. First, obtain the initial state in levels from the second order solution:

```julia
init_state_2nd = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true,
    algorithm = :second_order)
```

Then set `nu` to 0.1:

```julia
init_state_2nd(:nu, :, :) .= 0.1
```

and plot the IRF for `eps_a` starting from this initial state:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state_2nd),
    algorithm = :second_order)
```

While these components can be stacked, they won't add up linearly due to the non-linear solution. Instead, the IRF behavior can be compared across the two solution methods:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state))

plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state_2nd),
    algorithm = :second_order)
```

![Gali 2015 IRF - eps_a shock with initial state (multiple solutions)](../assets/multi_sol_init_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The legend shows two lines, with their input differences detailed in the table below. The first line corresponds to the initial state used for the first order solution as well as the IRF using the first order solution and the second line corresponds to the initial state used for the second order solution and using the second order solution. Note that the steady states differ between the two solution methods, which also affects the initial states (except for `nu`, which is set to 0.1 in both cases). A second table below the first one shows the relevant steady states for both solution methods. Since the relevant steady state of `A` is the same for both methods, the corresponding subplot shows the level on the left axis and percentage deviations on the right axis. For all other variables, the relevant steady state differs between methods, so only absolute level deviations appear (`abs. Δ`) on the left axis, with steady states listed in the table at the bottom.

For pruned solution methods the initial state can also be given as multiple state vectors (Vector{Vector{Float64}}). When providing a vector of vectors, values must be specified as differences from the non-stochastic steady state. When providing only one vector, values must be in levels, with the initial state having its full nonlinear effect in the first period. Using a vector of vectors allows setting the pruned higher-order auxiliary state vectors. While this can be useful in some cases, note that these higher-order auxiliary state vectors have only a linear impact on the dynamics. Start by assembling the vector of vectors:

```julia
init_state_pruned_3rd_in_diff = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true) - get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    algorithm = :pruned_third_order,
    levels = true)
```

First- and third-order dynamics don't affect the steady state through risk, so they are zero. The second-order steady state includes the risk adjustment. Assemble the vectors for the third order case:

```julia
init_states_pruned_3rd_vec = [
    zero(vec(init_state_pruned_3rd_in_diff)),
    vec(init_state_pruned_3rd_in_diff),
    zero(vec(init_state_pruned_3rd_in_diff)),
]
```

Then set `nu` to 0.1 in the first order terms. Inspecting `init_state_pruned_3rd_in_diff` shows that `nu` is the 18th variable in the vector:

```julia
init_states_pruned_3rd_vec[1][18] = 0.1

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = init_states_pruned_3rd_vec,
    algorithm = :pruned_third_order)
```

![Gali 2015 IRF - eps_a shock with pruned 3rd order vector](../assets/pruned_3rd_vec_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Alternatively, a simple vector can be used for the initial state. In this case the values must be in levels and the impact of the initial state is assumed to have the full nonlinear effect in the first period:

```julia
init_state_pruned_3rd = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true,
    algorithm = :pruned_third_order)

init_state_pruned_3rd(:nu, :,  :) .= 0.1

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state_pruned_3rd),
    algorithm = :pruned_third_order)
```

Compare this with the second- and first-order versions, each starting from their respective steady states.

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state_2nd),
    algorithm = :second_order)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    initial_state = vec(init_state))
```

![Gali 2015 IRF - eps_a shock with initial state (all solution methods)](../assets/all_sol_init_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

This shows that the pruned third-order solution changes the dynamics while maintaining the same steady states as the second-order solution.

### Shocks

The `shocks` argument (default: `:all_excluding_obc`) determines which IRFs to calculate. Shock names can be specified as either a `Symbol` or `String` (e.g., `:y` or `"y"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:simulate` triggers random draws of all shocks (excluding those related to occasionally binding constraints). `:all_excluding_obc` includes all shocks except OBC-related ones. `:all` also includes the OBC related shocks. A series of shocks can be passed using either a `Matrix{Float64}` or a `KeyedArray{Float64}`, with shocks (`Symbol` or `String`) in rows and periods in columns. The `KeyedArray` type is provided by the AxisKeys package. The simulation length equals the input length in the period dimension plus the number of periods specified in the `periods` argument. When using a `KeyedArray{Float64}`, ensure that rows are named with valid shock names of type `Symbol`. Any shocks not part of the model will trigger a warning. `:none` in combination with an `initial_state` can be used for deterministic simulations.

Individual shocks can be called by name:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock](../assets/single_shock_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The same works if the shock name is input as a string:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = "eps_a")
```

or multiple shocks at once (as strings or symbols):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = [:eps_a, :eps_z])
```

![Gali 2015 IRF - eps_a shock](../assets/multi_shocks_irf__Gali_2015_chapter_3_nonlinear__eps_a__3.png)

![Gali 2015 IRF - eps_z shock](../assets/multi_shocks_irf__Gali_2015_chapter_3_nonlinear__eps_z__3.png)

Multiple shocks can also be input as a `Tuple`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = (:eps_a,
    :eps_z))
```

or a matrix:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = [:eps_a :eps_z])
```

Then there are some predefined options:

`:all_excluding_obc` (default) plots all shocks not used to enforce occasionally binding constraints (OBC).

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :all_excluding_obc)
```

![Gali 2015 IRF - eps_nu shock](../assets/all_ex_obc_irf__Gali_2015_chapter_3_nonlinear__eps_nu__1.png)

`:all` plots all shocks including the OBC related ones.

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :all)
```

`:simulate` triggers random draws of all shocks (excluding OBC-related shocks). The seed can be set to get reproducible results (e.g. `import Random; Random.seed!(10)`).

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :simulate)
```

![Gali 2015 IRF - simulated shocks](../assets/simulated_irf__Gali_2015_chapter_3_nonlinear__simulation__1.png)

`:none` can be used in combination with an `initial_state` for deterministic simulations. See the section on `initial_state` for more details. Start by getting the initial state in levels:

```julia
init_state = get_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    variables = :all,
    periods = 1,
    levels = true)
```

Only state variables will have an impact on the IRF. To check which variables are state variables:

```julia
get_state_variables(Gali_2015_chapter_3_nonlinear)
```

Now modify the initial state and set `nu` to 0.1:

```julia
init_state(:nu, :, :) .= 0.1
```

Now input the modified initial state into the `plot_irf` function as a vector and set `shocks` to `:none`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :none,
    initial_state = vec(init_state))
```

![Gali 2015 IRF - deterministic simulation from initial state](../assets/deterministic_irf__Gali_2015_chapter_3_nonlinear__no_shock__1.png)

Note how this is similar to a shock to `eps_nu` but instead `nu` is set to 0.1 in the initial state and the model evolves deterministically from there. In the title the reference to the shock disappeared as it was set to `:none`.

Shocks can also be compared:

```julia
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = shocks[1])

for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear,
        shocks = s)
end
```

![Gali 2015 IRF - all shocks compared](../assets/compare_shocks_irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1.png)

All three shocks now appear overlaid in the same plot. The legend below the plot indicates which color corresponds to which shock and in the title the plot shows that all shocks are positive and includes multiple shocks.

A series of shocks can be passed on using either a `Matrix{Float64}`, or a `KeyedArray{Float64}` as input with shocks (`Symbol` or `String`) in rows and periods in columns. Start with a `KeyedArray`:

```julia
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)

n_periods = 3

shock_keyedarray = KeyedArray(
    zeros(length(shocks), n_periods),
    Shocks = shocks,
    Periods = 1:n_periods,
)
```

and then set a one standard deviation shock to `eps_a` in period 1, a negative 1/2 standard deviation shock to `eps_z` in period 2 and a 1/3 standard deviation shock to `eps_nu` in period 3:

```julia
shock_keyedarray("eps_a", [1]) .= 1
shock_keyedarray("eps_z", [2]) .= -1/2
shock_keyedarray("eps_nu", [3]) .= 1/3

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = shock_keyedarray)
```

![Gali 2015 IRF - shock series from KeyedArray](../assets/shock_series_irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2.png)

In the title it is now mentioned that the input is a series of shocks and the values of the shock processes Z and `nu` move with the shifted timing and note that the impact of the `eps_z` shock has a - in front of it in the model definition, which is why they both move in the same direction. Note also that the number of periods is prolonged by the number of periods in the shock input. This example defines 3 periods of shocks and the default number of periods is 40, so the result shows 43 periods in total.

The same can be done with a `Matrix`:

```julia
shock_matrix = zeros(length(shocks), n_periods)

shock_matrix[1, 1] = 1
shock_matrix[3, 2] = -1/2
shock_matrix[2, 3] = 1/3

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = shock_matrix)
```

In certain circumstances a shock matrix might correspond to a certain scenario and when working with linear solutions, the IRF can be stacked for different scenarios or components of scenarios. Consider two scenarios defined by two different shock matrices:

```julia
shock_matrix_1 = zeros(length(shocks), n_periods)

shock_matrix_1[1, 1] = 1
shock_matrix_1[3, 2] = -1/2
shock_matrix_1[2, 3] = 1/3


shock_matrix_2 = zeros(length(shocks), n_periods * 2)

shock_matrix_2[1, 4] = -1
shock_matrix_2[3, 5] = 1/2
shock_matrix_2[2, 6] = -1/3
```

Plot them on top of each other using the :stack option:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = shock_matrix_1)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = shock_matrix_2,
    plot_type = :stack)
```

![Gali 2015 IRF - stacked shock matrices](../assets/stacked_matrices_irf__Gali_2015_chapter_3_nonlinear__shock_matrix__2.png)

The blue bars correspond to the first shock matrix and the red to the second shock matrix and they are labeled accordingly in the legend below the plot. The solid line corresponds to the sum of both components. The result shows 46 periods as the second shock matrix has 6 periods and the first one 3 periods and the default number of periods is 40.

### Simulation Periods

The `periods` argument (default: `40`, type: `Int`) specifies the number of periods for which to calculate the output. When a matrix of shocks is provided, this defines how many periods to continue after the shock series.
To set the number of periods to 10 (for the `eps_a` shock):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    periods = 10,
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock (10 periods)](../assets/ten_periods_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The x-axis automatically adjusts to show only 10 periods.

The following example uses a shock matrix with 15 periods as input, sets the `periods` argument to 20, and compares it to the previous 10-period plot:

```julia
shock_matrix = zeros(length(shocks), 15)

shock_matrix[1, 1] = .1
shock_matrix[3, 5] = -1/2
shock_matrix[2, 15] = 1/3

plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = shock_matrix,
    periods = 20)
```

![Gali 2015 IRF - mixed period lengths](../assets/mixed_periods_irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__1.png)

The x-axis adjusts to 35 periods, with the first plot ending after 10 periods and the second plot ending after 35 periods. The legend indicates which color corresponds to each shock, and the title shows that multiple shocks are in the plot.

### Shock Size

The `shock_size` argument (default: `1.0`, type: `Real`) controls the size of the shocks in standard deviations. Only affects shocks not provided as a matrix or `KeyedArray` or set to `:none`. Negative values flip the sign of the shock.

The shock size can be set using the `shock_size` argument. The following example sets it to -2 standard deviations:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    shock_size = -2)
```

![Gali 2015 IRF - eps_a shock (size -2)](../assets/shock_size_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Note how the shock's sign is flipped and the response magnitude is increased.

### Negative Shocks

The `negative_shock` argument (default: `false`, type: `Bool`), when `true`, calculates IRFs for a negative shock. Only affects shocks not provided as a matrix or `KeyedArray` or set to `:none`.


Alternatively, set `negative_shock` to `true` to obtain the IRF for a negative one-standard-deviation shock:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    negative_shock = true)
```

![Gali 2015 IRF - eps_z shock (negative)](../assets/negative_shock_irf__Gali_2015_chapter_3_nonlinear__eps_z__1.png)

### Variables to Plot

The `variables` argument (default: `:all_excluding_obc`) specifies which variables to show results. Variable names can be specified as either a `Symbol` or `String` (e.g. `:y` or `"y"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` includes all variables except auxiliary variables and those related to occasionally binding constraints (OBC). `:all_excluding_obc` includes all variables except those related to occasionally binding constraints. `:all` includes all variables.

Specific variables can be selected to plot. The following example selects only output (`Y`) and inflation (`Pi`) using a `Vector` of `Symbol`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = [:Y, :Pi])
```

![Gali 2015 IRF - selected variables (Y, Pi)](../assets/var_select_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The plot now displays only the two selected variables (sorted alphabetically), with two subplots for each shock.
The same can be done using a `Tuple`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = (:Y, :Pi))
```

a `Matrix`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = [:Y :Pi])
```

or providing the variable names as strings:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = ["Y", "Pi"])
```

or a single variable as a `Symbol`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = :Y)
```

or as a string:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = "Y")
```

Then there are some predefined options:

`:all_excluding_auxiliary_and_obc` (default) plots all variables except auxiliary variables and those used to enforce occasionally binding constraints (OBC).

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = :all_excluding_auxiliary_and_obc)
```

`:all_excluding_obc` plots all variables except those used to enforce occasionally binding constraints (OBC).

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
plot_irf(FS2000, 
	variables = :all_excluding_obc)
```

![FS2000 IRF - e_a shock with auxiliary variables](../assets/with_aux_vars_irf__FS2000__e_a__1.png)

Both `c` and `P` appear twice: once as the variable itself and once as an auxiliary variable with the `L(1)` superscript, representing the value of the variable in t+1 as expected in t.

`:all` plots all variables including auxiliary variables and those used to enforce occasionally binding constraints (OBC).

Use the `Gali_2015_chapter_3` model with an effective lower bound (note the max statement in the Taylor rule):

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
plot_irf(Gali_2015_chapter_3_obc,
    variables = :all)
```

![Gali 2015 OBC IRF - eps_z shock with OBC variables](../assets/with_obc_vars_irf__Gali_2015_chapter_3_obc__eps_z__3.png)

The OBC-related variables appear in the last subplot.
Note that with the `eps_z` shock, the interest rate `R` hits the effective lower bound in period 1:

![Gali 2015 OBC IRF - eps_z shock hitting lower bound](../assets/with_obc_vars_irf__Gali_2015_chapter_3_obc__eps_z__2.png)

The effective lower bound is enforced using shocks to the equation containing the max statement. See the documentation for details on constructing occasionally binding constraints. For this specific model, examine the equations the parser generated to enforce the OBC:

```julia
get_equations(Gali_2015_chapter_3_obc)
```

### Parameter Values

When no parameters are provided, the solution uses the previously defined parameter values. Parameters can be provided as a `Vector` of values, or as a `Vector` or `Tuple` of Pairs mapping parameter Symbols or Strings to values. The solution is recalculated when new parameter values differ from the previous ones.

Start by changing the discount factor `β` from 0.99 to 0.95:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = :β => 0.95,
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock (`β = 0.95`)](../assets/beta_095_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The steady states and dynamics changed as a result of changing the discount factor. To better visualize the differences between `β = 0.99` and `β = 0.95`, the two IRFs can be overlaid. Since parameter changes are permanent, first reset `β = 0.99` before overlaying the IRF with `β = 0.95` on top of it:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = :β => 0.99,
    shocks = :eps_a)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = :β => 0.95,
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock comparing β values](../assets/compare_beta_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The legend below the plot indicates which color corresponds to each `β` value, with the table underneath showing the relevant steady states. Note that both the steady states and dynamics differ across the two `β` values, even when the steady state remains the same (e.g., for `Y`).

Multiple parameters can also be changed simultaneously to compare the results to previous plots. This example changes `β` to 0.97 and `τ` to 0.5 using a `Tuple` of Pairs and define the variables with Symbols:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.97, :τ => 0.5),
    shocks = :eps_a)
```

![Gali 2015 IRF - eps_a shock with multiple parameter changes](../assets/multi_params_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

Since the plot function calls now differ in multiple input arguments, the legend indicates which color corresponds to each input combination, with the table showing steady states for all three combinations.

A `Vector` of Pairs can also be used:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = [:β => 0.98, :τ => 0.25],
    shocks = :eps_a)
```

Alternatively, use a `Vector` of parameter values in the order they were defined in the model. To obtain them:

```julia
params = get_parameters(Gali_2015_chapter_3_nonlinear, values = true)

param_vals = [p[2] for p in params]

plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = param_vals,
    shocks = :eps_a)
```

### Ignoring Occasionally Binding Constraints

The `ignore_obc` argument (default: `false`, type: `Bool`), when `true`, ignores occasionally binding constraints (OBC) even if they are part of the model. This is useful for comparing dynamics with and without OBC.
For models with defined OBC, use the `ignore_obc` argument to ignore them. The following example compares the IRF of the `Gali_2015_chapter_3_obc` model with and without OBC. First, examine the IRF for a 3-standard-deviation `eps_z` shock with OBC enforced. See the `shock_size` and `variables` sections for more details on these input arguments. Since OBC is enforced by default, call:

```julia
plot_irf(Gali_2015_chapter_3_obc,
    shocks = :eps_z,
    variables = [:Y, :R, :Pi, :C],
    shock_size = 3)
```

Then overlay the IRF ignoring the OBC:

```julia
plot_irf!(Gali_2015_chapter_3_obc,
    shocks = :eps_z,
    variables = [:Y, :R, :Pi, :C],
    shock_size = 3,
    ignore_obc = true)
```

![Gali 2015 OBC IRF - eps_z shock comparing with and without OBC](../assets/compare_obc_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

The legend indicates which color corresponds to each `ignore_obc` value. Note how the interest rate `R` hits the effective lower bound in periods 1-3 when OBC is enforced (blue line) but not when OBC is ignored (orange line). The dynamics of other variables also change when OBC is enforced. Enforcing the OBC results in a deeper and longer recession. The length of the lower bound period depends on the size of the shock.

### Generalized Impulse Response Functions

The `generalised_irf` argument (default: `false`, type: `Bool`), when `true`, calculates generalized IRFs (GIRFs) instead of standard IRFs. GIRFs are computed by simulating the model with and without the shock, taking the difference, and averaging over multiple draws. GIRFs are particularly useful for models with non-linearities or state-dependent dynamics, such as higher-order solutions or models with occasionally binding constraints (OBC).

The following example compares the IRF of the `Gali_2015_chapter_3_obc` model for a 3-standard-deviation `eps_z` shock with and without using `generalised_irf`. First, examine the GIRF:

```julia
plot_irf(Gali_2015_chapter_3_obc,
    generalised_irf = true,
    shocks = :eps_z,
    variables = [:Y, :R, :Pi, :C],
    shock_size = 3)
```

![Gali 2015 OBC IRF - eps_z shock GIRF](../assets/obc_girf_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

then overlay the standard IRF:

```julia
plot_irf!(Gali_2015_chapter_3_obc,
    shocks = :eps_z,
    variables = [:Y, :R, :Pi, :C],
    shock_size = 3)
```

![Gali 2015 OBC IRF - eps_z shock comparing GIRF vs standard](../assets/obc_girf_compare_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

The legend indicates which color corresponds to each `generalised_irf` value. Note how the interest rate `R` hits the effective lower bound in periods 1-3 when using the standard IRF (orange line). This suggests that the GIRF's accepted draws include many cases where the OBC is not binding. This can be confirmed by also overlaying the IRF ignoring the OBC.

```julia
plot_irf!(Gali_2015_chapter_3_obc,
    shocks = :eps_z,
    variables = [:Y, :R, :Pi, :C],
    shock_size = 3,
    ignore_obc = true)
```

![Gali 2015 OBC IRF - eps_z shock GIRF vs standard vs no OBC](../assets/obc_all_compare_irf__Gali_2015_chapter_3_obc__eps_z__1.png)

The IRF ignoring the OBC shows `R` falling more, confirming that the GIRF draws include cases where the OBC is binding. Enforcing the OBC results in a deeper and longer recession. The length of the lower bound period depends on the size of the shock.

Another use case for GIRFs is examining the IRF of a model with a higher-order solution. The following example examines the IRF of the `Gali_2015_chapter_3_nonlinear` model solved with pruned second-order perturbation for a 1-standard-deviation `eps_a` shock with and without using `generalised_irf`. First, examine the GIRF:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    generalised_irf = true,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

![Gali 2015 IRF - eps_a shock GIRF (pruned 2nd order)](../assets/girf_2nd_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

Some lines are quite jittery, highlighting the state-dependent nature of the GIRF and the dominant effect of randomness (e.g., `N` or `MC`).

Now overlay the standard IRF for the pruned second order solution:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

![Gali 2015 IRF - eps_a shock GIRF vs standard (pruned 2nd order)](../assets/girf_compare_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The comparison reveals that the response of `S` is highly state-dependent and can go either way depending on the economy's state when the shock occurs. The same applies to `W_real`, while other variables are less state-dependent, making the GIRF and standard IRF more similar.

### GIRF Configuration

The `generalised_irf_draws` (default: `50`, type: `Int`) and `generalised_irf_warmup_iterations` (default: `100`, type: `Int`) arguments control the number of draws and warmup iterations. Increasing the number of draws improves GIRF accuracy but increases computation time. Warmup iterations ensure that the starting points of individual draws adequately explore the state space and represent the model's ergodic distribution.

Start with the GIRF that had the wiggly lines above:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    generalised_irf = true,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

then overlay the GIRF with 1000 draws:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    generalised_irf = true,
    generalised_irf_draws = 1000,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

The lines are now less jittery due to the increased number of draws:

![Gali 2015 IRF - eps_a shock GIRF with 1000 draws](../assets/girf_1000_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

then overlay the GIRF with 5000 draws:

```julia
plot_irf!(Gali_2015_chapter_3_nonlinear,
    generalised_irf = true,
    generalised_irf_draws = 5000,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

The lines are even smoother with the further increase in draws:

![Gali 2015 IRF - eps_a shock GIRF with 5000 draws](../assets/girf_5000_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

To fully cover the model's ergodic distribution, consider increasing the number of warmup iterations as well. The following example overlays the standard IRF for the pruned second-order solution with the GIRF using 5000 draws and 500 warmup iterations:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :pruned_second_order)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    generalised_irf = true,
    generalised_irf_draws = 5000,
    generalised_irf_warmup_iterations = 500,
    shocks = :eps_a,
    algorithm = :pruned_second_order)
```

![Gali 2015 IRF - eps_a shock GIRF with 5000 draws and 500 warmup](../assets/girf_5000_500_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

With this configuration, the difference between the GIRF and standard IRF is minimal. This suggests minimal state dependence in the model with a second-order pruned solution for a one-standard-deviation `eps_a` shock. The apparent insight from the GIRF with 100 draws and 50 warmup iterations was primarily driven by randomness.

### Plot Labels

The `label` argument (type: `Union{String,Symbol,Real}`) controls labels that appear in plots when using the `plot_irf!` function to overlay multiple IRFs. By default, labels are sequential numbers, but custom labels can be provided using this argument. Acceptable inputs are a `String`, `Symbol`, or a Real.

Custom labels are particularly useful when inputs differ in complex ways (e.g., shock matrices or multiple input changes).
For example, let's compare the IRF of the `Gali_2015_chapter_3_nonlinear` model for a 1 standard deviation `eps_a` shock with `β = 0.99` and `τ = 0` to the IRF with `β = 0.95` and `τ = 0.5` using custom labels `String` input:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.99, :τ => 0.0),
    shocks = :eps_a,
    label = "Std. params")

plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.95, :τ => 0.5),
    shocks = :eps_a,
    label = "Alt. params")
```

![Gali 2015 IRF - eps_a shock with custom labels](../assets/custom_labels_irf__Gali_2015_chapter_3_nonlinear__eps_a__2.png)

The legend now displays the custom label names instead of sequential numbers (1 and 2). Additionally, the tables showing input differences and steady states use the custom labels in the first column instead of sequential numbers.

The same result can be achieved using Symbols (though they are less expressive):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.99, :τ => 0.0),
    shocks = :eps_a,
    label = :standard)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.95, :τ => 0.5),
    shocks = :eps_a,
    label = :alternative)
```

or with Real inputs:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.99, :τ => 0.0),
    shocks = :eps_a,
    label = 0.99)

plot_irf!(Gali_2015_chapter_3_nonlinear,
    parameters = (:β => 0.95, :τ => 0.5),
    shocks = :eps_a,
    label = 0.95,
    save_plots = true,
    save_plots_format = :svg)
```

### Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) accepts a dictionary of attributes passed on to the plotting function. See the Plots.jl documentation for details.

The color palette can be customized using the `plot_attributes` argument. The following example defines a custom color palette (inspired by the European Commission's economic reports) to plot and stack all IRFs for the `Gali_2015_chapter_3_nonlinear` model:
First, define the custom color palette using hex color codes:

```julia
ec_color_palette =
[
	"#FFD724", 	# "Sunflower Yellow"
	"#353B73", 	# "Navy Blue"
	"#2F9AFB", 	# "Sky Blue"
	"#B8AAA2", 	# "Taupe Grey"
	"#E75118", 	# "Vermilion"
	"#6DC7A9", 	# "Mint Green"
	"#F09874", 	# "Coral"
	"#907800"  	# "Olive"
]
```

Next, retrieve all shocks defined in the model:

```julia
shocks = get_shocks(Gali_2015_chapter_3_nonlinear)
```

Then plot the IRF for the first shock:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = shocks[1])
```

Finally, overlay the IRFs for the remaining shocks using the custom color palette:

```julia
for s in shocks[2:end]
    plot_irf!(Gali_2015_chapter_3_nonlinear,
        shocks = s,
        plot_attributes = Dict(:palette => ec_color_palette),
        plot_type = :stack)
end
```

![Gali 2015 IRF - all shocks with custom color palette](../assets/custom_colors_irf__Gali_2015_chapter_3_nonlinear__multiple_shocks__2.png)

The colors of the shocks now follow the custom color palette.

Other attributes such as the font family can also be modified (see [here](https://github.com/JuliaPlots/Plots.jl/blob/v1.41.1/src/backends/gr.jl#L61) for options):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    plot_attributes = Dict(:fontfamily => "computer modern"))
```

![Gali 2015 IRF - eps_a shock with custom font](../assets/custom_font_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

All text in the plot now uses the Computer Modern font. Note that font rendering inherits the constraints of the plotting backend (GR in this case)—for example, the superscript + is not rendered properly for this font.

### Plots Per Page

The `plots_per_page` argument (default: `6`, type: `Int`) controls the number of subplots per page. When the number of variables exceeds this value, multiple pages are created.
The following example selects 9 variables and sets `plots_per_page` to 2:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    variables = [:Y, :Pi, :R, :C, :N, :W_real, :MC, :i_ann, :A],
    shocks = :eps_a,
    plots_per_page = 2)
```

![Gali 2015 IRF - eps_a shock (2 plots per page)](../assets/two_per_page_irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png)

The first page displays the first two variables (sorted alphabetically) with two subplots for each shock. The title indicates this is page 1 of 5.

### Display Plots

The `show_plots` argument (default: `true`, type: `Bool`), when `true`, displays the plots; otherwise, they are only returned as an object.

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    show_plots = false)
```

### Saving Plots

The `save_plots` argument (default: `false`, type: `Bool`), when `true`, saves the plots to disk; otherwise, they are only displayed and returned as an object. 

Related arguments control the saving behavior:

- `save_plots_format` (default: `:pdf`, type: `Symbol`): output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `save_plots_path` (default: `"."`, type: `String`): path where plots are saved. If the path does not exist, it will be created automatically.
- `save_plots_name` (default: `"irf"`, type: `Union{String, Symbol}`): prefix prepended to the filename when saving plots to disk.

Each plot is saved as a separate file with a name indicating the prefix, model name, shocks, and a sequential number for multiple plots (e.g., `irf__ModelName__shock__1.pdf`).

The following example saves all IRFs for the `Gali_2015_chapter_3_nonlinear` model as PNG files in the `../plots` directory with the filename prefix `:impulse_response`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    save_plots = true,
    save_plots_format = :png,
    save_plots_path = "./../plots",
    save_plots_name = :impulse_response)
```

The plots appear in the specified folder with the specified prefix. Each plot is saved in a separate file with a name reflecting the model, the shock, and a sequential index when the number of variables exceeds the plots per page.

### Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps variable or shock symbols to custom display names in plots. This is particularly useful when comparing models with different variable naming conventions, allowing them to be displayed with consistent labels.

For example, to rename variables for clearer display:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    rename_dictionary = Dict(:Y => "Output", :Pi => "Inflation", :R => "Interest Rate"))
```

This feature is especially valuable when overlaying IRFs from different models. Consider comparing FS2000 (which uses lowercase variable names like `c`) with Gali_2015_chapter_3_nonlinear (which uses uppercase like `C`). The `rename_dictionary` allows harmonizing these names when plotting them together:

```julia
# First model (FS2000) with lowercase variable names
plot_irf(FS2000,
    shocks = :e_a,
    rename_dictionary = Dict(:c => "Consumption", :y => "Output", :R => "Interest Rate"))

# Overlay second model (Gali_2015_chapter_3_nonlinear) with different naming, mapped to same display names
plot_irf!(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    rename_dictionary = Dict(:C => "Consumption", :Y => "Output", :R => "Interest Rate"))
```

Both models now appear in the plot with consistent, readable labels, making comparison straightforward.

The `rename_dictionary` also works with shocks. For example, Gali_2015_chapter_3_nonlinear has shocks `eps_a` and `nu`, while FS2000 has `e_a` and `e_m`. To compare these with consistent labels:

```julia
# Gali model with shocks eps_a and nu
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = [:eps_a, :nu],
    rename_dictionary = Dict(:eps_a => "Technology Shock", :nu => "Monetary Policy Shock"))

# FS2000 model with shocks e_a and e_m  
plot_irf!(FS2000,
    shocks = [:e_a, :e_m],
    rename_dictionary = Dict(:e_a => "Technology Shock", :e_m => "Monetary Policy Shock"))
```

The `rename_dictionary` accepts flexible type combinations for keys and values—both `Symbol` and `String` types work interchangeably:

```julia
# All of these are valid and equivalent:
Dict(:Y => "Output")              # Symbol key, String value
Dict("Y" => "Output")             # String key, String value
Dict(:Y => :Output)               # Symbol key, Symbol value
Dict("Y" => :Output)              # String key, Symbol value
```

This flexibility is particularly useful for models like Backus_Kehoe_Kydland_1992, which uses both internal symbol representations and more accessible string names with special characters:

```julia
# Define the Backus model (abbreviated for clarity)
@model Backus_Kehoe_Kydland_1992 begin
    # ... (model equations)
end

@parameters Backus_Kehoe_Kydland_1992 begin
    # ... (parameter values)
end

# Backus model example showing Symbol to String mapping
plot_irf(Backus_Kehoe_Kydland_1992,
    shocks = Symbol("E{H}"),
    rename_dictionary = Dict(Symbol("C{H}") => "Home Consumption", 
                             Symbol("C{F}") => "Foreign Consumption",
                             Symbol("Y{H}") => "Home Output",
                             Symbol("Y{F}") => "Foreign Output"))
```

This flexibility allows natural usage regardless of whether variables are referenced as symbols or strings in the code. Variables or shocks not included in the dictionary retain their default names. The renaming applies to all plot elements including legends, axis labels, and tables.

### Verbose Output

The `verbose` argument (default: `false`, type: `Bool`), when `true`, enables verbose output related to solving the model

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    verbose = true)
```

The code outputs information about solving the steady state blocks.
When parameters change, the first-order solution is recomputed; otherwise, it uses the cached solution:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    parameters = :β => 0.955,
    verbose = true)
```

### Numerical Tolerances

The `tol` argument (default: `Tolerances()`, type: `Tolerances`) defines various tolerances for the algorithm used to solve the model. See the Tolerances documentation for more details: `?Tolerances`
The tolerances used by the numerical solvers can be adjusted. The Tolerances object allows setting tolerances for the non-stochastic steady state solver (NSSS), Sylvester equations, Lyapunov equation, and quadratic matrix equation (QME). For example, to set tighter tolerances (this example also changes parameters to force recomputation):

```julia
custom_tol = Tolerances(qme_acceptance_tol = 1e-12,
    sylvester_acceptance_tol = 1e-12)

plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    tol = custom_tol,
    algorithm = :second_order,
    parameters = :β => 0.9555,
    verbose = true)
```

This is useful when higher precision is needed or when the default tolerances are insufficient for convergence. Use this argument for specific needs or encounter issues with the default solver.

### Quadratic Matrix Equation Solver

The `quadratic_matrix_equation_algorithm` argument (default: `:schur`, type: `Symbol`) specifies the algorithm to solve quadratic matrix equation (`A * X ^ 2 + B * X + C = 0`). Available algorithms: `:schur`, `:doubling`
The quadratic matrix equation solver is used internally when solving the model to first order. Different algorithms are available. The `:schur` algorithm is generally faster and more reliable, while `:doubling` can be more precise in some cases (this example also changes parameters to force recomputation):

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    quadratic_matrix_equation_algorithm = :doubling,
    parameters = :β => 0.95555,
    verbose = true)
```

For most use cases, the default `:schur` algorithm is recommended. Use this argument for specific needs or encounter issues with the default solver.

### Sylvester Equation Solver

[Default: selector that uses `:doubling` for smaller problems and switches to `:bicgstab` for larger problems, Type: `Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}}`]: Algorithm to solve the Sylvester equation (`A * X * B + C = X`). Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:dqgmres`, `:gmres`. The input argument can contain up to two elements in a `Vector` or `Tuple`. The first (second) element corresponds to the second (third) order perturbation solution's Sylvester equation. When only one element is provided, it corresponds to the second-order perturbation solution's Sylvester equation.
The algorithm to use can be specified for solving Sylvester equations in higher-order solutions. For example, select the `:bartels_stewart` algorithm for solving the second-order perturbation problem:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :second_order,
    sylvester_algorithm = :bartels_stewart,
    verbose = true)
```

For third-order solutions, different algorithms can be specified for the second- and third-order Sylvester equations using a `Tuple`:

```julia
plot_irf(Gali_2015_chapter_3_nonlinear,
    shocks = :eps_a,
    algorithm = :third_order,
    sylvester_algorithm = (:doubling, :bicgstab),
    verbose = true)
```

The choice of algorithm affects both speed and precision: `:doubling` and `:bartels_stewart` are generally faster, while `:bicgstab`, `:dqgmres`, and `:gmres` are better for large sparse problems. Use this argument for specific needs or encounter issues with the default solver.
