# Model Estimates

`plot_model_estimates` visualizes the variables that enter an estimation problem, the corresponding filtered or smoothed estimates, and—optionally—the contribution of each shock. Every subplot displays an observable (line plot) or the contribution of a shock (stacked bars) measured against the non‑stochastic or stochastic steady state that is relevant for the chosen solution algorithm. The function returns a `Vector{Plots.Plot}` so you can display the figures, save them, or combine them further.

The figures are built with StatsPlots/Plots.jl and expect a `KeyedArray` from the AxisKeys package as data input. Axis 1 must contain the observable names, axis 2 the period labels. Observables are automatically matched to model variables, renamed (if desired), and sorted alphabetically in the plot legends.

## Example Setup

The examples below reuse a simple RBC model. Any other model created with `@model`/`@parameters` can be substituted.

```julia
using MacroModelling, StatsPlots

@model RBC_CME begin
    y[0] = A[0] * k[-1]^alpha
    1 / c[0] = beta * 1 / c[1] * (alpha * A[1] * k[0]^(alpha-1) + (1 - delta))
    1 / c[0] = beta * 1 / c[1] * (R[0] / Pi[1])
    R[0] * beta = (Pi[0] / Pibar)^phi_pi
    A[0] * k[-1]^alpha = c[0] + k[0] - (1 - delta * z_delta[0]) * k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

simulation = simulate(RBC_CME, periods = 200)
data = simulation([:y, :c, :k, :R], :, :simulate)                # KeyedArray (Variables × Periods)
data = rekey(data, :Periods => 1:size(data, 2))                   # optional: replace the period axis labels

plot_model_estimates(RBC_CME, data)                               # returns a Vector{Plot}
```

You can also build a `KeyedArray` from empirical data:

```julia
using AxisKeys, Dates

observations = [:y, :c, :k]
quarters = Date(1960):Month(3):Date(2009, 9)
matrix = Matrix(df[:, observations])'                            # observables × periods

data_empirical = KeyedArray(matrix,
    Variables = observations,
    Periods = collect(quarters))
```

`plot_model_estimates` sorts the variables on the first axis to match the model’s internal order, so the exact ordering of `observations` does not matter, but the provided names must match valid model observables (symbols or strings that can be parsed into symbols).

## Required Arguments

### Model (`𝓂`)
Any model created with `@model`/`@parameters`. The routine internally calls `solve!` with `algorithm` and `parameters`, so you do not have to precompute a solution. If the model includes occasionally binding constraints, they are respected whenever the chosen algorithm supports them.

### Data (`data::KeyedArray{Float64}`)
Two-dimensional `KeyedArray` with observables on axis 1 and periods on axis 2. Axis keys can be `Symbol`s or `String`s; string keys are parsed into symbols using `Meta.parse`, so `:y` and `"y"` are both valid. The second axis labels are displayed on the x-axis (dates, integers, etc.). If the data originate from `simulate`, select the desired observables and slice away the `:simulate` shock dimension, as shown in the example. Use `rekey` to rename axes if necessary.

## Keyword Arguments

### `parameters`
Additional parameter sets. Accepts a vector of parameter values ordered as listed by `get_parameters(model)`, tuples/vectors of `Pair`s (`[:beta => .995, :phi_pi => 1.75]`), matrices of values, or `Dict`s (see `ParameterType`). Whenever the provided values differ from the current calibration, `solve!` recomputes the solution:

```julia
plot_model_estimates(RBC_CME, data,
    parameters = [:beta => .995, :phi_pi => 1.75])
```

### `algorithm`
Solution algorithm (`:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`). Higher-order algorithms require the `:inversion` filter; the helper automatically enforces that constraint.

```julia
plot_model_estimates(RBC_CME, data,
    algorithm = :pruned_second_order)
```

### `filter`
Filtering method. `:kalman` is available for first-order linear models, while `:inversion` works for both linear and nonlinear (higher-order) solutions. Passing an invalid combination is corrected automatically (a message is logged explaining the adjustment).

```julia
plot_model_estimates(RBC_CME, data, filter = :inversion)
```

### `warmup_iterations`
Number of additional simulated periods prepended before the first observation. This option is **only** honored when `algorithm = :first_order` and `filter = :inversion`; otherwise it is ignored with a warning.

```julia
plot_model_estimates(RBC_CME, data,
    filter = :inversion,
    warmup_iterations = 5)
```

### `variables`
Select the endogenous variables to plot. Accepted inputs: a single `Symbol`/`String`, tuples, vectors, matrices of names, or one of the built-in selectors `:all`, `:all_excluding_obc`, or `:all_excluding_auxiliary_and_obc`. Variables not present in the model raise a warning.

```julia
plot_model_estimates(RBC_CME, data,
    variables = [:y, :c, :k])

plot_model_estimates(RBC_CME, data,
    variables = :all_excluding_auxiliary_and_obc)
```

### `shocks`
Select which shocks to display in the decomposition (ignored when `shock_decomposition = false`). Inputs mirror `variables` and accept `:all`, `:all_excluding_obc`, explicit name collections, or `:none` to skip shock plots even if `shock_decomposition = true`.

```julia
plot_model_estimates(RBC_CME, data,
    shock_decomposition = true,
    shocks = [:eps_z, :delta_eps])
```

### `presample_periods`
Drop the first `presample_periods` periods **after** filtering/smoothing to focus on the main sample:

```julia
plot_model_estimates(RBC_CME, data,
    presample_periods = 20)
```

### `data_in_levels`
Set to `false` if the provided data are already expressed as deviations from the steady state. When `true` (default) the data are demeaned using the relevant steady state before filtering.

```julia
plot_model_estimates(RBC_CME, data,
    data_in_levels = false)
```

### `smooth`
Toggle backward smoothing. Only the Kalman filter supports smoothing; requesting `smooth = true` with the inversion filter automatically flips the flag back to `false`.

```julia
plot_model_estimates(RBC_CME, data,
    filter = :kalman,
    smooth = true)
```

### `shock_decomposition`
Controls whether stacked bars with the contribution of each shock, the initial condition, and—when using pruned solutions—the “Nonlinearities” bucket are plotted below the estimates. The default is `true` for first-order and pruned solutions and `false` for unpruned higher-order solutions (the inversion filter cannot provide a consistent decomposition there). The convenience wrapper `plot_shock_decomposition` always sets this flag to `true`.

```julia
plot_shock_decomposition(RBC_CME, data,
    shocks = [:eps_z, :delta_eps])
```

### `label`
Legend label for the entire run. When you later call `plot_model_estimates!` the new label is shown alongside the previous runs:

```julia
plot_model_estimates(RBC_CME, data, label = "Baseline")
plot_model_estimates!(RBC_CME, data[:, 50:end], label = "Post-break")
```

### `show_plots`
Display the plots as they are created. Disable this flag when running batch jobs or saving figures without rendering to the screen.

```julia
plots = plot_model_estimates(RBC_CME, data,
    show_plots = false)
```

### `save_plots`, `save_plots_format`, `save_plots_name`, `save_plots_path`
Save each page of the output (`save_plots = true`). `save_plots_name` becomes part of the filename (`<name>__<model>__<page>.<format>`), `save_plots_format` is any format understood by Plots.jl (e.g. `:pdf`, `:png`), and `save_plots_path` controls the directory (created automatically if it does not exist).

```julia
plot_model_estimates(RBC_CME, data,
    save_plots = true,
    save_plots_format = :png,
    save_plots_name = "sw_estimates",
    save_plots_path = "estimation_output")
```

### `plots_per_page`
Number of subplots before a new page is started (default: 6). Increase this to pack more variables per page or reduce it to emphasize individual series.

```julia
plot_model_estimates(RBC_CME, data,
    plots_per_page = 4)
```

### `transparency`
Alpha channel applied to the stacked bars in the decomposition (0–1). Only relevant when `shock_decomposition = true`.

```julia
plot_shock_decomposition(RBC_CME, data,
    transparency = 0.35)
```

### `max_elements_per_legend_row`
Upper bound on the number of legend entries per row (useful when overlaying multiple runs or showing many shocks). The function automatically computes the number of columns subject to this cap.

```julia
plot_shock_decomposition(RBC_CME, data,
    shocks = :all,
    max_elements_per_legend_row = 6)
```

### `extra_legend_space`
Adds extra vertical space (in fraction of the total plot height) below the subplots for the legend. This is helpful when period labels contain long strings (dates) or when the legend spans multiple rows.

```julia
plot_model_estimates(RBC_CME, data,
    extra_legend_space = 0.1)
```

### `rename_dictionary`
Map variable or shock names to prettier labels. Keys and values can be `Symbol`s or `String`s. All resulting names must be unique because the plots use them as labels.

```julia
plot_model_estimates(RBC_CME, data,
    rename_dictionary = Dict(
        :y => "Output",
        :c => "Consumption",
        :k => "Capital",
        :eps_z => "Technology shock"))
```

### `plot_attributes`
Additional Plots.jl attributes merged into the defaults (size, palette, legend position, etc.). `:framestyle` is kept for the subplots; all other entries are applied to the combined layout.

```julia
plot_model_estimates(RBC_CME, data,
    plot_attributes = Dict(
        :size => (900, 600),
        :palette => :Dark2,
        :legend => :bottom))
```

### `verbose`
When `true`, prints progress information from the steady-state solver, the filtering routines, and the linear algebra backends (quadratic matrix equation, Sylvester, Lyapunov). Useful while debugging convergence problems.

```julia
plot_model_estimates(RBC_CME, data,
    verbose = true)
```

### `tol`
Instance of `Tolerances`, letting you tighten or relax the acceptance thresholds of the steady-state solver and the matrix-equation solvers.

```julia
plot_model_estimates(RBC_CME, data,
    tol = Tolerances(NSSS_xtol = 1e-13, qme_tol = 1e-12))
```

### `quadratic_matrix_equation_algorithm`
Select the solver for the quadratic matrix equation that underpins the linear solution (`:schur`, `:doubling`, `:linear_time_iteration`, `:quadratic_iteration`). The default `:schur` is fast and reliable; `:doubling` is more precise for difficult calibrations.

```julia
plot_model_estimates(RBC_CME, data,
    quadratic_matrix_equation_algorithm = :doubling)
```

### `sylvester_algorithm`
Choose the solver for the Sylvester equations that appear in higher-order solutions. Pass a single symbol (e.g. `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:dqgmres`) to reuse it everywhere, or a tuple/vector with two entries to specify separate algorithms for the first- and third-order Sylvester problems. Large models automatically fallback to `:bicgstab` unless you override this behavior.

```julia
plot_model_estimates(RBC_CME, data,
    sylvester_algorithm = (:doubling, :bicgstab))
```

### `lyapunov_algorithm`
Solver for the Lyapunov equation (`:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`, `:iterative`, `:speedmapping`). The default `:doubling` is typically fastest and most accurate.

```julia
plot_model_estimates(RBC_CME, data,
    lyapunov_algorithm = :bartels_stewart)
```

## Return Value and Side Effects

The function returns a vector of plots (one entry per page). Each page contains the subplots plus a legend panel that lists the line and bar colors, the model name, and the page count. When `show_plots = true`, each page is displayed. When `save_plots = true`, each page is written to disk after it is rendered.

## Comparing Runs with `plot_model_estimates!`

`plot_model_estimates!` shares the same signature but appends its results to the last call of `plot_model_estimates`/`plot_model_estimates!`. Use it to compare different parameterizations, filters, or samples without rebuilding the plot manually.

```julia
baseline = plot_model_estimates(RBC_CME, data,
    parameters = [:beta => .999], label = "β = 0.999")

plot_model_estimates!(RBC_CME, data,
    parameters = [:beta => .995], label = "β = 0.995",
    transparency = 0.6)
```

The helper keeps an internal registry of all arguments so that follow-up calls can use the same variable ordering, labels, and color palette. Remember to change `label` for each overlay when you intend to show multiple runs.

## Shock-Decomposition Wrapper

`plot_shock_decomposition(args...; kwargs...)` is a lightweight wrapper around `plot_model_estimates` with `shock_decomposition = true`. Use it when you only care about the stacked contributions and do not want to retype the keyword argument.
