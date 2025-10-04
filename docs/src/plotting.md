# Plotting

MacroModelling.jl integrates a comprehensive plotting toolkit on top of [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl). The plotting API is exported together with the modelling macros, so once you define a model you can immediately visualise impulse responses, simulations, conditional forecasts, model estimates, variance decompositions, and policy functions. All plotting functions live in the `StatsPlotsExt` extension, which is loaded automatically when StatsPlots is available.

## Setup

Load the packages once per session:

```julia
using MacroModelling, StatsPlots
```

The helpers return a `Vector{Plots.Plot}`, so you can post-process or save figures with the standard StatsPlots / Plots APIs.

## IRFs at a glance (quick start)

A minimal call computes standard impulse responses for **every exogenous shock** and **every endogenous variable**, using the model’s default solution (first-order perturbation) and a **one-standard-deviation positive** shock. The example below reproduces the RBC tutorial illustration.

```julia
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1)  (1 - δ))
    c[0]  k[0] = (1 - δ) * k[-1]  q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρᶻ * z[-1]  σᶻ * ϵᶻ[x]
end

@parameters RBC begin
    σᶻ = 0.01
    ρᶻ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

plot_irf(RBC)
```

![Default RBC impulse responses](assets/irf__RBC__eps_z__1.png)

### Defaults to remember

- `periods = 40` → the x‑axis spans 40 periods.
- `shocks = :all_excluding_obc` → plots all shocks not used to enforce OBCs.
- `variables = :all_excluding_auxiliary_and_obc` → hides auxiliary and OBC bookkeeping variables.

The plot shows every endogenous variable affected by each exogenous shock and annotates the title with the model name, shock identifier, sign of the impulse (positive by default), and the page indicator (e.g. `(1/2)`). Each subplot overlays the steady state as a horizontal reference line (non‑stochastic for first‑order solutions, stochastic otherwise) and, when the variable is strictly positive, adds a secondary axis with percentage deviations.

## Selecting shocks, variables, and horizon

You can focus the figure on a subset of shocks, select specific variables, and change the horizon. The `shocks` keyword accepts `Symbol`s, collections (vectors/tuples), string equivalents, `:simulate`, or explicit shock paths via matrices and `AxisKeys.KeyedArray`s. Variable names can be passed as symbols or strings.

```julia
# Deterministic shock sequence with a focused variable set and shorter horizon
shock_path = zeros(1, 16)
shock_path[1, 2] = 0.5
shock_path[1, 8] = -0.25
plot_irf(RBC;
         shocks = shock_path,
         variables = [:c, :k],
         periods = 16,
         plots_per_page = 2)
```

![Custom shock path and variable selection](assets/irf__RBC__shock_matrix__1.png)

The horizon automatically extends when the provided shock matrix is longer than `periods`. To switch from IRFs to **stochastic simulations**, set `shocks = :simulate` and choose the horizon with `periods`.

## Changing parameters and comparing runs

Temporarily override parameter values with `parameters`. Provide a single `Pair`, a vector of pairs, or a dictionary from parameter symbols to values.

```julia
plot_irf(RBC; parameters = [:α => 0.65, :β => 0.98])
```

![Impulse responses after a parameter change](assets/irf__RBC_new__eps_z__1.png)

To compare scenarios in the **same** figure, call the mutating companion [`plot_irf!`](@ref) after the initial plot and supply `label` entries. New lines are overlaid (or stacked if `plot_type = :stack`) using the StatsPlots backend.

```julia
plot_irf(RBC; label = "baseline")
plot_irf!(RBC; parameters = :α => 0.6, label = "higher capital share")
```

![Overlaying IRFs](assets/irf__SW03_new__eta_R__1.png)

## Shock magnitude, sign, and higher-order algorithms

- `shock_size` rescales the one‑σ impulse;  
- `negative_shock = true` flips its sign.

These are helpful for stress testing or exploring asymmetries (e.g., with OBCs).

`plot_irf` supports all perturbation orders implemented by the solver; select them via:

```julia
plot_irf(RBC; algorithm = :pruned_second_order)
```

Available options include `:first_order`, `:second_order`, `:third_order`, `:pruned_second_order`, and `:pruned_third_order`.

## Generalised IRFs and stochastic starting points

Set `generalised_irf = true` to compute **generalised impulse responses** (Monte Carlo averages over simulated histories). Control the workflow with:

- `generalised_irf_warmup_iterations` (burn‑in length),
- `generalised_irf_draws` (number of draws).

For higher‑order solutions, a tailored `initial_state` (or a vector of state arrays for pruned second/third order) is often informative when studying IRFs around non‑zero deviations from steady state.

## Occasionally binding constraints (OBCs)

When models include OBCs, the helper injects the constraint shocks by default so that the displayed paths respect the imposed limits. Override with `ignore_obc = true` to study the unconstrained response to the same disturbance.

```julia
plot_irf(Gali_2015_chapter_3_obc; shocks = :eps_z, parameters = :R̄ => 1.0)
plot_irf(Gali_2015_chapter_3_obc;
         shocks = :eps_z,
         parameters = :R̄ => 1.0,
         ignore_obc = true)
```

![IRFs with occasionally binding constraint enforced](assets/Gali_2015_chapter_3_obc__eps_z.png)
![Ignoring the occasionally binding constraint](assets/Gali_2015_chapter_3_obc__simulation__no.png)

## Layout, styling, and persistence

Quality‑of‑life keywords:

- `plots_per_page` controls how many subplots stack in a single figure.
- `plot_attributes::Dict` forwards arbitrary StatsPlots attributes (line styles, palettes, legends, fonts, grid lines, etc.).
- `label` customises legend entries when comparing scenarios.
- `show_plots` (default `true`) suppresses display in batch/documentation builds when set to `false`.
- `save_plots`, `save_plots_format`, and `save_plots_path` persist figures directly from the helper.

Because calls return a `Vector{Plot}`, you can always post‑process or save with the standard APIs.

## Solver and accuracy options

Solver‑related keywords flow into an internal `CalculationOptions` object. Use:

- `verbose` to print progress and diagnostics,
- `tol::Tolerances` to control steady‑state, Sylvester, Lyapunov, and quadratic‑equation tolerances,
- `quadratic_matrix_equation_algorithm`, `sylvester_algorithm`, and `lyapunov_algorithm` to switch linear‑algebra routines (helpful for large systems or when trying iterative solvers).

Defaults already adapt to model size (e.g., large systems may prefer iterative Sylvester solvers).

## Option reference

The table below summarises all `plot_irf` keywords. Defaults correspond to the constants in `MacroModelling.default_options`.

| Keyword | Default | Description |
| --- | --- | --- |
| `periods::Int` | `40` | Number of periods shown on the x‑axis. Automatically increased when shock matrices with more periods are supplied. |
| `shocks` | `:all_excluding_obc` | Shock selector (`Symbol`, vector/tuple of symbols, string inputs, `:simulate`, `Matrix`, or `AxisKeys.KeyedArray`). Matrices and keyed arrays provide fully specified shock paths. |
| `variables` | `:all_excluding_auxiliary_and_obc` | Which endogenous variables to plot. Accepts symbols, strings, and collections thereof. |
| `parameters` | `nothing` | Replacement parameters applied before solving and plotting. Provide a `Pair`, vector of pairs, or a dictionary. |
| `label` | `1` | Legend label attached to this run (relevant when combining plots via `plot_irf!`). |
| `show_plots::Bool` | `true` | Whether to display figures as they are created. Set to `false` for scripted runs. |
| `save_plots::Bool` | `false` | Persist the generated figures to `save_plots_path`. |
| `save_plots_format::Symbol` | `:pdf` | File format passed to `StatsPlots.savefig`. |
| `save_plots_path::String` | `"."` | Output directory used when `save_plots = true`. |
| `plots_per_page::Int` | `9` | Number of subplots per page for the returned figure collection. |
| `algorithm::Symbol` | `:first_order` | Perturbation algorithm (`:first_order`, `:second_order`, `:third_order`, `:pruned_second_order`, `:pruned_third_order`). |
| `shock_size::Real` | `1` | Scale factor applied to the shock standard deviation. |
| `negative_shock::Bool` | `false` | Flips the sign of the impulse. |
| `generalised_irf::Bool` | `false` | Computes generalised IRFs based on Monte‑Carlo simulations. |
| `generalised_irf_warmup_iterations::Int` | `100` | Number of warm‑up simulations discarded when `generalised_irf = true`. |
| `generalised_irf_draws::Int` | `50` | Number of simulated draws averaged for generalised IRFs. |
| `initial_state` | `[0.0]` | Initial deviation from the reference steady state. Supply a vector of vectors for pruned higher‑order solutions. |
| `ignore_obc::Bool` | `false` | Skip adding the shocks that enforce occasionally binding constraints. |
| `plot_attributes::Dict` | `Dict()` | Additional StatsPlots attributes merged into the plotting recipe. |
| `verbose::Bool` | `false` | Enable detailed logging for the underlying solution routine. |
| `tol::Tolerances` | `Tolerances()` | Numerical tolerances passed to the solver (see [`Tolerances`](@ref)). |
| `quadratic_matrix_equation_algorithm::Symbol` | `:schur` | Algorithm used to solve quadratic matrix equations that arise in higher‑order perturbations. |
| `sylvester_algorithm` | `DEFAULT_SYLVESTER_SELECTOR(𝓂)` | Solver used for Sylvester equations. Accepts a `Symbol`, vector, or tuple to specify second‑ and third‑order choices. |
| `lyapunov_algorithm::Symbol` | `:doubling` | Algorithm used for discrete Lyapunov equations. |

## Related helpers

The following wrappers reuse the same interface:

- [`plot_irf!`](@ref) overlays or stacks multiple runs.
- [`plot_irfs`](@ref) and [`plot_irfs!`](@ref) are aliases of the IRF routines.
- [`plot_simulation`](@ref) / [`plot_simulations`](@ref) run stochastic simulations (equivalent to `shocks = :simulate`).
- [`plot_girf`](@ref) / [`plot_girf!`](@ref) specialise in generalised IRFs.

With these tools you can produce all impulse‑response visualisations used in the tutorials and tailor them for quick calibration feedback, policy experiments, or publication‑ready figures.
