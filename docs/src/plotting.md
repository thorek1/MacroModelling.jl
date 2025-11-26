# Plotting

MacroModelling.jl integrates a comprehensive plotting toolkit based on [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl). The plotting API is exported alongside the modelling macros, enabling visualization of impulse responses, simulations, conditional forecasts, model estimates, variance decompositions, and policy functions immediately after model definition. All plotting functions are implemented in the `StatsPlotsExt` extension, which loads automatically when importing or using StatsPlots.

## Available Plotting Functions

MacroModelling.jl provides several plotting functions for analyzing and visualizing model behavior:

- **[Impulse Response Functions (IRF)](plot_irf.md)**: Visualize the dynamic response of endogenous variables to exogenous shocks using `plot_irf`
- **[Policy Functions](plot_solution.md)**: Plot the relationship between state variables and endogenous variables using `plot_solution`
- **[Conditional Forecasting](plot_conditional_forecast.md)**: Generate model projections conditional on future paths for endogenous variables or exogenous shocks using `plot_conditional_forecast`
- **[Conditional Variance Decomposition](plot_conditional_variance_decomposition.md)**: Visualize the forecast error variance decomposition (FEVD) showing shock contributions to variable variance using `plot_conditional_variance_decomposition` (also available as `plot_fevd`)
- **[Model Estimates](plot_model_estimates.md)**: Display filtered or smoothed estimates of endogenous variables and exogenous shocks, with optional shock decomposition and unconditional forecasts using `plot_model_estimates` (also available as `plot_shock_decomposition`)
