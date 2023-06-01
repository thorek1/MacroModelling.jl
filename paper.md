---
title: 'MacroModelling.jl: A Julia package for developing and solving dynamic stochastic general equilibrium models'
tags:
  - DSGE
  - macroeconomics  
  - perturbation 
  - difference equations
  - dynamical systems
authors:
  - name: Thore Kockerols
    orcid: 0000-0002-0068-1809
    affiliation: "1" 
affiliations:
 - name: Norges Bank, Norway
   index: 1
date: 1 June 2023
bibliography: paper.bib
---

# Summary

`MacroModelling.jl` is a Julia [@Bezanson:2012] package for developing and solving dynamic stochastic general equilibrium (DSGE) models. Its goal is to reduce coding time and speed up model development by providing functions for creating, calibrating, simulating, and estimating discrete-time DSGE models. The package includes several pre-defined models from prominent economic papers, providing an immediate starting point for economic researchers and students.

The package can parse a model written with user-friendly syntax, solve the model knowing only the model equations and parameter values, calculate first, second, and third-order perturbation solutions  [@villemot2011solving; @levintal2017fifth] using symbolic and automatic differentiation. It can also calibrate parameters, match model moments, and estimate the model on data using the Kalman filter [@durbin2012time]. The package is designed to be user-friendly and efficient, enabling users to generate outputs or change parameters rapidly once the functions are compiled and the non-stochastic steady state has been found.

The user can implement a simple real business cycle model with the following code:

```julia
using MacroModelling
import StatsPlots

@model RBC begin
    1 / c[0] = (β / c[1]) * (α * z[1] * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = z[0] * k[-1]^α
    z[0] = (1 - ρ) + ρ * z[-1] + σ_z * eps_z[x]
end

@parameters RBC begin
    σ_z= 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

plot_irf(RBC)
```

![Impulse response to a positive 1 standard deviation shock.\label{fig:irf__RBC__eps_z__1}](irf__RBC__eps_z__1.png)
The plot shows both the level, percent deviation from the non stochastic steady steady (NSSS) as well as the NSSS itself. Note that the code to generate the impulse response function (IRF) plot contains only the equations, parameter values, and the command to plot.

# Statement of need

DSGE models are a type of models used in academia and policy institutions to explain economic phenomena such as business cycles, or the effects of economic policy. The forward looking expectations, nonlinearity, and high dimensionality of these models necessitate efficient numerical tools for their solution, calibration, simulation, and estimation. This is where `MacroModelling.jl` comes in.

This package supports the user especially in the model development phase. The intuitive syntax, automatic variable declaration, and effective steady state solver facilitate fast prototyping of models.

Once the model is solved the package provides user-friendly functions to generate output. The package stands out for its ability to calculate sensitivities of model moments, its automatic variable declaration, and effective steady state solver.

# Acknowledgements

The author wants to thank everybody who opened issues, reported bugs, contributed ideas, and was supportive in driving `MacroModelling.jl` forward.

# References