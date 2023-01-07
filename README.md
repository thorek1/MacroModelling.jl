# MacroModelling.jl

[![codecov](https://codecov.io/gh/thorek1/MacroModelling.jl/branch/main/graph/badge.svg?token=QOANGF5MSX)](https://codecov.io/gh/thorek1/MacroModelling.jl)
[![CI](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://thorek1.github.io/MacroModelling.jl/stable)
<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://thorek1.github.io/MacroModelling.jl/dev) -->

The goal of `MacroModelling.jl` is to reduce coding time and speed up model development.

The package currently supports dicsrete-time DSGE models with end-of-period timing.

As of now the package can:
- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (attempt to) solve the model only knowing the model equations and parameter values (no steady state file, no variable and parameter declaration needed)
- calculate first, second, and third order perturbation solutions using (forward) automatic differentiation (AD)
- calculate (generalised) impulse response functions, and simulate the model
- calibrate parameters using (non stochastic) steady state relationships
- match model moments
- estimate the model on data (kalman filter using first order perturbation)
- **differentiate** (forward AD) the model solution (first order perturbation), kalman filter loglikelihood, model moments, steady state, **with respect to the parameters**

# Getting started
## Installation
`MacroModelling.jl` requires [`julia`](https://julialang.org/downloads/) version 1.8 or higher and an IDE is recommended (e.g. [`VS Code`](https://code.visualstudio.com/download) with the [`julia extension`](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)).

Once set up you can install `MacroModelling.jl` by typing the following in the julia REPL:
```julia
using Pkg; Pkg.add("MacroModelling")
```
## Example
See below for example code of a simple RBC model. For more details see the [documentation](https://thorek1.github.io/MacroModelling.jl/stable).
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

plot_irfs(RBC)
```


# Comparison with other packages
||MacroModelling.jl|[dynare](https://www.dynare.org)|[RISE](https://github.com/jmaih/RISE_toolbox)|[DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)|[StateSpaceEcon.jl](https://bankofcanada.github.io/DocsEcon.jl/dev/)|[SolveDSGE.jl](https://github.com/RJDennis/SolveDSGE.jl)|[dolo.py](https://www.econforge.org/dolo.py/)|[DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl)|[gEcon](http://gecon.r-forge.r-project.org)|[GDSGE](https://www.gdsge.com)
|---|---|---|---|---|---|---|---|---|---|---|
**Host language**|julia|MATLAB|MATLAB|julia|julia|julia|Python|julia|R|MATLAB|
**Non stochastic steady state solver**|*symbolic* or numerical solver of recursive blocks; symbolic removal of variables redundant in steady state; inclusion of calibration equations in problem|numerical solver of recursive blocks or user-supplied values/functions|numerical solver of recursive blocks or user-supplied values/functions||numerical solver of recursive blocks or user-supplied values/functions|numerical solver|numerical solver or user supplied values/equations|numerical solver or user supplied values/equations|numerical solver; inclusion of calibration equations in problem||
**Automatic declaration of variables and parameters**|yes||||||||||
**Derivatives (Automatic Differentiation) wrt parameters**|yes - for all 1st order perturbation solution related output||||||yes - for all 1st, 2nd order perturbation solution related output *if user supplied steady state equations*|||
**Perturbation solution order**|1, 2, 3 |k|1 to 5 |1|1|1, 2, 3 |1, 2, 3 |1, 2 |1||
**Automatic derivation of first order conditions**||||||||yes||
**Handles occasionally binding constraints**||yes|yes|yes||yes|yes|||yes|
**Global solution**||||||yes|yes|||yes|
**Estimation**|yes|yes|yes|yes|||||yes||
**Balanced growth path**||yes|yes|yes|yes||||||
**Model input**|macro (julia)|text file|text file|text file|module (julia)|text file|text file|macro (julia)|text file|text file|
**Timing convention**|end-of-period|end-of-period|end-of-period||end-of-period|start-of-period|end-of-period|start-of-period|end-of-period|start-of-period|


# Bibliography

Levintal, O., (2017), "Fifth-Order Perturbation Solution to DSGE models", Journal of Economic Dynamics and Control, 80, pp. 1---16.

Villemot, S., (2011), "Solving rational expectations models at first order: what Dynare does", Dynare Working Papers 2, CEPREMAP.
