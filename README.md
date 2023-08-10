# MacroModelling.jl

**Documentation**: [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://thorek1.github.io/MacroModelling.jl/stable)

[![codecov](https://codecov.io/gh/thorek1/MacroModelling.jl/branch/main/graph/badge.svg?token=QOANGF5MSX)](https://codecov.io/gh/thorek1/MacroModelling.jl)
[![CI](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml)
<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://thorek1.github.io/MacroModelling.jl/dev) -->
[![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/MacroModelling)](https://pkgs.genieframework.com?packages=MacroModelling).


**Author: Thore Kockerols (@thorek1)**

`MacroModelling.jl` is a package for developing and solving dynamic stochastic general equilibrium (DSGE) models. The package provides functions for creating, calibrating, simulating and estimating discrete-time DSGE models. These kind of models are typicaly used to decsribe the behaviour of a macroeconomy and are particularly suited for counterfactual analysis (economic policy evaluation) and exploring/quantifying specific mechanisms (academic research). These models are difficult to work with because they consist of a nonlinear system of equations describing a stochastic control problem.

The goal of `MacroModelling.jl` is to reduce coding time and speed up model development.

As of now the package can:

- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (tries to) solve the model only knowing the model equations and parameter values (no steady state file needed)
- calculate first, second, and third order perturbation solutions using (forward or reverse-mode) automatic differentiation (AD)
- calculate (generalised) impulse response functions, simulate the model, or do conditional forecasts
- calibrate parameters using (non stochastic) steady state relationships
- match model moments
- estimate the model on data (Kalman filter using first order perturbation)
- **differentiate** (forward AD) the model solution, Kalman filter loglikelihood (reverse-mode AD), model moments, steady state, **with respect to the parameters**

The package is not:

- guaranteed to find the non stochastic steady state (solving systems of nonlinear equations is an active area of research)
- the fastest package around if you already have a fast way to find the NSSS (time to first plot is long, time to second plot (with new parameters) is very short)

For more details have a look at the [documentation](https://thorek1.github.io/MacroModelling.jl/stable).

## Getting started

### Installation

`MacroModelling.jl` requires [`julia`](https://julialang.org/downloads/) version 1.8 or higher and an IDE is recommended (e.g. [`VS Code`](https://code.visualstudio.com/download) with the [`julia extension`](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)).

Once set up you can install `MacroModelling.jl` (and `StatsPlots` in order to plot) by typing the following in the Julia REPL:

```julia
using Pkg; Pkg.add(["MacroModelling", "StatsPlots"])
```

### Example

See below an implementation of a simple RBC model. You can find more detailed tutorials in the [documentation](https://thorek1.github.io/MacroModelling.jl/stable).

```julia
using MacroModelling
import StatsPlots

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

plot_irf(RBC)
```

![RBC IRF](docs/src/assets/irf__RBC__eps_z__1.png)

The package contains the following models in the `models` folder:

- [Aguiar and Gopinath (2007)](https://www.journals.uchicago.edu/doi/10.1086/511283) `Aguiar_Gopinath_2007.jl`
- [Ascari and Sbordone (2014)](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.679) `Ascari_sbordone_2014.jl`
- [Baxter and King (1993)](https://www.jstor.org/stable/2117521) `Baxter_and_King_1993.jl`
- [Caldara et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S1094202511000433) `Caldara_et_al_2012.jl`
- [Gali (2015)](https://press.princeton.edu/books/hardcover/9780691164786/monetary-policy-inflation-and-the-business-cycle) - Chapter 3 `Gali_2015_chapter_3_nonlinear.jl`
- [Gali and Monacelli (2005)](https://crei.cat/wp-content/uploads/users/pages/roes8739.pdf) - CPI inflation-based Taylor rule `Gali_Monacelli_2005_CITR.jl`
- [Gerali, Neri, Sessa, and Signoretti (2010)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4616.2010.00331.x) `GNSS_2010.jl`
- [Ghironi and Melitz (2005)](https://faculty.washington.edu/ghiro/GhiroMeliQJE0805.pdf) `Ghironi_Melitz_2005.jl`
- [Ireland (2004)](http://irelandp.com/pubs/tshocksnk.pdf) `Ireland_2004.jl`
- [Jermann and Quadrini (2012)](https://www.aeaweb.org/articles?id=10.1257/aer.102.1.238) - RBC `JQ_2012_RBC.jl`
- [New Area-Wide Model (2008)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp944.pdf) - Euro Area - US `NAWM_EAUS_2008.jl`
- [Schmitt-Grohé and Uribe (2003)](https://www.sciencedirect.com/science/article/abs/pii/S0022199602000569) - debt premium `SGU_2003_debt_premium.jl`
- [Schorfheide (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.582) `FS2000.jl`
- [Smets and Wouters (2003)](https://onlinelibrary.wiley.com/doi/10.1162/154247603770383415) `SW03.jl`
- [Smets and Wouters (2007)](https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586) `SW07.jl`

## Comparison with other packages

||MacroModelling.jl|[dynare](https://www.dynare.org)|[RISE](https://github.com/jmaih/RISE_toolbox)|[NBTOOLBOX](https://github.com/Coksp1/NBTOOLBOX/tree/main/Documentation)|[IRIS](https://iris.igpmn.org)|[DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)|[StateSpaceEcon.jl](https://bankofcanada.github.io/DocsEcon.jl/dev/)|[SolveDSGE.jl](https://github.com/RJDennis/SolveDSGE.jl)|[dolo.py](https://www.econforge.org/dolo.py/)|[DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl)|[gEcon](http://gecon.r-forge.r-project.org)|[GDSGE](https://www.gdsge.com)|[Taylor Projection](https://sites.google.com/site/orenlevintal/taylor-projection)
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
**Host language**|julia|MATLAB|MATLAB|MATLAB|MATLAB|julia|julia|julia|Python|julia|R|MATLAB|MATLAB|
**Non stochastic steady state solver**|*symbolic* or numerical solver of independent blocks; symbolic removal of variables redundant in steady state; inclusion of calibration equations in problem|numerical solver of independent blocks or user-supplied values/functions|numerical solver of independent blocks or user-supplied values/functions|user-supplied steady state file or numerical solver|numerical solver of independent blocks or user-supplied values/functions||numerical solver of independent blocks or user-supplied values/functions|numerical solver|numerical solver or user supplied values/equations|numerical solver or user supplied values/equations|numerical solver; inclusion of calibration equations in problem|||
**Automatic declaration of variables and parameters**|yes|||||||||||||
**Derivatives (Automatic Differentiation) wrt parameters**|yes|||||||||yes - for all 1st, 2nd order perturbation solution related output *if user supplied steady state equations*|||
**Perturbation solution order**|1, 2, 3|k|1 to 5|1|1|1|1|1, 2, 3|1, 2, 3|1, 2|1||1 to 5|
**Automatic derivation of first order conditions**|||||||||||yes||
**Handles occasionally binding constraints**||yes|yes|||yes||yes|yes|||yes||
**Global solution**||||||||yes|yes|||yes||
**Estimation**|yes|yes|yes|yes|yes|yes|||||yes|||
**Balanced growth path**||yes|yes|yes|yes|yes|yes|||||||
**Model input**|macro (julia)|text file|text file|text file|text file|text file|module (julia)|text file|text file|macro (julia)|text file|text file|text file|
**Timing convention**|end-of-period|end-of-period|end-of-period|end-of-period|end-of-period||end-of-period|start-of-period|end-of-period|start-of-period|end-of-period|start-of-period|start-of-period|

## Bibliography

Durbin, J, and Koopman, S. J. (2012), "Time Series Analysis by State Space Methods, 2nd edn", Oxford University Press.

Levintal, O., (2017), "Fifth-Order Perturbation Solution to DSGE models", Journal of Economic Dynamics and Control, 80, pp. 1---16.

Villemot, S., (2011), "Solving rational expectations models at first order: what Dynare does", Dynare Working Papers 2, CEPREMAP.
