# MacroModelling.jl

**Documentation**: [![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://thorek1.github.io/MacroModelling.jl/stable)

[![codecov](https://codecov.io/gh/thorek1/MacroModelling.jl/branch/main/graph/badge.svg?token=QOANGF5MSX)](https://codecov.io/gh/thorek1/MacroModelling.jl)
[![CI](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thorek1/MacroModelling.jl/actions/workflows/ci.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05598/status.svg)](https://doi.org/10.21105/joss.05598)
[![Package Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FMacroModelling&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/MacroModelling)
<!-- [![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/MacroModelling)](https://pkgs.genieframework.com?packages=MacroModelling) -->
<!-- [![DOI](https://zenodo.org/badge/571475096.svg)](https://zenodo.org/badge/latestdoi/571475096) -->
<!-- [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://thorek1.github.io/MacroModelling.jl/dev) -->

**Author: Thore Kockerols (@thorek1)**

`MacroModelling.jl` is a Julia package for developing and solving dynamic stochastic general equilibrium (DSGE) models.

These kinds of models describe the behaviour of a macroeconomy and are particularly suited for counterfactual analysis (economic policy evaluation) and exploring / quantifying specific mechanisms (academic research). Due to the complexity of these models, efficient numerical tools are required, as analytical solutions are often unavailable. `MacroModelling.jl` serves as a tool for handling the complexities involved, such as forward-looking expectations, nonlinearity, and high dimensionality.

The goal of this package is to reduce coding time and speed up model development by providing functions for working with discrete-time DSGE models. The user-friendly syntax, automatic variable declaration, and effective steady state solver facilitate fast prototyping of models. Furthermore, the package allows the user to work with nonlinear model solutions (up to third order (pruned) perturbation) and estimate the model using gradient based samplers (e.g. NUTS, or HMC). Currently, `DifferentiableStateSpaceModels.jl` is the only other package providing functionality to estimate using gradient based samplers but they use the start-of-period timing convention instead of the end-of-period timing convention used in most other packages. The target audience for the package includes central bankers, regulators, graduate students, and others working in academia with an interest in DSGE modelling.

As of now the package can:

- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (tries to) solve the model only knowing the model equations and parameter values (no steady state file needed; possibility to define custom steady state function)
- calculate **first, second, and third order (pruned) perturbation** solutions using symbolic derivatives
- handle **occasionally binding constraints** for linear and nonlinear solutions
- calculate (generalised) impulse response functions, simulate the model, or do conditional forecasts for linear and nonlinear solutions
- calibrate parameters using (non-stochastic) steady state relationships
- calculate **(unconditional and conditional) variance decompositions** including pruned **second- and third-order** solutions; per-shock contributions can additionally be reported as **marginal contributions (Shapley values)** so that the cross-shock interaction is allocated across the individual shocks
- compute **shock decompositions** of filtered data via the Kalman or inversion filter, including pruned **second- and third-order** solutions where the nonlinear interaction term can optionally be allocated across shocks via **marginal contributions (Shapley values)**
- **match model moments** (also for pruned **higher order** solutions)
- estimate the model on data (Kalman filter using first order perturbation) with **gradient based samplers** (e.g. NUTS, HMC), **estimate nonlinear models** using the inversion filter, or the filter-free joint likelihood
- **differentiate** the model solution, loglikelihood (Kalman and inversion filters), model moments, and steady state **with respect to the parameters** using forward-mode AD ([ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)) and reverse-mode AD ([Mooncake.jl](https://github.com/compintell/Mooncake.jl) recommended; other ChainRules-compatible backends such as Zygote.jl also work via custom rrules)
- **modify a model after it has been defined** — model and calibration equations can be updated, added, or removed in place (`update_equations!`, `add_equation!`, `remove_equation!`, and the `*_calibration_equation!` variants) without having to re-run the `@model` / `@parameters` macros. A chronological revision history is kept (`get_revision_history`); this mirrors the equation-revision workflow familiar from `TROLL`

The package is not:

- guaranteed to find the non-stochastic steady state (solving systems of nonlinear equations is an active area of research)
- the fastest package around if there is already a fast way to find the NSSS (time to first plot is long, time to second plot (with new parameters) is very short)

For more details have a look at the [documentation](https://thorek1.github.io/MacroModelling.jl/stable).

## Speed

MacroModelling.jl is genuinely faster than Dynare at computing perturbation solutions, which makes it well suited for tasks where runtime matters most — in particular estimation.

For first-order solves, Dynare is most competitive when using compiled MATLAB mex files. MacroModelling.jl is still faster in that case, but the gap narrows as model size grows and the QZ decomposition starts to dominate the runtime; at that point performance is largely determined by the underlying BLAS/LAPACK rather than by the surrounding code.

For higher-order perturbation the gap widens substantially. MacroModelling.jl is typically close to an order of magnitude faster at second order, and around two orders of magnitude faster at third order (e.g. roughly `40x`–`115x` on the bundled third-order timings for `Caldara_et_al_2012` and `Gali_2015_chapter_3_nonlinear`). Derivative construction (Jacobians and Hessians) shows even larger relative speedups — often `100x`–`1000x` — but because those operations take only microseconds in absolute terms for most models, they are not the main driver of end-to-end runtime. The speedups that actually change the user experience are in the solve steps themselves.

For full benchmarks across operating systems, CPU architectures, and models, see the [Speed Benchmarks](https://thorek1.github.io/MacroModelling.jl/stable/speed/) page.

Wall-clock user experience is a different story. MacroModelling.jl inherits Julia's just-in-time compilation cost: time-to-first-output can take a minute or more on a small model while the relevant functions are compiled. Once compiled, every subsequent call is extremely fast. Dynare, being built on a compiled host, has the edge on a single cold call, but this advantage erodes as model size grows and disappears entirely as soon as functions are reused.

In practice: if you only want one output from a model once, Dynare will deliver it sooner. If you want to iterate on a model — changing parameters or equations interactively, running estimation, or doing anything that calls the solver many times — MacroModelling.jl is the better fit, thanks to its interactive design, the speed of precompiled functions, and its compatibility with Julia's rich ecosystem of gradient-based samplers.

## Getting started

### Installation

`MacroModelling.jl` requires [`julia`](https://julialang.org/downloads/) version 1.10 or higher and an IDE is recommended (e.g. [`VS Code`](https://code.visualstudio.com/download) with the [`julia extension`](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)).

Once set up `MacroModelling.jl` can be installed (and `StatsPlots` in order to plot) by typing the following in the Julia REPL:

```julia
using Pkg; Pkg.add(["MacroModelling", "StatsPlots"])
```

### Optional extensions

`MacroModelling.jl` uses Julia's package extension mechanism to provide additional functionality when certain packages are loaded. Install the ones relevant for the intended workflow:

```julia
using Pkg; Pkg.add(["Turing", "Mooncake"])         # Bayesian estimation with gradient-based samplers
using Pkg; Pkg.add("ForwardDiff")                   # Forward-mode AD for derivatives of loglikelihood, solutions, IRFs, and moments
using Pkg; Pkg.add("MatrixEquations")               # Bartels-Stewart algorithm for Sylvester/Lyapunov equations
using Pkg; Pkg.add("Optim")                         # LBFGS for conditional forecasts; SAMIN for steady state solver tuning
```

**Automatic differentiation backends:**

- **[Mooncake.jl](https://github.com/compintell/Mooncake.jl)** (reverse-mode) is the recommended backend for gradient-based estimation with Turing.jl (NUTS, HMC). Custom ChainRules `rrule` definitions ensure efficient reverse-mode differentiation through all solvers and filters. Other ChainRules-compatible backends (e.g. Zygote.jl) also work through these same rrules.
- **[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)** (forward-mode) is supported via a package extension and provides `ForwardDiff.jacobian` / `ForwardDiff.gradient` compatibility for `get_solution`, `get_irf`, and `get_statistics` (steady state, mean, variance, standard deviation, covariance, correlation, autocorrelation) across all perturbation orders (first, second, third, and pruned variants), as well as the current likelihood entry points `get_loglikelihood` (Kalman and inversion filter) and `get_filter_free_loglikelihood`.

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

#### Custom steady state function

By default, the package automatically solves for the non-stochastic steady state. For complex models, you can provide a custom steady state function to ensure efficient computation or in case the automatic solver fails:

```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

function rbc_steady_state!(ss, params)
    std_z, rho, delta, alpha, beta = params

    k_ss = ((1 / beta - 1 + delta) / alpha)^(1 / (alpha - 1))
    q_ss = k_ss^alpha
    c_ss = q_ss - delta * k_ss
    z_ss = 0.0

    ss[1] = c_ss
    ss[2] = k_ss
    ss[3] = q_ss
    ss[4] = z_ss
end

@parameters RBC steady_state_function = rbc_steady_state! begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;
```

#### Delayed parameter definition

Parameters do not need to be defined upfront in the `@parameters` block. You can define the model first and provide parameter values later when calling functions. This is useful for loading parameter values from external sources (e.g., CSV files).

```julia
using MacroModelling
import StatsPlots

@model RBC_delayed begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))

    c[0] + k[0] = (1 - δ) * k[-1] + q[0]

    q[0] = exp(z[0]) * k[-1]^α

    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC_delayed begin
end;

plot_irf(RBC_delayed, parameters = (:std_z => 0.01, :ρ => 0.2, :δ => 0.02, :α => 0.5, :β => 0.95))
```

![RBC IRF](docs/src/assets/irf__RBC__eps_z__1.png)

Note: Calibration equations (using `|` syntax) and parameters defined as functions of other parameters must be declared in the `@parameters` block.

Steady state options:

- Automatic solver (default) from the model equations.
- Custom steady state function.

Parameter values can also be supplied later (delayed parameter definition) as illustrated above.

See the documentation for more details on the [steady state](https://thorek1.github.io/MacroModelling.jl/stable/steady_state/).

#### Modifying a model after definition

Model and calibration equations can be edited in place after the `@model` /
`@parameters` block has been evaluated, without having to re-declare the
model. The `update_equations!`, `add_equation!`, and `remove_equation!`
functions (plus their `*_calibration_equation!` counterparts) mutate the
model object, invalidate cached solver results, and re-solve the
non-stochastic steady state. Every change is appended to a revision log
exposed via `get_revision_history`.

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

# Replace the technology-shock process with a more persistent one
update_equations!(RBC, :(z[0] = ρ * z[-1] + std_z * eps_z[x]),
                       :(z[0] = 0.9 * z[-1] + std_z * eps_z[x]))

# Append an auxiliary equation defining log output
add_equation!(RBC, :(log_q[0] = log(q[0])))

# Drop it again
remove_equation!(RBC, :(log_q[0] = log(q[0])))

# Inspect what has been changed
get_revision_history(RBC)
```

See the [how-to guide on modifying models](https://thorek1.github.io/MacroModelling.jl/stable/how-to/modify_equations/)
for details, and calibration-equation examples.

## Models

The package contains the following models in the `models` folder:

- [Aguiar and Gopinath (2007)](https://www.journals.uchicago.edu/doi/10.1086/511283) `Aguiar_Gopinath_2007.jl`
- [Ascari and Sbordone (2014)](https://www.aeaweb.org/articles?id=10.1257/jel.52.3.679) `Ascari_Sbordone_2014.jl`
- [Backus, Kehoe, and Kydland (1992)](https://www.jstor.org/stable/2138686) `Backus_Kehoe_Kydland_1992.jl`
- [Baxter and King (1993)](https://www.jstor.org/stable/2117521) `Baxter_King_1993.jl`
- [Caldara et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S1094202511000433) `Caldara_et_al_2012.jl`
- [Gali (2015)](https://press.princeton.edu/books/hardcover/9780691164786/monetary-policy-inflation-and-the-business-cycle) - Chapter 3 `Gali_2015_chapter_3_nonlinear.jl`
- [Gali and Monacelli (2005)](https://crei.cat/wp-content/uploads/users/pages/roes8739.pdf) - CPI inflation-based Taylor rule `Gali_Monacelli_2005_CITR.jl`
- [Federal Reserve Board U.S. model (FRB/US) - LINVER (2024))](https://www.federalreserve.gov/econres/us-models-about.htm) `FRBUS.jl`
- [Gerali, Neri, Sessa, and Signoretti (2010)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4616.2010.00331.x) `GNSS_2010.jl`
- [Ghironi and Melitz (2005)](https://faculty.washington.edu/ghiro/GhiroMeliQJE0805.pdf) `Ghironi_Melitz_2005.jl`
- [Ireland (2004)](http://irelandp.com/pubs/tshocksnk.pdf) `Ireland_2004.jl`
- [Jermann and Quadrini (2012)](https://www.aeaweb.org/articles?id=10.1257/aer.102.1.238) - RBC `JQ_2012_RBC.jl`
- [New Area-Wide Model (2008)](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp944.pdf) - Euro Area - US `NAWM_EAUS_2008.jl`
- [QUEST3 (2009)](https://www.sciencedirect.com/science/article/abs/pii/S026499930800076X)  `QUEST3_2009.jl`
- [Schmitt-Grohé and Uribe (2003)](https://www.sciencedirect.com/science/article/abs/pii/S0022199602000569) - debt premium `SGU_2003_debt_premium.jl`
- [Schorfheide (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.582) `FS2000.jl`
- [Smets and Wouters (2003)](https://onlinelibrary.wiley.com/doi/10.1162/154247603770383415) `SW03.jl`
- [Smets and Wouters (2007)](https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586) `SW07.jl`

## Comparison with other packages

||MacroModelling.jl|[dynare](https://www.dynare.org)|[DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)|[dolo.py](https://www.econforge.org/dolo.py/)|[SolveDSGE.jl](https://github.com/RJDennis/SolveDSGE.jl)|[DifferentiableStateSpaceModels.jl](https://github.com/HighDimensionalEconLab/DifferentiableStateSpaceModels.jl)|[StateSpaceEcon.jl](https://bankofcanada.github.io/DocsEcon.jl/dev/)|[IRIS](https://iris.igpmn.org)|[RISE](https://github.com/jmaih/RISE_toolbox)|[NBTOOLBOX](https://github.com/Coksp1/NBTOOLBOX/tree/main/Documentation)|[gEcon](http://gecon.r-forge.r-project.org)|[GDSGE](https://www.gdsge.com)|[Taylor Projection](https://sites.google.com/site/orenlevintal/taylor-projection)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Host language**|julia|MATLAB|julia|Python|julia|julia|julia|MATLAB|MATLAB|MATLAB|R|MATLAB|MATLAB|
|**Non-stochastic steady state solver**|*symbolic* or numerical solver of independent blocks; symbolic removal of variables redundant in steady state; inclusion of calibration equations in problem; custom steady state function can be provided|numerical solver of independent blocks or user-supplied values/functions||numerical solver of independent blocks or user-supplied values/functions|numerical solver|numerical solver or user supplied values/equations|numerical solver of independent blocks or user-supplied values/functions|numerical solver of independent blocks or user-supplied values/functions|numerical solver of independent blocks or user-supplied values/functions|user-supplied steady state file or numerical solver|numerical solver; inclusion of calibration equations in problem|||
|**Automatic declaration of variables and parameters**|yes|||||||||||||
|**Derivatives wrt parameters**|yes|||||yes||||||||
|**Perturbation solution order**|1, 2, 3|k|1|1, 2, 3|1, 2, 3|1, 2|1|1|1 to 5|1|1||1 to 5|
|**Pruning**|yes|yes||||yes|||yes|||||
|**Automatic derivation of first order conditions**|||||||||||yes|||
|**Occasionally binding constraints**|yes|yes|yes|yes|yes||||yes|||yes||
|**Global solution**||||yes|yes|||||||yes||
|**Estimation**|yes|yes|yes|||yes||yes|yes|yes|yes|||
|**Balanced growth path**||yes|yes||||yes|yes|yes|yes||||
|**Model input**|macro (julia)|text file|text file|text file|text file|macro (julia)|module (julia)|text file|text file|text file|text file|text file|text file|
|**Timing convention**|end-of-period|end-of-period||end-of-period|start-of-period|start-of-period|end-of-period|end-of-period|end-of-period|end-of-period|end-of-period|start-of-period|start-of-period|

## Bibliography

Andreasen, M. M., Fernández-Villaverde, J., and Rubio-Ramírez, J. F.  (2018), "The pruned state-space system for non-linear DSGE models: Theory and empirical applications.", The Review of Economic Studies, 85.1, p. 1-49.

Durbin, J, and Koopman, S. J. (2012), "Time Series Analysis by State Space Methods, 2nd edn", Oxford University Press.

Levintal, O., (2017), "Fifth-Order Perturbation Solution to DSGE models", Journal of Economic Dynamics and Control, 80, p. 1---16.

Villemot, S., (2011), "Solving rational expectations models at first order: what Dynare does", Dynare Working Papers 2, CEPREMAP.

## Questions and Contributions

If there are questions regarding the usage or unexpected behaviour is encountered please file an issue.

Contributions are very welcome, as are feature requests and suggestions. Please open an issue or even better become a contributor and create a pull request.
