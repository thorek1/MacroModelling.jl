# Project Context (On-Demand)

Read this file only when project background or codebase orientation is needed.

## Overview

`MacroModelling.jl` is a Julia package for developing and solving dynamic stochastic general equilibrium (DSGE) models.

Key capabilities:

- Parse models with time-indexed syntax (`[0]`, `[-1]`, `[1]`)
- Solve models automatically from equations and parameters
- Compute first-, second-, and third-order (pruned) perturbation solutions
- Handle occasionally binding constraints
- Compute IRFs, simulations, and conditional forecasts
- Estimate models using gradient-based samplers (NUTS/HMC) or inversion filters
- Differentiate solutions and moments w.r.t. parameters

Target audience: central banks, regulators, graduate students, and researchers.

Timing convention: end-of-period.

## High-Level Repository Structure

```text
MacroModelling.jl/
├── src/                    # Core package code
├── test/                   # Test suite
├── models/                 # Example DSGE models
├── docs/                   # Documenter-based docs
├── benchmark/              # Benchmark scripts
└── ext/                    # Package extensions
```

Common files in `src/`:

- `MacroModelling.jl` (module/exports/types)
- `macros.jl` (`@model`, `@parameters`)
- `get_functions.jl` (user-facing API)
- `perturbation.jl` (1st-3rd order solvers)
- `moments.jl`, `structures.jl`, `options_and_caches.jl`
- `dynare.jl`, `inspect.jl`, `solver_parameters.jl`, `default_options.jl`
- `algorithms/`, `filter/`, `custom_autodiff_rules/`

## Model Syntax Quick Reference

- Variables use time indices: `...[2], [1], [0], [-1], [-2]...`
- Shocks use `[x]`: `eps_z[x]`
- Calibration equations use `|` in `@parameters`
- Custom steady state can be provided via `steady_state_function`

## Design Considerations

- Performance is critical (type stability and allocations matter)
- Symbolic stack uses Symbolics.jl and SymPyPythonCall
- Supports forward/reverse AD for parameter gradients
- Thread safety matters for estimation workloads
