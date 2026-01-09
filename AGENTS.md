# Agent Guide for MacroModelling.jl

This file summarizes repository-specific guidance for automated agents.

## Project Overview

MacroModelling.jl is a Julia package for building and solving DSGE models.
Models are defined with `@model` and `@parameters` macros and use time indices
like `[0]` (present), `[1]` (future), `[-1]` (past), and `[x]` (shock).

## Repository Layout

- `src/`: core package code and algorithms
- `models/`: example DSGE models used by tests and docs
- `test/`: targeted tests and model runners
- `benchmark/`: benchmark scripts
- `docs/`: documentation sources
- `ext/`: optional dependency integrations

## Development Setup

- Julia 1.10+ is required (see `README.md` and `docs/src/tutorials/install.md`).
- Typical REPL setup:
  ```julia
  using Pkg
  Pkg.activate(".")
  using MacroModelling
  ```

## Tests

Tests are organized by test sets specified via `TEST_SET`:

```bash
TEST_SET=basic julia --project -e 'using Pkg; Pkg.test()'
TEST_SET=estimation julia --project -e 'using Pkg; Pkg.test()'
```

See `test/runtests.jl`, `test/functionality_tests.jl`, and `test/test_models.jl`
for model-level checks and examples.

## Documentation

Build docs with:

```bash
julia --project=docs docs/make.jl
```

## Benchmarks

Benchmarks are defined in `benchmark/benchmarks.jl` using BenchmarkTools.
Typical run:

```bash
julia --project -e 'using BenchmarkTools; include("benchmark/benchmarks.jl"); run(SUITE)'
```

## Code Map and Conventions

- `src/MacroModelling.jl`: main module, exports, and reserved names list
- `src/macros.jl`: `@model` and `@parameters` macros
- `src/get_functions.jl`: public API functions (`get_irf`, `simulate`, etc.)
- `src/perturbation.jl`: perturbation solution algorithms (1st to 3rd order)
- `src/structures.jl`: core model data structures
- `src/algorithms/`: matrix equation solvers and nonlinear solver
- `src/filter/`: Kalman and inversion filters
- `ext/`: optional integrations (Optim, StatsPlots, Turing)

## Writing Style

- Avoid second-person phrasing (“you”) in docs and docstrings.

## Common Change Points

- New API: add in `src/get_functions.jl` and export from `src/MacroModelling.jl`.
- New model: add a file under `models/` using the model macros.
- Solver changes: look in `src/perturbation.jl` and `src/algorithms/`.
