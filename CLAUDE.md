# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with this repository.

## Project Overview

MacroModelling.jl is a Julia package for developing and solving Dynamic Stochastic General Equilibrium (DSGE) models used in macroeconomic research and policy analysis. It provides tools for model specification, steady-state solving, perturbation solutions (up to 3rd order), estimation, and analysis.

**Author**: Thore Kockerols (@thorek1)
**Documentation**: https://thorek1.github.io/MacroModelling.jl/stable

## Key Commands

### Running Tests

Tests are organized by test sets specified via the `TEST_SET` environment variable:

```bash
# Run basic functionality tests
TEST_SET=basic julia --project -e 'using Pkg; Pkg.test()'

# Run estimation tests
TEST_SET=estimation julia --project -e 'using Pkg; Pkg.test()'

# Other test sets: jet, higher_order_1, higher_order_2, higher_order_3,
# plots_1-5, estimate_sw07, 1st_order_inversion_estimation,
# 2nd_order_estimation, pruned_2nd_order_estimation, 3rd_order_estimation, etc.
```

### Building Documentation

```bash
julia --project=docs docs/make.jl
```

### Loading the Package for Development

```julia
using Pkg
Pkg.activate(".")
using MacroModelling
```

## Architecture Overview

### Source Code (`src/`)

| File | Purpose |
|------|---------|
| `MacroModelling.jl` | Main module, exports, type definitions |
| `macros.jl` | `@model` and `@parameters` macro definitions |
| `get_functions.jl` | Public API functions (`get_irf`, `get_solution`, `simulate`, etc.) |
| `perturbation.jl` | Perturbation solution algorithms (1st, 2nd, 3rd order) |
| `structures.jl` | Core data structures (`ℳ` model type, `timings`, etc.) |
| `moments.jl` | Model moment calculations |
| `dynare.jl` | Dynare compatibility layer |
| `inspect.jl` | Model inspection utilities |
| `default_options.jl` | Default solver options and constants |
| `options_and_caches.jl` | Options structs and caching mechanisms |

### Algorithms (`src/algorithms/`)

- `sylvester.jl` - Sylvester equation solvers
- `lyapunov.jl` - Lyapunov equation solvers
- `quadratic_matrix_equation.jl` - QME solvers (Schur, doubling)
- `nonlinear_solver.jl` - Nonlinear equation solvers for steady state

### Filters (`src/filter/`)

- `kalman.jl` - Kalman filter for linear estimation
- `inversion.jl` - Inversion filter for nonlinear models
- `find_shocks.jl` - Shock finding algorithms for conditional forecasting

### Models (`models/`)

Contains 25+ pre-built DSGE models including RBC variants, Smets-Wouters (2003, 2007), and other academic models. These serve as examples and test cases.

### Extensions (`ext/`)

Optional integrations loaded when dependencies are present:
- `OptimExt.jl` - Optimization with Optim.jl
- `StatsPlotsExt.jl` - Plotting support
- `TuringExt.jl` - Bayesian inference with Turing.jl

## Key Concepts

### Model Definition

Models are defined using two macros:

```julia
@model ModelName begin
    # Equations with time indices: [0] = present, [1] = future, [-1] = past, [x] = shock
    c[0] + k[0] = (1 - δ) * k[-1] + y[0]
    y[0] = z[0] * k[-1]^α
    z[0] = ρ * z[-1] + σ * ε[x]
end

@parameters ModelName begin
    α = 0.33
    δ = 0.025
    # Calibration equations use | syntax
    β | r[ss] = 0.04  # β calibrated so steady-state r = 0.04
end
```

### Core Data Structure

The main model type is `ℳ` (mutable struct) containing:
- Variable/parameter names and values
- Parsed equations (symbolic and compiled)
- Solution matrices (perturbation coefficients)
- Steady state values
- Timings (variable classifications by timing)
- Solver caches and options

### Variable Timing Classification

Variables are classified by their timing in equations:
- `present_only` - Only appear at t=0
- `past_not_future` - Appear at t-1 but not t+1
- `future_not_past` - Appear at t+1 but not t-1
- `mixed` - Appear at both t-1 and t+1

### Solution Algorithms

- `:first_order` - Linear perturbation
- `:second_order` / `:pruned_second_order` - Quadratic perturbation
- `:third_order` / `:pruned_third_order` - Cubic perturbation

## Coding Conventions

### Import Aliases

The codebase uses Unicode aliases for common imports:
```julia
import LinearAlgebra as ℒ
import DifferentiationInterface as 𝒟
import ForwardDiff as ℱ
import LinearSolve as 𝒮
```

### Type Annotations

- Use `Float64` for numerical computations
- `KeyedArray` (from AxisKeys.jl) for labeled arrays in return values
- `SparseMatrixCSC` for sparse Jacobians/Hessians

### Function Documentation

Functions use DocStringExtensions macros:
```julia
"""
$(SIGNATURES)
Description here.
# Arguments
- `arg1`: description
# Keyword Arguments
- `kwarg1`: description
# Returns
- Description of return value
# Examples
```jldoctest
...
```
"""
```

### Reserved Names

The constant `SYMPYWORKSPACE_RESERVED_NAMES` in `MacroModelling.jl` lists names that cannot be used as variables/parameters (mathematical functions like `exp`, `log`, `sin`, etc.).

## Testing Patterns

Tests use Julia's `Test` module with `@testset` blocks:

```julia
@testset verbose = true "Test Name" begin
    include("path/to/model.jl")
    # Test assertions
    @test some_condition
end
```

The `functionality_tests.jl` file contains a comprehensive `functionality_test()` function that tests models across multiple algorithm and solver combinations.

## Common Tasks

### Adding a New Function

1. Add the function to `src/get_functions.jl` (or appropriate file)
2. Export it in `src/MacroModelling.jl`
3. Add docstring with examples
4. Add tests in `test/` directory

### Adding a New Model

1. Create file in `models/` directory
2. Use `@model` and `@parameters` macros
3. Include in relevant test sets

### Modifying Solution Algorithms

Perturbation solutions are in `src/perturbation.jl`. Matrix equation solvers are in `src/algorithms/`.

## Dependencies

Key dependencies (see Project.toml for full list):
- **Symbolic**: SymPyPythonCall, Symbolics.jl
- **Linear Algebra**: LinearAlgebra, SparseArrays, LinearSolve, Krylov, MatrixEquations
- **AD**: ForwardDiff, DifferentiationInterface, ChainRulesCore
- **Optimization**: NLopt
- **Data**: AxisKeys, DataStructures

## CI/CD

GitHub Actions runs tests across:
- Julia versions: 1.10 (min), LTS, latest, pre-release
- OS: Ubuntu, macOS, Windows
- Architectures: x64, arm64

Test sets are parallelized across the matrix. JET.jl is used for static analysis.
