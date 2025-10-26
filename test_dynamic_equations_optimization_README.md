# Dynamic Equations Nonlinear Solver Test Script

This script demonstrates how to use the `get_dynamic_residuals` function to solve for variables across stochastic shock realizations using NonlinearSolve.jl.

## Purpose

The script shows an advanced use case where:
1. A model is loaded and solved
2. Stochastic and non-stochastic steady states are computed
3. Shock draws are generated using Sobol quasi-random sequences
4. Variables (past, present, future) are solved to zero out average residuals across all shock realizations

## Requirements

```julia
using MacroModelling
using QuasiMonteCarlo
using NonlinearSolve
using SpecialFunctions
using Statistics
```

Install missing packages with:
```julia
using Pkg
Pkg.add("QuasiMonteCarlo")
Pkg.add("NonlinearSolve")
Pkg.add("SpecialFunctions")
```

## Usage

```bash
julia test_dynamic_equations_optimization.jl
```

## What it does

1. **Model Setup**: Defines and solves a simple RBC model
2. **Steady States**: Computes both stochastic and non-stochastic steady states
3. **Shock Generation**: Uses Sobol sequences from QuasiMonteCarlo.jl to generate quasi-random shock draws, then transforms them to standard normal distribution
4. **Nonlinear Solving**: Uses NonlinearSolve.jl to find values for past, present, and future variables that zero out average residuals across all shock draws
5. **Results**: Displays solved variables and compares them with steady states

## Key Features

- Uses **Sobol sequences** for better coverage of the shock space compared to random sampling
- Transforms uniform Sobol draws to **standard normal** distribution using inverse error function
- Solves for **all time dimensions** (past, present, future) simultaneously
- Uses **non-stochastic steady state** for the steady_state input
- Returns **per-equation average residuals** compatible with NonlinearSolve.jl
- Evaluates dynamic equations efficiently using the pre-compiled function
- Uses **Newton-Raphson** method with automatic differentiation

## Expected Output

The script will:
- Display model dimensions
- Show initial and final residual norms
- Print solved variables
- Compare solved values with steady states
- Display per-draw residuals at the solution

## Notes

- The nonlinear solver finds variables that satisfy the dynamic equations (on average) across many shock realizations
- This approach could be used for:
  - Computing stochastic steady states numerically
  - Analyzing equilibria under uncertainty
  - Implementing custom solution algorithms
  - Model estimation with moment matching
- The residual function returns per-equation averages (not a scalar), making it compatible with nonlinear solvers
