# Dynamic Equations Optimization Test Script

This script demonstrates how to use the `get_dynamic_residuals` function to optimize variables across stochastic shock realizations.

## Purpose

The script shows an advanced use case where:
1. A model is loaded and solved
2. Stochastic and non-stochastic steady states are computed
3. Shock draws are generated using Sobol quasi-random sequences
4. Variables (past, present, future) are optimized to minimize average residuals across all shock realizations

## Requirements

```julia
using MacroModelling
using QuasiMonteCarlo
using Optimization
using OptimizationOptimJL
```

Install missing packages with:
```julia
using Pkg
Pkg.add("QuasiMonteCarlo")
Pkg.add("Optimization")
Pkg.add("OptimizationOptimJL")
```

## Usage

```bash
julia test_dynamic_equations_optimization.jl
```

## What it does

1. **Model Setup**: Defines and solves a simple RBC model
2. **Steady States**: Computes both stochastic and non-stochastic steady states
3. **Shock Generation**: Uses Sobol sequences from QuasiMonteCarlo.jl to generate quasi-random shock draws
4. **Optimization**: Minimizes the sum of squared residuals across all shock draws by choosing optimal values for past, present, and future variables
5. **Results**: Displays optimized variables and compares them with steady states

## Key Features

- Uses **Sobol sequences** for better coverage of the shock space compared to random sampling
- Optimizes over **all time dimensions** (past, present, future) simultaneously
- Uses **non-stochastic steady state** for the steady_state input (as requested)
- Evaluates dynamic equations efficiently using the pre-compiled function
- Shows how to integrate the dynamic equations function with optimization workflows

## Expected Output

The script will:
- Display model dimensions
- Show initial and final objective values
- Print optimized variables
- Compare optimized values with steady states
- Display residuals at the optimal point

## Notes

- The optimization finds variables that best satisfy the dynamic equations across many shock realizations
- This approach could be used for:
  - Finding robust policies
  - Analyzing stochastic equilibria
  - Implementing custom solution algorithms
  - Estimating models with moment matching
