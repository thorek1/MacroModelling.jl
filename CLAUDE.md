# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`MacroModelling.jl` is a Julia package for developing and solving dynamic stochastic general equilibrium (DSGE) models. It provides first, second, and third order (pruned) perturbation solutions, handles occasionally binding constraints, and supports estimation using gradient-based samplers (NUTS, HMC) or inversion filters.

**Timing convention:** End-of-period (not start-of-period like some other packages).

## Common Commands

### Package Setup
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Running Julia
Always use `julia -t auto` to enable multi-threading.

### Testing

**Do NOT run the full test suite** - it takes too long. Instead:

1. **Quick feature testing** - Write a bespoke script using a simple model:
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Test your changes here
get_irf(RBC)
```

2. **Specific test sets** (CI only) - Set `TEST_SET` environment variable:
   - `basic`, `estimation`, `higher_order_1-3`, `plots_1-5`, `estimate_sw07`, `jet`
   - Various estimation tests: `1st_order_inversion_estimation`, `2nd_order_estimation`, `pruned_2nd_order_estimation`, `3rd_order_estimation`, `pruned_3rd_order_estimation`

### Documentation
```bash
julia --project=docs docs/make.jl
```

### Benchmarking
```julia
include("benchmark/benchmarks.jl")
run(SUITE)
```

## Revise-Based Development Workflow (REQUIRED)

**ALWAYS use Revise.jl for interactive development.** This enables hot-reloading of code changes without restarting Julia, which is essential for efficient iteration.

### Setup Steps

1. **Start Julia REPL** with multi-threading enabled:
   ```bash
   cd /path/to/MacroModelling.jl
   julia -t auto --project=.
   ```

2. **Load Revise FIRST**, then MacroModelling:
   ```julia
   using Revise
   using MacroModelling
   ```

3. **Define a test model** for quick testing:
   ```julia
   @model RBC begin
       1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
       c[0] + k[0] = (1 - δ) * k[-1] + q[0]
       q[0] = exp(z[0]) * k[-1]^α
       z[0] = ρ * z[-1] + std_z * eps_z[x]
   end

   @parameters RBC begin
       std_z = 0.01
       ρ = 0.2
       δ = 0.02
       α = 0.5
       β = 0.95
   end
   ```

### Development Workflow

1. **Keep the Julia REPL running** throughout the session - never restart between edits
2. **Edit source files** in `src/` directory
3. **Revise automatically detects changes** and recompiles only affected functions
4. **Test changes immediately** in the same REPL session
5. **Iterate rapidly** - edit, test, fix, repeat without restarting

### Practical Example

```julia
# Initial call (before any edits)
julia> get_equations(RBC)
4-element Vector{String}:
 "1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0] ^ (α - 1) + (1 - δ))"
 ...

# Now edit src/inspect.jl to add a print statement:
# println("🔍 get_equations called - Revise is working!")
# Save the file - Revise detects the change automatically

# Call again - no restart needed!
julia> get_equations(RBC)
🔍 get_equations called - Revise is working!
4-element Vector{String}:
 "1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0] ^ (α - 1) + (1 - δ))"
 ...
```

### Why This Matters

- **Eliminates precompilation delays** - changes apply in seconds, not minutes
- **Preserves session state** - models, variables, and computations persist
- **Enables rapid debugging** - add/remove print statements instantly
- **Essential for this package** - MacroModelling has significant compile times

### Important Caveats

- **Revise must be loaded BEFORE MacroModelling** - order matters!
- **Structural changes require restart** - new types, module reorganization, or changing `__init__` functions
- **Manual refresh available** - if a change isn't detected, run `Revise.revise()`
- **Julia 1.12+ note** - may show world age warnings, but hot-reload still works

## Architecture Overview

### Core Source Files (`src/`)

| File | Purpose |
|------|---------|
| `MacroModelling.jl` | Main module, exports, type definitions |
| `macros.jl` | `@model` and `@parameters` macro implementations |
| `get_functions.jl` | User-facing API (IRFs, simulations, forecasts, solutions) |
| `perturbation.jl` | First, second, third order perturbation solution algorithms |
| `moments.jl` | Model moment calculations |
| `structures.jl` | Core data structures and types |
| `options_and_caches.jl` | Solution caching and calculation options |
| `dynare.jl` | Dynare file import support |
| `inspect.jl` | Model inspection utilities |

### Algorithms (`src/algorithms/`)

| File | Purpose |
|------|---------|
| `sylvester.jl` | Sylvester equation solvers for perturbation |
| `lyapunov.jl` | Lyapunov equation solvers (doubling, Bartels-Stewart, iterative) |
| `quadratic_matrix_equation.jl` | QME solvers (Schur, doubling) |
| `nonlinear_solver.jl` | Steady state solver |

### Filters (`src/filter/`)

| File | Purpose |
|------|---------|
| `kalman.jl` | Kalman filter for first-order estimation |
| `inversion.jl` | Inversion filter for nonlinear estimation |
| `find_shocks.jl` | Shock identification for conditional forecasting |

### Extensions (`ext/`)

| File | Purpose |
|------|---------|
| `StatsPlotsExt.jl` | Plotting functionality (loaded with StatsPlots) |
| `TuringExt.jl` | Bayesian estimation (loaded with Turing) |
| `OptimExt.jl` | Optimization support (loaded with Optim) |

## Model Syntax

- Variables use time indices: `...[2], [1], [0], [-1], [-2]...`
- Shocks use `[x]`: `eps_z[x]`
- Calibration equations use `|` syntax in `@parameters` block
- Custom steady state functions can be provided via `steady_state_function` parameter

## Key Design Considerations

- **Performance critical** - Package competes with Dynare/RISE. Be mindful of type stability and allocations.
- **Symbolic mathematics** - Uses Symbolics.jl and SymPyPythonCall for symbolic derivatives compiled to efficient numerical code.
- **Automatic differentiation** - Supports forward and reverse-mode AD for gradients w.r.t. parameters.
- **Thread safety** - Important for estimation tasks.

## Test Models

Quick testing models in `test/models/`:
- `RBC_CME.jl` - Simple RBC model
- `Backus_Kehoe_Kydland_1992.jl` - International RBC
- `SW07_nonlinear.jl` - Smets-Wouters 2007

Reference models in `models/` directory include implementations from published papers (SW03, SW07, FS2000, NAWM, etc.).
