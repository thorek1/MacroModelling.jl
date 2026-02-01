# Agent Guide for MacroModelling.jl

This file provides guidance for AI coding agents (GitHub Copilot, Claude, etc.) when working with this repository.

## Project Overview

`MacroModelling.jl` is a Julia package for developing and solving dynamic stochastic general equilibrium (DSGE) models. These models describe macroeconomic behavior and are used for counterfactual analysis, economic policy evaluation, and quantifying specific mechanisms in academic research.

**Key capabilities:**
- Parse models with user-friendly syntax (time indices like `[0], [-1], [1]`)
- Solve models automatically from equations and parameter values
- Calculate first, second, and third order (pruned) perturbation solutions
- Handle occasionally binding constraints
- Calculate impulse response functions, simulations, and conditional forecasts
- Estimate models using gradient-based samplers (NUTS, HMC) or inversion filters
- Differentiate solutions and moments with respect to parameters

**Target audience:** Central bankers, regulators, graduate students, and researchers in DSGE modeling.

**Timing convention:** End-of-period (not start-of-period like some other packages).

## Project Structure

```
MacroModelling.jl/
├── src/                      # Main source code
│   ├── MacroModelling.jl    # Main module, exports, type definitions
│   ├── macros.jl            # @model and @parameters macros
│   ├── get_functions.jl     # User-facing API (IRFs, simulations, forecasts)
│   ├── perturbation.jl      # Perturbation solution algorithms (1st-3rd order)
│   ├── moments.jl           # Model moment calculations
│   ├── structures.jl        # Core data structures and types
│   ├── options_and_caches.jl # Solution caching and calculation options
│   ├── dynare.jl            # Dynare file import support
│   ├── inspect.jl           # Model inspection utilities
│   ├── solver_parameters.jl # Solver configuration parameters
│   ├── default_options.jl   # Default option values
│   ├── common_docstrings.jl # Shared documentation strings
│   ├── algorithms/          # Matrix equation solvers (sylvester, lyapunov, quadratic_matrix_equation, nonlinear_solver)
│   ├── filter/              # Kalman and inversion filters (kalman, inversion, find_shocks)
│   └── custom_autodiff_rules/ # AD rules (forwarddiff, zygote)
├── test/                     # Test suite with multiple test sets
├── models/                   # Example DSGE models from literature
├── docs/                     # Documentation (Documenter.jl)
├── benchmark/                # Benchmark scripts (BenchmarkTools)
└── ext/                      # Package extensions (StatsPlots, Turing, Optim)
```

## Development Setup

### Julia Requirements

- **Julia version:** 1.10 or higher (tested on 1.10+, lts, and pre-release versions)
- **Running Julia:** Always use `julia -t auto` to enable multi-threading

### Package Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
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

## Testing

**Do NOT run the full test suite** - it takes too long. Instead:

### Quick Feature Testing

Write a bespoke script using the simple RBC model shown above, then test your changes:

```julia
# Test your changes here
get_irf(RBC)
simulate(RBC)
```

### Test Sets (CI Only)

Tests are organized by test sets specified via `TEST_SET` environment variable:

- `basic`, `estimation`, `higher_order_1-3`, `plots_1-5`, `estimate_sw07`, `jet`
- Estimation tests: `1st_order_inversion_estimation`, `2nd_order_estimation`, `pruned_2nd_order_estimation`, `3rd_order_estimation`, `pruned_3rd_order_estimation`
- Pigeons estimation tests: `estimation_pigeons`, `1st_order_inversion_estimation_pigeons`, `2nd_order_estimation_pigeons`, `pruned_2nd_order_estimation_pigeons`, `3rd_order_estimation_pigeons`, `pruned_3rd_order_estimation_pigeons`

```bash
TEST_SET=basic julia --project -e 'using Pkg; Pkg.test()'
```

### Test Environment Setup

```julia
using Pkg
Pkg.activate("test")
Pkg.instantiate()
```

## Documentation

Build documentation locally:

```bash
julia --project=docs docs/make.jl
```

Documentation is built with Documenter.jl and deployed to GitHub Pages.

## Benchmarking

```julia
using BenchmarkTools
include("benchmark/benchmarks.jl")
run(SUITE)
```

## Model Syntax

- **Variables** use time indices: `...[2], [1], [0], [-1], [-2]...`
- **Shocks** use `[x]`: `eps_z[x]`
- **Calibration equations** use `|` syntax in `@parameters` block
- **Custom steady state** can be provided via `steady_state_function` parameter

## Code Style and Conventions

### General Principles

1. **Minimal changes:** Make the smallest possible changes to accomplish the task
2. **Testing:** Test changes with simple models rather than running the full test suite
3. **Performance:** This package emphasizes performance - be mindful of type stability and allocations
4. **Documentation:** Update docstrings when modifying public APIs

### Writing Style

- Avoid second-person phrasing ("you") in docs and docstrings

### Caching Guidance

- For constant calculations that can be computed once and reused, compute lazily on first use and store in the model struct cache; subsequent use must read from the cache

## Key Design Considerations

- **Performance critical** - Package competes with Dynare/RISE. Be mindful of type stability and allocations.
- **Symbolic mathematics** - Uses Symbolics.jl and SymPyPythonCall for symbolic derivatives compiled to efficient numerical code.
- **Automatic differentiation** - Supports forward and reverse-mode AD for gradients w.r.t. parameters.
- **Thread safety** - Important for estimation tasks.

## Common Tasks

### Adding a New Feature

1. Write the feature in the appropriate `src/` file
2. Create a minimal test script (don't rely on full test suite)
3. Test with the simple RBC model
4. Update documentation if it's a user-facing feature

### Fixing a Bug

1. Identify the issue location in `src/`
2. Write a minimal reproduction case
3. Fix and verify with test script
4. Ensure existing functionality isn't broken

### Adding a New Model

1. Place in `models/` directory
2. Follow existing model structure
3. Include citation information
4. Test that it solves and produces IRFs

### Common Change Points

- **New API:** add in `src/get_functions.jl` and export from `src/MacroModelling.jl`
- **New model:** add a file under `models/` using the model macros
- **Solver changes:** look in `src/perturbation.jl` and `src/algorithms/`

## CI/CD Pipeline

- **CI runs on:** push (pull requests are commented out in workflow)
- **Platforms:** Ubuntu, macOS, Windows (x64 and arm64 where applicable)
- **Coverage:** Uploaded to Codecov
- **Matrix testing:** Multiple test sets run in parallel across different OS/architecture combinations

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary.

## Workflow Orchestration

### Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

### Session Progress Log

- Always take stock of what was done and what remains, and save it in `AGENT_PROGRESS.md`
- At the start of a new session, always read `AGENT_PROGRESS.md` before making changes

### Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## CRITICAL WORKFLOW REQUIREMENTS

**These rules are non-negotiable.**

1. **NEVER claim something works without running a test to prove it.** After writing any code, immediately write and run a test. If you cannot test it, say so explicitly.

2. **Work modularly.** Complete one module at a time. After each module, report what you built, show test results.

3. **Iterate and fix errors yourself.** Do not rely on the user to report errors back to you. Run the code, observe the output, and fix problems before presenting results.

4. **Be explicit about unknowns.** If you're uncertain about something, say so. Don't guess.

5. **Verify before done.** Never mark a task complete without proving it works. Diff behavior between main and your changes when relevant. Ask yourself: "Would a staff engineer approve this?"

## Additional Resources

- **Documentation:** https://thorek1.github.io/MacroModelling.jl/stable
- **Issue tracker:** GitHub Issues
- **Contributing guidelines:** See CONTRIBUTING.md
- **Code of Conduct:** See CODE_OF_CONDUCT.md
