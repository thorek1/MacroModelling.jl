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

## Revise-Based Development Workflow (REQUIRED)

**ALWAYS use Revise.jl for interactive development.** This allows hot-reloading of code changes without restarting Julia.

### Setup

1. Start Julia REPL with multi-threading:
   ```bash
   cd /path/to/MacroModelling.jl
   julia -t auto --project=.
   ```

2. Load Revise FIRST, then MacroModelling:
   ```julia
   using Revise
   using MacroModelling
   ```

3. Define a test model:
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

### Workflow

1. **Keep the Julia REPL running** - do not restart it between edits
2. **Edit source files** in `src/` using your editor
3. **Revise automatically detects changes** and recompiles affected functions
4. **Test your changes** immediately in the same REPL session
5. **Iterate** - make more edits, test again, without restarting

### Example

```julia
# First call (before editing)
julia> get_equations(RBC)
4-element Vector{String}:
 "1 / c[0] = ..."
 ...

# Edit src/inspect.jl to add: println("Debug: get_equations called")
# Revise detects the change automatically

# Second call (after editing) - no restart needed!
julia> get_equations(RBC)
Debug: get_equations called
4-element Vector{String}:
 "1 / c[0] = ..."
 ...
```

### Benefits

- **No precompilation wait** after each change
- **Preserves model state** - no need to re-define models
- **Fast iteration** - edit, test, repeat in seconds
- **Essential for debugging** - add print statements, test, remove them

### Important Notes

- Revise must be loaded BEFORE MacroModelling
- Structural changes (new types, module reorganization) may require restart
- If Revise misses a change, run `Revise.revise()` manually

## Tests

Before running tests, activate the test environment and instantiate dependencies:

```julia
using Pkg
Pkg.activate("test")
Pkg.instantiate()
```

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

## Caching Guidance

- For constant calculations that can be computed once and reused, compute lazily on first use and store in the model struct cache; subsequent use must read from the cache.

## Session Progress Log

- Always take stock of what was done and what remains, and save it in `AGENT_PROGRESS.md`.
- At the start of a new session, always read `AGENT_PROGRESS.md` before making changes.

## Common Change Points

- New API: add in `src/get_functions.jl` and export from `src/MacroModelling.jl`.
- New model: add a file under `models/` using the model macros.
- Solver changes: look in `src/perturbation.jl` and `src/algorithms/`.

## CRITICAL WORKFLOW REQUIREMENTS

**YOU MUST FOLLOW THESE RULES. THEY ARE NON-NEGOTIABLE.**

1. **NEVER claim something works without running a test to prove it.** After writing any code, immediately write and run a test. If you cannot test it, say so explicitly.

2. **Work modularly.** Complete one module at a time. After each module, report what you built, show test results.

3. **Iterate and fix errors yourself.** Do not rely on the user to report errors back to you. Run the code, observe the output, and fix problems before presenting results.

4. **Be explicit about unknowns.** If you're uncertain about something, say so. Don't guess.
