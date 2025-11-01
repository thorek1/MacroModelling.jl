# Copilot Instructions for MacroModelling.jl

## Repository Overview

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

## Project Structure

```
MacroModelling.jl/
├── src/                      # Main source code
│   ├── MacroModelling.jl    # Main module file
│   ├── macros.jl            # @model and @parameters macros
│   ├── get_functions.jl     # User-facing functions (IRFs, simulations)
│   ├── perturbation.jl      # Perturbation solution algorithms
│   ├── moments.jl           # Model moment calculations
│   ├── algorithms/          # Core solution algorithms
│   └── filter/              # Kalman and inversion filters
├── test/                     # Test suite
│   ├── runtests.jl          # Test runner with multiple test sets
│   └── models/              # Test models
├── models/                   # Example DSGE models from literature
├── docs/                     # Documentation
├── benchmark/                # Benchmark scripts
├── ext/                      # Package extensions (StatsPlots, Turing)
└── AGENTS.md                # Agent-specific instructions
```

## Development Guidelines

### Testing

**Important:** This repository has an extensive test suite organized into multiple test sets. Tests are automatically run via CI on push.

- **Test sets:** `basic`, `estimation`, `higher_order_1-3`, `plots_1-4`, `estimate_sw07`, various estimation tests, and `jet`
- **Running tests:**
  - **Do NOT run the full test suite** (`julia --project -e "using Pkg; Pkg.test()"`) as it takes too long
  - Instead, test specific functionality with a bespoke script using a few models from `test/models/`
  - Example models for quick testing: Look at `test/models/test_models.jl` or simple models like RBC
  - Set `TEST_SET` environment variable to run specific test sets during CI
  
- **When testing changes:**
  ```julia
  # Example: Test a specific feature with a simple model
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
  ```
- **Avoid precompiling** the whole package when testing new functionality; use standalone scripts instead

### Julia Setup

- **Julia version:** Requires Julia 1.10 or higher (tested on 1.10+, lts, and pre-release versions)
- **Installation:**
  ```bash
  curl -fsSL https://install.julialang.org -o juliaup.sh
  sh juliaup.sh --yes --default-channel release
  export PATH="$HOME/.juliaup/bin:$PATH"
  ```
- **Running Julia:** Always use `julia -t auto` to enable multi-threading

### Building

- **Package installation:**
  ```julia
  using Pkg
  Pkg.activate(".")
  Pkg.instantiate()
  ```
- **Development mode:** Use `Pkg.develop(".")` if working on package code

### Benchmarking

- Use the `BenchmarkTools` package for performance testing
- See `benchmark/benchmarks.jl` for existing benchmark scripts
- When comparing implementations, follow the structure in `benchmark/benchmarks.jl`

### Documentation

- Documentation is built with Documenter.jl and deployed to GitHub Pages
- Source files are in `docs/src/`
- Build documentation locally: `julia --project=docs docs/make.jl`

### Dependencies

- **R packages:** Try installing with micromamba first, then fall back to `install.packages()`
- **Micromamba installation:**
  ```bash
  curl -Ls https://micro.mamba.pm/install.sh | MAMBA_NO_BANNER=1 MAMBA_NO_PROMPT=1 bash -s -- -b -u -p ~/micromamba
  ```

## Code Style and Conventions

### General Principles

1. **Minimal changes:** Make the smallest possible changes to accomplish the task
2. **Testing:** Test changes with simple models rather than running the full test suite
3. **Performance:** This package emphasizes performance - be mindful of type stability and allocations
4. **Documentation:** Update docstrings when modifying public APIs

### Julia Conventions

- Follow standard Julia style guidelines
- Use meaningful variable names
- Time indices in models: `...[2], [1], [0], [-1], [-2]...` for variables, `[x]` for shocks
- Parameter definitions use the `@parameters` macro
- Model definitions use the `@model` macro

### Mathematical Notation

- The package uses end-of-period timing convention (not start-of-period like some other packages)
- Models work with symbolic mathematics via Symbolics.jl
- Solutions use perturbation methods (first, second, third order)

## Working with the Repository

### CI/CD Pipeline

- **CI runs on:** push (pull requests are commented out in workflow)
- **Platforms:** Ubuntu, macOS, Windows (x64 and arm64 where applicable)
- **Coverage:** Uploaded to Codecov
- **Matrix testing:** Multiple test sets run in parallel across different OS/architecture combinations

### Common Tasks

1. **Adding a new feature:**
   - Write the feature in the appropriate `src/` file
   - Create a minimal test script (don't rely on full test suite)
   - Test with a simple model like RBC
   - Update documentation if it's a user-facing feature

2. **Fixing a bug:**
   - Identify the issue location in `src/`
   - Write a minimal reproduction case
   - Fix and verify with test script
   - Ensure existing functionality isn't broken

3. **Adding a new model:**
   - Place in `models/` directory
   - Follow existing model structure
   - Include citation information
   - Test that it solves and produces IRFs

## Additional Resources

- **Documentation:** https://thorek1.github.io/MacroModelling.jl/stable
- **Issue tracker:** GitHub Issues
- **Contributing guidelines:** See CONTRIBUTING.md
- **Code of Conduct:** See CODE_OF_CONDUCT.md
- **Agent-specific instructions:** See AGENTS.md for specialized agent guidance

## Notes for Copilot Agents

- This is a specialized numerical package for macroeconomics - solutions involve perturbation methods, Kalman filtering, and nonlinear equation solving
- The package has complex dependencies (SymPy, Symbolics, optimization libraries) that must be handled carefully
- Performance is critical - the package competes with established tools like Dynare and RISE
- Many operations use symbolic mathematics that are compiled to efficient numerical code
- Thread safety and multi-threading are important for estimation tasks
- See AGENTS.md for specific instructions on testing, benchmarking, and package management
