# MacroModelling.jl Examples

This directory contains example scripts demonstrating various features of MacroModelling.jl.

## Running the Examples

To run an example, navigate to the package directory and execute:

```bash
julia --project=. examples/partial_parameters_example.jl
```

## Available Examples

### partial_parameters_example.jl

Demonstrates the new partial parameter definition feature, which allows you to:
- Define models without specifying all parameters upfront
- Add parameters incrementally via multiple `@parameters` calls
- Provide parameters dynamically via function arguments
- Load parameters from external sources

This is particularly useful for:
- Large models where parameter definition is cumbersome
- Scenarios where parameters come from different sources or files
- Experimentation with different parameter values

## Requirements

Make sure the MacroModelling.jl package is properly installed and activated:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
