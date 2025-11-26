# Examples

This directory contains example scripts demonstrating various features of MacroModelling.jl.

## Finch-based Higher Order Solutions

The `finch_example.jl` script demonstrates the usage of the new Finch-based implementations of higher order perturbation solutions.

### Running the Example

```julia
using MacroModelling
include("examples/finch_example.jl")
```

### What it demonstrates

- How to define a simple RBC model
- How to solve it using the standard implementation
- How the new Finch-based functions are available for use
- Conceptual integration patterns for future development

## Additional Examples

For more comprehensive examples, please see:
- The `models/` directory in the root of the repository
- The official documentation at https://thorek1.github.io/MacroModelling.jl/stable
