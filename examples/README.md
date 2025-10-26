# MacroModelling.jl Examples

This directory contains example scripts demonstrating various features of MacroModelling.jl.

## Available Examples

### calibration_tracking_example.jl

Demonstrates how to track and document changes to calibration equations over time. This is useful for:
- Maintaining an audit trail of calibration decisions
- Documenting different calibration scenarios
- Facilitating collaboration and reproducibility

Run with:
```julia
julia --project=. examples/calibration_tracking_example.jl
```

## Adding New Examples

When adding new examples:
1. Create a descriptive filename (e.g., `feature_name_example.jl`)
2. Include comments explaining what the example demonstrates
3. Add the example to this README
4. Ensure the example is self-contained and can run independently
