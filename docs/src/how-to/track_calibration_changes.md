# Tracking Calibration Equation Changes

MacroModelling.jl provides functionality to track and document changes to calibration equations over time. This is useful for:

- Maintaining a history of model calibration decisions
- Documenting sensitivity analyses 
- Tracking different calibration scenarios
- Facilitating collaboration and reproducibility

## Overview

The package provides three main functions for tracking calibration equation revisions:

1. `modify_calibration_equations!` - Document a change to calibration equations
2. `get_calibration_revision_history` - Retrieve the revision history
3. `print_calibration_revision_history` - Display the revision history in a readable format

## Basic Usage

Here's a complete example showing how to track calibration equation changes:

```julia
using MacroModelling

# Define a model
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

# Set up initial parameters with calibration equation
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    k[ss] / q[ss] = 2.5 | δ  # Initial calibration target
    α = 0.5
    β = 0.95
end

# Document a change to the calibration target
modify_calibration_equations!(RBC, 
    [:δ => :(k[ss] / q[ss] - 3.0)],
    "Updated capital-to-output ratio target from 2.5 to 3.0 based on empirical evidence",
    verbose = true)

# View the revision history
print_calibration_revision_history(RBC)
```

Output:
```
Documented revision for parameter :δ
  New target: k[ss] / q[ss] - 3.0

Revision recorded. To apply these changes, re-run the @parameters macro with the new calibration equations.

Calibration Equation Revision History:
============================================================

Revision 1: 2024-01-15T10:30:45.123 - Updated capital-to-output ratio target from 2.5 to 3.0 based on empirical evidence
------------------------------------------------------------
  δ: k[ss] / q[ss] - 3.0
```

## Documenting Multiple Changes

You can document changes to multiple calibration equations at once:

```julia
modify_calibration_equations!(RBC, 
    [
        :δ => :(k[ss] / q[ss] - 3.0),
        :α => :(y[ss] / k[ss] - 0.35)
    ],
    "Updated both depreciation and productivity calibration targets")
```

## Retrieving Revision History Programmatically

To access the revision history programmatically:

```julia
history = get_calibration_revision_history(RBC)

for (timestamp_note, equations, parameters) in history
    println("Revision: ", timestamp_note)
    for (param, eq) in zip(parameters, equations)
        println("  ", param, " => ", eq)
    end
end
```

## Important Notes

### Applying Changes

The `modify_calibration_equations!` function **documents** changes for tracking purposes but does not automatically apply them to the model. To apply the documented changes, you need to:

1. Document the change using `modify_calibration_equations!`
2. Re-run the `@parameters` macro with the new calibration equations

```julia
# Document the change
modify_calibration_equations!(RBC, 
    [:δ => :(k[ss] / q[ss] - 3.0)],
    "Updated calibration target")

# Apply the change by re-running @parameters
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    k[ss] / q[ss] = 3.0 | δ  # New calibration target
    α = 0.5
    β = 0.95
end
```

### Use Cases

This functionality is particularly useful for:

1. **Sensitivity Analysis**: Document different calibration scenarios you've tested
   ```julia
   # Test scenario 1
   modify_calibration_equations!(model, 
       [:δ => :(k[ss] / q[ss] - 2.0)],
       "Scenario 1: Low capital intensity")
   
   # Test scenario 2
   modify_calibration_equations!(model, 
       [:δ => :(k[ss] / q[ss] - 4.0)],
       "Scenario 2: High capital intensity")
   ```

2. **Collaboration**: Share revision history with team members to communicate calibration decisions

3. **Reproducibility**: Maintain a complete audit trail of calibration changes

## See Also

- [`get_calibration_equations`](@ref) - Get current calibration equations
- [`get_calibrated_parameters`](@ref) - Get parameters determined by calibration
- [`@parameters`](@ref) - Define model parameters and calibration equations
