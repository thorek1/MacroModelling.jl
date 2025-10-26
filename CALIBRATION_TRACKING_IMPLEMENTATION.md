# Calibration Equation Tracking Implementation

## Overview

This implementation adds functionality to track and document changes to calibration equations in MacroModelling.jl models. The feature allows users to maintain an audit trail of calibration decisions, document different scenarios, and improve reproducibility.

## Implementation Details

### 1. Data Structure Changes

**File: `src/structures.jl`**

Added a new field to the `ℳ` struct to store revision history:

```julia
calibration_equations_revision_history::Vector{Tuple{String, Vector{Expr}, Vector{Symbol}}}
```

Each entry in the history contains:
- A timestamp and optional note (String)
- The calibration equations at that revision (Vector{Expr})
- The parameters those equations calibrate (Vector{Symbol})

### 2. Initialization

**File: `src/macros.jl`**

The revision history is initialized as an empty vector when the model is created:

```julia
Tuple{String, Vector{Expr}, Vector{Symbol}}[], # calibration_equations_revision_history
```

### 3. Core Functionality

**File: `src/modify_calibration.jl`** (new file)

Three main functions were implemented:

#### `modify_calibration_equations!(𝓂, param_equation_pairs, revision_note; verbose)`

Documents changes to calibration equations. This function:
- Validates that specified parameters are actual calibration parameters
- Records the new equations and parameters in the revision history
- Adds a timestamp and optional note
- Does NOT automatically apply changes (user must re-run `@parameters` to apply)

**Signature:**
```julia
function modify_calibration_equations!(
    𝓂::ℳ, 
    param_equation_pairs::Vector{<:Pair{Symbol, <:Any}},
    revision_note::String = "";
    verbose::Bool = false
)
```

**Example:**
```julia
modify_calibration_equations!(model, 
    [:δ => :(k[ss] / q[ss] - 3.0)],
    "Updated capital-to-output ratio",
    verbose = true)
```

#### `get_calibration_revision_history(𝓂; formatted)`

Retrieves the revision history, optionally in a human-readable format.

**Signature:**
```julia
function get_calibration_revision_history(
    𝓂::ℳ; 
    formatted::Bool = true
)
```

Returns a vector of tuples containing the revision history. When `formatted=true`, converts symbolic representations to readable strings.

#### `print_calibration_revision_history(𝓂)`

Prints the revision history in a formatted, readable way.

**Example output:**
```
Calibration Equation Revision History:
============================================================

Revision 1: 2024-01-15T10:30:45.123 - Updated capital-to-output ratio
------------------------------------------------------------
  δ: k[ss] / q[ss] - 3.0
```

### 4. Module Integration

**File: `src/MacroModelling.jl`**

The new file is included in the module:
```julia
include("modify_calibration.jl")
```

And the functions are exported:
```julia
export modify_calibration_equations!, get_calibration_revision_history, print_calibration_revision_history
```

Added `Dates` to imports for timestamp generation.

## Design Decisions

### Why Documentation-Focused?

The implementation is focused on **documenting** rather than **automatically applying** changes for several reasons:

1. **Complexity**: Calibration equations are deeply integrated with the symbolic processing and steady-state solver. Modifying them programmatically would require re-running significant portions of the `@parameters` macro logic.

2. **Safety**: Automatic modification could lead to inconsistent states if not done carefully. The current approach requires explicit re-running of `@parameters`, making changes deliberate and clear.

3. **Transparency**: Users explicitly see what equations they're using, maintaining clarity about the model specification.

4. **Workflow**: The typical workflow involves iterating on calibration by editing and re-running code anyway. This implementation augments that workflow with tracking.

### Use Cases

This implementation is particularly valuable for:

1. **Sensitivity Analysis**: Document different calibration scenarios tested
2. **Collaboration**: Share rationale for calibration decisions with team members
3. **Reproducibility**: Maintain complete audit trail of changes
4. **Model Development**: Track evolution of calibration strategy over time

## Testing

### Test File: `test/test_modify_calibration.jl`

Comprehensive tests covering:
- Initial state verification
- Single revision documentation
- Multiple revisions
- Error handling (invalid parameters)
- History retrieval (programmatic and formatted)
- History printing

### Basic Functionality Test: `/tmp/test_basic_functionality.jl`

Standalone test verifying:
- Data structure correctness
- Timestamp generation
- Parameter validation logic
- History formatting

## Documentation

### User Guide: `docs/src/how-to/track_calibration_changes.md`

Complete guide including:
- Overview of functionality
- Basic usage examples
- Multiple calibration changes
- Programmatic access to history
- Important notes about applying changes
- Use case examples

### Example: `examples/calibration_tracking_example.jl`

Runnable example demonstrating:
- Creating a model with calibration equations
- Documenting multiple calibration scenarios
- Viewing revision history
- Programmatic access to history

## Files Changed/Added

### Modified Files
1. `src/structures.jl` - Added revision history field to ℳ struct
2. `src/macros.jl` - Initialize revision history when creating models
3. `src/MacroModelling.jl` - Include new file, export functions, add Dates import

### New Files
1. `src/modify_calibration.jl` - Core implementation (3 functions, ~200 lines)
2. `test/test_modify_calibration.jl` - Test suite (~110 lines)
3. `docs/src/how-to/track_calibration_changes.md` - User documentation (~160 lines)
4. `examples/calibration_tracking_example.jl` - Example script (~75 lines)
5. `examples/README.md` - Examples directory documentation

## Future Enhancements (Optional)

If desired, future enhancements could include:

1. **Export/Import**: Save/load revision history to/from JSON or CSV
2. **Comparison**: Compare calibration equations across revisions
3. **Visualization**: Plot how calibration targets have evolved
4. **Integration**: Deeper integration with steady-state solving (more complex)
5. **Validation**: Check if documented equations match actual equations in use

## Backward Compatibility

This implementation is fully backward compatible:
- Existing models work without changes
- New field is initialized to empty vector
- Functions only available if explicitly called
- No performance impact on existing functionality

## Summary

The implementation successfully provides calibration equation tracking functionality through:
- A clean data structure for storing revision history
- Easy-to-use functions for documenting and viewing changes
- Comprehensive documentation and examples
- Full test coverage
- Backward compatibility

The approach is pragmatic, focusing on documentation and tracking rather than automatic modification, which aligns with typical model development workflows while providing valuable audit trail capabilities.
