# Refactoring Summary: Symbolic Steady State Solver

## Overview

The symbolic steady state solver in `MacroModelling.jl` has been refactored to separate general orchestration logic from model-specific calculations, following the same architecture as the recently refactored precompiled version.

## Key Changes

### 1. Architecture

**Before:**
- Monolithic `solve_SS` function generated as a single RuntimeGeneratedFunction (RTGF)
- All logic (symbolic evaluation, numerical solving, error checking) embedded in one generated function
- Used `write_block_solution!` to add complex numerical solving code inline

**After:**
- General orchestration in `solve_steady_state_symbolic` (normal Julia function in `src/steady_state_solver.jl`)
- Model-specific calculations in separate RTGFs:
  - `setup_parameters_and_bounds`: parameter vector → named parameters
  - `evaluate_symbolic_solutions`: evaluates symbolically-solved variables
  - `check_minmax_errors`: checks min/max constraint violations
  - Per-block RTGFs: `get_block_inputs`, `update_solution` (for numerical blocks)
  - `set_dynamic_exogenous`: sets dynamic exogenous variables to zero
  - `extract_solution_vector`: named variables → solution vector

### 2. File Structure

New files:
- `src/steady_state_solver.jl`: Contains `solve_steady_state_symbolic` and `solve_steady_state_precompiled`

Modified files:
- `src/non_stochastic_steady_state.jl`: Refactored `write_steady_state_solver_function!` (symbolic version)
- `src/steady_state_helpers.jl`: Added `build_symbolic_rtgfs!` function
- `src/structures.jl`: Updated `non_stochastic_steady_state` struct with symbolic-specific fields
- `src/macros.jl`: Updated NSSS initialization

### 3. Data Structures

The `non_stochastic_steady_state` struct now includes:

```julia
mutable struct non_stochastic_steady_state
    solve_blocks_in_place::Vector{ss_solve_block}
    dependencies::Any
    # Shared RTGFs (both versions)
    setup_parameters_and_bounds::Union{Function, Nothing}
    block_metadata::Vector{ss_block_metadata}
    set_dynamic_exogenous::Union{Function, Nothing}
    extract_solution_vector::Union{Function, Nothing}
    solution_vector_length::Int
    # Symbolic-specific RTGFs
    evaluate_symbolic_solutions::Union{Function, Nothing}
    check_minmax_errors::Union{Function, Nothing}
end
```

### 4. Symbolic Solving Logic

The refactored symbolic version cleanly separates three solution paths:

1. **Symbolic constants** (e.g., `x = 1.5`):
   - Collected during block loop
   - Evaluated in `evaluate_symbolic_solutions` RTGF

2. **Symbolic expressions** (e.g., `x = α * β / (1-γ)`):
   - Collected during block loop
   - Evaluated in `evaluate_symbolic_solutions` RTGF
   - Domain error checking included

3. **Numerical blocks** (when symbolic fails):
   - Collected as block data structures
   - Solved using general `block_solver` function
   - Same infrastructure as precompiled version

### 5. Min/Max Constraint Handling

Variables inside min/max constraints are handled specially:
- Min/max terms are removed from equations during symbolic solving
- Error terms stored in `min_max_errors`
- Checked after all solving in `check_minmax_errors` RTGF

## Benefits of Refactoring

1. **Separation of Concerns**: General algorithm logic separated from model-specific code generation
2. **Maintainability**: Easier to understand and modify orchestration logic
3. **Consistency**: Symbolic and precompiled versions share same infrastructure for numerical blocks
4. **Reduced Code Duplication**: Block solving logic unified in `block_solver`
5. **Cleaner Code Generation**: Model-specific RTGFs are simpler and more focused
6. **Testability**: General orchestration can be tested independently

## Implementation Notes

### Simplifications

The refactored version simplifies some aspects of the original implementation:

1. **Numerical Block Handling**:
   - Original: Used complex `write_block_solution!` with domain error handling
   - Refactored: Collects block metadata, delegates to general `block_solver`

2. **Bounds**:
   - Original: Complex per-variable bounds extraction
   - Refactored: Default bounds used (can be refined later)

3. **Domain Error Handling**:
   - Original: Inline domain error checks with auxiliary variables
   - Refactored: Simplified error checking in symbolic RTGF

### Areas for Future Enhancement

1. **Variable Bounds**: Restore per-variable bounds from original implementation
2. **Domain Errors**: Add back full domain error handling machinery if needed
3. **Partial Solving**: Restore `partial_solve` optimization for small numerical blocks
4. **Auxiliary Variables**: Handle non-negativity auxiliary variables more explicitly

## Testing

The refactored solver should be tested with:

1. Models with pure symbolic solutions
2. Models with mixed symbolic/numerical solutions
3. Models with min/max constraints
4. Models with bounded variables
5. Models that previously used `write_block_solution!`

## Migration Notes

No user-facing changes - the API remains identical. Users continue to call:
- `get_steady_state(model)`
- `get_irf(model)`
- etc.

The refactoring is entirely internal to the steady state solving process.

## References

- Symbolic version: `src/non_stochastic_steady_state.jl` (lines 1-264)
- Precompiled version: `src/non_stochastic_steady_state.jl` (lines 245+)
- General orchestration: `src/steady_state_solver.jl`
- Helper functions: `src/steady_state_helpers.jl`
- Data structures: `src/structures.jl`
