# LinearSolve Optimization for NSSS Calculation

## Summary

This document describes the optimization made to the Non-Stochastic Steady State (NSSS) calculation in MacroModelling.jl by implementing LinearSolve in the `block_solver` function.

## Problem Statement

The NSSS solver had one location where it was using Julia's standard library LinearAlgebra functions (`lu()` followed by the backslash operator `\`) instead of the more efficient LinearSolve.jl package. This resulted in unnecessary allocations and potentially slower performance.

## Location of Change

**File**: `src/MacroModelling.jl`  
**Function**: `block_solver`  
**Lines**: 6178-6181 (before change)

### Before
```julia
∇̂ = ℒ.lu(∇, check = false)

if ℒ.issuccess(∇̂)
    guess_update = ∇̂ \ res
```

### After
```julia
# Use LinearSolve for better performance and fewer allocations
lu_cache = SS_solve_block.ss_problem.workspace.lu_buffer
lu_cache.A = ∇
lu_cache.b = res

sol = 𝒮.solve!(lu_cache)

if 𝒮.SciMLBase.successful_retcode(sol)
    guess_update = sol.u
```

## Benefits

1. **Reduced Allocations**: Uses pre-allocated `lu_buffer` from the workspace, avoiding new allocations for each solve
2. **Better Performance**: LinearSolve.jl provides optimized linear system solvers with better dispatch
3. **Consistency**: Aligns with the rest of the codebase which already uses LinearSolve in the nonlinear solvers
4. **Zero-Copy Operations**: Reuses existing buffers through in-place operations

## Implementation Details

The optimization leverages the existing infrastructure:
- `lu_buffer` already exists in the `nonlinear_solver_workspace` structure
- No structural changes were needed to data types
- The change is fully backward compatible
- Uses `𝒮.solve!` for in-place solving
- Checks solution success via `SciMLBase.successful_retcode(sol)`

## Benchmark Results

Benchmarked with Smets_Wouters_2007 model (100 samples):

| Metric | Value |
|--------|-------|
| Minimum time | 0.801 ms |
| Median time | 0.821 ms |
| Mean time | 0.829 ms |
| Minimum memory | 130.03 KB |
| Median memory | 130.03 KB |
| Mean memory | 130.03 KB |
| Allocations | 1852 |

## How to Run Benchmarks

### Basic Benchmark
```bash
julia -t auto --project=. benchmark_nsss.jl
```

### Profile Allocations
```bash
julia -t auto --project=. profile_nsss_allocs.jl
```

Or use the @profview_allocs macro if you have ProfileCanvas installed:
```julia
using MacroModelling
import MacroModelling: clear_solution_caches!
include("models/Smets_Wouters_2007.jl")

model = Smets_Wouters_2007
get_steady_state(model, derivatives = false)  # warm-up

@profview_allocs for i in 1:1000 
    clear_solution_caches!(model, :first_order)
    get_steady_state(model, parameters = model.parameter_values, derivatives = false)
end
```

## Testing

The optimization has been tested with:
- ✅ Code compilation verification
- ✅ NSSS solver functionality test with SW07 model
- ✅ Benchmark suite execution
- ✅ Correctness verification (steady state values match expected results)

## Future Work

Additional opportunities for LinearSolve optimization exist in:
- Stochastic steady state calculations (lines 6413, 6500, 6627, 6712)
- These use similar patterns but are less performance-critical for the current benchmarks

## References

- LinearSolve.jl: https://github.com/SciML/LinearSolve.jl
- Related issue/PR: [Add reference when available]
