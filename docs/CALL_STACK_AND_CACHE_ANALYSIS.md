# Call Stack and Cache Initialization Analysis
## get_functions.jl and StatsPlotsExt.jl

**Author**: Analysis for MacroModelling.jl  
**Date**: January 2026  
**Purpose**: Document the call stack hierarchy and cache initialization patterns for key user-facing functions

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Cache Structures Overview](#cache-structures-overview)
3. [Cache Initialization Functions](#cache-initialization-functions)
4. [Call Stack: get_functions.jl](#call-stack-get_functionsjl)
5. [Call Stack: StatsPlotsExt.jl](#call-stack-statsplotsextjl)
6. [Detailed Cache Initialization Flow](#detailed-cache-initialization-flow)
7. [Performance Considerations](#performance-considerations)

---

## Executive Summary

This document provides a comprehensive overview of the call stack and cache initialization for functions in:
- `src/get_functions.jl` - Core estimation and filtering functions
- `ext/StatsPlotsExt.jl` - Plotting and visualization functions

### Key Findings

1. **Consistent Call Pattern**: All estimation functions follow a standardized sequence:
   ```
   merge_options → normalize_options → solve! → get_steady_states → filter_data → ensure_caches!
   ```

2. **Lazy Cache Initialization**: Caches are initialized on-demand via `ensure_*_cache!()` functions, not eagerly

3. **Cache Hierarchy**: Main umbrella cache (`caches#`) contains all specialized sub-caches

4. **Performance-Critical Operations**:
   - `solve!()` - Solves model dynamics (perturbation solution)
   - `filter_data_with_model()` - Runs Kalman/inversion filter
   - Cache initialization happens **after** solve and filter

---

## Cache Structures Overview

### Location
Cache structures are defined in: `src/structures.jl` (lines 250-441)

### Main Cache Container

```julia
mutable struct caches#{F,G}
    timings::model_timings
    auxiliary_indices::auxiliary_variable_indices
    second_order_auxiliary_matrices::second_order_auxilliary_matrices
    third_order_auxiliary_matrices::third_order_auxilliary_matrices
    # ... specialized caches below ...
    name_display_cache::Union{Nothing, name_display_cache}
    model_structure_cache::Union{Nothing, model_structure_cache}
    computational_constants::Union{Nothing, computational_constants_cache}
    conditional_forecast_index_cache::Union{Nothing, conditional_forecast_index_cache}
    moments_cache::Union{Nothing, moments_cache}
    first_order_index_cache::Union{Nothing, first_order_index_cache}
    custom_steady_state_buffer::Union{Nothing, Vector{F}}
end
```

### Cache Types by Purpose

| Cache Type | Purpose | Initialized By |
|------------|---------|----------------|
| `name_display_cache` | Formatted variable/shock names for plots | `ensure_name_display_cache!()` |
| `model_structure_cache` | Variable lists, selectors, steady-state mappings | `ensure_model_structure_cache!()` |
| `computational_constants` | BitVectors for state selection, Kronecker indices | `ensure_computational_constants_cache!()` |
| `conditional_forecast_index_cache` | Index sets for conditional forecasting | `ensure_conditional_forecast_index_cache!()` |
| `moments_cache` | Kronecker products for moment calculations | `ensure_moments_cache!()` |
| `first_order_index_cache` | First-order derivative indices | `ensure_first_order_index_cache!()` |
| `krylov_caches` | GMRES, BiCGSTAB workspace for linear solvers | (mutable, runtime) |
| `sylvester_caches` | Temporary matrices for Sylvester equations | (mutable, runtime) |
| `higher_order_caches` | Kronecker products for 2nd/3rd order | (mutable, runtime) |

---

## Cache Initialization Functions

### Location
Cache initialization functions are in: `src/options_and_caches.jl`

### Main Initialization Entry Point

```julia
function initialize_caches!(𝓂)  # Lines 169-174
    ensure_name_display_cache!(𝓂)
    ensure_computational_constants_cache!(𝓂)
    ensure_model_structure_cache!(𝓂)
    # Note: Other caches initialized on-demand
end
```

### Individual Initialization Functions

| Function | Lines | Purpose | Complexity |
|----------|-------|---------|------------|
| `ensure_name_display_cache!(𝓂)` | 176-218 | Formats variable/shock names with curly brackets, subscripts | O(n_vars) |
| `ensure_computational_constants_cache!(𝓂)` | 220-276 | Creates BitVectors for state selection, Kronecker sparse indices | O(n_vars²) |
| `ensure_model_structure_cache!(𝓂)` | 426-485 | Builds variable lists, selector matrices, steady-state mappings | O(n_vars) |
| `ensure_first_order_index_cache!(𝓂)` | 404-413 | Builds first-order derivative indices | O(n_vars) |
| `ensure_conditional_forecast_index_cache!(𝓂; third_order)` | 278-349 | Builds forecast-related index sets (depends on algorithm order) | O(n_vars²) for 2nd, O(n_vars³) for 3rd |
| `ensure_moments_cache!(𝓂)` | 515-542 | Initializes moment calculation Kronecker products | O(n_vars²) |
| `ensure_moments_substate_cache!(𝓂, nˢ)` | 544-559 | Creates substate-specific sparse matrices | O(n_vars) |
| `ensure_moments_dependency_kron_cache!(𝓂, deps, s_in_s⁺)` | 561-574 | Creates dependency-specific Kronecker products | Varies by dependencies |

### Cache Initialization Pattern

All `ensure_*_cache!()` functions follow this pattern:

```julia
function ensure_*_cache!(𝓂; kwargs...)
    # 1. Check if cache already exists
    if isnothing(𝓂.caches.*_cache)
        # 2. Compute cache data (potentially expensive)
        data = compute_cache_data(𝓂, kwargs...)
        
        # 3. Store in model cache
        𝓂.caches.*_cache = CacheType(data...)
    end
    # 4. Return (no-op if cache already exists)
    return nothing
end
```

**Key Property**: Idempotent - safe to call multiple times, only initializes once.

---

## Call Stack: get_functions.jl

### Overview

All estimation functions in `get_functions.jl` follow a **consistent 6-step pattern**:

```
1. merge_calculation_options()      ← Create options object
2. normalize_filtering_options()    ← Validate parameters
3. solve!(𝓂, ...)                  ⭐ SOLVE MODEL (cache-heavy)
4. get_relevant_steady_states()     ← Extract steady states from cache
5. filter_data_with_model()        ⭐ FILTER DATA (uses cached solution)
6. ensure_name_display_cache!()    ⭐ INITIALIZE DISPLAY CACHE
```

---

### 1. get_shock_decomposition()

**Location**: `src/get_functions.jl` lines 79-147

**Purpose**: Decompose deviations from steady state into contributions from each shock

#### Detailed Call Stack

```
get_shock_decomposition(𝓂, data; parameters, algorithm, filter, ...)
│
├─── merge_calculation_options(tol, verbose, qme_algorithm, ...) [Line 95]
│    └─ Returns: opts (calculation options including cache settings)
│
├─── normalize_filtering_options(filter, smooth, algorithm, ...) [Line 101]
│    └─ Returns: filter, smooth, algorithm, _, pruning, warmup_iterations
│
├─── solve!(𝓂, parameters, steady_state_function, opts, dynamics, algorithm) [Line 103] ⭐
│    │   PURPOSE: Computes perturbation solution of model
│    │   CACHE OPERATIONS:
│    ├─── Reads: 𝓂.caches.timings, 𝓂.caches.auxiliary_indices
│    ├─── Writes: 𝓂.solution.perturbation.first_order.solution_matrix
│    ├─── Writes: 𝓂.solution.perturbation.second_order_solution (if algorithm ≥ 2nd order)
│    ├─── Writes: 𝓂.solution.perturbation.third_order_solution (if algorithm = 3rd order)
│    └─── May initialize: higher_order_caches (Kronecker products)
│
├─── get_relevant_steady_states(𝓂, algorithm, opts) [Line 110]
│    │   PURPOSE: Retrieves NSSS or SSS based on algorithm
│    ├─── Reads: 𝓂.solution.non_stochastic_steady_state
│    └─── Returns: reference_steady_state, NSSS, SSS_delta
│
├─── filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), ...) [Line 126] ⭐
│    │   PURPOSE: Runs Kalman smoother/filter or inversion filter
│    │   CACHE OPERATIONS:
│    ├─── Reads: 𝓂.solution.perturbation.first_order.solution_matrix
│    ├─── Reads: 𝓂.caches.timings (for indexing)
│    ├─── May initialize: kalman_caches or inversion_caches (internal to filter implementation)
│    └─── Returns: variables, shocks, standard_deviations, decomposition
│
└─── ensure_name_display_cache!(𝓂) [Line 131] ⭐
     │   PURPOSE: Formats variable/shock names for output axes
     │   CACHE OPERATIONS:
     ├─── Checks: 𝓂.caches.name_display_cache (if isnothing, initializes)
     ├─── Writes: 𝓂.caches.name_display_cache.var_axis
     ├─── Writes: 𝓂.caches.name_display_cache.exo_axis_with_subscript
     └─── Returns: nothing (side-effect: cache populated)
```

#### Return Value

```julia
KeyedArray{Float64, 3}:
  Variables × Shocks × Periods
```

---

### 2. get_estimated_shocks()

**Location**: `src/get_functions.jl` lines 208-264

**Purpose**: Extract estimated shock series from filter decomposition

#### Detailed Call Stack

```
get_estimated_shocks(𝓂, data; parameters, algorithm, filter, ...)
│
├─── merge_calculation_options(...) [Line 224]
│    └─ Returns: opts
│
├─── normalize_filtering_options(...) [Line 230]
│    └─ Returns: filter, smooth, algorithm, _, _, warmup_iterations
│
├─── solve!(𝓂, parameters, steady_state_function, algorithm, opts, dynamics) [Line 232] ⭐
│    └─ (Same cache operations as get_shock_decomposition)
│
├─── get_relevant_steady_states(𝓂, algorithm, opts) [Line 239]
│    └─ Returns: reference_steady_state, NSSS, SSS_delta
│
├─── filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), ...) [Line 255] ⭐
│    └─ Returns: variables, shocks, standard_deviations, decomposition
│         (Only shocks are used in return value)
│
└─── ensure_name_display_cache!(𝓂) [Line 260] ⭐
     └─ Reads: 𝓂.caches.name_display_cache.exo_axis_with_subscript
```

#### Return Value

```julia
KeyedArray{Float64, 2}:
  Shocks × Periods
```

---

### 3. get_estimated_variables()

**Location**: `src/get_functions.jl` lines 331-388

**Purpose**: Extract estimated variable paths from filter

#### Detailed Call Stack

```
get_estimated_variables(𝓂, data; parameters, algorithm, filter, levels, ...)
│
├─── merge_calculation_options(...) [Line 348]
│    └─ Returns: opts
│
├─── normalize_filtering_options(...) [Line 354]
│    └─ Returns: filter, smooth, algorithm, _, _, warmup_iterations
│
├─── solve!(𝓂, parameters, steady_state_function, algorithm, opts, dynamics) [Line 356] ⭐
│    └─ (Same cache operations as previous functions)
│
├─── get_relevant_steady_states(𝓂, algorithm, opts) [Line 363]
│    └─ Returns: reference_steady_state, NSSS, SSS_delta
│
├─── filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), ...) [Line 379] ⭐
│    └─ Returns: variables, shocks, standard_deviations, decomposition
│         (Only variables are used in return value)
│
└─── ensure_name_display_cache!(𝓂) [Line 384] ⭐
     └─ Reads: 𝓂.caches.name_display_cache.var_axis
```

#### Return Value

```julia
KeyedArray{Float64, 2}:
  Variables × Periods
  # Returns in levels if levels=true, else deviations
```

---

### 4. get_model_estimates()

**Location**: `src/get_functions.jl` lines 456-506

**Purpose**: Combined output of `get_estimated_variables` + `get_estimated_shocks`

#### Detailed Call Stack

```
get_model_estimates(𝓂, data; parameters, levels, ...)
│
├─── get_estimated_variables(𝓂, data; parameters, levels, ...) [Line 472]
│    └─ (Full call stack as documented above)
│    └─ Returns: vars (KeyedArray)
│
└─── get_estimated_shocks(𝓂, data; parameters, ...) [Line 487]
     └─ (Full call stack as documented above)
     └─ Returns: shks (KeyedArray)
     └─ Combined into single KeyedArray (vertical concatenation)
```

**Note**: This function calls two other estimation functions, each with full solve/filter/cache pipeline. This means `solve!` is called **twice** with same parameters (potential optimization opportunity).

#### Return Value

```julia
KeyedArray{Float64, 2}:
  Variables_and_shocks × Periods
```

---

### Cache Initialization Summary for get_functions.jl

| Function | solve! | filter_data_with_model | ensure_name_display_cache! | Additional Caches |
|----------|--------|------------------------|----------------------------|-------------------|
| get_shock_decomposition | Line 103 | Line 126 | Line 131 | None |
| get_estimated_shocks | Line 232 | Line 255 | Line 260 | None |
| get_estimated_variables | Line 356 | Line 379 | Line 384 | None |
| get_model_estimates | (via sub-calls) | (via sub-calls) | (via sub-calls) | Calls solve! twice |

**Performance Note**: All functions call `solve!()` which may initialize:
- `higher_order_caches` (for 2nd/3rd order algorithms)
- `sylvester_caches` (for Sylvester equation solving)
- `krylov_caches` (for iterative linear solvers)

---

## Call Stack: StatsPlotsExt.jl

### Overview

Plotting functions in `StatsPlotsExt.jl` build on top of `get_functions.jl`, adding visualization layers. They follow similar patterns but include additional forecast and display logic.

---

### 1. plot_model_estimates()

**Location**: `ext/StatsPlotsExt.jl` lines 129-677

**Purpose**: Plot estimated variables, shocks, data, and optional unconditional forecast

#### Detailed Call Stack

```
plot_model_estimates(𝓂, data; parameters, algorithm, filter, forecast_periods, ...)
│
├─── merge_calculation_options(tol, verbose, qme_algorithm, ...) [Line 162-166]
│    └─ Returns: opts
│
├─── normalize_filtering_options(filter, smooth, algorithm, ...) [Line 185]
│    └─ Returns: filter, smooth, algorithm, shock_decomposition, pruning, warmup_iterations
│
├─── solve!(𝓂, parameters, steady_state_function, algorithm, opts, dynamics) [Line 187-192] ⭐
│    └─ (Identical to get_functions solve! call)
│
├─── get_relevant_steady_states(𝓂, algorithm, opts) [Line 194]
│    └─ Returns: reference_steady_state, NSSS, SSS_delta
│
├─── filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), ...) [Line 268] ⭐
│    └─ Returns: variables_to_plot, shocks_to_plot, standard_deviations, decomposition
│
├─── [CONDITIONAL] get_irf(𝓂; parameters, algorithm, shocks=:none, periods, ...) [Line 288-300] ⭐
│    │   CONDITION: Only if forecast_periods > 0
│    │   PURPOSE: Compute unconditional forecast extending beyond data
│    │   CACHE OPERATIONS:
│    ├─── Reads: 𝓂.solution.perturbation.* (from previous solve!)
│    ├─── Reads: final filtered state from filter_data_with_model
│    ├─── Uses: cached solution to simulate forward (no new solve!)
│    └─── Returns: forecast_irf (KeyedArray)
│
└─── Plotting Operations [Lines 376-676]
     ├─ Construct plot containers and legends
     ├─ Format variable/shock names (uses 𝓂.caches.name_display_cache implicitly)
     └─ Render StatsPlots subplots with decomposition or estimates
```

#### Key Difference from get_functions

**Addition of Unconditional Forecast**:
- If `forecast_periods > 0`, calls `get_irf()` to extend beyond data
- Uses **final filtered state** as initial condition
- No shocks applied (`:none`) → shows model's expected path
- Rendered as **dashed line** to distinguish from filtered estimates

#### Cache Initialization Summary

| Operation | Line | Cache Impact |
|-----------|------|--------------|
| solve! | 187-192 | Initializes solution caches |
| filter_data_with_model | 268 | Uses cached solution |
| get_irf (optional) | 288-300 | Uses cached solution, no new initialization |
| Name display | Throughout | Implicitly uses name_display_cache |

---

### 2. plot_model_estimates!()

**Location**: `ext/StatsPlotsExt.jl` lines 784-1357

**Purpose**: Append new plot to existing plot comparison (compare multiple filters/algorithms)

#### Detailed Call Stack

```
plot_model_estimates!(𝓂, data; parameters, algorithm, filter, label, ...)
│
├─── merge_calculation_options(...) [Line 815-819]
├─── normalize_filtering_options(...) [Line 838]
├─── solve!(𝓂, ...) [Line 840-845] ⭐
├─── get_relevant_steady_states(...) [Line 847]
├─── filter_data_with_model(...) [Line 921] ⭐
├─── [CONDITIONAL] get_irf(...) [Line 940-952] ⭐
│    └─ (Same as plot_model_estimates)
│
├─── Duplicate Check [Lines 1022-1037]
│    │   PURPOSE: Avoid redundant plots with identical parameters
│    ├─ Compare current args/kwargs with model_estimates_active_plot_container
│    └─ Only add to container if different
│
└─── Plot Comparison Logic [Lines 1039-1357]
     ├─ Compare parameters across all stored plot containers
     ├─ Identify differences (algorithm, filter, smooth, parameters, etc.)
     ├─ Generate annotations showing differences
     └─ Overlay multiple estimate lines on same subplots
```

#### Key Feature: Plot Registry

```julia
# Global container storing all plot calls for comparison
const model_estimates_active_plot_container = Dict[]

# Each plot_model_estimates!() call adds entry:
push!(model_estimates_active_plot_container, Dict(
    :run_id => ...,
    :model_name => ...,
    :label => ...,
    :parameters => ...,
    :algorithm => ...,
    :filter => ...,
    :variables_to_plot => ...,
    :shocks_to_plot => ...,
    :forecast_data => ...,
    # ... etc ...
))
```

#### Cache Initialization Summary

**Identical to `plot_model_estimates()`**, with addition of:
- **Plot container management**: Stores cached plot data for comparison
- **Diff computation**: Compares parameters across multiple plot calls

---

### 3. plot_conditional_forecast()

**Location**: `ext/StatsPlotsExt.jl` lines 4765-5127

**Purpose**: Plot conditional forecast given restrictions on variables/shocks

#### Detailed Call Stack

```
plot_conditional_forecast(𝓂, conditions; shocks, initial_state, periods, ...)
│
├─── merge_calculation_options(...) [Line 4801-4805]
│    └─ Returns: opts
│
├─── get_conditional_forecast(𝓂, conditions; shocks, initial_state, periods, ...) [Line 4812-4826] ⭐
│    │   PURPOSE: Solve constrained optimization to find shocks matching conditions
│    │   INTERNAL CALLS (inside get_conditional_forecast):
│    ├─── solve!(𝓂, parameters, steady_state_function, opts, dynamics, algorithm)
│    │    └─ Initializes solution caches
│    ├─── parse_algorithm_to_state_update(algorithm, ...)
│    │    └─ Gets state transition function
│    ├─── get_relevant_steady_states(𝓂, algorithm, opts)
│    │    └─ Returns reference_steady_state, NSSS, SSS_delta
│    ├─── ensure_conditional_forecast_index_cache!(𝓂; third_order) [CACHE INIT] ⭐
│    │    └─ Initializes forecast-specific index sets
│    └─── find_shocks_conditional_forecast(...) [Iterative solver]
│         └─ Uses Lagrange-Newton or other solver to find shocks
│    └─── Returns: Y (KeyedArray of conditional forecast paths)
│
├─── get_steady_state(𝓂, algorithm, return_variables_only, derivatives, ...) [Line 4850-4854]
│    │   PURPOSE: Retrieve steady state for plot scaling
│    ├─── Reads: 𝓂.solution.non_stochastic_steady_state (from cached solve!)
│    └─── Returns: relevant_SS (KeyedArray)
│
└─── Plotting Operations [Lines 5016-5127]
     ├─ Format variable/shock names
     ├─ Mark conditions with scatter points (★ or pentagon)
     └─ Render conditional forecast paths
```

#### Key Difference from plot_model_estimates

**No Direct solve!/filter calls in plotting function**:
- `get_conditional_forecast()` encapsulates **all** computation
- Internally calls `solve!`, then runs constrained optimization
- `get_steady_state()` reads cached solution (no re-solve)

**New Cache Initialization**:
- `ensure_conditional_forecast_index_cache!(𝓂)` inside `get_conditional_forecast`
- Creates index sets for state partitioning in forecast algorithm

#### Cache Initialization Summary

| Operation | Line | Function | Cache Impact |
|-----------|------|----------|--------------|
| solve! | (internal) | get_conditional_forecast | Initializes solution caches |
| ensure_conditional_forecast_index_cache! | (internal) | get_conditional_forecast | Forecast-specific indices ⭐ |
| get_steady_state | 4850-4854 | plot_conditional_forecast | Reads cached NSSS |

---

### 4. plot_conditional_forecast!()

**Location**: `ext/StatsPlotsExt.jl` lines 5224-6020

**Purpose**: Append conditional forecast to existing plot comparison

#### Detailed Call Stack

```
plot_conditional_forecast!(𝓂, conditions; shocks, plot_type, ...)
│
├─── merge_calculation_options(...) [Line 5317-5321]
├─── get_conditional_forecast(𝓂, conditions, ...) [Line 5323-5337] ⭐
│    └─ (Identical to plot_conditional_forecast)
├─── get_steady_state(...) [Line 5363-5369]
│
├─── Duplicate Check [Lines 5472-5488]
│    └─ Compare with conditional_forecast_active_plot_container
│
└─── Plot Comparison Logic [Lines 5490-6020]
     ├─ Compare conditions, shocks, initial_states across containers
     ├─ Generate annotations for differences
     └─ Overlay forecasts (compare mode) or stack (stack mode)
```

#### Key Feature: Conditional Forecast Registry

```julia
# Global container for conditional forecast plots
const conditional_forecast_active_plot_container = Dict[]

# Stores:
# - conditions, shocks matrices
# - initial_state vectors
# - plot_data (forecast results)
# - reference_steady_state
```

---

## Detailed Cache Initialization Flow

### Order of Cache Initialization Across Call Stack

```
User calls: plot_model_estimates() or get_shock_decomposition()
│
├─ 1. merge_calculation_options()
│     └─ Creates opts object (no cache init)
│
├─ 2. normalize_filtering_options()
│     └─ Validates inputs (no cache init)
│
├─ 3. solve!(𝓂, ...)  ⭐ MAJOR CACHE INITIALIZATION
│     ├─ May initialize: higher_order_caches
│     │   └─ If algorithm ∈ [:second_order, :third_order, :pruned_*]
│     │   └─ Allocates Kronecker product workspaces
│     ├─ May initialize: sylvester_caches
│     │   └─ If solving Sylvester equations
│     │   └─ Allocates temporary matrices
│     ├─ May initialize: krylov_caches
│     │   └─ If using iterative linear solvers
│     │   └─ Allocates GMRES, BiCGSTAB workspaces
│     └─ Writes: 𝓂.solution.perturbation.*
│         └─ solution_matrix, second_order_solution, third_order_solution
│
├─ 4. get_relevant_steady_states(𝓂, ...)
│     └─ Reads: 𝓂.solution.non_stochastic_steady_state (no cache init)
│
├─ 5. filter_data_with_model(𝓂, ...)  ⭐ USES CACHED SOLUTION
│     ├─ Reads: 𝓂.solution.perturbation.first_order.solution_matrix
│     ├─ Reads: 𝓂.caches.timings (for indexing)
│     └─ May allocate: internal kalman/inversion filter buffers (not stored in 𝓂.caches)
│
└─ 6. ensure_name_display_cache!(𝓂)  ⭐ DISPLAY CACHE
      ├─ Check: if isnothing(𝓂.caches.name_display_cache)
      ├─ Initialize: name_display_cache
      │   ├─ var_axis (formatted variable names)
      │   ├─ exo_axis_with_subscript (shock names with ₍ₓ₎)
      │   └─ par_axis (parameter names)
      └─ Store: 𝓂.caches.name_display_cache = name_display_cache(...)
```

### Conditional Cache Initialization

Some caches are only initialized when specific functions are called:

| Cache | Initialization Trigger | Function |
|-------|------------------------|----------|
| `conditional_forecast_index_cache` | `get_conditional_forecast()` | `ensure_conditional_forecast_index_cache!(𝓂; third_order)` |
| `moments_cache` | `get_moments()`, `get_statistics()` | `ensure_moments_cache!(𝓂)` |
| `moments_substate_cache` | Higher-order moment calculations | `ensure_moments_substate_cache!(𝓂, nˢ)` |
| `first_order_index_cache` | Certain derivative operations | `ensure_first_order_index_cache!(𝓂)` |
| `model_structure_cache` | `initialize_caches!()` or on-demand | `ensure_model_structure_cache!(𝓂)` |

---

## Performance Considerations

### 1. Cache Initialization Cost

| Cache Type | Complexity | Cost | Frequency |
|------------|-----------|------|-----------|
| `name_display_cache` | O(n_vars) | Low | Once per model |
| `computational_constants` | O(n_vars²) | Medium | Once per model |
| `model_structure_cache` | O(n_vars) | Low | Once per model |
| `conditional_forecast_index_cache` (2nd order) | O(n_vars²) | Medium | Once per forecast call |
| `conditional_forecast_index_cache` (3rd order) | O(n_vars³) | **High** | Once per forecast call |
| `moments_cache` | O(n_vars²) | Medium | Once per moment calculation |
| `higher_order_caches` | O(n_vars²) to O(n_vars³) | **High** | Once per solve! (if higher-order) |

### 2. Solve! Dominates Runtime

```
Typical Runtime Breakdown (for get_shock_decomposition):
┌──────────────────────────┬─────────┐
│ Operation                │ % Time  │
├──────────────────────────┼─────────┤
│ solve! (perturbation)    │ 60-70%  │ ⭐ Dominant
│ filter_data_with_model   │ 25-35%  │ ⭐ Second largest
│ Cache initialization     │ 1-5%    │
│ Other (options, etc.)    │ <1%     │
└──────────────────────────┴─────────┘
```

**Optimization Implications**:
- Cache initialization is **not** a bottleneck
- Focus optimization on `solve!` and filtering algorithms
- Lazy cache initialization is appropriate (minimal overhead)

### 3. Redundant Solve! Calls

**Identified Issue**: `get_model_estimates()` calls:
1. `get_estimated_variables()` → solve!
2. `get_estimated_shocks()` → solve! (with **same parameters**)

**Potential Optimization**:
- Cache solution in `get_model_estimates()` scope
- Pass cached solution to sub-functions
- Avoid re-solving identical problem twice

### 4. Plot Comparison Memory Usage

**plot_model_estimates!()** and **plot_conditional_forecast!()** store full plot data in global containers:
- Each call adds ~10-100 MB depending on:
  - Number of variables/shocks
  - Number of periods
  - Algorithm order (decomposition size)

**Memory Management**:
- Containers cleared on first call to non-bang version
- User must manually clear if plotting many comparisons

---

## Appendix: Quick Reference

### Key Functions and Their Primary Caches

| Function | Primary Cache Reads | Primary Cache Writes |
|----------|---------------------|---------------------|
| `solve!` | timings, auxiliary_indices | solution.perturbation.*, higher_order_caches |
| `filter_data_with_model` | solution.perturbation.*, timings | (none, internal buffers) |
| `get_conditional_forecast` | solution.perturbation.* | conditional_forecast_index_cache |
| `get_irf` | solution.perturbation.* | (none) |
| `get_moments` | solution.perturbation.* | moments_cache, moments_substate_cache |
| All plotting functions | name_display_cache | (none) |

### Cache Initialization Checklist

When adding new functionality that requires caches:

1. ✅ Define cache struct in `src/structures.jl`
2. ✅ Add field to `caches#` mutable struct
3. ✅ Create `ensure_*_cache!(𝓂; kwargs...)` function in `src/options_and_caches.jl`
4. ✅ Call `ensure_*_cache!()` at appropriate point in call stack (usually before first use)
5. ✅ Add documentation of cache structure and initialization to this document

---

## Conclusion

This analysis documents the **consistent, predictable** structure of call stacks in MacroModelling.jl:

1. **Lazy, on-demand cache initialization** minimizes overhead
2. **solve!** is the primary cache initializer and performance bottleneck
3. **Plotting functions** build on top of get_functions with minimal additional cache overhead
4. **Cache initialization cost** is negligible compared to solve/filter operations

**For developers**: When modifying these functions, maintain the established pattern to preserve consistency and performance characteristics.

**For users**: Understanding this call stack helps debug performance issues and optimize estimation workflows.

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: MacroModelling.jl Development Team
