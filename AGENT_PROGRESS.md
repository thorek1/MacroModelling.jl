# Agent Progress Log

## Current Session (2026-01-25) - Added find_shocks_workspace for Conditional Forecast Caching

### Summary

Added workspace struct to cache Kronecker product buffers used in `find_shocks_conditional_forecast`. The buffers depend only on `n_exo` (number of shocks) and are now lazily allocated once per model rather than every call.

### Changes Made This Session

1. **Added `find_shocks_workspace` struct in structures.jl:**
   - `n_exo::Int` - dimension tracking for reallocation checks
   - `kron_buffer::Vector{T}` - size n_exo^2, for ℒ.kron(x, x)
   - `kron_buffer2::Matrix{T}` - size n_exo^2 × n_exo, for ℒ.kron(J, x)
   - `kron_buffer²::Vector{T}` - size n_exo^3, for ℒ.kron(x, kron_buffer) (3rd order)
   - `kron_buffer3::Matrix{T}` - size n_exo^3 × n_exo, for ℒ.kron(J, kron_buffer) (3rd order)
   - `kron_buffer4::Matrix{T}` - size n_exo^3 × n_exo^2, for ℒ.kron(kron(J,J), x) (3rd order)

2. **Updated `workspaces` struct in structures.jl:**
   - Added `find_shocks::find_shocks_workspace{Float64}` field

3. **Added functions in options_and_caches.jl:**
   - `Find_shocks_workspace(;T::Type = Float64)` - constructor with 0-dimensional lazy init
   - `ensure_find_shocks_buffers!(ws, n_exo; third_order = false)` - lazy allocation for given n_exo
   - Updated `Workspaces()` to include `Find_shocks_workspace(T = T)`

4. **Updated find_shocks_conditional_forecast in find_shocks.jl:**
   - Added `ws::find_shocks_workspace{Float64}` as required positional parameter
   - Calls `ensure_find_shocks_buffers!(ws, n_exo; third_order = ...)` at start
   - Uses workspace buffers instead of allocating fresh ones each call

5. **Updated call sites in get_functions.jl:**
   - Both `get_conditional_forecast` call sites now pass `𝓂.workspaces.find_shocks`

6. **Updated OptimExt.jl:**
   - Added `find_shocks_workspace` to imports
   - Updated LBFGS method signature to include workspace parameter (for consistency, unused)

### Tests Verified

- ✅ Smets Wouters 2007 model: 2nd order conditional forecast
- ✅ Smets Wouters 2007 model: 3rd order conditional forecast

---

## Previous Session (2026-01-25) - Expanded Sylvester Workspace for Allocation Caching

### Summary

Extended `sylvester_workspace` struct to cache more allocations, similar to the lyapunov workspace pattern. The workspace now caches buffers for the doubling algorithm and properly manages dimensions.

### Changes Made This Session

1. **Expanded `sylvester_workspace` struct in structures.jl:**
   - Added dimension fields: `n` (rows), `m` (cols) for tracking buffer sizes
   - Added doubling algorithm buffers: `𝐀`, `𝐀¹`, `𝐁`, `𝐁¹`, `𝐂_dbl`, `𝐂¹`, `𝐂B`
   - Renamed existing Krylov buffers section for clarity
   - All buffers lazy-initialized to 0-dimensional arrays

2. **Updated `Sylvester_workspace` constructor in options_and_caches.jl:**
   - Now initializes all new fields (dimensions + doubling buffers)

3. **Added ensure functions in options_and_caches.jl:**
   - `ensure_sylvester_doubling_buffers!(ws, n, m)` - allocates A/B/C buffers for doubling
   - `ensure_sylvester_krylov_buffers!(ws, n, m)` - allocates Krylov method buffers

4. **Updated dense-dense doubling method in sylvester.jl:**
   - Now uses workspace buffers instead of allocating fresh copies
   - Calls `ensure_sylvester_doubling_buffers!` at start
   - Returns `copy(𝐂)` to avoid aliasing workspace

5. **Updated bartels_stewart method in sylvester.jl:**
   - Now uses `ensure_sylvester_krylov_buffers!` for `𝐂¹` and `tmp̄`

6. **Updated Krylov methods (bicgstab, dqgmres, gmres) in sylvester.jl:**
   - Replaced ad-hoc `if length(...) == 0` checks with `ensure_sylvester_krylov_buffers!`
   - Cleaner, more consistent allocation pattern

### Tests Verified

- ✅ RBC model 2nd order solution (uses Sylvester equation)
- ✅ RBC model 3rd order solution (uses Sylvester equation)

---

## Previous Session (2026-01-26) - Fixed AD Compatibility for Workspace Refactoring

### Summary

Completed all workspace refactoring with full AD (ForwardDiff and Zygote) compatibility:
- All workspaces (QME, Sylvester, Lyapunov) now use positional arguments
- No convenience/fallback wrappers
- Lazy initialization with 0-dimensional arrays
- Full ForwardDiff and Zygote gradient support through Kalman filter

### Fixes Made This Session

1. **Fixed rrule pullback return counts in perturbation.jl:**
   - Lines 173, 197, 225, 264, 271: Changed from 4 `NoTangent()` to 5 `NoTangent()`
   - With 4 positional args (∇₁, constants, qme_ws, sylv_ws), pullback needs 5 returns (1 for function + 4 for args)

2. **Fixed ForwardDiff.Dual method for sylvester in sylvester.jl:**
   - Added `tol::AbstractFloat = 1e-14` keyword parameter to match base method signature

3. **Fixed Lyapunov workspace dimension mismatch in lyapunov.jl:**
   - Added dynamic workspace dimension update at start of `solve_lyapunov_equation`:
   ```julia
   n = size(A, 1)
   if workspace.n != n
       workspace.n = n
   end
   ```

4. **Added @ignore_derivatives in MacroModelling.jl:**
   - Wrapped all `ensure_*_workspace!` calls in `@ignore_derivatives` to prevent Zygote from trying to differentiate through workspace initialization

5. **Fixed test imports in test_standalone_function.jl:**
   - Added `ensure_qme_workspace!, ensure_sylvester_1st_order_workspace!` to imports

### Tests Verified

- ✅ Basic model solution (1st order)
- ✅ ForwardDiff gradient on IRF
- ✅ Log-likelihood computation (Kalman filter)
- ✅ ForwardDiff gradient on Kalman filter
- ✅ Zygote gradient on Kalman filter  
- ✅ ForwardDiff and Zygote gradients match (within 1e-5 tolerance)

---

## Previous Session (2026-01-26) - Made QME workspace required positional argument

### Refactored QME workspace to eliminate convenience/fallback wrappers

**User Request:** Apply same pattern as Lyapunov - don't use convenience or fallback wrappers for QME workspace.

**Changes Made:**

1. **Updated perturbation.jl signatures:**
   - Main `calculate_first_order_solution(∇₁, constants, qme_ws; ...)` - now requires qme_ws as positional
   - `rrule` method - now requires qme_ws as positional
   - `Dual` method (ForwardDiff) - now requires qme_ws as positional

2. **Updated quadratic_matrix_equation.jl:**
   - `Dual` method now accepts workspace as positional and passes it through to inner call

3. **Updated call sites in moments.jl (2 locations):**
   - Uses `ensure_qme_workspace!(𝓂)` before call
   - Passes qme_ws as positional argument

4. **Updated call sites in get_functions.jl (4 locations):**
   - Uses `ensure_qme_workspace!(𝓂)` before each call
   - Passes qme_ws as positional argument

5. **Updated call sites in MacroModelling.jl (5 locations):**
   - Lines ~6451, ~6789, ~7251, ~7274, ~10594
   - All converted from `qme_workspace = qme_ws` keyword to positional

6. **Updated call site in kalman.jl:**
   - Uses `ensure_qme_workspace!(𝓂)` before call
   - Passes qme_ws as positional argument

7. **Updated call site in inversion.jl:**
   - Uses `ensure_qme_workspace!(𝓂)` before call
   - Passes qme_ws as positional argument

### Nonlinear solver workspace

**Analysis:** The nonlinear solver workspace is already embedded in the `function_and_jacobian.workspace` field. This is acceptable since it's part of the function struct passed through the code - no convenience wrappers needed.

### Tests verified

- ✅ RBC model: `solve!`
- ✅ RBC model: `get_irf`
- ✅ RBC model: `simulate`
- ✅ RBC model: `get_moments` (variance)
- ✅ RBC model with doubling algorithm: `get_irf`
- ✅ RBC model with doubling algorithm: `simulate`

---

## Previous Session (2026-01-26) - Made Lyapunov workspace required positional argument

### Refactored to eliminate convenience/fallback wrappers

**User Request:** Don't use convenience or fallback wrappers - thread workspace through all code paths.

**Changes Made:**

1. **Removed convenience wrapper from lyapunov.jl:**
   - Eliminated the no-workspace version that created temporary workspace
   - Now only one signature: `solve_lyapunov_equation(A, C, workspace; ...)`

2. **Updated Kalman filter chain (kalman.jl):**
   - `calculate_loglikelihood(::Val{:kalman}, ...)` - added `lyap_ws::lyapunov_workspace` as final positional arg
   - `calculate_kalman_filter_loglikelihood` (Vector{Symbol} version) - added workspace as positional arg
   - `calculate_kalman_filter_loglikelihood` (Vector{String} version) - added workspace as positional arg
   - `calculate_kalman_filter_loglikelihood` (Vector{Int} version) - added workspace as positional arg
   - `get_initial_covariance(::Val{:theoretical}, ...)` - added workspace, passes to solve_lyapunov_equation
   - `get_initial_covariance(::Val{:diagonal}, ...)` - added workspace for API consistency (unused)

3. **Updated inversion filter (inversion.jl):**
   - `calculate_loglikelihood(::Val{:inversion}, ...)` - added `lyap_ws::lyapunov_workspace` for API consistency (unused by inversion filter)

4. **Updated caller in get_functions.jl:**
   - `get_loglikelihood` now calls `ensure_lyapunov_workspace_1st_order!(𝓂)` before calling `calculate_loglikelihood`
   - Workspace passed through to Kalman/inversion filter

5. **All call sites in moments.jl already updated (previous work):**
   - 4 locations pass workspace as positional argument

6. **All call sites in get_functions.jl already updated (previous work):**
   - 2 variance decomposition locations pass workspace as positional argument

### Tests verified

- ✅ RBC model: `get_moments` (uses Lyapunov solver)
- ✅ RBC model: `get_variance_decomposition`
- ✅ RBC model: `get_conditional_variance_decomposition`

---

## Previous Session (2026-01-26) - Added Lyapunov workspace for all moment orders

### Created lyapunov_workspace struct to avoid allocations in Lyapunov solver

**User Request:** Add a workspace for each moment order (1st, 2nd, 3rd) and include as many algorithms as useful, including Krylov method caches.

**Analysis:**

The Lyapunov equation solver (`A * X * A' + C = X`) supports multiple algorithms:
- **:doubling** - Iterative doubling algorithm (fast, precise) - 5 n×n matrix allocations
- **:bartels_stewart** - Uses MatrixEquations.lyapd (dense only, precise)
- **:bicgstab** - Krylov iterative method (less precise)
- **:gmres** - Krylov iterative method (less precise)

Matrix dimensions vary by moment order:
- 1st order: `nVars × nVars`
- 2nd order: `(nˢ + nˢ + nˢ²) × (nˢ + nˢ + nˢ²)` - augmented state
- 3rd order: Even larger augmented state

**Changes Made:**

1. **New struct in structures.jl:**
   - Added `lyapunov_workspace{T}` struct with:
     - Doubling algorithm buffers: `𝐂`, `𝐂¹`, `𝐀`, `𝐂A`, `𝐀²` (5 n×n matrices)
     - Krylov buffers: `tmp̄`, `𝐗` (n×n matrices), `b` (n² vector)
     - Pre-allocated Krylov solvers: `bicgstab_workspace`, `gmres_workspace`

2. **Updated `workspaces` struct (structures.jl):**
   - Added `lyapunov_1st_order::lyapunov_workspace{Float64}`
   - Added `lyapunov_2nd_order::lyapunov_workspace{Float64}`
   - Added `lyapunov_3rd_order::lyapunov_workspace{Float64}`

3. **Added constructors in options_and_caches.jl:**
   - `Lyapunov_workspace(n::Int; T=Float64)` - creates workspace with Krylov solver pre-allocation
   - `ensure_lyapunov_workspace!(workspaces, n, order)` - ensure workspace is properly sized by order
   - `ensure_lyapunov_workspace_1st_order!(𝓂)` - convenience for first order

4. **Updated solve_lyapunov_equation functions (lyapunov.jl):**
   - Main dispatch function: now requires workspace as positional argument
   - Added keyword-argument version for backward compatibility and AD (creates temporary workspace)
   - `:bartels_stewart` - accepts workspace for API consistency
   - `:doubling` (dense-dense) - uses workspace buffers, returns `copy(𝐂)`
   - `:doubling` (sparse variants) - accepts workspace for API consistency
   - `:bicgstab` - uses workspace buffers and pre-allocated `Krylov.BicgstabWorkspace`
   - `:gmres` - uses workspace buffers and pre-allocated `Krylov.GmresWorkspace`

5. **Updated call sites in moments.jl (4 locations):**
   - `calculate_covariance`: uses workspace for Float64, keyword version for AD types
   - `calculate_second_order_moments_with_covariance`: same pattern
   - `calculate_third_order_moments_with_autocorrelation`: same pattern
   - `calculate_third_order_moments`: same pattern

6. **Updated call sites in get_functions.jl (2 locations):**
   - Variance decomposition calls now use workspace

7. **Note on kalman.jl:**
   - The call in `get_initial_covariance` uses keyword arguments
   - This is acceptable since Kalman filter is typically called during estimation/AD

### QME workspace simplification (earlier in this session)

**User Request:** Make QME workspace required (not optional), remove `use_workspace` conditional logic.

**Changes Made:**
- Updated QME function signatures: workspace is now required positional argument
- Updated `calculate_first_order_solution`: creates temporary workspace if none provided
- Added same pattern to rrule for AD

### Files modified

- **src/structures.jl**: Added `lyapunov_workspace` struct, updated `workspaces` struct with 3 lyapunov fields
- **src/options_and_caches.jl**: Added `Lyapunov_workspace` constructor, `ensure_lyapunov_workspace!` functions
- **src/algorithms/lyapunov.jl**: Updated all function signatures to accept workspace, updated doubling and Krylov algorithms to use workspace buffers
- **src/moments.jl**: Updated 4 call sites to use workspace with type dispatch (Float64 vs AD types)
- **src/get_functions.jl**: Updated 2 call sites to use workspace
- **src/perturbation.jl**: Updated workspace creation logic for QME

### Tests verified

- ✅ RBC model: `get_covariance`, `get_variance_decomposition`
- ✅ Smets_Wouters_2007 model: `get_covariance`, `get_variance_decomposition`
- ✅ Smets_Wouters_2007 model: `get_moments(algorithm = :pruned_second_order)` with derivatives (AD)

---

## Previous Session (2026-01-26) - Added QME workspace for doubling algorithm

### Created qme_workspace struct to avoid allocations in QME doubling algorithm

**User Request:** Analyze quadratic_matrix_equation.jl for workspace opportunities similar to nonlinear_solver_workspace.

**Analysis:**

The quadratic matrix equation solver has two main algorithms:
- **:schur** - Uses Schur decomposition, mostly creates views/slices
- **:doubling** - Iterative algorithm with many n×n matrix allocations

The **doubling algorithm** had 13 n×n matrix allocations per call:
- `E`, `F` - working matrices for recurrence
- `X`, `Y` - current iteration solution matrices
- `X_new`, `Y_new`, `E_new`, `F_new` - next iteration matrices
- `temp1`, `temp2`, `temp3` - temporary matrices for intermediate computations
- `B̄` - copy of B for LU factorization
- `AXX` - temporary for residual computation

**Changes Made:**

1. **New struct in structures.jl:**
   - Added `qme_workspace{T}` struct with all 13 matrices
   - Documented purpose and usage

2. **Updated `workspaces` struct (structures.jl):**
   - Added `qme::qme_workspace{Float64}` field

3. **Added constructors in options_and_caches.jl:**
   - `Qme_workspace(n::Int; T=Float64)` - creates workspace of dimension n×n
   - `ensure_qme_workspace!(𝓂)` and `ensure_qme_workspace!(workspaces, n)` - ensure workspace is properly sized

4. **Updated `Workspaces` constructor (options_and_caches.jl):**
   - Now initializes QME workspace with size 0 (resized when needed)

5. **Updated solve_quadratic_matrix_equation functions (quadratic_matrix_equation.jl):**
   - Added optional `workspace::Union{Nothing, qme_workspace{R}}` parameter to all function signatures
   - Main dispatch function passes workspace to algorithm-specific functions
   - `:doubling` algorithm uses workspace if provided and correctly sized
   - `:schur` algorithm accepts but ignores workspace (for API consistency)

6. **Updated calculate_first_order_solution (perturbation.jl):**
   - Added optional `qme_workspace` parameter
   - Passes workspace to solve_quadratic_matrix_equation

7. **Updated call sites in MacroModelling.jl (5 locations):**
   - All calls to `calculate_first_order_solution` now call `ensure_qme_workspace!(𝓂)` first
   - Pass `qme_workspace = qme_ws` to the function

8. **Fixed minor issue in nonlinear_solver.jl:**
   - Replaced `iters = [0,0]` array allocation with scalar variables `grad_iter`, `func_iter`

### Files modified

- **src/structures.jl**: Added `qme_workspace` struct, updated `workspaces` struct
- **src/options_and_caches.jl**: Added `Qme_workspace` constructor, `ensure_qme_workspace!` functions, updated `Workspaces`
- **src/algorithms/quadratic_matrix_equation.jl**: Added workspace parameter to all function signatures, updated doubling algorithm to use workspace
- **src/perturbation.jl**: Added `qme_workspace` parameter to `calculate_first_order_solution` and rrule
- **src/MacroModelling.jl**: Updated 5 call sites to pass workspace
- **src/algorithms/nonlinear_solver.jl**: Replaced `iters` array with scalar variables

### Tests verified

- ✅ RBC model IRF computation works
- ✅ RBC model with doubling algorithm works
- ✅ Smets_Wouters_2007 model IRF computation works
- ✅ Smets_Wouters_2007 with doubling algorithm works
- ✅ Workspace correctly sized (e.g., 44×44 for Smets_Wouters_2007)

---

## Previous Session (2026-01-26) - Moved buffers into nonlinear_solver_workspace

### Consolidated all solver buffers into workspace struct

**User Request:** Move `jac_buffer`, `chol_buffer`, `lu_buffer`, and `func_buffer` from `function_and_jacobian` into `nonlinear_solver_workspace`.

**Changes Made:**

1. **Updated `nonlinear_solver_workspace` struct (structures.jl):**
   - Added `func_buffer::Vector{T}` - buffer for function evaluation
   - Added `jac_buffer::AbstractMatrix{T}` - buffer for Jacobian
   - Added `chol_buffer::LinearCache` - Cholesky factorization cache
   - Added `lu_buffer::LinearCache` - LU factorization cache

2. **Simplified `function_and_jacobian` struct (structures.jl):**
   - Now only contains: `func`, `jac`, `workspace`
   - All buffers are accessed via `workspace`

3. **Updated constructor `Nonlinear_solver_workspace` (options_and_caches.jl):**
   - Now takes buffers as arguments instead of just problem dimension
   - Signature: `Nonlinear_solver_workspace(func_buffer, jac_buffer, chol_buffer, lu_buffer)`

4. **Updated initialization in MacroModelling.jl:**
   - `function_and_jacobian` now constructed with just `(func, jac, workspace)`
   - Buffers passed to workspace constructor

5. **Updated nonlinear_solver.jl:**
   - All buffer accesses changed from `fnj.X` to `ws.X`
   - Both `levenberg_marquardt` and `newton` updated

6. **Updated buffer accesses in MacroModelling.jl `solve_ss` and `block_solver`:**
   - Changed `SS_solve_block.ss_problem.func_buffer` to `SS_solve_block.ss_problem.workspace.func_buffer`
   - Changed `SS_solve_block.ss_problem.jac_buffer` to `SS_solve_block.ss_problem.workspace.jac_buffer`

### Files modified

- **src/structures.jl**: Moved buffer fields to workspace, simplified `function_and_jacobian`
- **src/options_and_caches.jl**: Updated constructor signature
- **src/MacroModelling.jl**: Updated constructor calls and buffer accesses
- **src/algorithms/nonlinear_solver.jl**: Updated all buffer accesses to go through `ws`

### Tests verified

- ✅ RBC model works
- ✅ Smets_Wouters_2007 model works (3 solve blocks, uses nonlinear solver)
- ✅ Workspace contains all buffers correctly

---

## Previous Session (2026-01-26) - Added nonlinear_solver_workspace struct

### Created workspace struct to avoid allocations in nonlinear solvers

**User Request:** Move temporary arrays from `levenberg_marquardt` and `newton` functions to a new `nonlinear_solver_workspace` struct to avoid per-call allocations.

**Changes Made:**

1. **New struct in structures.jl:**
   - Added `nonlinear_solver_workspace{T}` struct with:
     - `current_guess`, `previous_guess`, `guess_update` - iteration vectors
     - `current_guess_untransformed`, `previous_guess_untransformed` - for coordinate transformation
     - `best_previous_guess`, `best_current_guess` - output vectors
     - `factor` - multipurpose vector (transformation jacobian diagonal / temp storage)
     - `u_bounds`, `l_bounds` - transformed bounds

2. **Updated `function_and_jacobian` struct (structures.jl):**
   - Made generic with type parameter `{T <: Real}`
   - Added `workspace::nonlinear_solver_workspace{T}` field

3. **Added constructor in options_and_caches.jl:**
   - `Nonlinear_solver_workspace(n; T=Float64)` - creates workspace of size n

4. **Updated MacroModelling.jl:**
   - Both `function_and_jacobian` constructor calls now create and pass workspace
   - Regular problem workspace has size `ng` (number of unknowns)
   - Extended problem workspace has size `ng + nx` (unknowns + parameters)

5. **Updated nonlinear_solver.jl:**
   - `levenberg_marquardt`: Now uses `fnj.workspace` vectors instead of allocating
   - `newton`: Now uses `fnj.workspace.current_guess` instead of mutating input

### Files modified

- **src/structures.jl**: Added `nonlinear_solver_workspace` struct, updated `function_and_jacobian`
- **src/options_and_caches.jl**: Added `Nonlinear_solver_workspace` constructor
- **src/MacroModelling.jl**: Updated 2 `function_and_jacobian` constructor calls
- **src/algorithms/nonlinear_solver.jl**: Updated both `levenberg_marquardt` and `newton`

### Tests verified

- ✅ RBC model IRF computation works
- ✅ Smets_Wouters_2007 model IRF computation works (3 solve blocks)
- ✅ Workspace correctly initialized with proper sizes

---

## Previous Session (2026-01-26) - Renamed cache structs to indices (they are constant)

### Renaming: `*_cache` structs/fields to `*_indices`

**User Request:** The `moments_substate_cache`, `moments_dependency_kron_cache`, `substate_cache`, and `dependency_kron_cache` are actually constant and not caches. Rename accordingly.

**Changes Made:**

1. **Struct renames in structures.jl:**
   - `moments_substate_cache` → `moments_substate_indices`
   - `moments_dependency_kron_cache` → `moments_dependency_kron_indices`

2. **Field renames in `third_order` struct (structures.jl):**
   - `substate_cache` → `substate_indices`
   - `dependency_kron_cache` → `dependency_kron_indices`

3. **Function renames in options_and_caches.jl:**
   - `ensure_moments_substate_cache!` → `ensure_moments_substate_indices!`
   - `ensure_moments_dependency_kron_cache!` → `ensure_moments_dependency_kron_indices!`

4. **Updated usages in moments.jl:**
   - Two occurrences in `calculate_third_order_moments_with_autocorrelation` and `calculate_third_order_moments`
   - Changed local variable names from `substate_cache` to `substate_indices`
   - Updated function calls from `ensure_moments_substate_cache!` to `ensure_moments_substate_indices!`
   - Updated function calls from `ensure_moments_dependency_kron_cache!` to `ensure_moments_dependency_kron_indices!`

### Files modified
- **src/structures.jl**: Renamed structs and fields in `third_order`
- **src/options_and_caches.jl**: Renamed functions, updated Dict type declarations and field accesses
- **src/moments.jl**: Updated function calls and local variable names in both `calculate_third_order_moments_with_autocorrelation` and `calculate_third_order_moments`

### Tests verified
- ✅ IRF computation works
- ✅ `pruned_third_order` moments computation works (tests the renamed structs)

---

## Previous Session (2026-01-26) - Verified CI test functionality locally

**Test Results - All Pass:**

1. **RBC model:** Zygote/ForwardDiff gradients match (max diff ~1.7e-14)
2. **FS2000 model:** Zygote/ForwardDiff gradients match (max diff ~9e-13)
3. **Smets_Wouters_2007 model:** Zygote/ForwardDiff gradients match (max diff ~1e-10)
4. **Backus_Kehoe_Kydland_1992 model (CI test model):**
   - `standard_deviation`: Zygote vs FiniteDiff max diff: 1.04e-4, passes `isapprox(..., rtol=1e-5)` ✅
   - `variance`: Zygote vs FiniteDiff max diff: 3.1e-5, passes `isapprox(..., rtol=1e-5)` ✅

**Conclusion:** The struct-based interface for `calculate_jacobian`, `calculate_hessian`, and `calculate_third_order_derivatives` is working correctly. The rrules properly propagate gradients through Zygote. All local tests pass. The CI failure may have been from a previous code state or environment-specific issues.

### Files verified working
- **src/structures.jl**: `jacobian_functions`, `hessian_functions`, `third_order_derivatives_functions` structs
- **src/MacroModelling.jl**: rrules for derivative calculation functions
- **src/moments.jl**: Calls `calculate_jacobian` with `𝓂.functions.jacobian` struct
- **src/get_functions.jl**: Derivative function calls use struct interface

---

## Previous Session (2026-01-25) - Refactored derivative functions to use structs instead of kwargs

### Refactoring: Pass derivative functions via struct instead of keyword arguments

**Problem:** The previous fix for Zygote gradient propagation used kwargs to pass derivative functions (`jacobian_parameters_func`, `jacobian_SS_and_pars_func`, etc.). User requested a cleaner interface using structs.

**Solution:** Created three new structs to bundle derivative functions:

1. `jacobian_functions` - Contains:
   - `f::Function` - Main jacobian function
   - `f_parameters::Function` - Jacobian w.r.t. parameters
   - `f_SS_and_pars::Function` - Jacobian w.r.t. SS_and_pars

2. `hessian_functions` - Contains:
   - `f::Function` - Main hessian function  
   - `f_parameters::Function` - Hessian w.r.t. parameters
   - `f_SS_and_pars::Function` - Hessian w.r.t. SS_and_pars

3. `third_order_derivatives_functions` - Contains:
   - `f::Function` - Main third order derivatives function
   - `f_parameters::Function` - Third order derivatives w.r.t. parameters  
   - `f_SS_and_pars::Function` - Third order derivatives w.r.t. SS_and_pars

Updated `model_functions` struct to use these instead of individual function fields.

**New function signatures:**
```julia
calculate_jacobian(parameters, SS_and_pars, caches_obj::caches, jacobian_funcs::jacobian_functions)
calculate_hessian(parameters, SS_and_pars, caches_obj::caches, hessian_funcs::hessian_functions)  
calculate_third_order_derivatives(parameters, SS_and_pars, caches_obj::caches, third_order_funcs::third_order_derivatives_functions)
```

**Simplified rrule:** The rrule pullback no longer needs to check for `nothing` - the struct guarantees all functions exist.

### Files modified
- **src/structures.jl**: Added 3 new structs, updated `model_functions` struct (removed 9 individual function fields, added 3 struct fields)
- **src/MacroModelling.jl**: Updated function signatures, rrules, all call sites, and function assignment code
- **src/moments.jl**: Updated all 5 derivative function calls
- **src/get_functions.jl**: Updated all derivative function calls
- **src/macros.jl**: Updated `model_functions` constructor to use struct instances

### Tests verified
- ✅ Model compilation successful (RBC model)
- ✅ IRF computation works
- ✅ Struct types correctly created (jacobian_functions, hessian_functions, third_order_derivatives_functions)
- ✅ `calculate_jacobian` with struct works correctly
- ✅ ForwardDiff and Zygote gradients match (max difference ~8e-15, machine precision)

---

## Previous Session (2026-01-25) - Fixed Zygote gradient propagation in moments.jl

### Bug Fix: Kalman filter gradient test failing - ForwardDiff and Zygote gradients don't match

**Problem:** The "Kalman filter and gradient" test in test_standalone_function.jl was failing because ForwardDiff gradients had non-zero values for all parameters, but Zygote gradients had zeros for parameters 4-8.

**Root Cause:** When `calculate_jacobian`, `calculate_hessian`, and `calculate_third_order_derivatives` are called without the derivative function kwargs (`jacobian_parameters_func`, `hessian_parameters_func`, etc.), the rrule returns `NoTangent()` causing Zygote to return zeros.

Several call sites in moments.jl were missing these kwargs:
- Line 17: `calculate_jacobian` in `calculate_covariance`
- Line 69: `calculate_jacobian` in `calculate_mean`
- Line 81: `calculate_hessian` in `calculate_mean`
- Line 183: `calculate_hessian` in `calculate_second_order_moments`
- Line 305: `calculate_hessian` in `calculate_second_order_moments_with_covariance`

**Fix:** Added the derivative function kwargs to all calls:
```julia
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian,
                        jacobian_parameters_func = 𝓂.functions.jacobian_parameters,
                        jacobian_SS_and_pars_func = 𝓂.functions.jacobian_SS_and_pars)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian,
                        hessian_parameters_func = 𝓂.functions.hessian_parameters,
                        hessian_SS_and_pars_func = 𝓂.functions.hessian_SS_and_pars)
```

### Files modified
- **src/moments.jl**: Added kwargs to 5 derivative function calls

### Tests verified
- ✅ ForwardDiff and Zygote gradients now match (max difference < 1e-6)

---

## Previous Session (2026-01-25) - Reworked outdated logic with new outdated_caches struct

### Refactoring: Replaced Set-based algorithm tracking with Bool fields

**Problem:** The previous approach used `outdated_algorithms::Set{Symbol}` to track which algorithms needed recalculation. This was cumbersome with Set operations like `push!`, `setdiff`, and `∈` checks.

**Solution:** Created a new `outdated_caches` struct with Bool fields for each cache element that can be outdated:
- `non_stochastic_steady_state::Bool`
- `jacobian::Bool`
- `hessian::Bool`
- `third_order_derivatives::Bool`
- `first_order_solution::Bool`
- `second_order_solution::Bool`
- `pruned_second_order_solution::Bool`
- `third_order_solution::Bool`
- `pruned_third_order_solution::Bool`

The `solution` struct now contains `outdated::outdated_caches` instead of `outdated_algorithms::Set{Symbol}` and `outdated_NSSS::Bool`.

### Files modified
- **src/structures.jl**: 
  - Added new `outdated_caches` struct with Bool fields
  - Simplified `solution` struct to contain `outdated::outdated_caches` and `functions_written::Bool`
  - Fixed pre-existing type invariance bug: `CircularBuffer{Vector{Vector{<: Real}}}` → `CircularBuffer{Vector{Vector{Float64}}}`
- **src/macros.jl**: Updated `solution` initialization to create `outdated_caches` with all fields set to `true`
- **src/MacroModelling.jl**: 
  - Updated `clear_solution_caches!` to set all Bool fields to `true`
  - Updated `set_custom_steady_state_function!` to set all Bool fields
  - Updated `write_symbolic_derivatives!` to mark solution fields as outdated
  - Updated `solve!` function to check Bool fields instead of Set membership
  - Updated `write_parameters_input!` to mark appropriate fields as outdated when parameters change
- **src/get_functions.jl**: Updated `get_moments` to use `outdated.non_stochastic_steady_state`

### Tests verified
- ✅ Initial outdated state correctly set to `true` for solutions, `false` for NSSS after model creation
- ✅ First order solution correctly marks `first_order_solution = false` after solve
- ✅ Second and third order solutions correctly mark their respective fields as `false` after solve
- ✅ `write_parameters_input!` correctly marks solutions as outdated when parameters change
- ✅ `clear_solution_caches!` correctly marks all fields as `true` (outdated)

---

## Previous Session (2026-01-25) - Fixed Zygote AD derivatives

### Bug Fix: Zygote returning zeros for higher-order derivatives

**Problem:** `get_solution` with parameter input was returning all zeros for Zygote jacobians of the first order solution matrix (element 2) and higher when using `:third_order` algorithm. FiniteDifferences produced correct non-zero values.

**Root Cause:** The `rrule` definitions for `calculate_jacobian`, `calculate_hessian`, and `calculate_third_order_derivatives` had keyword arguments (`jacobian_parameters_func`, etc.) that defaulted to `nothing`. When these kwargs were `nothing`, the pullback returned `NoTangent()` for all arguments, causing Zygote to return zeros.

The calls from `get_solution` did not pass these kwargs, so the rrule always used the default `nothing` values.

**Fix:**
1. Added kwargs to the primary function signatures for all three derivative calculation functions
2. Updated calls in `get_solution` to pass the derivative functions as kwargs
3. The derivative functions are stored in `𝓂.functions.jacobian_parameters`, `𝓂.functions.jacobian_SS_and_pars`, etc.

### Files modified
- **src/MacroModelling.jl**: Added kwargs to `calculate_jacobian`, `calculate_hessian`, `calculate_third_order_derivatives` function signatures
- **src/get_functions.jl**: Updated `get_solution` to pass the derivative function kwargs when calling these functions

### Tests verified
- ✅ Zygote jacobian for `:third_order` solution on FS2000 model now matches FiniteDifferences
- ✅ All three solution elements (SS, first order, second order) have correct non-zero Zygote derivatives

---

## Previous Session (2026-01-25)

### Completed (Further caches consolidation)

- Moved `non_stochastic_steady_state::Vector{Float64}` from `solution` struct to `caches`
- Moved `solver_cache`, `∂equations_∂parameters`, `∂equations_∂SS_and_pars` from `non_stochastic_steady_state` struct to `caches`
- Deleted the `perturbation` struct (was already empty)
- Simplified `solution` struct (now only contains `outdated_algorithms`, `outdated_NSSS`, `functions_written`)
- Simplified `non_stochastic_steady_state` struct (now only contains `solve_blocks_in_place`, `dependencies`)

### Files modified
- **src/structures.jl**: Added 4 new fields to `caches`, deleted `perturbation` struct, simplified `solution` and `non_stochastic_steady_state` structs
- **src/macros.jl**: Updated constructors for `caches`, `non_stochastic_steady_state`, and `solution`
- **src/MacroModelling.jl**: Updated all `𝓂.solution.non_stochastic_steady_state` → `𝓂.caches.non_stochastic_steady_state`, `𝓂.NSSS.solver_cache` → `𝓂.caches.solver_cache`, etc.
- **src/get_functions.jl**: Updated all references
- **src/moments.jl**: Updated all references
- **src/inspect.jl**: Updated all references
- **test/test_standalone_function.jl**: Updated test references
- **test/functionality_tests.jl**: Updated test references
- **test/runtests.jl**: Updated test references

### Tests verified (2026-01-25)
- ✅ RBC model with first, second, and third order perturbation solutions
- ✅ NSSS vector correctly stored in `caches.non_stochastic_steady_state`
- ✅ Stochastic steady states stored correctly in `caches.second_order_stochastic_steady_state`, `caches.third_order_stochastic_steady_state`

---

## Previous Session (2026-01-26)

### Completed (struct consolidation - rename perturbation_derivatives to caches)

- Renamed `perturbation_derivatives` struct to `caches` since it holds all cached buffers
- Moved perturbation solution fields into `caches` struct:
  - `first_order_solution_matrix` (was in `perturbation_solution`)
  - `qme_solution` (was in `perturbation.qme_solution`)
  - `second_order_stochastic_steady_state` (was in `second_order_perturbation_solution`)
  - `second_order_solution` (was in `perturbation.second_order_solution`)
  - `pruned_second_order_stochastic_steady_state` (was in `second_order_perturbation_solution`)
  - `third_order_stochastic_steady_state` (was in `third_order_perturbation_solution`)
  - `third_order_solution` (was in `perturbation.third_order_solution`)
  - `pruned_third_order_stochastic_steady_state` (was in `third_order_perturbation_solution`)
- Removed individual perturbation solution structs: `perturbation_solution`, `second_order_perturbation_solution`, `third_order_perturbation_solution`
- Emptied `perturbation` struct (kept for potential future use)
- Changed `ℳ` model struct field from `derivatives::perturbation_derivatives` to `caches::caches`

### Files modified:
- **src/structures.jl**: Renamed struct, added 9 new fields, removed old solution structs
- **src/macros.jl**: Updated constructor to create `caches()` with all 18 fields
- **src/MacroModelling.jl**: Updated all `𝓂.derivatives` → `𝓂.caches`, `𝓂.solution.perturbation.X` → `𝓂.caches.X`
- **src/get_functions.jl**: Updated all references
- **src/moments.jl**: Updated all references
- **src/filter/kalman.jl**: Updated all references
- **src/filter/inversion.jl**: Updated all references
- **benchmark/benchmarks.jl**: Added `calculate_jacobian_for_bench` wrapper, updated references

### Tests verified (2026-01-26):
- ✅ RBC model with all 5 perturbation algorithms:
  - First order
  - Second order
  - Pruned second order
  - Third order
  - Pruned third order

---

### Previous (2026-01-26, derivative function signature refactoring - no backward compatibility wrappers)

- Refactored `calculate_jacobian`, `calculate_hessian`, and `calculate_third_order_derivatives` to accept specific objects instead of full model struct
- **Removed backward-compatible wrappers** per user request - only new signatures are available
- New primary signatures allow better type inference:
  - `calculate_jacobian(parameters, SS_and_pars, caches_obj::caches, jacobian_func::Function)`
  - `calculate_hessian(parameters, SS_and_pars, caches_obj::caches, hessian_func::Function)`
  - `calculate_third_order_derivatives(parameters, SS_and_pars, caches_obj::caches, third_order_derivatives_func::Function)`
- Updated `rrule` methods to use new signatures with optional parameter functions for pullback support

### Updated call sites across codebase:

- **MacroModelling.jl**: `calculate_second_order_stochastic_steady_state`, `calculate_third_order_stochastic_steady_state`, `solve!`, `get_relevant_steady_state_and_state_update`, OBC section
- **moments.jl**: `calculate_covariance`, `calculate_mean`, `calculate_second_order_moments`, `calculate_second_order_moments_with_covariance`, `calculate_third_order_moments_with_autocorrelation`, `calculate_third_order_moments`
- **get_functions.jl**: `get_irf`, `get_solution`, FEVD functions, variance decomposition functions
- **filter/kalman.jl**: `calculate_kalman_filter_loglikelihood`
- **filter/inversion.jl**: shock decomposition function
- **benchmark/benchmarks.jl**: benchmark definitions for jacobian calculation

### Tests verified (2026-01-26)
- RBC model with first, second, and third order perturbation solutions
- Moments calculation at all orders

---

## Previous Session (2026-01-24)

### Completed (derivative function signature refactoring - initial version)

- Initial refactoring with backward-compatible wrappers (later removed)

## 2026-01-25

### Completed (2026-01-25, bug fix for ForwardDiff NSSS derivatives)

- Fixed bug where ForwardDiff-based derivative calculations would fail with `FieldError: type Array has no field 'nzval'`
- Root cause: Two lines at 8182 and 8210 in `write_functions_mapping!` were assigning the wrong function to `𝓂.functions.NSSS_∂equations_∂parameters` and `𝓂.functions.NSSS_∂equations_∂SS_and_pars`
- The surrounding code that generated the correct NSSS derivative functions was commented out, but these assignment lines were left active, causing them to assign the jacobian function (which expects sparse output) instead
- Fix: Commented out the erroneous assignment lines since the correct functions are already set in `write_ss_check_function!`
- Affected models: Any model with enough variables to make the NSSS Jacobian dense (e.g., Guerrieri_Iacoviello_2017 with 126x39 matrix)
- Tests verified: `get_steady_state(Guerrieri_Iacoviello_2017)` and Zygote gradient on RBC_CME model

### Completed (2026-01-25, state_update functions refactoring)

- Moved `state_update` and `state_update_obc` Function fields from perturbation solution structs into `model_functions`:
  - Removed from `perturbation_solution` (first order)
  - Removed from `second_order_perturbation_solution`
  - Removed from `third_order_perturbation_solution`

- Added 10 new function fields to `model_functions` for state updates:
  - `first_order_state_update`, `first_order_state_update_obc`
  - `second_order_state_update`, `second_order_state_update_obc`
  - `pruned_second_order_state_update`, `pruned_second_order_state_update_obc`
  - `third_order_state_update`, `third_order_state_update_obc`
  - `pruned_third_order_state_update`, `pruned_third_order_state_update_obc`

- Updated [src/macros.jl](src/macros.jl) constructor:
  - Added all 10 state_update function initializers (as `(x,y)->nothing`)
  - Simplified perturbation solution struct constructors (only pass data, no functions)

- Updated all references in [src/MacroModelling.jl](src/MacroModelling.jl):
  - `𝓂.solution.perturbation.first_order.state_update` → `𝓂.functions.first_order_state_update`
  - `𝓂.solution.perturbation.first_order.state_update_obc` → `𝓂.functions.first_order_state_update_obc`
  - Similar changes for all 5 perturbation orders (first, second, pruned_second, third, pruned_third)
  - Updated solution struct assignments to set functions separately from data

### Tests (2026-01-25, state_update functions)

- Verified compilation and basic RBC model functionality
- Verified IRF generation at all perturbation orders (1st, 2nd, 3rd order)

### Completed (2026-01-25, model_functions struct refactoring)

- Extended `model_functions` struct in [src/structures.jl](src/structures.jl) to hold all Function-type fields:
  - Existing: `NSSS_solve`, `NSSS_check`, `NSSS_custom`, `obc_violation`
  - Added: `NSSS_∂equations_∂parameters`, `NSSS_∂equations_∂SS_and_pars`, `jacobian`, `jacobian_parameters`, `jacobian_SS_and_pars`, `hessian`, `hessian_parameters`, `hessian_SS_and_pars`, `third_order_derivatives`, `third_order_derivatives_parameters`, `third_order_derivatives_SS_and_pars`

- Updated `non_stochastic_steady_state` struct to hold only buffer matrices (no tuples with functions):
  - `∂equations_∂parameters::AbstractMatrix{<: Real}` (was `Tuple{AbstractMatrix, Function}`)
  - `∂equations_∂SS_and_pars::AbstractMatrix{<: Real}` (was `Tuple{AbstractMatrix, Function}`)

- Updated `perturbation_derivatives` struct to hold only buffer matrices (no tuples with functions):
  - All 9 fields changed from `Tuple{AbstractMatrix, Function}` to `AbstractMatrix{<: Real}`

- Updated [src/macros.jl](src/macros.jl) constructor:
  - Split tuple initializations into separate buffer and function variables
  - Updated struct constructors to pass buffers to data structs and functions to `model_functions`

- Updated all references in [src/MacroModelling.jl](src/MacroModelling.jl):
  - Assignment patterns: `𝓂.derivatives.field = buffer, func` → separate assignments to `𝓂.derivatives.field` and `𝓂.functions.field`
  - Buffer access: `𝓂.derivatives.field[1]` → `𝓂.derivatives.field`
  - Function access: `𝓂.derivatives.field[2]` → `𝓂.functions.field`
  - NSSS patterns: `𝓂.NSSS.∂equations_∂parameters[1]` → `𝓂.NSSS.∂equations_∂parameters`
  - NSSS patterns: `𝓂.NSSS.∂equations_∂parameters[2]` → `𝓂.functions.NSSS_∂equations_∂parameters`
  - Similar changes for `∂equations_∂SS_and_pars`

### Tests (2026-01-25)

- Verified compilation and basic RBC model functionality
- Verified IRF generation at all perturbation orders (1st, 2nd, 3rd order)

## 2026-01-19

### In Progress (2026-01-19)

- Migrating parameter/calibration constants to `post_parameters_macro` across the codebase (partial; compatibility accessors added).

### Completed (2026-01-19)

- Removed `Base.getproperty`/`Base.setproperty!` overrides on `ℳ` and updated call sites to use `post_parameters_macro` fields directly.
- Removed `Base.getproperty`/`Base.setproperty!` overrides on `constants`; inlined cache fields into `post_parameters_macro` and moved computational/auxiliary caches to `constants`.
- Updated `options_and_caches`, `get_functions`, `inspect`, and `dynare` to use `post_parameters_macro` fields.
- Updated `MacroModelling.jl` in several key sections (model display, symbol creation, SS check, parameter input handling) to use `post_parameters_macro`.
- Updated steady-state docs to reference `post_parameters_macro.calibration_equations_parameters`.
- Updated `options_and_caches`, `perturbation`, `moments`, `filter`, `get_functions`, and tests to use inlined cache fields and constants-level caches.
- Replaced prefixed cache fields with unprefixed fields in `post_complete_parameters` and moved completion-time caches out of `post_parameters_macro`.
- Rebuilt `@parameters` macro output to recreate `post_parameters_macro` and `post_complete_parameters` immutably, and fixed `guess_dict` initialization order.
- Renamed auxiliary matrix caches to `second_order`/`third_order`, moved higher-order cache fields into those structs, dissolved `moments_cache`/`conditional_forecast_index_cache`/`computational_constants_cache`, and added parametric axes plus `diag_nVars` to `post_complete_parameters`.
- Fixed `get_std` on `Caldara_et_al_2012` by guarding aux bounds lookup and updating parameter-axis references.
- Suppressed NSSS failure warnings unless verbose is enabled to avoid spurious FS2000 setup warnings.
- Moved FS2000 custom steady state functions before `@parameters` and wired `steady_state_function` to avoid global solver-parameter optimization during setup.
- Added a helper to locate model-named custom steady state functions for fallback (unused in FS2000 after file reorder).

### Tests (2026-01-19)

- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`

## 2026-01-23

### Completed (2026-01-23)

- Fixed `write_mod_file` steady-state key lookup to handle mixed string/symbol keys (e.g., `beta{F}`) in Dynare export.

### Tests (2026-01-23)

- `julia -t auto --project -e 'using MacroModelling; include("models/FS2000.jl"); write_mod_file(FS2000); println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; include("models/FS2000.jl"); println("solver_params_len=", length(FS2000.solver_parameters));'`
- `julia -t auto --project -e 'using MacroModelling; include("models/Caldara_et_al_2012.jl"); stds = get_std(Caldara_et_al_2012); println(stds);'`
- `julia -t auto --project=test -e 'using MacroModelling, CSV, DataFrames, AxisKeys; include("models/FS2000.jl"); dat = CSV.read("test/data/FS2000_data.csv", DataFrame); dataFS2000 = KeyedArray(transpose(Array(dat)), Variable = Symbol.("log_" .* names(dat)), Time = 1:size(dat)[1]); dataFS2000 = log.(dataFS2000); observables = sort(Symbol.("log_" .* names(dat))); dataFS2000 = dataFS2000(observables, :); llh_k = get_loglikelihood(FS2000, dataFS2000, FS2000.parameter_values; filter = :kalman, algorithm = :first_order, steady_state_function = FS2000_custom_steady_state_function!); llh_i = get_loglikelihood(FS2000, dataFS2000, FS2000.parameter_values; filter = :inversion, algorithm = :pruned_second_order, steady_state_function = FS2000_custom_steady_state_function!); println("kalman=", llh_k, " inversion=", llh_i);'`
- `julia -t auto --project=test -e 'using MacroModelling, CSV, DataFrames, AxisKeys; include("models/FS2000.jl"); dat = CSV.read("test/data/FS2000_data.csv", DataFrame); dataFS2000 = KeyedArray(transpose(Array(dat)), Variable = Symbol.("log_" .* names(dat)), Time = 1:size(dat)[1]); dataFS2000 = log.(dataFS2000); observables = sort(Symbol.("log_" .* names(dat))); dataFS2000 = dataFS2000(observables, :); llh_k = get_loglikelihood(FS2000, dataFS2000, FS2000.parameter_values; filter = :kalman, algorithm = :first_order, steady_state_function = FS2000_custom_steady_state_function!); llh_i = get_loglikelihood(FS2000, dataFS2000, FS2000.parameter_values; filter = :inversion, algorithm = :pruned_second_order, steady_state_function = FS2000_custom_steady_state_function!); println("kalman=", llh_k, " inversion=", llh_i);'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(ss) > 0; println("ok")'`

## 2026-01-24

### Completed (2026-01-24, equations struct refactoring)

- Created a new `mutable struct equations` in [src/structures.jl](src/structures.jl) to consolidate equation storage with fields: `original`, `dynamic`, `steady_state`, `steady_state_aux`, `obc_violation`.
- Removed old equation fields from `ℳ` struct: `ss_aux_equations`, `dyn_equations`, `ss_equations`, `original_equations`, `obc_violation_equations`.
- Updated macros.jl constructor to create the nested `equations` struct.
- Updated all usages across the codebase:
  - `𝓂.dyn_equations` → `𝓂.equations.dynamic`
  - `𝓂.ss_equations` → `𝓂.equations.steady_state`
  - `𝓂.ss_aux_equations` → `𝓂.equations.steady_state_aux`
  - `𝓂.original_equations` → `𝓂.equations.original`
  - `𝓂.obc_violation_equations` → `𝓂.equations.obc_violation`
- Updated test file [test/functionality_tests.jl](test/functionality_tests.jl) to use `m.equations.obc_violation` instead of `m.obc_violation_equations`.

### Tests (2026-01-24, equations struct refactoring)

- Verified with RBC model test:
  - Model created successfully
  - `equations` struct type is `MacroModelling.equations`
  - All equation fields accessible: `original` (4), `dynamic` (4), `steady_state` (4), `steady_state_aux` (4), `obc_violation` (0)
  - IRF computed successfully

### Completed (2026-01-24, fixes)

- Fixed a `symbolics` constructor mismatch in [src/MacroModelling.jl](src/MacroModelling.jl) by adding the missing `dyn_exo_list` argument in `create_symbols_eqs!`.
- Fixed `Base.show(::IO, ::ℳ)` in [src/MacroModelling.jl](src/MacroModelling.jl) to stop referencing removed `post_model_macro` fields (`aux_present`, `aux_future`) and instead use `aux`.
- Validated with a minimal RBC smoke test (`get_steady_state(RBC)`) which now completes successfully.

### Completed (2026-01-24, constants usage audit)

- Added a repo-local analyzer script that maps each `constants` subfield to the functions that reference it and the possible exported entrypoints that can reach those functions (heuristic, regex-based).
- Added an audit view highlighting `constants` fields not reachable from any exported entrypoint and listing post-construction `.constants.* = ...` mutations.

### Output (2026-01-24, constants usage audit)

- Report: analysis/CONSTANTS_USAGE.md
- Audit: analysis/CONSTANTS_AUDIT.md
- Generator: analysis/constants_usage_report.py (run with `/opt/homebrew/bin/python3 analysis/constants_usage_report.py`)

### Completed (2026-01-24, later)

- Initialized `post_complete_parameters` with concrete `Symbol` axes and added axis-type inference/conversion so display axes (including `calib_axis`) are stored as `Vector{Symbol}` or `Vector{String}` rather than `Vector{Union{Symbol,String}}`.

### Tests (2026-01-24, later)

- `julia -t auto --project -e 'using MacroModelling; include("models/Backus_Kehoe_Kydland_1992.jl"); MacroModelling.ensure_name_display_cache!(Backus_Kehoe_Kydland_1992); @assert eltype(Backus_Kehoe_Kydland_1992.constants.post_complete_parameters.calib_axis) == String; println("ok")'`

### Completed (2026-01-24)

- Moved `custom_steady_state_buffer` storage from `post_complete_parameters` into the `workspaces` struct and updated buffer access/test usage.

### Tests (2026-01-24)

- `julia -t auto --project -e 'using MacroModelling; @model RBC_switch begin 1 / c[0] = (beta / c[1]) * (alpha * exp(z[1]) * k[0]^(alpha - 1) + (1 - delta)); c[0] + k[0] = (1 - delta) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^alpha; z[0] = rho * z[-1] + std_z * eps_z[x]; end; @parameters RBC_switch begin std_z = 0.01; rho = 0.2; delta = 0.02; alpha = 0.5; beta = 0.95; end; function _ss(out, params); out .= [1.0, 1.0, 1.0, 0.0]; return nothing; end; get_steady_state(RBC_switch, steady_state_function = _ss); @assert length(RBC_switch.workspaces.custom_steady_state_buffer) == length(RBC_switch.constants.post_model_macro.vars_in_ss_equations_no_aux) + length(RBC_switch.constants.post_parameters_macro.calibration_equations_parameters); println("ok")'`

## 2026-01-18

### Completed (2026-01-18)

- Renamed the `model` struct to `post_model_macro` and updated all direct type/constructor references.
- Verified all call sites reference `post_model_macro` and no `model` type references remain.
- Moved model-macro constant lists (`vars_in_ss_equations`, `dyn_var_*`, `dyn_*`) into `post_model_macro` and removed redundant fields from `ℳ`.
- Updated caches and call sites to read constants from `post_model_macro` (including tests).
- Restored `vars_in_ss_equations` field in `model_structure_cache` and aligned cache initialization with `post_model_macro`.
- Fixed standalone test imports and aligned `post_model_macro` usage in standalone tests.
- Adjusted steady-state indexing to exclude `➕_vars` in `get_steady_state`.
- Fixed dynare export to use the passed model argument.
- Updated FS2000 custom steady state to return full `vars_in_ss_equations` (including `➕` auxiliaries) in correct order.
- Added no-aux steady-state variable list to `post_model_macro` and caches; custom steady state functions now use no-aux list.
- Updated FS2000 custom steady state to exclude auxiliary `➕` variables.

### Tests (2026-01-18)

- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; ss = get_steady_state(RBC); @assert length(RBC.constants.post_model_macro.vars_in_ss_equations) > 0; @assert length(RBC.constants.post_model_macro.dyn_var_present_list) > 0; println("ok")'`
- `julia -t auto --project=test test/test_standalone_function.jl`
- `julia -t auto --project -e 'using MacroModelling; @model RBC begin 1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ)); c[0] + k[0] = (1 - δ) * k[-1] + q[0]; q[0] = exp(z[0]) * k[-1]^α; z[0] = ρ * z[-1] + std_z * eps_z[x]; end; @parameters RBC begin std_z = 0.01; ρ = 0.2; δ = 0.02; α = 0.5; β = 0.95; end; write_mod_file(RBC); println("dynare export ok")'`
- `julia -t auto --project=test -e 'using MacroModelling; include("models/FS2000.jl"); get_steady_state(FS2000, steady_state_function = FS2000_custom_steady_state_function!); println("custom ss ok")'`
- `julia -t auto --project=test -e 'using MacroModelling; include("models/FS2000.jl"); get_steady_state(FS2000, steady_state_function = FS2000_custom_steady_state_function!); println("custom ss ok")'`

### Notes

- No prior AGENT_PROGRESS.md existed in the workspace; created this log.
