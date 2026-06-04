- add filter free joint loglikelihood estimation
- add missing value support in estimation and filtering/smoothing
- add initial_state support in estimation
- added higher order variance decomposition
higher order oslution algorithms moved to compressed space (exploting symmetry)
- test system prior estimation
- compat with mooncake and switch to mooncake in docs
- all functions meant to be used with derivatives now can be used with forward and reverse mode autodiff across all options (higher order, filters, etc.)
- moved Bartels-Stewart sylvester and lyapunov equations solvers from MatrixEquations.jl to an extension
- forwarddiff is an extension now, as internal derivatives are now done using to rrules based on analytical derivatives
- Sylvester large systems solved with krylov methods now benefit from an ilu preconditioner
- fix correctness issue in inversion filter (second order)
- QME solver switches to doubling for large systems by default
- use preallocated BLAS/LAPACK calls throughout for better performance and reduced allocations (QZ,LU,QR)
- much more detailed tolerance settings for all solvers, with more robust defaults
- Lyapunov solver accepts initial guess, has early termination, better handles unstable systems, the Krylov solver now works with the upper triangular system (implicitly forcing symmetry of the solution), added dqgmres support
- overall much reduced allocations and better performance
- many indices precomputed. moved constants to a separate struct. workspaces reduce allocations and caches are used for repeated solves
- preallocated workspaces for all solvers, with better reuse and reduced allocations. use of LinearSolve and FastLapackInterface for matrix solves
- write_mod_file (dynare) now allows to modify order, pruning and irf length
- allow equation modification like in Troll
- add correlation to get_statistics (including derivatives)
- add `marginal_contribution` (Shapley) shock decomposition for the inversion filter and `marginal_contribution` variance decomposition at pruned 2nd/3rd order; both use a polynomial-coefficient algorithm (∑_{j≤2k} C(nᵉ,j) Lyapunov solves) instead of the 2^nᵉ exhaustive coalition enumeration, making the higher-order Shapley aggregation feasible for models with many shocks
- custom steady state function support  (including in-place functions)
- `get_equations` and `get_calibration_equations` accept a `filter` keyword argument
- equations are now returned as expressions instead of strings
- counters for steady state and perturbation solves
- replace MCMCChains with FlexiChains in estimation
- shock decomposition includes calibration parameters
- Lyapunov solver supports `has_unit_roots` parameter for unit root covariance handling
- added warmup iterations for first-order inversion loglikelihood rrule, fixed gradient accuracy
- NSSS solver refactored: struct dissolved into constants, functions, caches, and workspaces; `solve_nsss_wrapper` introduced as API layer
- DispatchDoctor type stability coverage expanded across numerical source files
- compat with Turing 0.45
- added FRBUS model
- get_irf with parameters now also works with higher order
- removed RecursiveFactorization and DifferentiationInterface direct dependency
- speed up doubling algos AD paths by caching matrix powers
- custom sparse matrix kernel to speed up higher order solves
- analytical OBC jacobian instead of ForwardDiff.jl


---

# Clustered release notes

## User-facing

### Estimation, filtering, and smoothing
- Filter-free joint loglikelihood estimation.
- Missing-value support in estimation, filtering, and smoothing.
- User-provided `initial_state` support in estimation.
- Fixed a correctness issue in the second-order inversion filter.
- Added warmup iterations to inversion-filter loglikelihood `rrule`.

### Statistics, moments, and decompositions
- Added higher-order (pruned 2nd/3rd) variance decomposition.
- Added correlation to `get_statistics`, including derivatives.
- Added `marginal_contribution` (Shapley) shock decomposition for the inversion filter and `marginal_contribution` variance decomposition at pruned 2nd/3rd order, using a polynomial-coefficient algorithm (∑_{j≤2k} C(nᵉ,j) Lyapunov solves) in place of the 2^nᵉ exhaustive coalition enumeration, making higher-order Shapley aggregation feasible for models with many shocks.

### Solutions and IRFs
- `get_irf` with parameters also works at higher order.

### API, model writing, and Dynare interop
- Custom steady-state function support (including in-place functions).
- `get_equations` and `get_calibration_equations` accept a `filter` keyword argument.
- `write_mod_file` (Dynare) allows modifying order, pruning, and IRF length.
- Equation modification supported (Troll-style).
- Counters for steady-state and perturbation solves.

### Autodiff and ecosystem compatibility
- All functions intended for use with derivatives now work with forward- and reverse-mode AD across all options (higher order, filters, etc.).
- Compatibility with Mooncake (and switched the docs to Mooncake).
- Compatibility with Turing <=0.45.

### New models
- Added the FRB/US model.

## Internal

### Performance and allocations
- Overall greatly reduced allocations and improved performance.
- Preallocated BLAS/LAPACK calls (QZ, LU, QR) throughout.
- Preallocated workspaces for all solvers, with better reuse; uses LinearSolve and FastLapackInterface for matrix solves.
- Many indices precomputed; constants moved to a separate struct; workspaces and caches reused across repeated solves.
- AD paths of the doubling algorithms sped up by caching matrix powers.
- Custom sparse-matrix kernel for higher-order solves.

### Solutions and Solvers
- Higher-order solution algorithms moved to compressed space (exploiting symmetry).
- QME solver switches to doubling for large systems by default.
- Analytical OBC Jacobian replaces ForwardDiff.
- Bartels–Stewart Sylvester and Lyapunov solvers moved from MatrixEquations.jl to an extension.
- Large Sylvester systems solved with Krylov methods now benefit from an ILU preconditioner.
- Lyapunov solver: accepts an initial guess, supports early termination, better handles unstable systems, the Krylov solver now operates on the upper-triangular system (implicitly enforcing symmetry of the solution), and adds dqgmres support.
- Lyapunov solver supports a `has_unit_roots` parameter for unit-root covariance handling.
- Much more detailed tolerance settings across solvers, with more robust defaults.

### Architecture and refactors
- ForwardDiff is now an extension; internal derivatives use `rrule`s based on analytical derivatives.
- NSSS solver refactored: struct dissolved into constants, functions, caches, and workspaces; `solve_nsss_wrapper` introduced as the API layer.
- DispatchDoctor type-stability coverage expanded across numerical source files.
- Removed direct dependencies on RecursiveFactorization and DifferentiationInterface.
- Equations are returned as expressions instead of strings.

### Tests
- System-prior estimation is now tested.
- Replaced MCMCChains with FlexiChains in estimation tests.