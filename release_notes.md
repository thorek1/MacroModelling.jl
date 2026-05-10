compat with mooncake and switch to mooncake in docs
all functions meant to be used with derivatives now can be used with forward and reverse mode autodiff across all options (higher order, filters, etc.)
moved Bartels-Stewart sylvester and lyapunov equations solvers from MatrixEquations.jl to an extension
forwarddiff is an extension now, as internal derivatives are now done using to rrules based on analytical derivatives
Sylvester large systems solved with krylov methods now benefit from an ilu preconditioner
fix correctness issue in inversion filter (second order)
QME solver switches to doubling for large systems by default
use preallocated BLAS/LAPACK calls throughout for better performance and reduced allocations (QZ,LU,QR)
much more detailed tolerance settings for all solvers, with more robust defaults
Lyapunov solver accepts initial guess, has early termination, better handles unstable systems, the Krylov solver now works with the upper triangular system (implicitly forcing symmetry of the solution), added dqgmres support
overall much reduced allocations and better performance
many indices precomputed. moved constants to a separate struct. workspaces reduce allocations and caches are used for repeated solves
preallocated workspaces for all solvers, with better reuse and reduced allocations. use of LinearSolve and FastLapackInterface for matrix solves
write_mod_file (dynare) now allows to modify order, pruning and irf length
allow equation modification like in Troll
custom steady state function support  (including in-place functions)
`get_equations` and `get_calibration_equations` accept a `filter` keyword argument
equations are now returned as expressions instead of strings
add correlation to `get_statistics`
counters for steady state and perturbation solves
replace MCMCChains with FlexiChains in estimation
shock decomposition includes calibration parameters
Lyapunov solver supports `has_unit_roots` parameter for unit root covariance handling
added warmup iterations for first-order inversion loglikelihood rrule, fixed gradient accuracy
NSSS solver refactored: struct dissolved into constants, functions, caches, and workspaces; `solve_nsss_wrapper` introduced as API layer
DispatchDoctor type stability coverage expanded across numerical source files
compat with Turing 0.45
added FRBUS model
get_irf with parameters now also works with higher order
removed RecursiveFactorization and DifferentiationInterface direct dependency

JET test on less functions (hot paths) so we get some coverage at least
see that all test scripts are actually run
there are various approaches of dealing with operations on sparse matrices and constructing them manually throughout th epackage. take stock of what approaches are out there, evaluate them in terms of performance gains and specific challenge they tackle and then use the best in class throughout appropriate applications

follow-ups:
revisit func test tols
describe FRBUS model in docs and add to index
inegrate speed section in docs with benchmarks
do triage of helpers. either make sure they are used across the package where applciable but then also check that there is no more consice or already existing implmentation in the ecosystem.
check that we need this BARTELS_STEWART_AVAILABLE thing. its a weird construct to me. check alternatives
with these large models being used, make returns that are scaling with the number of variables and shocks output only the selected variables and shocks in order not to bloat memory and speed up computations. this includes IRFs, variance decompositions, etc.
rethink these crazy long input types
more DD coverage and fixes as well as getting JET to work again on the whole package
checkout StaticArrays for filter, if they speed things up
time filter step as well (SW07) for speed docs
eliminate this nested spaghetti code in nsss_solver
get iterative SSS/mean analytically