compat with mooncake and switch to mooncake in docs
all functions meant to be used now can be used with forward and reverse mode autodiff across all option s(higher order, filters, etc.)
compat with latest Turing version
moved Bartesl-Stewart sylvester and lyapunov equations solvers from MatrixEquations.jl to an extension
forwarddiff is an extension now, asinternal derivatives are now done using to rrules
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