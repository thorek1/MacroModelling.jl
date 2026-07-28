# Agent Progress

## Current task

Migrate internal higher-order state updates to compressed second- and third-order coordinates.

## Completed

- Established the Julia environment with `Pkg.instantiate()` after the initial package load reported the missing `DocStringExtensions` dependency.
- Added allocation-free compressed pair/triple Kronecker-vector kernels, matrix overloads, compressed index maps, and analytical VJP helpers.
- Kept ordinary, pruned, OBC, IRF, filter-free, particle, inversion, shock-decomposition, Aumann–Shapley, and higher-order reverse-mode transition paths in compressed coordinates.
- Updated stochastic-steady-state Newton residual/Jacobian assembly to use compressed state/constant columns and triangular/cubic scratch buffers.
- Preserved compressed matrices at internal state-update and stochastic-steady-state interfaces; public `get_solution` still expands to its documented full tensor output.
- Added focused kernel/reference-equivalence tests and a static transition audit.

## Verification

- `test/test_compressed_kron.jl`: 50/50 vector checks, 5/5 analytical VJP checks, 3/3 state-update equivalence checks, and 52/52 static-audit checks.
- Inversion filter likelihood smoke: 7/7.
- Third-order and pruned-third-order transition/IRF smoke: 4/4.
- Particle-filter compressed transition smoke: 2/2.
- Filter-free likelihood smoke across second, pruned-second, third, and pruned-third order: 4/4.
- All edited Julia files parse cleanly and `git diff --check` is clean.

## Intentional full-space conversions

- Public `get_solution` expands with `𝐔₂`/`𝐔₃` for compatibility with its documented duplicated-axis tensor output.
- Existing moment formulas and their pullbacks use full pair/triple tensor coordinates where those formulas explicitly require them.
- Third-order Sylvester/RHS construction expands the second-order solution internally; this remains isolated to solver assembly and is not used by state updates.
- The compression matrices `𝐔₂`/`𝐔₃` remain as representation/solver constants, but no active ordinary/pruned transition, simulation, IRF, filtering, particle-filter, OBC, or filter-free transition path expands with them.

The full test suite was intentionally not run, per the task instructions. The large third-order Gali OBC probe was stopped during its expensive solve after the pruned-second-order compressed transition assertion passed; smaller OBC and higher-order probes covered the active compressed code paths.

## Performance benchmark follow-up

- Created a detached `HEAD` baseline worktree and ran the same benchmark harness against the pre-compressed full-Kronecker implementation.
- Added the reproducible harnesses [tasks/benchmark_compressed_vs_full.jl](tasks/benchmark_compressed_vs_full.jl) and [tasks/benchmark_particle_parallel_prototype.jl](tasks/benchmark_particle_parallel_prototype.jl).
- On a 5-variable RBC model (3 past states, 2 shocks, 32 periods, pruned third order), cached filter-free likelihood was 0.216 ms compressed versus 1.371 ms full, about 6.3x faster, with 165 kB versus 255 kB allocated.
- The same small-model 2048-particle bootstrap filter was 153 ms compressed versus 111 ms full. This is a crossover case: resampling, random draws, and measurement scoring dominate, and dense BLAS/full Kronecker work can be competitive at small augmented dimension.
- Preallocated direct transition sweeps (4,000 steps) measured the following compressed/full medians: augmented dimension 6, 1.91/2.76 ms; 13, 7.39/6.17 ms; 21, 82.1/995 ms; 33, 451/4,596 ms. The compressed loop allocated zero bytes during the sweep; the full reference allocated about 1.47 MB, 6.08 MB, 326 MB, and 1.20 GB respectively.
- Added automatic same-vector cubic specialization and allocation-free same-vector pair/triple VJPs. This reduced the large direct compressed cases further; the focused kernel test now passes 50/50, with the existing 5/5 VJP, 3/3 equivalence, and 52/52 audit checks still green.
- A four-thread particle propagation/scoring prototype with thread-local scratch achieved 1.88x speedup on 32,768 particles × 12 steps at augmented dimension 21, with identical output and 31 kB scheduler allocation. Production parallelism is not enabled yet.

## Particle parallelism assessment

The predict/score loops in bootstrap, auxiliary, and tempered filters are particle-independent within a period and can be threaded. Resampling, weight normalization/reduction, tempering level selection, and MH acceptance remain sequential synchronization points. A safe production design must pre-draw shocks sequentially (preserving `particle_rng` order), use a typed chunk function with one scratch/shock/measurement buffer per chunk, and either restrict the fast path to diagonal measurement error or give dense measurement-error state one cache per chunk. An opt-in threshold is preferable because the small RBC end-to-end benchmark did not benefit.

LoopVectorization is not applicable to the branchy triangular cubic kernel; a branchless candidate was slower. Existing preallocation, function barriers, `mul!`, `@inbounds`, and dense/sparse selection are already used where applicable.
