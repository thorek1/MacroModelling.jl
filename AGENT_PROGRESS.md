# Agent Progress

## Current task

Keep repeated-input higher-order contractions explicit and keep compressed index maps out of hot loops.

## Completed

- Established the Julia environment with `Pkg.instantiate()` after the initial package load reported the missing `DocStringExtensions` dependency.
- Added allocation-free compressed pair/triple Kronecker-vector kernels, matrix overloads, compressed index maps, and analytical VJP helpers.
- Added explicit `compressed_kron²_power`/`compressed_kron³_power` kernels and matching analytical power VJPs; legacy `*_same` names remain compatibility aliases.
- Precomputed triangular pair-coordinate maps and cubic shock-state row maps in model constants, then threaded those maps through inversion and higher-order warmup pullbacks.
- Kept ordinary, pruned, OBC, IRF, filter-free, particle, inversion, shock-decomposition, Aumann–Shapley, and higher-order reverse-mode transition paths in compressed coordinates.
- Updated stochastic-steady-state Newton residual/Jacobian assembly to use compressed state/constant columns and triangular/cubic scratch buffers.
- Preserved compressed matrices at internal state-update and stochastic-steady-state interfaces; public `get_solution` still expands to its documented full tensor output.
- Added focused kernel/reference-equivalence tests and a static transition audit.

## Verification

- `test/test_compressed_kron.jl`: 86/86 vector checks, 4/4 power edge-case checks, 6/6 analytical VJP checks, 3/3 state-update equivalence checks, 2/2 cached-row-map checks, and 67/67 static-audit checks.
- The same focused kernel/audit test passed again after threading cached maps through all third-order reverse-mode warmup callers; Julia rebuilt the package successfully.
- Third-order cache setup smoke on the RBC_CME model produced 20 shock-state-state and 12 shock-shock-state indices with matching cached row maps.
- `test/test_inversion_filter_likelihood.jl`: 7/7 focused likelihood-level checks (5m46.2s; includes the intentional 701-point profile).
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

A redundant `--compiled-modules=no` source-load probe was interrupted after 8.5 minutes while compiling the dependency graph; the normal package rebuild and focused tests completed successfully.

## Performance benchmark follow-up

- Created a detached `HEAD` baseline worktree and ran the same benchmark harness against the pre-compressed full-Kronecker implementation.
- Added the reproducible harnesses [tasks/benchmark_compressed_vs_full.jl](tasks/benchmark_compressed_vs_full.jl) and [tasks/benchmark_particle_parallel_prototype.jl](tasks/benchmark_particle_parallel_prototype.jl).
- On a 5-variable RBC model (3 past states, 2 shocks, 32 periods, pruned third order), cached filter-free likelihood was 0.216 ms compressed versus 1.371 ms full, about 6.3x faster, with 165 kB versus 255 kB allocated.
- The same small-model 2048-particle bootstrap filter was 153 ms compressed versus 111 ms full. This is a crossover case: resampling, random draws, and measurement scoring dominate, and dense BLAS/full Kronecker work can be competitive at small augmented dimension.
- Preallocated direct transition sweeps (4,000 steps) measured the following compressed/full medians: augmented dimension 6, 1.91/2.76 ms; 13, 7.39/6.17 ms; 21, 82.1/995 ms; 33, 451/4,596 ms. The compressed loop allocated zero bytes during the sweep; the full reference allocated about 1.47 MB, 6.08 MB, 326 MB, and 1.20 GB respectively.
- Added explicit same-vector power kernels and allocation-free power VJPs. Against a local pre-specialization permutation reference, cubic power kernels were 2.6–4.7x faster for augmented sizes 6–33; pair kernels were within measurement noise because their arithmetic is already minimal.
- A four-thread particle propagation/scoring prototype with thread-local scratch achieved 1.88x speedup on 32,768 particles × 12 steps at augmented dimension 21, with identical output and 31 kB scheduler allocation. Production parallelism is not enabled yet.

## Particle parallelism assessment

The predict/score loops in bootstrap, auxiliary, and tempered filters are particle-independent within a period and can be threaded. Resampling, weight normalization/reduction, tempering level selection, and MH acceptance remain sequential synchronization points. A safe production design must pre-draw shocks sequentially (preserving `particle_rng` order), use a typed chunk function with one scratch/shock/measurement buffer per chunk, and either restrict the fast path to diagonal measurement error or give dense measurement-error state one cache per chunk. An opt-in threshold is preferable because the small RBC end-to-end benchmark did not benefit.

## CI repair follow-up

- Added `Statistics` to the test target so the particle-filter test environment can load the stdlib explicitly.
- Made particle-pool cross-representation copy methods explicit for JET's nested-vector union split, and narrowed validated particle measurement error before the keyword dispatch boundary.
- Added the missing cached shock–shock–state row bindings in ordinary and pruned third-order inversion paths; pure shock-coordinate selectors now use global compressed maps without state offsets.
- Reproduced the Windows inversion `BoundsError` with the affected selector shape and verified second-order RBC calibration and inversion calls return finite results after the global-map fix. The focused missing-data likelihood and shock-estimation probes also pass.
- The old CI run's later SSS finite-difference and Turing initialisation failures were not treated as solved by masking them; they require separate numerical reproductions if they recur after these direct dispatch and index fixes.

LoopVectorization is not applicable to the branchy triangular cubic kernel; a branchless candidate was slower. Existing preallocation, function barriers, `mul!`, `@inbounds`, and dense/sparse selection are already used where applicable.

## Review follow-up: invariant compressed index maps

- Added lazy constant caches for all pair maps used by inversion, `find_shocks`, and reverse-mode warmup paths, including the distinct no-volatility shock×state selector.
- Added lazy constant caches for the four third-order selector maps used by those paths.
- Removed every runtime `compressed_pair_indices`/`compressed_triple_indices` call from transition, filtering, inversion, and rrule code; the only remaining calls are one-time cache initialization and the helper definitions.
- Extended the static audit to reject runtime pair/triple map construction.
- Package-load/constructor smoke, the compressed-kernel suite (83/83 static-audit checks), and a pruned-third-order RBC model-level map-equivalence smoke all pass.

## Review follow-up: symmetric tangents, SSS filtering, and scratch reuse

- Corrected Aumann–Shapley higher-order tangents: the symmetric compressed pair derivative uses coefficient `1` after the outer `1/2`, the symmetric cubic derivative uses coefficient `1/2` after the outer `1/6`, and mixed pair derivatives retain both compressed terms with coefficient `1`.
- Added regression checks against full `𝐔₂`/`𝐔₃` directional derivatives, including the repeated-input multiplicities.
- Added allocation-free `compressed_triple_state_to_pair!` and `compressed_triple_state_pair_to_shock!` kernels and reused existing third-order inversion workspace matrices in repeated forward and reverse-mode call sites. The bang kernels measure only 32 bytes of call overhead after warm-up and do not allocate their output matrices.
- The selector matrices are structurally sparse (the state-to-pair example has 110 nonzeros out of 1100 entries), but their current consumers accept dense workspace matrices and the sparse alternative would require a cached CSC pattern. The current change removes repeated matrix allocation without changing matrix representation; sparse CSC caching remains a separate optimization candidate.
- Stochastic-steady-state Newton now receives only past/mixed rows and the state/constant prefix of compressed policy matrices from the calculating interface. The full compressed matrices remain available for returned interfaces and final state evaluation; the public Newton wrapper retains its full-matrix-compatible default behavior for rrules.

Verification for this follow-up:

- `test/test_compressed_kron.jl`: 89/89 checks, including directional derivative identities, bang-helper equivalence, and the static audit.
- `test/test_inversion_filter_likelihood.jl`: 7/7 checks.
- User-facing RBC smoke: second-, third-, and pruned-third-order stochastic steady states all returned finite values.
- `test/test_filter_free_gradients.jl` could not start because `ForwardDiff` is absent from the test environment; no code failure was observed.
