# Lessons

## 2026-07-28

- The policy solution cache already stores second- and third-order coefficients in compressed symmetric coordinates; the main duplication occurs at state-update consumers.
- The compressed vector convention is the extractor convention represented by `𝐔₂` and `𝐔₃`, with unique nondecreasing index tuples and summed mixed permutations.
- Mixed quadratic state terms do not receive an extra `1/2`: the compressed kernel already sums the two distinct permutations. Same-variable Taylor terms retain their outer `1/2` and `1/6` factors; their derivatives therefore use the corresponding differentiated factors.
- Full global pair/triple indices cannot be sliced directly into compressed policy columns. Inversion and conditional-forecast paths must map them through compressed index helpers.
- Stochastic-steady-state Newton Jacobians use the augmented state identity (`nPast + 1` columns) and must select only the state/constant prefix of compressed augmented coordinates; shock columns are not part of the Newton unknown.
- Third-order stochastic-steady-state assembly may return a dense compressed matrix, while the Newton implementation historically expects sparse storage. Normalize that compressed matrix at the interface instead of expanding it.
- The ordinary third-order inversion pullback can delegate to the missing-observation compressed path, avoiding a second full-coordinate implementation.
- Full-Kronecker removal is most valuable once the augmented dimension is moderate: dense BLAS can win at small dimensions, while cubic scratch and matrix-vector work dominate by augmented dimensions around 20–30.
- Same-vector cubic contractions are common in pruned transitions. A dedicated `3!`/`6!`-weighted kernel removes repeated permutation arithmetic; same-vector VJPs should also write directly into the cotangent to avoid temporary vectors.
- LoopVectorization cannot consume the branchy triangular cubic kernel, and a branchless triangular rewrite was slower. Existing package patterns—typed function barriers, preallocated scratch, `mul!`, and `@inbounds`—remain the better fit.
- Particle prediction/scoring is parallelizable only inside each period. Resampling and RNG sequencing are global boundaries; pre-drawing shocks and assigning thread-local scratch preserves reproducibility and avoids unsafe shared mutable buffers.
- In Julia 1.12, default worker thread IDs may be offset when an interactive pool exists; thread-local arrays should not assume IDs are `1:nthreads()` unless they use an explicit chunk index or account for the offset.
- Explicit power kernels make repeated-input intent visible. The cubic power path avoids permutation-case arithmetic and measured 2.6–4.7x faster than the pre-specialization reference for augmented sizes 6–33; the pair path is already so small that dispatch removal is not a reliable speedup.
- Compressed global cubic coordinates are not loop-order rows: selected policy columns must be sorted once, with an inverse permutation cached for the loop order. The shock-state-state helper carries the policy convention's factor of one third relative to the fully symmetrized cubic power vector.
- Static hot-path audits must exempt particle resampling's `searchsortedfirst`; that operation is unrelated to compressed state-coordinate indexing and remains part of the sequential resampling boundary.
- A validated `Union{Nothing, measurement-error}` value still needs an explicit type assertion before keyword dispatch when the callee accepts only the non-`nothing` branch; JET does not always propagate an error-based narrowing through a large caller.
- JET can report a missing nested-vector method even when a parametric subtype method exists; concrete `Vector{Float64}`/`Vector{Vector{Float64}}` cross-representation methods make the invalid branch visible and total.
- Compressed pair/triple selector maps are model invariants, but the no-volatility shock×state selector is distinct from the augmented shock×state selector and needs its own cached map. Keeping all selector maps on the second-/third-order constants prevents repeated sorting and allocation during inversion setup and pullbacks.

## 2026-07-29

- A compressed symmetric directional derivative already contains the distinct permutation sum: `d(C₂(a,a))/2 = C₂(da,a)` and `d(C₃(a,a,a))/6 = C₃(da,a,a)/2`. Applying the old full-coordinate prefactors again silently drops a factor of two or three.
- Existing third-order inversion workspaces already have matrices with the exact selector dimensions needed by state-to-pair and state-pair-to-shock helpers. Adding bang overloads and reusing those buffers removes the recurring output allocation without changing downstream matrix types.
- Sparse selector matrices can be much cheaper for the subsequent multiplication when the selector pattern is very thin, but a useful sparse implementation needs a cached CSC pattern and value-position map. Do not rebuild sparse structure in a period loop; measure the full workload before adding that cache.
