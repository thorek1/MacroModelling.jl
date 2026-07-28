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
