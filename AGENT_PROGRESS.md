# Agent progress

The BGP implementation now uses symbolic multiplicative stationarization
before perturbation. Focused regression coverage passes for multiplicative
trends, expectations, IRFs, moments, derivatives, and equation rebuilds.

Higher-order moment calculations now reconstruct hidden growth-factor
variables in internal solver order before combining them with perturbation
moments. The implementation was committed and pushed to the feature branch.
`test/bgp_math.md` contains the mathematical explanation of the
stationarization, growth restrictions, expectations, IRFs, covariances, and
differences from the former additive approach. The comparison with IRIS,
NBToolbox, Dynare, and RISE is documented in
`test/bgp_implementation_comparison.md`.

Estimation-oriented BGP dispatch is now implemented. Structural candidate
metadata records trigger parameters and parameter indices, stationary models
stay on the existing fast path, and parameter updates only rebuild the model
representation when the BGP mode or active driver set changes. Growth
exponents are retained as symbolic parameter expressions and their numeric
values are refreshed in place, so estimation draws do not freeze
parameter-dependent trends or rerun the parser. Focused BGP coverage now
includes parameter-dependent growth and stationary/BGP mode switching.

Estimation likelihood and pullback entry points now forward candidate
parameter vectors before solving, and hidden growth variables are excluded
from the observation-to-steady-state index map. Mixed additive/multiplicative
trend candidates are rejected consistently, and completing missing parameters
initializes the same BGP dispatch path as a fully specified model.

`test/bgp_math.md` now documents the current implementation in detail. It
covers the structural BGP metadata and mode detector, symbolic restriction
matrix and parameter-dependent coefficient solve, generated lead/lag
stationarization, estimation-time mode switching, steady-state growth output,
stationary IRFs, finite covariances, and hidden growth-variable handling.

`test/bgp_steady_state.md` separately documents the standard NSSS problem,
the normalized BGP NSSS problem, internal growth unknowns, free trend-level
anchoring, block solving and continuation, package comparisons, advantages,
limitations, and failure conditions.

`test/bgp_steady_state_automation.md` explains that BGP stationarization
precedes the automatic NSSS solve, identifies the reused versus adapted
machinery, documents structural trend detection and trigger caching, and
clarifies why this restricted automation does not contradict the Canova
et al. argument about general unit-root identification.

The unused parameter-name heuristic for `ρ`/`rho` persistence was removed.
`scripts/inspect_bgp_internals.jl` is a top-level, cell-by-cell internal call
trace for the BGP path. It deliberately avoids helper functions and leaves
each parser, stationarization, NSSS, residual, Jacobian, and first-order
solution intermediate in the REPL namespace.
