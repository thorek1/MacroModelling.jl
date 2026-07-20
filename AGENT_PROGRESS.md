# Agent progress

The BGP implementation now uses symbolic multiplicative stationarization
before perturbation. Focused regression coverage passes for multiplicative
trends, expectations, IRFs, moments, derivatives, and equation rebuilds.

Higher-order moment calculations now reconstruct hidden growth-factor
variables in internal solver order before combining them with perturbation
moments. The implementation was committed locally on the feature branch;
remote push is pending explicit permission for the configured external
GitHub remote.
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

The structural BGP profile now caches each candidate's parsed growth factor,
whether that factor contains timed variables, and additive candidates. Trigger
updates reclassify these cached factors directly instead of rescanning the raw
equations. The multi-driver regression in
`test/test_balanced_growth_path.jl` covers independent technology and
population trends, combined output growth, a stable first-order solution, and
parameter-dependent exponent refresh without representation rebuild. The
focused suite passes 60/60; `tasks/reproduce_multiple_trends.jl` also reports a
cached numeric refresh of approximately 0.125 seconds in the current run.

The native perturbation path only special-cases BGP steady-state input. The
raw two-point solve produces normalized levels and gross growth factors; the
ordinary stationary Jacobian, Hessian, and third-order derivative functions
then evaluate the processed stationary equations on the full internal vector.
The QME, moment, filtering, inversion, and estimation paths therefore reuse
the main infrastructure. Hidden growth variables are reconstructed before
perturbation and analytical pullbacks map their cotangents to public
steady-state variables. The focused BGP suite passes 64/64.

The corrected SW07/Caldara stress-test now runs at
`tasks/reproduce_sw07_caldara_bgp.jl` without modifying either fixture's
equations. It sets three SW07 persistence parameters (`crhoa`, `crhob`, and
`crhoms`) and Caldara's `ρ` to `1.01`, then checks finite solutions. Both
models remain classified as stationary, with their original equation counts
and expressions unchanged; Caldara reaches pruned third order. The focused
BGP suite passes 64/64. The fixture run also exposed and fixed mixed-node
additive-term accumulation and real-valued simplification type annotations.

An additional stress case now runs at
`tasks/reproduce_complex_multiple_trends.jl`. It combines three independent
multiplicative drivers (`a`, `n`, and `h`), nonlinear cross-scaling, a forward
expectation equation, and a stationary shock. First-, pruned second-, and
pruned third-order solutions are finite; second- and third-order moments are
finite; and the ordinary stationary Hessian and third-order tensors agree
through the native BGP wrapper to `1e-9`.

The native BGP versus stationary benchmark is in
`tasks/benchmark_bgp_approaches.jl`. On the three-driver model, the ordinary
stationary and BGP-wrapper Jacobian/Hessian/third-order tensors agree to
`1e-9`. Warm BGP-wrapper calls take about 0.0017--0.0034 ms versus
0.0005--0.0019 ms for the ordinary calls, with a small allocation overhead.
The cold full first-order solve was about 5.27 s on the BGP path versus
0.31 s for the stationary path; this is the raw two-point NSSS cost, not a
second perturbation algorithm.

The default BGP NSSS route now uses a cached raw-equation representation.
It evaluates two consecutive BGP points, assigns a gross factor to every
timed endogenous variable, applies exact \(x^*/G_x\) lag and \(x^*G_x\) lead
identities directly to the original equations, replaces each driver law with
its growth-factor equation and level anchor, and reuses the existing automatic
NSSS solver. The symbolic stationary representation is used by the ordinary
perturbation path; it is not a selectable alternative BGP NSSS route. The
focused regression passes 64/64. The three-driver expectation/nonlinear
higher-order stress case passes with first-, pruned second-, and pruned
third-order solutions, finite moments, and native BGP/stationary tensor
equality. The SW07/Caldara fixture checks pass 5/5 for each model, and the raw
direct NSSS prototype and native perturbation probe pass with residuals below
`1e-15`.

Added `test/bgp_nsss_rbc_walkthrough.md`, which documents the complete direct
BGP NSSS construction with a four-equation trend-growth RBC example. It lists
the exact transformed residuals, the full level/growth unknown vector, the
temporary raw-model setup, the ordinary NSSS block/continuation path, and the
public/internal output mapping. The displayed numerical solution reproduces
the residual system with maximum absolute residual below `2e-15`.
The walkthrough also distinguishes arbitrary trend-level normalization from
growth-factor identification and documents compatibility with stationary
log-space shock/growth-factor processes versus unsupported additive log-level
unit roots.
