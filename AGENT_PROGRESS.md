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
log-space shock/growth-factor processes versus the raw affine handling of
additive log-level roots.

The BGP dispatch is now staged rather than eager: the raw ordinary NSSS path
is attempted first, and a failed or unanchored active-trend root activates the
direct BGP fallback. A successful fallback records
`equations.bgp_detection.prefer_bgp = true`, so subsequent solves try BGP
first while retaining the ordinary path as a fallback. Additive log-coordinate
unit roots used inside `exp(...)` are represented with a gross factor
`exp(Δell)` and a zero log-level anchor; generic additive-level roots remain in
the raw representation. Runtime verification now passes:
`tasks/reproduce_bgp_fallback.jl`
passes the raw-first ratio fallback and additive log-coordinate reproduction,
and `test/test_balanced_growth_path.jl` passes all 70 focused regression checks.

The direct BGP NSSS representation is now a generic three-date affine system.
For every endogenous variable it solves a level, an intercept (`xᴬ`), and a
multiplier (`xᴳ`) from all original equations evaluated at shifts 0, 1, and 2.
This removes the driver-law skip and allows the solved coefficients to classify
stationary, additive, or multiplicative growth. Rank-aware level normalizations
are retained for the active symbolic representation, and additive log drivers
map their raw intercept to the stationary gross factor with `exp(xᴬ)`. The
forward-looking additive-shock reproduction in
`tasks/reproduce_affine_bgp_forward.jl` passes, including `c[1]`, `zᴬ=μ`,
`zᴳ=1`, and induced `cᴳ=yᴳ=exp(μ)`. The focused BGP regression remains 70/70.

Final verification after the affine-solver robustness fixes passes cleanly:
`tasks/reproduce_bgp_fallback.jl`, `tasks/reproduce_affine_bgp_forward.jl`, and
`test/test_balanced_growth_path.jl` (70/70). The generic raw route now builds
its affine solver with numerical blocks for lag-coupled equations, keeps
multipliers away from zero, and seeds/validates the raw 3N root from the
active stationary solve when that representation is available. This prevents
the solver from accepting the singular `xᴳ≈0` boundary while retaining the
generic raw fallback for models without a recognized symbolic trend.

The dispatch was subsequently made direct-first for recognized active BGPs:
the affine \(3N\) raw system is attempted first, then the existing ordinary
raw NSSS route is used if it fails. A successful affine solve records the
cached BGP preference for later calls. Stationary models still use the
ordinary raw route directly. Pure additive candidates remain on that ordinary
route because attempting a fully generic affine solve on the large FRBUS
fixture was numerically impractical; the affine representation remains a
fallback after an ordinary failure.

The mixed forward-looking reproduction now combines additive log growth,
multiplicative technology growth, an expectation term, and a stationary shock.
It passes finite first-, second-, pruned second-, third-, and pruned
third-order perturbation solves. The default NSSS audit passes all 24 model
fixtures, and a second audit passes all 24 after parameter-only trend stress
overrides (mostly AR coefficients set to `1.01`). The model source calibrations
were left unchanged; the stress script copies and modifies parameter vectors.

Additional focused reproductions pass for pure additive large fixtures,
small additive raw affine setup, affine forward-looking BGPs, and the fallback
dispatch. The balanced-growth regression passes 72/72. The public steady-state
wrapper was also corrected to label only the variables selected by the NSSS
problem, so the default and trend-stressed audits now pass all 24 fixtures
through both the internal NSSS and public `get_SS` paths.

The perturbation boundary was tightened after the affine solver changes. When a
raw model has a solved affine cache, the perturbation code uses only the
numerically solved `(xᴬ, xᴳ)` pairs to detrend every timed endogenous reference,
then calls `process_model_equations`, `set_up_steady_state_solver!`, and the
ordinary symbolic derivative generator. The transformed raw clone is used for
first-, second-, and third-order derivative tensors; it does not inspect
`exp`, shock names, or equation forms. A regression restores a recognized
model to its raw equations and verifies a finite Jacobian through this route.
The already processed stationarized representation remains the dimensional
authority for active symbolic BGP models; this is why the raw clone is not
used after stationarization, rather than padding an incomplete derivative
system with hidden growth equations.

The explicit cold/warm benchmark is in
`tasks/benchmark_bgp_nsss_explicit.jl`. With `cold_start=true`, the mixed
additive/multiplicative forward model measured 0.235 s for direct 3N versus
0.295 s for the ordinary raw solve, 0.019 s warm; the three-trend forward
model measured 0.377 s versus 0.0025 s, 0.028 s warm. The latter is about
152x slower on a cold uncached solve because the 3N construction and coupled
root are included; warm calls are about 10x the sub-millisecond ordinary call.
The comparison is route-cost-oriented: the ordinary timing reuses the raw
model setup, while the direct timing includes affine-model construction.

The 24-model parameter-stress dispatch audit completed with zero residuals for
all 24 models. None selected `direct_3N`: the AR-coefficient overrides do not
create a recognized deterministic BGP in these equations, and the ordinary
raw route solved first. The OBC raw-copy diagnostics still report their
pre-existing duplicate-equation limitation, but the public dispatch route
solved those fixtures; the forced direct-first FRBUS experiment was stopped
after more than six minutes, so unsupported large models remain ordinary-first
with direct fallback for performance.

Latest direct-first audit correction: the route is now attempted without using
the detector to select it. The accepted `direct_3N` models were Baxter--King,
Gali--Monacelli, Iacoviello linear, Ireland, SGU debt-premium, and Smets--Wouters
2007 linear (6/24). Thirteen models solved through ordinary fallback. Five
large direct attempts exceeded the 90-second audit bound: FRBUS, GNSS, NAWM,
QUEST3, and Smets--Wouters 2007. OBC residual expressions that are not
equalities are capability-guarded and immediately use ordinary NSSS. Completed
fallback residuals were below `3e-14`; the direct route independently checks
the full affine residual before acceptance. The focused BGP regression passes
76/76 after these changes.
