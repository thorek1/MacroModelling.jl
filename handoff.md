# MacroModelling.jl BGP Handoff

Date: 2026-07-18

## Current repository state

- Repository: `thorek1/MacroModelling.jl`
- Branch: `pr/304`
- HEAD: `c15ad94d Add step-through BGP inspection script`
- The branch is aligned with `github-desktop-ylevch/feature/balanced-growth-path`.
- `test/bgp_steady_state_automation.md` is currently untracked and contains the
  latest BGP automation explanation. Do not discard it; decide whether to add
  and commit it.
- The latest committed change includes removal of the unused rho-name
  heuristic and the interactive inspection script.

## Implemented BGP work

The current implementation uses symbolic multiplicative stationarization:

```text
X[t] = H[t] * Xhat[t]
H[t] = product(H_driver[t]^B_driver(theta))
```

The main implementation is in:

- `src/parser/stationarization.jl`
  - Structural trend-driver detection.
  - Additive-trend rejection.
  - Symbolic growth-form extraction.
  - Growth restriction matrix construction and solve.
  - Lead/lag AST rewriting.
  - BGP mode detection and lazy stationary/BGP switching.
- `src/steady_state/nsss_solver.jl`
  - Hidden growth-variable NSSS handling.
  - Rank-based anchoring of free trend levels.
  - Reuse of the ordinary block-triangular NSSS solver.
- `src/structures.jl`
  - `BGP_STATIONARY_MODE`, `BGP_ACTIVE_MODE`,
    `BGP_UNSUPPORTED_MODE`.
  - `bgp_detection_metadata`.
- `src/get_functions.jl`
  - Public `Growth_rate` output.
  - BGP-relative IRFs and level reconstruction.
  - Hidden growth-variable filtering.
- `src/moments.jl`
  - Hidden growth-variable reconstruction and finite covariance handling.
- `src/rrules.jl`
  - BGP-aware likelihood and derivative paths.

The ordinary solver and perturbation machinery remains the core numerical
path after stationarization, but BGP-specific handling remains necessary for
internal variables, public axes, IRFs, moments, covariance, and estimation
mode switches.

## Detection and estimation behavior

Detection is structural, not a general eigenvalue/unit-root detector. It
recognizes explicit multiplicative laws such as:

```julia
x[0] = x[-1] * g[0]
x[0] / x[-1] = g[0]
```

Candidate parameter dependencies are collected once. Runtime updates compare
only trigger values and refresh the numeric growth state when needed. A
stationary model stays on the ordinary fast path; a BGP representation is
built lazily when a parameter draw crosses into the active mode.

The old runtime scan that treated parameter names beginning with `rho` or
the Greek rho name as persistence parameters was unused and has been deleted. Do not
reintroduce name-based unit-root detection.

Parameter-dependent growth exponents are retained as symbolic expressions,
then evaluated numerically on parameter updates. This avoids freezing
expressions such as `alpha * log(growth)` at the initial calibration.

## Step-through inspection script

Use:

```bash
julia --project=. -i scripts/inspect_bgp_internals.jl
```

The script is intentionally top-level "spaghetti" code rather than a set of
large helper functions. Execute each `# %%` block manually in VS Code or
continue line-by-line in the REPL. It exposes named intermediates for:

1. Raw model parsing.
2. Parameter parsing and application.
3. Structural BGP detection.
4. Symbolic growth restrictions and generated stationary equations.
5. Parsing of the generated stationary representation.
6. Application of the active BGP representation.
7. NSSS solver setup and step metadata.
8. NSSS solution and hidden growth variables.
9. Direct residual evaluation.
10. Symbolic derivative generation.
11. Jacobian evaluation.
12. First-order perturbation solution.
13. Public API comparison.

The script was smoke-tested by including it non-interactively and checking
that the NSSS solution, residuals, and first-order solution are finite and
successful.

## Validation already completed

The focused symbolic BGP suite passed:

```text
test/test_balanced_growth_path.jl
43 / 43 tests passed
```

The script smoke test also passed:

```bash
julia --project=. -e 'include("scripts/inspect_bgp_internals.jl")'
```

The full test suite has intentionally not been run.

## Important historical context

Before the current paper-aligned implementation, BGP support used an
IRIS-like additive two-time-point NSSS augmentation. Each variable received a
level and additive growth unknown, and equations were evaluated at two time
origins. That implementation was replaced by symbolic multiplicative
stationarization in commit `c36a6d2d`.

Relevant documentation:

- `test/bgp_math.md`
- `test/bgp_implementation_comparison.md`
- `test/bgp_steady_state.md`
- `test/bgp_steady_state_automation.md`

## Remaining work

The estimation-performance plan is not fully implemented. The main pending
design items are:

- Add a typed, parameter-independent BGP structural profile with precomputed
  index maps and dependencies.
- Replace repeated expression/dictionary work in hot estimation paths with
  compiled evaluators and reusable numeric buffers.
- Add a dedicated BGP workspace for trigger values, growth coefficients,
  restriction solves, and reconstruction maps.
- Add representation generation or mode stamps so all dimension-dependent
  caches are invalidated exactly once on a stationary/BGP switch.
- Cache both raw and BGP representations if repeated switching is important.
- Add focused estimation benchmarks and regressions for repeated draws,
  trigger changes, mode switching, likelihoods, and gradients.

The session plan in
`~/.copilot/session-state/39947ae5-4fea-42e2-aa91-ffcb4642ca1b/plan.md`
contains the longer design and risk discussion.

## Suggested first continuation steps

1. Read this file, `AGENT_PROGRESS.md`, and the session plan.
2. Inspect the untracked `test/bgp_steady_state_automation.md`.
3. Run the focused BGP suite before changing behavior.
4. Use `scripts/inspect_bgp_internals.jl` to trace any solver issue.
5. Decide whether to commit the automation document separately.
6. Work on one estimation optimization item at a time and validate with
   focused tests only.
