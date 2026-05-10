# AGENT_PROGRESS

## Current task
Fix NaN-inconsistent correlation test failures for pruned second order.

## Progress
- Reproduced the test failure: `check_isapprox(corrl, CORRL, rtol=1e-5, nans=true)` at `functionality_tests.jl:2134`.
- Identified root cause: variable `eps_zᴸ⁽¹⁾` (lead-1 auxiliary for an exogenous shock) has exactly zero variance. The correlation matrix correctly has NaN for this variable. However, off-diagonal correlations between `z_delta` and other variables are **theoretically zero** (independent shock channels) but have machine-precision noise (~5e-16) whose sign flips between QME algorithms (`:schur` vs `:doubling`), causing `rtol=1e-5` to fail.
- All 40 failures correspond to `qme=doubling`; `qme=schur` passes because it matches the reference.
- Fix: Added noise clamping in `covariance_to_correlation` (`src/MacroModelling.jl`). Off-diagonal correlation entries with `|c| < eps(T)^(2/3)` (≈3.7e-11 for Float64) are set to zero.
- Verified: all 80 correlation test combinations now pass; autocorrelation tests unaffected.

## Next steps
- No remaining work for this task.
