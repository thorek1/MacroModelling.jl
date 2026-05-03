# Agent Progress Log

## 2026-02-06
- Normalized equation input handling in update/add/remove model and calibration APIs to strip line-number nodes and collapse single-expression blocks.
- Added `normalize_equation_expr`/`normalize_equation_input` helpers in `src/inspect.jl`.
- Kept equation revision history clean of `begin` blocks from parsing.

### Tests Run
- Basic model update/add/remove equation script (RBC) via Julia `-e`.
- Basic calibration update/add/remove script (RBC_cal) via Julia `-e`.
