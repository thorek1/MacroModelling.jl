# Todo

## NSSS_solve summary (main vs branch)

- [x] Locate and read `NSSS_solve` logic on main branch (scaling, bounds, errors, caches).
- [x] Read current-branch NSSS solving logic (`solve_nsss_steps` and wrapper) and identify matching concerns.
- [x] Write a three-part markdown summary: main branch logic, current branch logic, and differences.
- [x] Record results in AGENT_PROGRESS.md and add review notes.

- [x] Investigate why `get_dynamic_equations` filters return empty results for timing subscripts.
- [x] Update dynamic filtering logic to match subscripted and auxiliary symbols.
- [x] Run targeted test(s) for `get_dynamic_equations` filtering.
- [ ] Record results in AGENT_PROGRESS.md and review notes.

- [ ] Update remove_calibration_equation! APIs to drop new_value(s) and rely on parameters input.
- [ ] Adjust tests and docstrings for the new parameters-based behavior.
- [ ] Run targeted tests for remove_calibration_equation! coverage.
- [ ] Update AGENT_PROGRESS.md with changes and test results.

## Review

- Tests: `julia -t auto --project=. -e 'using MacroModelling, Test; include("test/test_filter_equations.jl")'`
- Notes: First two attempts failed due to missing `Test` and macro imports.
- Notes: NSSS_solve summary captured in tasks/nsss_solve_summary.md.
