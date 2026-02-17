# Task Runbook (On-Demand)

Read this file only for operational heuristics, orchestration style, or common task checklists.

## Common Change Points

- New API: update `src/get_functions.jl` and exports in `src/MacroModelling.jl`
- New model: add file under `models/` using model macros
- Solver changes: inspect `src/perturbation.jl` and `src/algorithms/`

## Typical Task Flows

### Add a feature

1. Implement in the appropriate `src/` location
2. Create a minimal targeted check script
3. Validate behavior with lightweight model(s)
4. Update documentation if user-facing

### Fix a bug

1. Reproduce minimally
2. Locate root cause
3. Implement smallest robust fix
4. Verify with focused check

### Add a model

1. Add model file under `models/`
2. Follow existing model conventions
3. Include citation metadata/context
4. Verify solve + IRFs

## Workflow Orchestration Heuristics

### Plan mode default

- Use plan mode for non-trivial tasks (3+ steps / architecture choices)
- Re-plan quickly if assumptions fail
- Include verification steps in plan, not only implementation

### Subagent usage

- Offload exploration/research for complex tasks
- Keep one focused goal per subagent

### Elegance check (for non-trivial changes)

- Reassess whether a cleaner root-cause solution exists before finalizing
- Avoid over-engineering for obvious/simple fixes

### Autonomous bug-fix expectation

- Drive issue resolution end-to-end without requiring user handholding
- Use logs/errors/tests to iterate quickly to a verified result

## Task and Learning Files

- Plan and execution tracking: `tasks/todo.md`
- Lessons from corrections: `tasks/lessons.md`
- Session status handoff: `AGENT_PROGRESS.md`

## CI/CD Reference

- CI runs on push
- Matrix includes Ubuntu/macOS/Windows (x64 and arm64 where applicable)
- Coverage uploaded to Codecov
- Test sets run in parallel by matrix configuration
