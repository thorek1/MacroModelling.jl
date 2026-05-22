# Agent Guide for MacroModelling.jl

This file is the concise default guide for AI coding agents (GitHub Copilot, Claude, etc.).
Read this file first. Read companion guides only when needed.

## Governing Behavior Contract (11 Rules)

1. Think before coding. State your assumptions. Surface tradeoffs. Ask before guessing. Push back when a simpler approach exists.
2. Simplicity first. Minimum code that solves the problem. No speculative features. No abstractions for single-use code.
3. Surgical changes. Touch only what is asked. Do not "improve" adjacent code, comments, or formatting. Match existing style.
4. Goal-driven execution. Define success criteria. Loop until verified. Do not narrate steps; tell me what success looks like.
5. Do not make the model do non-language work. Retry policies, routing, escalation thresholds belong in deterministic code.
6. Surface conflicts, do not average them. If two parts of the codebase disagree, flag the disagreement and ask which to follow.
7. Read before you write. Understand adjacent code (the file and nearby siblings) before adding new code.
8. Tests are required but are not the goal. A passing test that tests nothing useful is a failure. Tests must check behavior.
9. Long-running operations require checkpoints. After every significant step, summarize what was done and confirm before proceeding.
10. Convention beats novelty. In an established codebase, match the existing pattern even if a "better" one exists.
11. Fail visibly, not silently. Surface every skipped record, every rolled-back transaction, every constraint violation. Never report success when something was bypassed.

If any project-specific instruction below conflicts with the 11 rules above, follow the 11 rules above.

## Mandatory Workflow (Always Follow)

1. **Read session context first:** At session start, read `AGENT_PROGRESS.md` before making changes.
2. **Start with a minimal targeted script/test:** For new features or bug fixes, first create/run a minimal script or focused test that reproduces the exact error or validates the feature's correctness before editing code.
3. **Use plan mode for non-trivial work:** If a task has 3+ steps or architecture decisions, write and maintain a clear plan.
<!-- 3. **Use Revise-based development:** Keep one Julia REPL running persistently. **Never use one-shot `julia -e` or `julia script.jl` commands** — they discard the session and force full recompilation. AI agents must use the named-pipe pattern described in `docs/agent-guides/development-workflow.md` to maintain a persistent session: write Julia code to a `.jl` file, then `include()` it via the pipe. On Linux machines, Julia installed via juliaup can be found in `~/.juliaup/bin`. Install missing packages when they are not present in the active environment. -->
4. **Fix root cause when addressing errors:** Do not stop at symptom-level patches when a deeper cause can be identified and corrected.
5. **Prove changes by testing:** Never claim success without running a relevant test/check. For bug fixes and new features, accept code changes only if the initial minimal script/test passes after the implementation. If a test cannot be run, state that explicitly.
6. **Do not run the full test suite:** Use focused scripts and minimal reproductions unless a targeted test set is explicitly required.
7. **Fix issues end-to-end:** Reproduce, diagnose, implement, and verify without handing debugging back to the user.

## Core Engineering Principles

- Write all output/log files to the project folder (e.g. `tasks/`), never to `/tmp`.
- Keep changes minimal, focused, and at root cause.
- Keep code parsimonious and readable; apply Occam's razor to code changes.
- Preserve performance characteristics (type stability, allocations, threading behavior).
- Performance-critical code should live inside functions, not global scope.
- Avoid untyped global variables and abstractly typed containers in hot code paths.
- Update user-facing docs/docstrings when public APIs change.
- Avoid second-person phrasing ("you") in docs/docstrings.
- Cache reusable constants lazily in model caches when appropriate.
- Avoid try-catch statements for control flow. Use explicit checks and validation; reserve try-catch for unavoidable numerical failures.
- **rrule implementation:** Always derive analytical results for pullback functions. Never use AD inside a pullback—compute adjoints directly via mathematical derivation.

## Task Files (Required Discipline)

- Track plan/progress in `tasks/todo.md`.
- After corrections, capture reusable lessons in `tasks/lessons.md`.
- Keep `AGENT_PROGRESS.md` updated with what was done and what remains.

## Critical Non-Negotiables

1. Never claim something works without test evidence.
2. Work modularly and verify each completed module.
3. Iterate on failures independently; do not rely on user retesting loops.
4. Be explicit about unknowns; do not guess.
5. Verify before marking tasks complete.

## On-Demand Companion Guides (Read Only If Needed)

All companion guides live in `docs/agent-guides/`:

- Development setup, Revise workflow, testing, docs, benchmarking: `docs/agent-guides/development-workflow.md`
- Project overview, structure, model syntax, design context: `docs/agent-guides/project-context.md`
- Task runbook, orchestration heuristics, common change points: `docs/agent-guides/task-runbook.md`
- Code style conventions, naming, formatting, performance patterns: `docs/agent-guides/STYLE_GUIDE.md`

## Additional Resources

- Documentation: https://thorek1.github.io/MacroModelling.jl/stable
- Issue tracker: GitHub Issues
- Contributing guidelines: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
