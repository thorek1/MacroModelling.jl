# Agent Guide for MacroModelling.jl

This file is the concise default guide for AI coding agents (GitHub Copilot, Claude, etc.).
Read this file first. Read the companion files only when needed.

## Mandatory Workflow (Always Follow)

1. **Read session context first:** At session start, read `AGENT_PROGRESS.md` before making changes.
2. **Use plan mode for non-trivial work:** If a task has 3+ steps or architecture decisions, write and maintain a clear plan.
<!-- 3. **Use Revise-based development:** Keep one Julia REPL running persistently. **Never use one-shot `julia -e` or `julia script.jl` commands** — they discard the session and force full recompilation. AI agents must use the named-pipe pattern described in `docs/agent-guides/development-workflow.md` to maintain a persistent session: write Julia code to a `.jl` file, then `include()` it via the pipe. On Linux machines, Julia installed via juliaup can be found in `~/.juliaup/bin`. Install missing packages when they are not present in the active environment. -->
4. **Prove changes by testing:** Never claim success without running a relevant test/check. If a test cannot be run, state that explicitly.
5. **Do not run the full test suite:** Use focused scripts and minimal reproductions unless a targeted test set is explicitly required.
6. **Fix issues end-to-end:** Reproduce, diagnose, implement, and verify without handing debugging back to the user.

## Core Engineering Principles

- Write all output/log files to the project folder (e.g. `tasks/`), never to `/tmp`.
- Keep changes minimal, focused, and at root cause.
- Preserve performance characteristics (type stability, allocations, threading behavior).
- Update user-facing docs/docstrings when public APIs change.
- Avoid second-person phrasing ("you") in docs/docstrings.
- Cache reusable constants lazily in model caches when appropriate.

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

- Development setup, Revise workflow, testing, docs, benchmarking: `docs/agent-guides/development-workflow.md`
- Project overview, structure, model syntax, design context: `docs/agent-guides/project-context.md`
- Task runbook, orchestration heuristics, common change points: `docs/agent-guides/task-runbook.md`

## Additional Resources

- Documentation: https://thorek1.github.io/MacroModelling.jl/stable
- Issue tracker: GitHub Issues
- Contributing guidelines: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
