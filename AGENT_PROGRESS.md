Current task: enable analytical parameter gradients for `norm(get_solution(model, params)[2])` on the positional-parameter path and make the reverse-mode path work with Mooncake.

Done:
- Read the agent guidance and confirmed there was no existing `AGENT_PROGRESS.md`.
- Located the positional `get_solution(model, params)` implementation in `src/get_functions.jl`.
- Confirmed `calculate_first_order_solution` already has an analytical `rrule` in `src/custom_autodiff_rules/rrules.jl`.
- Identified likely work area as Mooncake compatibility and/or positional-wrapper composition rather than missing first-order adjoints.
- Created a session plan and SQL todos.
- Added `ext/MooncakeExt.jl` so Mooncake treats `MacroModelling.ℳ` as non-differentiable and reuses the existing analytical `get_solution` `rrule` via `Mooncake.@from_rrule`.
- Added a focused standalone-function regression comparing `ForwardDiff`, `Zygote`, and `FiniteDifferences` on `x -> norm(get_solution(model, x)[2])`, with an additional Mooncake check when Mooncake is available in the environment.
- Validated in `tasks/mooncake_env` that the extension loads and that Mooncake, Zygote, ForwardDiff, and FiniteDifferences all agree on the target gradient.

Next:
- If desired, revisit Mooncake test integration in `Pkg.test`; the current project test extras have a resolver conflict with Mooncake's compat bounds, so the Mooncake assertion in the repo test file is optional rather than mandatory.
