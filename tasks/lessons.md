Lessons learned:
- Mooncake can reuse an existing analytical ChainRules `rrule` cleanly via `Mooncake.@from_rrule`; the missing piece here was not new calculus, but a primitive registration layer for the positional `get_solution` call.
- `MacroModelling.ℳ` needs a Mooncake `NoTangent` override because its caches contain solver internals that Mooncake should not recursively differentiate through.
- In this repo, direct `test/runtests.jl` execution does not expose extras, and adding Mooncake to the default test extras currently triggers a resolver conflict with existing test-only dependencies. Optional Mooncake assertions inside focused tests are a safer interim pattern.
