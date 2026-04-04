Current task: analytical parameter gradients for `norm(get_solution(model, params)[2])` with Mooncake compatibility.

Plan:
1. Reproduce the current differentiation behavior on a focused first-order solution objective.
2. Identify the exact composition point that breaks for Mooncake.
3. Implement the narrowest analytical reverse-mode fix.
4. Add focused tests against Zygote, ForwardDiff, and FiniteDifferences, and verify Mooncake.

Status:
- Reproduction completed.
- Implemented a Mooncake extension for the positional `get_solution(model, params)` path.
- Added focused gradient comparisons to `test/test_standalone_function.jl`.
- Verified Mooncake, Zygote, ForwardDiff, and FiniteDifferences agreement in `tasks/mooncake_env`.
- Remaining caveat: Mooncake cannot currently be added to the default `Pkg.test` extras without introducing an existing resolver conflict with other test-only dependencies.
