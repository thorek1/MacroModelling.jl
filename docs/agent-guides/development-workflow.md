# Development Workflow (On-Demand)

Read this file only when setup, runtime workflow, testing, docs, or benchmarking details are needed.

## Julia Setup

- Julia version: 1.10+
- Run Julia with threads enabled: `julia -t auto`
- If Julia is not on PATH (Linux), check `~/.juliaup/bin/julia`

### Environment setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

If packages are missing, install them first (for example with `Pkg.add(...)`).

## Revise-Based Iteration (Required for Interactive Work)

Always use Revise for iterative development.

### One-time session setup

1. Start one REPL and keep it running:

```bash
cd /path/to/MacroModelling.jl
julia -t auto --project=.
```

2. In the REPL, load Revise before MacroModelling:

```julia
using Revise
using Pkg
Pkg.activate(".")
using MacroModelling
```

3. Edit source files and run code in the same session.

### Why

- Avoids repeated precompilation cost
- Preserves session/model state between edits
- Enables rapid edit-test-fix loops

### Caveats

- Structural changes (new type layouts, module reorganization, `__init__` changes) may require restart
- If updates are missed, run `Revise.revise()`

## Quick Testing Strategy

Do not run the full test suite for normal iteration.

### Preferred approach

- Use a bespoke script or quick reproduction with a small model
- Validate only the impacted behavior first

Example RBC model for lightweight checks:

```julia
@model RBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(RBC)
simulate(RBC)
```

## CI Test Sets (Reference)

Only use targeted sets when needed:

- `basic`, `estimation`, `higher_order_1-3`, `plots_1-5`, `estimate_sw07`, `jet`
- Estimation sets: `1st_order_inversion_estimation`, `2nd_order_estimation`, `pruned_2nd_order_estimation`, `3rd_order_estimation`, `pruned_3rd_order_estimation`
- Pigeons estimation sets: `estimation_pigeons`, `1st_order_inversion_estimation_pigeons`, `2nd_order_estimation_pigeons`, `pruned_2nd_order_estimation_pigeons`, `3rd_order_estimation_pigeons`, `pruned_3rd_order_estimation_pigeons`

```bash
TEST_SET=basic julia --project -e 'using Pkg; Pkg.test()'
```

Test environment setup:

```julia
using Pkg
Pkg.activate("test")
Pkg.instantiate()
```

## Documentation Build

```bash
julia --project=docs docs/make.jl
```

## Benchmarking

```julia
using BenchmarkTools
include("benchmark/benchmarks.jl")
run(SUITE)
```
