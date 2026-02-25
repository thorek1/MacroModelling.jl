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

Always use Revise for iterative development. **Never use one-shot `julia -e` commands** — they discard the session and force full recompilation on every call.

### Persistent REPL via Named Pipe (for AI Agents)

AI agents cannot type into a REPL interactively. Use a named-pipe pattern to maintain a persistent Julia session across tool calls.

#### 1. Start the session (once per conversation)

Use `.julia_repl/` inside the project directory (already in `.gitignore`) instead of `/tmp/` to avoid VS Code trusted-folder approval prompts.

```bash
# Create infrastructure (inside the project — no approval needed)
mkdir -p .julia_repl
rm -f .julia_repl/pipe .julia_repl/out
mkfifo .julia_repl/pipe
touch .julia_repl/out

# Start Julia reading from pipe (background process)
tail -f .julia_repl/pipe | julia -t auto --project=. 2>&1 | tee .julia_repl/out &
```

Start this with `isBackground=true` so the terminal stays alive.

#### 2. Load packages (once)

```bash
echo 'using Revise; using MacroModelling; println("REPL_READY")' > .julia_repl/pipe
sleep 30 && tail -3 .julia_repl/out
```

Wait for `REPL_READY` in the output before proceeding. Package loading takes 10-30 seconds.

#### 3. Execute code

**Preferred method** — write code to a file, then include it:

```bash
# Step A: Write Julia code to a .jl file (using create_file tool — no terminal command needed)
# File: tasks/_repl_cmd.jl

# Step B: Run it in the persistent session (one terminal command)
echo 'include("tasks/_repl_cmd.jl")' > .julia_repl/pipe
sleep 5 && tail -20 .julia_repl/out
```

**For short one-liners**, send directly:

```bash
echo 'println(1 + 1)' > .julia_repl/pipe
sleep 2 && tail -3 .julia_repl/out
```

#### 4. Read output

Always end code with a sentinel `println` (e.g., `println("DONE")`) and check for it:

```bash
tail -30 .julia_repl/out   # recent output
grep "DONE" .julia_repl/out  # verify completion
```

To reset the output file (avoid stale reads):

```bash
: > .julia_repl/out
```

#### 5. Key rules

- **Always use sentinel markers** — end every code block with `println("STEP_NAME_DONE")` so the agent can confirm execution completed.
- **Adjust sleep durations** — use longer sleeps for compilation-heavy first calls (~30s), shorter for cached calls (~2-5s).
- **The session persists** — variables, models, compiled methods all survive between `echo` commands. This is the whole point.
- **Revise picks up edits** — after editing `src/` files with the editor tool, the running session sees the changes automatically.
- **For test project deps**, use `--project=test` instead of `--project=.` when tests need extra packages (Zygote, Turing, etc.).
- **To reset the session**, send `exit()` to the pipe, then re-run steps 1-2:
  ```bash
  echo 'exit()' > .julia_repl/pipe && sleep 2
  rm -f .julia_repl/pipe .julia_repl/out && mkfifo .julia_repl/pipe && touch .julia_repl/out
  # Then restart with tail -f ... & and reload packages
  ```

### Human Developer REPL Setup

1. Start one REPL and keep it running:

```bash
cd /path/to/MacroModelling.jl
julia -t auto --project=.
```

2. In the REPL, load Revise before MacroModelling:

```julia
using Revise
using MacroModelling
```

3. Edit source files and run code in the same session.

### Why

- Avoids repeated precompilation cost (minutes per call → zero)
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
