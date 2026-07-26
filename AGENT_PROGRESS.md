# Agent progress

## Current task

Reduce fresh command-line latency for the standard NSSS, IRF, and first-order moments workflow.

## Status

- Added the generic command-line fixture at `benchmark/ttfx_basic.jl`.
- Extended the package workload to precompile the standard first-order NSSS, IRF, and moments calls.
- Baseline: 523.42 seconds without the package cache; post-precompile runs: 28.58, 23.12, and 22.43 seconds.
- All post-change output checks passed with the baseline NSSS, IRF, and moments checksums.
- Profiled the remaining silent time and checked inner NSSS/Jacobian/IRF type behavior.
- Replaced the IRF `Array{Any}` state store with `Array{Vector{S}}`; default-mode ex-post runs were 25.26, 22.91, and 23.28 seconds with unchanged outputs.
- Measured `--compile=min` separately at 10.44–11.26 seconds, but it changes the IRF checksum at the last floating-point bit and is not the default.
- Added `benchmark/ttfx_load_breakdown.jl` to time direct imports in the package's source order.
- Stable cached package import: 17.05 seconds; no-op `Pkg.precompile()`: 2.75 seconds.
- Incremental load breakdown: direct imports 16.432 seconds, with SymPyPythonCall 7.361 seconds and Symbolics 5.297 seconds; MacroModelling itself added 1.852 seconds after those imports.
- Forced source loading with `--compiled-modules=no --compile=min` took 135.62 seconds for package import alone. Repeated CLI startup is therefore dominated by package image restoration and module initialization, not the NSSS/IRF/moments kernels.
- Added `benchmark/ttfx_frbus.jl` and documented the command in `README.md`.
- FRBUS has 428 variables, 316 states, 116 shocks, and 1,139 parameters. Standard-mode external runs were 109.24 s and 98.79 s with identical output checksums.
- FRBUS definition generation took 77.53–85.73 s; after that, NSSS took 0.040–0.066 s, the all-shock 40-period IRF took 3.42–3.83 s, and supported mean/NSSS moments took 0.0003 s.
- Full covariance-based FRBUS moments are not valid because the model contains unit-root variables; the benchmark reports only mean/NSSS moments and documents this limitation.
- A FRBUS `--compile=min` probe was terminated after 561.69 s before completion, confirming that this mode is worse for the large model.
- Profiled SW07 NSSS setup. The pre-change baseline was 32.61 s for model generation, with 12.01 s in redundant-variable elimination, 10.50 s in NSSS setup, 2.24 s in NSSS search, and 3.12 s in first-order derivatives. The candidate workload contained 66 redundant-variable candidates.
- Added automatic `ss_symbolic_mode` selection. The default `:auto` path keeps `:single_equation` for small candidate workloads and selects numerical-only redundancy setup above 20 candidates. Added `benchmark/ttfx_sw07.jl` and documented explicit-mode comparison commands.
- SW07 automatic mode: 46.88 s external wall time, 26.40 s model definition, 20.6 s internal setup, 0.028 s NSSS, 0.801 s IRF, and 0.043 s moments. Explicit `:single_equation`: 53.39 s external, 32.99 s model definition, and 27.9 s internal setup. Full output checks passed after allowing the documented 875 undefined correlations.
- Generic RBC benchmark still passes with unchanged checksums. FRBUS automatic mode completed in 88.67 s and retained the earlier IRF checksum.

## Remaining work

- A custom Julia sysimage is the remaining high-impact startup optimization; lazy-loading SymPyPythonCall/PythonCall would be a larger API/architecture change because they are currently used in core types and parser code. The SW07 parser bottleneck is now reduced generically by automatic symbolic-mode selection, while explicit modes remain available when exact symbolic redundancy reduction is preferred.
