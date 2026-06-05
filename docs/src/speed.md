# Speed Benchmarks

This page compares the runtime of `MacroModelling.jl` against [Dynare](https://www.dynare.org), the most widely used DSGE toolbox, on a set of example models. Both tools are timed performing the same tasks on identical models, across three operating systems and CPU architectures.

## How to read the results

- **Models and their size.** Each model is identified by its name together with a `states / variables` pair: the number of state variables and the total number of model variables. This pair gives a sense for model size and difficulty — for example `Caldara_et_al_2012` (`3/12`) is among the smallest systems, while `FRBUS` (`316/428`) is the largest case shown here. Larger models shift the runtime towards the underlying linear-algebra (BLAS/LAPACK) cost.
- **Components.** Timings are split by the step being measured. *Jacobian* and *Hessian* are the first- and second-order derivatives of the model equations that the perturbation solution needs; *First-order solve* and *Second-order solve* are the perturbation solutions at those orders; *Third-order bundled* is a combined timing for the full third-order pipeline (see the note below the table).
- **Speedup.** A value of `Nx` means `MacroModelling.jl` is `N` times faster than Dynare for that component on that operating system — e.g. `4.8x` is roughly five times faster, while `0.1x` is about ten times slower (Dynare faster). The speedup and timing columns both run left-to-right as Linux (Ubuntu 24), macOS 26, and Windows 11.
- **Bold values** mark representative conservative Windows cases. On Windows, Dynare runs on MATLAB with compiled mex files and Intel MKL — its fastest configuration — whereas the Linux and macOS rows run Dynare on Octave. The bold Windows speedups therefore act as a lower bound on the advantage of `MacroModelling.jl`.

## Summary

- Across the Windows solve rows, `MacroModelling.jl` still usually remains faster, from `1.1x` on the largest stress case `FRBUS` (`316/428` states/variables) up to `4.8x` on the `Smets_Wouters_2007` (`26/66`) second-order solve. In absolute terms, that spans from `111.79 ms` versus `125.05 ms` for the `FRBUS` first-order solve down to `18.3 μs` versus `395.2 μs` for the `Caldara_et_al_2012` (`3/12`) second-order solve. Third-order bundled timings remain decisively in `MacroModelling.jl`'s favour on Windows at `80.2x` to `115.6x`, corresponding to `736.3 μs` versus `59.02 ms` for `Gali_2015_chapter_3_nonlinear` and `176.6 μs` versus `20.41 ms` for `Caldara_et_al_2012`.
- On Ubuntu 24 and macOS 26, the largest gains appear in derivative construction (the Jacobian and Hessian builds), and the effect is visible from very small to fairly large models. Small systems such as `Caldara_et_al_2012` (`3/12`), `FS2000` (`4/16`), and `Gali_2015_chapter_3_nonlinear` (`4/23`) build Jacobians in `0.5-1.2 μs` and Hessians in `1.2-2.8 μs` under `MacroModelling.jl`, while Dynare needs roughly `280 μs-2.5 ms` for Jacobians and `891 μs-2.51 ms` for Hessians. The same pattern extends to larger systems such as `GNSS_2010` (`38/66`), `QUEST3_2009` (`58/107`), and `NAWM_EAUS_2008` (`106/224`), where `MacroModelling.jl` still stays in the `3.0-64.3 μs` Jacobian range while Dynare takes `1.18-9.0 ms`.
- As model size grows, absolute solve times move from tens of microseconds for the smallest models to milliseconds and then low hundreds of milliseconds for the largest ones, but the ordering remains broadly stable across operating systems. The main exception is the Windows `FRBUS` Jacobian, where MATLAB mex files reduce Dynare's derivative cost to `49.1 μs` versus `722.8 μs` for `MacroModelling.jl` (a `0.1x` speedup, i.e. Dynare is faster in that single case). Outside that case, Linux and macOS show the clearest speedups, while Windows narrows the gap without changing the overall picture that `MacroModelling.jl` scales better across the benchmark set.

## Measurement notes

All timings reported below are single-thread measurements. Separate multithreaded thread-count sweeps increased runtime across the board for both `MacroModelling.jl` and Dynare, so only the single-thread results are shown here: in those multithreaded runs, orchestration and parallelisation overhead outweighed computational gains from additional threads.

The table is organised in perturbation-order blocks. The opening rows summarise the full machine and software stack, and each order block then repeats a compact header so the per-platform triplets remain readable in plain markdown.

## Benchmark Timings

Speedup columns report how many times faster MacroModelling.jl is than Dynare for that component.

| Perturbation Order | Model | Component | Speedup | Speedup | Speedup | States / Variables | MacroModelling.jl | MacroModelling.jl | MacroModelling.jl | Dynare | Dynare | Dynare |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | | OS | Ubuntu 24 | macOS 26 | Windows 11 | | Ubuntu 24 | macOS 26 | Windows 11 | Ubuntu 24 | macOS 26 | Windows 11 |
| | | CPU | AMD EPYC 7R13 | Apple M2 | Intel Alder Lake | | AMD EPYC 7R13 | Apple M2 | Intel Alder Lake | AMD EPYC 7R13 | Apple M2 | Intel Alder Lake |
| | | CPU Architecture | x86_64 | aarch64 | x86_64 | | x86_64 | aarch64 | x86_64 | x86_64 | aarch64 | x86_64 |
| | | Package version | | | | | 0.1.47 | 0.1.47 | 0.1.47 | 7.0.1 | 7.0.1 | 7.0 |
| | | Language | | | | | Julia 1.12.6 | Julia 1.12.6 | Julia 1.12.6 | Octave 11.1.0 | Octave 11.1.0 | MATLAB R2024b Update 6 |
| | | BLAS/LAPACK | | | | | OpenBLAS 0.3.29 | OpenBLAS 0.3.29 | OpenBLAS 0.3.29 | OpenBLAS 0.3.33 | OpenBLAS 0.3.33 | MKL 2024.1 / LAPACK 3.11.0 |
| | | | | | | | | | | | | |
| **Perturbation Order** | **Model** | **Component** | **Speedup** | **Speedup** | **Speedup** | **States / Variables** | **MacroModelling.jl** | **MacroModelling.jl** | **MacroModelling.jl** | **Dynare** | **Dynare** | **Dynare** |
| | | OS | Ubuntu 24 | macOS 26 | Windows 11 | | Ubuntu 24 | macOS 26 | Windows 11 | Ubuntu 24 | macOS 26 | Windows 11 |
| First | Caldara_et_al_2012 | Jacobian | 466.2x | 422.7x | 20.1x | 3/12 | 1.1 μs | 0.7 μs | 1.5 μs | 512.8 μs | 295.9 μs | 30.2 μs |
| | | First-order solve | 17.5x | 11.7x | 7.0x | 3/12 | 21.5 μs | 15.4 μs | 13.3 μs | 376.9 μs | 180.0 μs | 92.7 μs |
| | FRBUS | Jacobian | 14.8x | 52.5x | 0.1x | 316/428 | 195.6 μs | 34.3 μs | 722.8 μs | 2.9 ms | 1.8 ms | 49.1 μs |
| | | First-order solve | 3.5x | 4.0x | **1.1x** | 316/428 | 102.65 ms | 84.2 ms | 111.79 ms | 356.89 ms | 337.31 ms | 125.05 ms |
| | FS2000 | Jacobian | 617.5x | 558.4x | 17.7x | 4/16 | 0.8 μs | 0.5 μs | 0.6 μs | 494.0 μs | 279.2 μs | 10.6 μs |
| | | First-order solve | 9.3x | 7.2x | 2.8x | 4/16 | 43.2 μs | 26.7 μs | 34.6 μs | 402.0 μs | 191.9 μs | 97.7 μs |
| | GNSS_2010 | Jacobian | 274.7x | 453.3x | 5.1x | 38/66 | 8.3 μs | 3.0 μs | 9.4 μs | 2.28 ms | 1.36 ms | 48.4 μs |
| | | First-order solve | 1.9x | 1.7x | 2.2x | 38/66 | 1.24 ms | 967.9 μs | 1.13 ms | 2.33 ms | 1.6 ms | 2.48 ms |
| | Gali_2015_chapter_3_nonlinear | Jacobian | 420.0x | 351.1x | 13.4x | 4/23 | 1.2 μs | 0.8 μs | 0.8 μs | 504.0 μs | 280.9 μs | 10.7 μs |
| | | First-order solve | 7.8x | 5.7x | 3.9x | 4/23 | 57.1 μs | 37.0 μs | 37.5 μs | 448.0 μs | 212.0 μs | 145.1 μs |
| | NAWM_EAUS_2008 | Jacobian | 140.0x | 154.5x | 1.2x | 106/224 | 64.3 μs | 24.6 μs | 75.2 μs | 9.0 ms | 3.8 ms | 92.2 μs |
| | | First-order solve | 2.7x | 2.9x | **1.5x** | 106/224 | 16.59 ms | 13.29 ms | 13.67 ms | 44.07 ms | 39.1 ms | 21.06 ms |
| | QUEST3_2009 | Jacobian | 115.7x | 295.0x | 1.6x | 58/107 | 15.3 μs | 4.0 μs | 18.1 μs | 1.77 ms | 1.18 ms | 28.6 μs |
| | | First-order solve | 2.2x | 2.6x | **2.3x** | 58/107 | 2.45 ms | 1.68 ms | 2.07 ms | 5.37 ms | 4.41 ms | 4.85 ms |
| | Smets_Wouters_2003 | Jacobian | 319.7x | 308.9x | 3.8x | 19/54 | 7.1 μs | 4.5 μs | 7.1 μs | 2.27 ms | 1.39 ms | 26.7 μs |
| | | First-order solve | 1.9x | 1.8x | **2.2x** | 19/54 | 698.5 μs | 474.9 μs | 589.4 μs | 1.34 ms | 831.8 μs | 1.3 ms |
| | Smets_Wouters_2007 | Jacobian | 207.5x | 154.5x | 1.9x | 26/66 | 9.3 μs | 7.7 μs | 10.5 μs | 1.93 ms | 1.19 ms | 19.9 μs |
| | | First-order solve | 1.8x | 2.0x | **2.1x** | 26/66 | 1.08 ms | 657.0 μs | 779.1 μs | 1.91 ms | 1.29 ms | 1.61 ms |
| | | | | | | | | | | | | |
| **Perturbation Order** | **Model** | **Component** | **Speedup** | **Speedup** | **Speedup** | **States / Variables** | **MacroModelling.jl** | **MacroModelling.jl** | **MacroModelling.jl** | **Dynare** | **Dynare** | **Dynare** |
| | | OS | Ubuntu 24 | macOS 26 | Windows 11 | | Ubuntu 24 | macOS 26 | Windows 11 | Ubuntu 24 | macOS 26 | Windows 11 |
| Second | Caldara_et_al_2012 | Hessian | 525.0x | 524.1x | 65.9x | 3/12 | 2.8 μs | 1.7 μs | 1.6 μs | 1.47 ms | 891.0 μs | 105.4 μs |
| | | Second-order solve | 16.0x | 5.1x | 21.6x | 3/12 | 27.9 μs | 40.5 μs | 18.3 μs | 446.1 μs | 206.9 μs | 395.2 μs |
| | FS2000 | Hessian | 1321.1x | 1291.7x | 22.1x | 4/16 | 1.9 μs | 1.2 μs | 2.4 μs | 2.51 ms | 1.55 ms | 53.0 μs |
| | | Second-order solve | 10.0x | 4.4x | 8.0x | 4/16 | 53.0 μs | 57.6 μs | 70.6 μs | 531.0 μs | 255.1 μs | 564.6 μs |
| | Gali_2015_chapter_3_nonlinear | Hessian | 595.8x | 605.4x | 33.0x | 4/23 | 2.4 μs | 1.5 μs | 1.6 μs | 1.43 ms | 908.1 μs | 52.8 μs |
| | | Second-order solve | 8.1x | 3.6x | 6.4x | 4/23 | 78.4 μs | 90.5 μs | 67.2 μs | 635.5 μs | 329.0 μs | 429.5 μs |
| | Smets_Wouters_2007 | Hessian | 1337.9x | 1251.1x | 36.5x | 26/66 | 6.6 μs | 4.5 μs | 4.2 μs | 8.83 ms | 5.63 ms | 153.1 μs |
| | | Second-order solve | 6.5x | 7.9x | **4.8x** | 26/66 | 3.11 ms | 2.45 ms | 3.79 ms | 20.29 ms | 19.44 ms | 18.19 ms |
| | | | | | | | | | | | | |
| **Perturbation Order** | **Model** | **Component** | **Speedup** | **Speedup** | **Speedup** | **States / Variables** | **MacroModelling.jl** | **MacroModelling.jl** | **MacroModelling.jl** | **Dynare** | **Dynare** | **Dynare** |
| | | OS | Ubuntu 24 | macOS 26 | Windows 11 | | Ubuntu 24 | macOS 26 | Windows 11 | Ubuntu 24 | macOS 26 | Windows 11 |
| Third | Caldara_et_al_2012 | Third-order bundled | 75.6x | 52.8x | **115.6x** | 3/12 | 275.4 μs | 235.9 μs | 176.6 μs | 20.81 ms | 12.46 ms | 20.41 ms |
| | Gali_2015_chapter_3_nonlinear | Third-order bundled | 42.6x | 33.8x | 80.2x | 4/23 | 828.2 μs | 597.3 μs | 736.3 μs | 35.3 ms | 20.18 ms | 59.02 ms |

Third-order bundled is not third-order-only on the MacroModelling side. In this harness it is first-order solve + Hessian + second-order solve + third-order derivatives + third-order solve, while Dynare reports the direct bundled `k_order_pert` timing.
