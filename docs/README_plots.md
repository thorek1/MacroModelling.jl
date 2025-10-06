# Generating Plots for Documentation

This directory contains the `generate_plots.jl` script which generates all the plot images referenced in the plotting documentation (`src/plotting.md`).

## Prerequisites

Before running the plot generation script, you need to have:

1. Julia installed (version 1.11 or later recommended)
2. MacroModelling.jl package installed (or be in the package development environment)
3. StatsPlots.jl package installed

## How to Generate Plots

### From the package development environment

If you're developing MacroModelling.jl:

```bash
cd docs
julia --project=.. generate_plots.jl
```

This will:
- Create the `assets` directory if it doesn't exist
- Generate all plot images referenced in `src/plotting.md`
- Save them as PNG files in `docs/assets/`

### From a fresh Julia installation

If you want to generate plots with a fresh installation:

```julia
# In Julia REPL
using Pkg
Pkg.add("MacroModelling")
Pkg.add("StatsPlots")

# Then run the script
include("docs/generate_plots.jl")
```

## Generated Files

The script generates approximately 28 different plot files, including:
- Basic IRF plots for different shocks
- Comparison plots for different solution methods
- Plots with different parameter values
- Plots with custom styling and colors
- OBC (occasionally binding constraints) examples
- GIRF (generalised impulse response functions) examples

All files follow the naming convention:
```
irf__<model_name>__<shock>__<number>[_<descriptor>].png
```

For example:
- `irf__Gali_2015_chapter_3_nonlinear__eps_a__1.png`
- `irf__Gali_2015_chapter_3_nonlinear__eps_a__2_second_order.png`
- `irf__Gali_2015_chapter_3_obc__eps_z__1_girf.png`

## Notes

- The script sets `show_plots = false` to avoid displaying plots during generation
- Plot generation can take several minutes as it needs to solve the models multiple times
- If the script fails, check that all dependencies are properly installed
- The script uses the `save_plots_name` argument to create unique filenames for each plot
