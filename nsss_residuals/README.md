# NSSS residual exports

Each file in this directory is a standalone module for one model in `models/`.
The equation constants contain residual expressions, so every expression is set
equal to zero in the corresponding NSSS problem.

- `ORIGINAL_NSSS_EQUATIONS` is the NSSS form without parser-added auxiliary variables.
- `AUXILIARY_NSSS_EQUATIONS` is the NSSS form with parser-added auxiliary variables.
- `BLOCKS` gives the auxiliary form separated according to the NSSS block decomposition.
- `CALIBRATION_EQUATIONS` is included in both residual systems.
- `ALL_AUXILIARY_VARIABLE_*` also includes domain-safety variables created while setting up the solver.

The functions `residuals_original`, `residuals_auxiliary`, and `residuals_blocks`
all take `(parameters, solution)` and return a residual vector. The parameter and
solution ordering is given by the matching `*_NAMES` constants. Box constraints,
parameter values, and stored NSSS values are provided as constants in each file.

For example:

```julia
include("RBC_baseline.jl")
using .RBC_baselineNsssResiduals

residuals_auxiliary(PARAMETER_VALUES, AUXILIARY_SOLUTION_VALUES)
```

`scripts/generate_nsss_residual_models.jl` regenerates the exports from the
current model files. `scripts/verify_nsss_residual_models.jl` loads every export
and checks all three residual variants at their stored NSSS values.
