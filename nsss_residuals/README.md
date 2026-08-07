# NSSS residual exports

Each file in this directory is a standalone module for one model in `models/`.
The equation constants contain residual expressions, so every expression is set
equal to zero in the corresponding NSSS problem.

- `ORIGINAL_NSSS_EQUATIONS` is the NSSS form without parser-added auxiliary variables.
- `AUXILIARY_NSSS_EQUATIONS` is the NSSS form with parser-added auxiliary variables.
- `BLOCKS` gives the auxiliary form separated according to the NSSS block decomposition.
  Each block identifies its current unknowns, previously solved inputs, any fixed
  or unmatched external inputs, domain auxiliary unknowns, defining equations,
  box constraints, and stored values.
- `CALIBRATION_EQUATIONS` is included in both residual systems.
- `ALL_AUXILIARY_VARIABLE_*` also includes domain-safety variables created while setting up the solver.

The functions `residuals_original` and `residuals_auxiliary` take
`(parameters, solution)`. Each `residuals_block_N` takes
`(parameters, previous_solution, external_solution, solution)`, where the three
solution vectors follow that block's name lists. Domain auxiliary definitions
are included in the block residual vector. `residuals_blocks` concatenates all
block residuals and takes arrays of the three corresponding vector types.
Box constraints, parameter values, and stored NSSS values are provided as
constants in each file.

For example:

```julia
include("RBC_baseline.jl")
using .RBC_baselineNsssResiduals

residuals_auxiliary(PARAMETER_VALUES, AUXILIARY_SOLUTION_VALUES)
residuals_block_1(PARAMETER_VALUES,
                  BLOCK_PREVIOUS_SOLUTION_VALUES[1],
                  BLOCK_EXTERNAL_SOLUTION_VALUES[1],
                  BLOCK_SOLUTION_VALUES[1])
```

`scripts/generate_nsss_residual_models.jl` regenerates the exports from the
current model files. `scripts/verify_nsss_residual_models.jl` loads every export
and checks all three residual variants at their stored NSSS values.
