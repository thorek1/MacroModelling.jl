# `MacroModelling.jl` - fast prototyping of dynamic stochastic general equilibrium (DSGE) models

## Author: Thore Kockerols (@thorek1)

The package currently supports dicsrete-time DSGE models and the timing of a variable reflects when the variable is decided (end of period for stock variables).

As of now the package can:

- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (tries to) solve the model only knowing the model equations and parameter values (no steady state file needed)
- calculate first, second, and third order perturbation solutions using (forward) automatic differentiation (AD)
- calculate (generalised) impulse response functions, and simulate the model
- calibrate parameters using (non stochastic) steady state relationships
- match model moments
- estimate the model on data (kalman filter using first order perturbation)
- **differentiate** (forward AD) the model solution (first order perturbation), kalman filter loglikelihood, model moments, steady state, **with respect to the parameters**

The package contains the following models in the `test/models` folder:

- Smets and Wouters (2003) `SW03.jl`
- Smets and Wouters (2007) `SW07.jl`
- Schorfheide (2000) `FS2000.jl`
  