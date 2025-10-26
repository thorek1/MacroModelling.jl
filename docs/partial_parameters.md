# Partial Parameter Definition

Starting from this version, MacroModelling.jl allows you to define a model without specifying all parameters upfront. This is useful when:
- You want to load parameters from a file later
- You're working with large models and want to define parameters incrementally
- You want to experiment with different parameter values via function arguments

## Usage

### Basic Example

```julia
using MacroModelling

# Define your model
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

# Define only some parameters initially
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    α = 0.5
    # β and δ are not defined yet
end
```

When you run this, you'll see an informative message:
```
[ Info: Model set up with undefined parameters: [:β, :δ]
Non-stochastic steady state and solution cannot be calculated until all parameters are defined.
```

### Checking Undefined Parameters

You can check which parameters are still undefined:

```julia
RBC.undefined_parameters  # Returns [:β, :δ]
```

### Completing Parameter Definition

You can define the remaining parameters later by calling `@parameters` again:

```julia
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    α = 0.5
    β = 0.95  # Now defined
    δ = 0.02  # Now defined
end
```

After this, `RBC.undefined_parameters` will be empty, and you can compute steady states and IRFs.

### Providing Parameters via Function Arguments

Alternatively, you can provide all parameters when calling functions:

```julia
# Model still has undefined parameters
@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
end

# Provide all parameters including the undefined ones
params = Dict(:α => 0.5, :β => 0.95, :δ => 0.02, :std_z => 0.01, :ρ => 0.2)

# This will work and update the undefined_parameters list
get_irf(RBC, parameters = params, periods = 40)
```

## Behavior with Undefined Parameters

When you try to compute outputs (IRFs, steady states, etc.) with undefined parameters:

1. **Functions that compute NSSS**: These will log an error message indicating which parameters are missing and return appropriate fallback values
2. **The `undefined_parameters` field**: Automatically tracked and updated when:
   - Parameters are defined via `@parameters`
   - Parameters are provided via function arguments (Dict or Vector)

## Loading Parameters from Files

A common use case is loading parameters from external sources:

```julia
using MacroModelling, CSV, DataFrames

# Define model
@model MyModel begin
    # ... model equations ...
end

# Set up with minimal parameters
@parameters MyModel begin
    # ... only essential parameters ...
end

# Load remaining parameters from file
params_df = CSV.read("model_parameters.csv", DataFrame)
params_dict = Dict(Symbol(row.parameter) => row.value for row in eachrow(params_df))

# Provide parameters via function call
get_irf(MyModel, parameters = params_dict)
```

## Notes

- The model structure is still fully validated when created with `@model`
- Only parameter values can be left undefined, not the model equations
- When all parameters are eventually defined, the model behaves exactly as if all parameters were defined from the start
- The non-stochastic steady state and solution will be computed once all parameters are provided
