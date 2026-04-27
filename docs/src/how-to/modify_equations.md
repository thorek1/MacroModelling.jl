# Modifying a model after definition

`MacroModelling.jl` lets you edit a model in place after the `@model` and
`@parameters` blocks have been evaluated. Equations (and calibration
equations) can be replaced, appended, or removed without re-declaring the
model: the package re-runs the equation-processing pipeline, invalidates
cached solver results, and recomputes the non-stochastic steady state. Each
change is recorded in a chronological revision log.

This is useful for iterating on model variants in the REPL, programmatically
generating model alternatives, swapping in observation equations for
estimation, or testing the impact of a single equation change without
rebuilding the entire model definition.

## API overview

| Operation                 | Model equations              | Calibration equations                  |
|:--------------------------|:-----------------------------|:---------------------------------------|
| Replace one (or many)     | `update_equations!`          | `update_calibration_equations!`        |
| Append                    | `add_equation!`              | `add_calibration_equation!`            |
| Remove                    | `remove_equation!`           | `remove_calibration_equation!`         |

Helpers:

- `get_revision_history(𝓂)` — inspect the chronological log of changes.
- `write_julia_model_file(𝓂, path)` — serialise the current model state to a
  Julia file that re-creates the (possibly heavily revised) model when
  `include`d.

Each modifying function accepts:

- The target equation as a 1-based **index**, an `Expr`, or a `String`
  (matching is canonical, so whitespace/parenthesisation is ignored).
- Either a single update or a `Vector` / `Tuple` of `(target, new)` pairs
  (or, for `add_*`, a vector of new equations) for batch operations.
- A `parameters` keyword that is forwarded to the re-solve step.

All of them mutate the model object (note the trailing `!`) and return
`nothing`.

## Working example

The examples below use a small RBC model.

```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ     = 0.2
    δ     = 0.02
    α     = 0.5
    β     = 0.95
end;
```

### Replace an equation by index

`get_equations` shows the equations in their stored order; that order is
what the index-based API refers to.

```julia
get_equations(RBC)

# Replace the AR(1) shock process (the 4th equation) with a more
# persistent one.
update_equations!(RBC, 4, :(z[0] = 0.9 * z[-1] + std_z * eps_z[x]))
```

### Replace an equation by matching `Expr` / `String`

You don't need to know the index — pass the old equation literally. The
matcher is canonicalising, so the spacing of the input does not matter.

```julia
update_equations!(RBC,
    :(q[0] = exp(z[0]) * k[-1]^α),
    :(q[0] = exp(z[0]) * k[-1]^α * l[0]^(1 - α)))   # add labour to production

# Strings work too:
update_equations!(RBC,
    "q[0] = exp(z[0]) * k[-1]^α * l[0]^(1 - α)",
    "q[0] = exp(z[0]) * k[-1]^α")                    # revert
```

### Add and remove equations

```julia
# Append a definitional equation
add_equation!(RBC, :(log_q[0] = log(q[0])))

# Remove it again (by Expr, by String, or by index)
remove_equation!(RBC, :(log_q[0] = log(q[0])))
```

### Batch updates

`update_equations!`, `add_equation!`, and `remove_equation!` all accept a
vector to apply several changes in a single re-solve:

```julia
update_equations!(RBC, [
    (4, :(z[0] = 0.95 * z[-1] + std_z * eps_z[x])),
    (:(c[0] + k[0] = (1 - δ) * k[-1] + q[0]),
     :(c[0] + k[0] + g[0] = (1 - δ) * k[-1] + q[0])),
])
```

### Modify calibration equations

Calibration equations use the `lhs = rhs | param` syntax and are edited via
their own functions. The parameter on the right of `|` must already be
declared in the model.

```julia
# Replace the calibration target for δ
update_calibration_equations!(RBC, 1, :(k[ss] / q[ss] = 10.0 | δ))

# Add a new calibration equation
add_calibration_equation!(RBC, :(c[ss] / q[ss] = 0.7 | β))

# Remove it again, fixing the freed parameter to a chosen value
remove_calibration_equation!(RBC, :(c[ss] / q[ss] = 0.7 | β),
                             parameters = :β => 0.95)
```

### Inspect the revision history

Every modification appends an entry to the revision log. Each entry is a
`NamedTuple` with `timestamp`, `action`, `equation_index`, `old_equation`,
and `new_equation` fields.

```julia
for entry in get_revision_history(RBC)
    println(entry.action, " @ ", entry.equation_index,
            " : ", entry.old_equation, " => ", entry.new_equation)
end
```

### Persist the revised model

Once happy with the changes, the current state can be written back to a
Julia source file that re-creates the (revised) model when `include`d:

```julia
write_julia_model_file(RBC, "RBC_revised.jl"; overwrite = true)
```
