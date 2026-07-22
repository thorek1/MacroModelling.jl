# Equation-modification reprocessing pipeline.
#
# The `@model` and `@parameters` macros build a model struct and compile
# helpers. After an equation modification this file rebuilds the relevant
# internal state surgically — without re-evaluating the macros — by calling
# the pure functions `process_model_equations` and
# `process_parameter_definitions`.
#
# The file is organised as follows:
#   1. User-facing API   — exported functions with full docstrings.
#   2. Internal helpers  — reprocessing pipeline and small utilities.

# ========================================================================
# 1. User-facing API
# ========================================================================

const EquationInput = Union{Expr, String}
const EquationOrIndex = Union{Int, Expr, String}


"""
$(SIGNATURES)
Replace an existing model equation with a new one.

The first equation argument selects which equation to update: pass either the
1-based index, the existing equation as an `Expr`, or as a `String`. The new
equation can be passed as an `Expr` or a `String`.

A batched form is also supported: pass a `Vector` (or `Tuple`) of
`(old_or_index, new_equation)` `Pair`s or 2-tuples to apply several updates
in one rebuild.

After the update the revision history is appended (see
[`get_revision_history`](@ref)), all caches are invalidated and — if all
parameters are defined — the non-stochastic steady state is resolved and the
first-order symbolic derivatives are rewritten.

[`replace_equations!`](@ref) is an alias of this function.

# Arguments
- $MODEL®
- `old_equation_or_index` [Type: `Union{Int, Expr, String}`]: the equation to
  replace, identified by its 1-based index, by the equation itself as an
  `Expr`, or by the equation as a `String` (parsed before matching). When
  matching by `Expr`/`String`, comparison is done on a whitespace- and
  brace-insensitive canonical form.
- `new_equation` [Type: `Union{Expr, String}`]: the replacement equation.

Alternatively a single positional argument can be supplied:
- `updates` [Type: `Union{Vector, Tuple}`]: a collection of
  `(old_or_index => new_equation)` `Pair`s or 2-tuples.

# Keyword Arguments
- $PARAMETERS®
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline (e.g. about missing parameters).

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

# replace by index
update_equations!(RBC, 3, :(q[0] = exp(z[0]) * k[-1]^α + 0))

# replace by old equation expression
update_equations!(RBC,
    :(z[0] = ρ * z[-1] + std_z * eps_z[x]),
    :(z[0] = ρ * z[-1] + std_z * eps_z[x] + 0))

# batched update
update_equations!(RBC, [
    1 => :(1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))),
    2 => :(c[0] + k[0] = (1 - δ) * k[-1] + q[0]),
])
```
"""
function update_equations!(𝓂::ℳ,
                           old_equation_or_index::EquationOrIndex,
                           new_equation::EquationInput;
                           parameters::ParameterType = nothing,
                           verbose::Bool = false,
                           silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr

    originals = copy(𝓂.equations.original)
    idx::Int = if old_equation_or_index isa Int
        @assert 1 <= old_equation_or_index <= length(originals) "Equation index $(old_equation_or_index) out of bounds (1:$(length(originals)))."
        old_equation_or_index
    else
        target = normalize_equation_input(old_equation_or_index)::Expr
        found = find_equation_index(originals, target)
        @assert found !== nothing "Equation not found in model: $(target)"
        found
    end

    old_eq = originals[idx]
    originals[idx] = new_eq

    push!(𝓂.revision_history, revision_entry(:update_equation;
        equation_index = idx, old_equation = old_eq, new_equation = new_eq))

    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function update_equations!(𝓂::ℳ,
                           updates::Union{Vector, Tuple};
                           parameters::ParameterType = nothing,
                           verbose::Bool = false,
                           silent::Bool = true)
    originals = copy(𝓂.equations.original)
    history_entries = RevisionEntry[]
    for upd in updates
        @assert upd isa Union{Tuple, Pair} && length(upd) == 2 "Each update entry must be a (old_or_index, new_equation) pair/tuple."
        first_el = upd isa Pair ? upd.first : upd[1]
        second_el = upd isa Pair ? upd.second : upd[2]
        new_eq = normalize_equation_input(second_el)::Expr
        idx::Int = if first_el isa Int
            @assert 1 <= first_el <= length(originals) "Equation index $(first_el) out of bounds."
            first_el
        else
            target = normalize_equation_input(first_el)::Expr
            found = find_equation_index(originals, target)
            @assert found !== nothing "Equation not found in model: $(target)"
            found
        end
        old_eq = originals[idx]
        originals[idx] = new_eq
        push!(history_entries, revision_entry(:update_equation;
            equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    end
    append!(𝓂.revision_history, history_entries)
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Append one or more equations to the model.

Pass either a single equation (`Expr` or `String`) or a `Vector`/`Tuple` of
equations to add several at once. Each addition is recorded in the revision
history (see [`get_revision_history`](@ref)).

# Arguments
- $MODEL®
- `new_equation` [Type: `Union{Expr, String}`]: the equation to add.

Alternatively:
- `new_equations` [Type: `Union{Vector, Tuple}`]: a collection of equations
  to add in one rebuild.

# Keyword Arguments
- $PARAMETERS®
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline.

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

add_equation!(RBC, :(y[0] = c[0] + k[0] - (1 - δ) * k[-1]))

add_equation!(RBC, [
    :(inv[0] = k[0] - (1 - δ) * k[-1]),
    :(log_c[0] = log(c[0])),
])
```
"""
function add_equation!(𝓂::ℳ,
                       new_equation::EquationInput;
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    originals = copy(𝓂.equations.original)
    push!(originals, new_eq)
    push!(𝓂.revision_history, revision_entry(:add_equation;
        equation_index = length(originals), old_equation = nothing, new_equation = new_eq))
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function add_equation!(𝓂::ℳ,
                       new_equations::Union{Vector, Tuple};
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    originals = copy(𝓂.equations.original)
    entries = RevisionEntry[]
    for ne in new_equations
        new_eq = normalize_equation_input(ne)::Expr
        push!(originals, new_eq)
        push!(entries, revision_entry(:add_equation;
            equation_index = length(originals), old_equation = nothing, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Remove one or more equations from the model.

Equations can be selected by 1-based index, by their `Expr`, or by their
`String` representation. Pass a `Vector`/`Tuple` of selectors to remove
several equations in one rebuild. The model must always retain at least one
equation.

# Arguments
- $MODEL®
- `equation_or_index` [Type: `Union{Int, Expr, String}`]: the equation to
  remove. When matching by `Expr`/`String`, comparison is done on a
  whitespace- and brace-insensitive canonical form.

Alternatively:
- `removals` [Type: `Union{Vector, Tuple}`]: a collection of selectors.

# Keyword Arguments
- $PARAMETERS®
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline.

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
    y[0] = c[0] + k[0] - (1 - δ) * k[-1]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# remove by index
remove_equation!(RBC, 5)

# remove by expression
remove_equation!(RBC, :(q[0] = exp(z[0]) * k[-1]^α))
```
"""
function remove_equation!(𝓂::ℳ,
                          equation_or_index::EquationOrIndex;
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    originals = copy(𝓂.equations.original)
    @assert length(originals) > 1 "Cannot remove the last equation from the model."
    idx::Int = if equation_or_index isa Int
        @assert 1 <= equation_or_index <= length(originals) "Equation index $(equation_or_index) out of bounds."
        equation_or_index
    else
        target = normalize_equation_input(equation_or_index)::Expr
        found = find_equation_index(originals, target)
        @assert found !== nothing "Equation not found in model: $(target)"
        found
    end
    old_eq = originals[idx]
    deleteat!(originals, idx)
    push!(𝓂.revision_history, revision_entry(:remove_equation;
        equation_index = idx, old_equation = old_eq, new_equation = nothing))
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function remove_equation!(𝓂::ℳ,
                          removals::Union{Vector, Tuple};
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    originals = copy(𝓂.equations.original)
    indices = Int[]
    old_eqs = Expr[]
    for item in removals
        @assert length(originals) - length(indices) > 1 "Cannot remove the last equation from the model."
        idx::Int = if item isa Int
            @assert 1 <= item <= length(originals) "Equation index $(item) out of bounds."
            item
        else
            target = normalize_equation_input(item)::Expr
            found = find_equation_index(originals, target)
            @assert found !== nothing "Equation not found in model: $(target)"
            found
        end
        push!(indices, idx)
        push!(old_eqs, originals[idx])
    end
    order = sortperm(indices, rev = true)
    updated = copy(originals)
    for i in order
        deleteat!(updated, indices[i])
    end
    entries = RevisionEntry[]
    for (i, idx) in enumerate(indices)
        push!(entries, revision_entry(:remove_equation;
            equation_index = idx, old_equation = old_eqs[i], new_equation = nothing))
    end
    append!(𝓂.revision_history, entries)
    reprocess_model_equations!(𝓂, updated; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Replace an existing calibration equation.

Calibration equations use the `lhs = rhs | param` syntax and pin down a
parameter so it is solved jointly with the non-stochastic steady state. The
new equation must contain the `| param` marker and reference a parameter
that is already part of the model.

A batched form is supported: pass a `Vector`/`Tuple` of
`(old_or_index => new_equation)` `Pair`s or 2-tuples.

[`replace_calibration_equations!`](@ref) is an alias of this function.

# Arguments
- $MODEL®
- `old_equation_or_index` [Type: `Union{Int, Expr, String}`]: the calibration
  equation to replace, identified by its 1-based index in the calibration
  list or by its `Expr`/`String` representation.
- `new_equation` [Type: `Union{Expr, String}`]: the replacement calibration
  equation. Must contain the `| param` syntax.

Alternatively:
- `updates` [Type: `Union{Vector, Tuple}`]: a collection of
  `(old_or_index => new_equation)` `Pair`s or 2-tuples.

# Keyword Arguments
- $PARAMETERS®
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline.

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    α = 0.5
    β = 0.95
    k[ss] / (4 * q[ss]) = 1.5 | δ
end

# retarget the calibration equation
update_calibration_equations!(RBC, 1, :(k[ss] / (4 * q[ss]) = 2.0 | δ))
```
"""
function update_calibration_equations!(𝓂::ℳ,
                                       old_equation_or_index::EquationOrIndex,
                                       new_equation::EquationInput;
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    @assert extract_calibrated_parameter(new_eq) !== nothing "Calibration equation must contain `| param` syntax."

    calib_orig = copy(𝓂.equations.calibration_original)
    idx::Int = if old_equation_or_index isa Int
        @assert 1 <= old_equation_or_index <= length(calib_orig) "Calibration index $(old_equation_or_index) out of bounds."
        old_equation_or_index
    else
        target = normalize_equation_input(old_equation_or_index)::Expr
        found = find_equation_index(calib_orig, target)
        @assert found !== nothing "Calibration equation not found: $(target)"
        found
    end

    new_param = extract_calibrated_parameter(new_eq)
    known_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
    union!(known_params, 𝓂.constants.post_complete_parameters.parameters)
    union!(known_params, 𝓂.equations.calibration_parameters)
    new_param !== nothing && !(new_param in known_params) &&
        error("Parameter `$(new_param)` is not part of the model.")

    old_eq = calib_orig[idx]
    calib_orig[idx] = new_eq
    push!(𝓂.revision_history, revision_entry(:update_calibration_equation;
        equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function update_calibration_equations!(𝓂::ℳ,
                                       updates::Union{Vector, Tuple};
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    entries = RevisionEntry[]
    for upd in updates
        first_el = upd isa Pair ? upd.first : upd[1]
        second_el = upd isa Pair ? upd.second : upd[2]
        new_eq = normalize_equation_input(second_el)::Expr
        @assert extract_calibrated_parameter(new_eq) !== nothing "Calibration equation must contain `| param` syntax."
        idx::Int = if first_el isa Int
            @assert 1 <= first_el <= length(calib_orig) "Calibration index $(first_el) out of bounds."
            first_el
        else
            target = normalize_equation_input(first_el)::Expr
            found = find_equation_index(calib_orig, target)
            @assert found !== nothing "Calibration equation not found: $(target)"
            found
        end
        old_eq = calib_orig[idx]
        calib_orig[idx] = new_eq
        push!(entries, revision_entry(:update_calibration_equation;
            equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Add a new calibration equation to the model.

The equation must use the `lhs = rhs | param` syntax, and `param` must
already be part of the model and not yet calibrated. A batched form is
supported by passing a `Vector`/`Tuple` of equations.

# Arguments
- $MODEL®
- `new_equation` [Type: `Union{Expr, String}`]: the calibration equation to
  add. Must contain the `| param` syntax.

Alternatively:
- `new_equations` [Type: `Union{Vector, Tuple}`]: a collection of
  calibration equations to add in one rebuild.

# Keyword Arguments
- $PARAMETERS®
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline.

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

# pin down β by targeting a value for c at the steady state
add_calibration_equation!(RBC, :(c[ss] = 1.0 | β))
```
"""
function add_calibration_equation!(𝓂::ℳ,
                                   new_equation::EquationInput;
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    new_param = extract_calibrated_parameter(new_eq)
    new_param === nothing && error("Calibration equation must contain `| param` syntax.")
    new_param in 𝓂.equations.calibration_parameters &&
        error("Parameter `$(new_param)` is already calibrated.")
    known_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
    union!(known_params, 𝓂.constants.post_complete_parameters.parameters)
    !(new_param in known_params) && error("Parameter `$(new_param)` is not part of the model.")

    calib_orig = copy(𝓂.equations.calibration_original)
    push!(calib_orig, new_eq)
    push!(𝓂.revision_history, revision_entry(:add_calibration_equation;
        equation_index = length(calib_orig), old_equation = nothing, new_equation = new_eq))
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function add_calibration_equation!(𝓂::ℳ,
                                   new_equations::Union{Vector, Tuple};
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    entries = RevisionEntry[]
    for ne in new_equations
        new_eq = normalize_equation_input(ne)::Expr
        new_param = extract_calibrated_parameter(new_eq)
        new_param === nothing && error("Calibration equation must contain `| param` syntax.")
        push!(calib_orig, new_eq)
        push!(entries, revision_entry(:add_calibration_equation;
            equation_index = length(calib_orig), old_equation = nothing, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Remove a calibration equation.

When a calibration equation is removed, the parameter previously solved by
it becomes a free parameter and needs a numeric value. By default that value
is taken from the parameter's current non-stochastic steady state value. Use
the `parameters` keyword to override this with explicit values.

A batched form is supported by passing a `Vector`/`Tuple` of selectors.

# Arguments
- $MODEL®
- `equation_or_index` [Type: `Union{Int, Expr, String}`]: the calibration
  equation to remove.

Alternatively:
- `removals` [Type: `Union{Vector, Tuple}`]: a collection of selectors.

# Keyword Arguments
- `parameters` [Default: `nothing`]: optional value(s) used as the new fixed
  value for the parameter freed by the removal. Accepts the same forms as
  the `parameters` argument elsewhere — a `Pair`, a `Dict`, or a
  `Vector`/`Tuple` of `Pair`s — but only the entries matching freed
  parameters are used.
- $VERBOSE®
- `silent` [Default: `true`, Type: `Bool`]: suppress informational warnings
  from the rebuild pipeline.

# Returns
- `nothing`. The model `𝓂` is updated in place.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    α = 0.5
    β = 0.95
    k[ss] / (4 * q[ss]) = 1.5 | δ
end

# remove the calibration equation, fixing δ to 0.02
remove_calibration_equation!(RBC, 1, parameters = :δ => 0.02)
```
"""
function remove_calibration_equation!(𝓂::ℳ,
                                      equation_or_index::EquationOrIndex;
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    @assert !isempty(calib_orig) "No calibration equations to remove."
    idx::Int = if equation_or_index isa Int
        @assert 1 <= equation_or_index <= length(calib_orig) "Calibration index $(equation_or_index) out of bounds."
        equation_or_index
    else
        target = normalize_equation_input(equation_or_index)::Expr
        found = find_equation_index(calib_orig, target)
        @assert found !== nothing "Calibration equation not found: $(target)"
        found
    end
    old_eq = calib_orig[idx]
    deleteat!(calib_orig, idx)
    push!(𝓂.revision_history, revision_entry(:remove_calibration_equation;
        equation_index = idx, old_equation = old_eq, new_equation = nothing))

    param_overrides = parameters_to_dict(parameters)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = nothing,
        parameter_overrides = param_overrides, verbose = verbose, silent = silent)
    return nothing
end

function remove_calibration_equation!(𝓂::ℳ,
                                      removals::Union{Vector, Tuple};
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    @assert !isempty(calib_orig) "No calibration equations to remove."
    indices = Int[]
    old_eqs = Expr[]
    for item in removals
        idx::Int = if item isa Int
            @assert 1 <= item <= length(calib_orig) "Calibration index $(item) out of bounds."
            item
        else
            target = normalize_equation_input(item)::Expr
            found = find_equation_index(calib_orig, target)
            @assert found !== nothing "Calibration equation not found: $(target)"
            found
        end
        push!(indices, idx)
        push!(old_eqs, calib_orig[idx])
    end
    updated = copy(calib_orig)
    for i in sort(indices, rev = true)
        deleteat!(updated, i)
    end
    entries = RevisionEntry[]
    for (i, idx) in enumerate(indices)
        push!(entries, revision_entry(:remove_calibration_equation;
            equation_index = idx, old_equation = old_eqs[i], new_equation = nothing))
    end
    append!(𝓂.revision_history, entries)

    param_overrides = parameters_to_dict(parameters)
    reprocess_calibration_equations!(𝓂, updated; parameters = nothing,
        parameter_overrides = param_overrides, verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Return the recorded history of equation modifications for the model.

Each entry is a `NamedTuple` with the fields:
- `timestamp`: the `DateTime` when the modification was applied.
- `action`: one of `:update_equation`, `:add_equation`, `:remove_equation`,
  `:update_calibration_equation`, `:add_calibration_equation`,
  `:remove_calibration_equation`.
- `equation_index`: the 1-based index that was affected.
- `old_equation`: the previous equation as an `Expr`, or `nothing` for
  additions.
- `new_equation`: the new equation as an `Expr`, or `nothing` for removals.

The list is append-only and ordered chronologically.

# Arguments
- $MODEL®

# Returns
- `Vector{RevisionEntry}` — a copy of the model's revision history.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

add_equation!(RBC, :(y[0] = c[0] + k[0] - (1 - δ) * k[-1]))
get_revision_history(RBC)
# output
1-element Vector{@NamedTuple{timestamp::Dates.DateTime, action::Symbol, equation_index::Union{Nothing, Int64}, old_equation::Union{Nothing, Expr}, new_equation::Union{Nothing, Expr}}}:
 (timestamp = ..., action = :add_equation, equation_index = 5, old_equation = nothing, new_equation = :(y[0] = (c[0] + k[0]) - (1 - δ) * k[-1]))
```
"""
function get_revision_history(𝓂::ℳ)::Vector{RevisionEntry}
    return copy(𝓂.revision_history)
end


"""
$(SIGNATURES)
Write the current model equations and parameter block to a Julia source file.

The generated file uses the `@model` and `@parameters` macros, so
`include`ing it re-creates a model equivalent to the current state of `𝓂`
(equations, calibration, parameter values and bounds).

# Arguments
- $MODEL®
- `filepath` [Type: `String`]: destination path for the generated `.jl` file.

# Keyword Arguments
- `overwrite` [Default: `false`, Type: `Bool`]: replace the file if it
  already exists. When `false` and the file exists, an error is raised.

# Returns
- `String` — the path written to (same as `filepath`).

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

write_julia_model_file(RBC, joinpath(tempdir(), "RBC.jl"), overwrite = true)
```
"""
function write_julia_model_file(𝓂::ℳ, filepath::String; overwrite::Bool = false)::String
    if isfile(filepath) && !overwrite
        error("File already exists: $filepath. Pass overwrite = true to replace it.")
    end

    parameter_block = reconstruct_parameter_block(𝓂)
    model_name = string(𝓂.model_name)

    open(filepath, "w") do io
        println(io, "using MacroModelling\n")
        println(io, "@model ", model_name, " begin")
        for eq in 𝓂.equations.original
            println(io, "    ", replace(string(eq), "◖" => "{", "◗" => "}"))
        end
        println(io, "end\n")

        println(io, "@parameters ", model_name, " begin")
        for line in parameter_block.args
            line isa LineNumberNode && continue
            println(io, "    ", replace(string(line), "◖" => "{", "◗" => "}"))
        end
        println(io, "end")
    end

    return filepath
end


# Aliases
const replace_equations! = update_equations!
const replace_calibration_equations! = update_calibration_equations!


# ========================================================================
# 2. Internal helpers (not exported)
# ========================================================================

"""
    reset_solver_state!(𝓂::ℳ)

Invalidate every cached solver result so the next `solve!` call recomputes
from scratch. Also marks the compiled model functions as outdated so the
rebuild pipeline rewrites them.
"""
function reset_solver_state!(𝓂::ℳ)
    𝓂.caches.valid_for = valid_for_caches()
    empty!(𝓂.caches.solver)

    # Reset size-dependent cache matrices so downstream code reallocates for
    # the (potentially) new model dimensions.
    𝓂.caches.jacobian = zeros(0, 0)
    𝓂.caches.jacobian_parameters = zeros(0, 0)
    𝓂.caches.jacobian_SS_and_pars = zeros(0, 0)
    𝓂.caches.hessian = zeros(0, 0)
    𝓂.caches.hessian_parameters = zeros(0, 0)
    𝓂.caches.hessian_SS_and_pars = zeros(0, 0)
    𝓂.caches.third_order_derivatives = zeros(0, 0)
    𝓂.caches.third_order_derivatives_parameters = zeros(0, 0)
    𝓂.caches.third_order_derivatives_SS_and_pars = zeros(0, 0)
    𝓂.caches.first_order_solution_matrix = zeros(0, 0)
    𝓂.caches.first_order_obc_solution_matrix = zeros(0, 0)
    𝓂.caches.qme_solution = zeros(0, 0)
    𝓂.caches.second_order_stochastic_steady_state = Float64[]
    𝓂.caches.second_order_solution = SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)
    𝓂.caches.pruned_second_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_stochastic_steady_state = Float64[]
    𝓂.caches.third_order_solution = SparseMatrixCSC{Float64, Int64}(ℒ.I, 0, 0)
    𝓂.caches.pruned_third_order_stochastic_steady_state = Float64[]
    𝓂.caches.non_stochastic_steady_state = Float64[]
    𝓂.caches.covariance_first_order = zeros(0, 0)
    𝓂.caches.covariance_second_order = zeros(0, 0)
    𝓂.caches.covariance_third_order = zeros(0, 0)
    𝓂.caches.covariance_third_order_autocorr = zeros(0, 0)
    𝓂.caches.has_unit_roots = false

    𝓂.functions.functions_written = false

    # Invalidate derived caches that depend on the equations / calibration.
    # Empty axes / name tables are the sentinel used by the `ensure_*!`
    # helpers to decide whether to recompute.
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        var_axis = Symbol[],
        calib_axis = Symbol[],
        exo_axis_plain = Symbol[],
        exo_axis_with_subscript = Symbol[],
        full_NSSS_display = Symbol[],
        SS_and_pars_names = Symbol[],
        initialized = false,
    )

    # Reset the workspace buffers, forcing `ensure_*!` helpers to resize them
    # on next use.
    𝓂.workspaces = Workspaces()

    return nothing
end


"""
    reconstruct_parameter_block(𝓂; calibration_original_override = nothing,
                                   parameter_overrides = nothing)

Return a `:block` `Expr` that reproduces a valid `@parameters` body from the
current model state. Optional overrides let callers preview the effect of a
calibration change before committing it to the model.
"""
function reconstruct_parameter_block(𝓂::ℳ;
                                     calibration_original_override::Union{Nothing, Vector{Expr}} = nothing,
                                     parameter_overrides::Union{Nothing, AbstractDict{Symbol, <:Real}} = nothing)
    lines = Any[]

    calibration_original = calibration_original_override === nothing ?
        𝓂.equations.calibration_original : calibration_original_override

    new_calib_params = Set{Symbol}()
    for eq in calibration_original
        p = extract_calibrated_parameter(eq)
        if p !== nothing
            push!(new_calib_params, p)
        end
    end

    old_calib_params = Set{Symbol}(𝓂.equations.calibration_parameters)
    params_becoming_calibrated = setdiff(new_calib_params, old_calib_params)
    params_no_longer_calibrated = setdiff(old_calib_params, new_calib_params)

    fixed_params = 𝓂.constants.post_complete_parameters.parameters
    fixed_values = 𝓂.parameter_values
    for (p, v) in zip(fixed_params, fixed_values)
        (isnan(v) || p in params_becoming_calibrated) && continue
        val = (parameter_overrides !== nothing && haskey(parameter_overrides, p)) ? parameter_overrides[p] : v
        push!(lines, Expr(:(=), p, val))
    end

    if !isempty(params_no_longer_calibrated)
        n_vars = 𝓂.constants.post_model_macro.nVars
        old_calib_list = 𝓂.equations.calibration_parameters
        for p in params_no_longer_calibrated
            if parameter_overrides !== nothing && haskey(parameter_overrides, p)
                push!(lines, Expr(:(=), p, parameter_overrides[p]))
                continue
            end
            idx = findfirst(==(p), old_calib_list)
            if idx !== nothing && length(𝓂.caches.non_stochastic_steady_state) >= n_vars + idx
                val = 𝓂.caches.non_stochastic_steady_state[n_vars + idx]
                push!(lines, Expr(:(=), p, val))
            end
        end
    end

    for eq in 𝓂.equations.calibration_no_var
        push!(lines, eq)
    end

    for eq in calibration_original
        push!(lines, eq)
    end

    for (p, (lo, hi)) in 𝓂.constants.post_parameters_macro.bounds
        push!(lines, Expr(:comparison, lo, :(<), p, :(<), hi))
    end

    return Expr(:block, lines...)
end


"""
    extract_calibrated_parameter(eq::Expr) -> Union{Symbol, Nothing}

Return the parameter on the right of the `|` in a calibration equation such
as `k[ss] = 1.5 | δ`. Returns `nothing` if no calibration marker is found.
"""
function extract_calibrated_parameter(eq::Expr)::Union{Symbol, Nothing}
    result = Ref{Union{Symbol, Nothing}}(nothing)
    postwalk(eq) do x
        if x isa Expr && x.head == :call && !isempty(x.args) && x.args[1] == :|
            if length(x.args) >= 3 && x.args[end] isa Symbol
                result[] = x.args[end]
            end
        end
        x
    end
    return result[]
end


"""
    finalize_model_update!(𝓂; verbose, silent)

Internal helper that finalises a model update by rebuilding the steady-state
solver and symbolic derivatives for the current `𝓂` state. Called by both
`reprocess_model_equations!` and `reprocess_calibration_equations!`.
"""
function finalize_model_update!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    has_missing_parameters = !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
    missing_params = 𝓂.constants.post_complete_parameters.missing_parameters

    if !isnothing(𝓂.functions.NSSS_custom)
        write_ss_check_function!(𝓂)
    else
        if !has_missing_parameters
            set_up_steady_state_solver!(
                𝓂;
                verbose = verbose,
                silent = silent,
                ss_symbolic_mode = 𝓂.constants.post_parameters_macro.ss_symbolic_mode,
            )
        end
    end

    if !has_missing_parameters
        opts = merge_calculation_options(verbose = verbose)
        solve_steady_state!(
            𝓂,
            opts,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime;
            silent = silent,
        )
        write_symbolic_derivatives!(𝓂; perturbation_order = 1, silent = silent)
        𝓂.functions.functions_written = true
    else
        if !silent
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(missing_params). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined."
        end
    end

    return nothing
end


"""
    reprocess_model_equations!(𝓂, new_equations; parameters, verbose, silent)

Rebuild the model from an updated equation list while preserving the
revision history. Equivalent to re-running `@model` and `@parameters` on the
current parameter state, but without re-evaluating the macros.
"""
function reprocess_model_equations!(𝓂::ℳ,
                                    new_equations::Vector{Expr};
                                    parameters::ParameterType = nothing,
                                    verbose::Bool = false,
                                    silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end

    updated_block = Expr(:block, new_equations...)
    parameter_block = reconstruct_parameter_block(𝓂)

    T, equations_struct, ℂ, 𝓦 = process_model_equations(
        updated_block,
        𝓂.constants.post_model_macro.max_obc_horizon,
        𝓂.constants.post_parameters_macro.precompile,
    )

    𝓂.constants = ℂ
    𝓂.workspaces = 𝓦
    𝓂.equations = equations_struct

    reset_solver_state!(𝓂)

    parsed_parameters = process_parameter_definitions(parameter_block, 𝓂.constants.post_model_macro)

    𝓂.constants.post_parameters_macro = update_post_parameters_macro(
        𝓂.constants.post_parameters_macro;
        parameters_as_function_of_parameters = parsed_parameters.calib_parameters_no_var,
        ss_calib_list = parsed_parameters.ss_calib_list,
        par_calib_list = parsed_parameters.par_calib_list,
        bounds = parsed_parameters.bounds,
    )

    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
    )

    𝓂.parameter_values = parsed_parameters.parameter_values

    finalize_model_update!(𝓂; verbose = verbose, silent = silent)

    return nothing
end


"""
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters,
                                     parameter_overrides, verbose, silent)

Rebuild the model's calibration with a modified `calibration_original` list.
`parameter_overrides` lets callers supply replacement fixed values for
parameters that leave the calibration set.
"""
function reprocess_calibration_equations!(𝓂::ℳ,
                                          updated_calibration_original::Vector{Expr};
                                          parameters::ParameterType = nothing,
                                          parameter_overrides::Dict{Symbol, Float64} = Dict{Symbol, Float64}(),
                                          verbose::Bool = false,
                                          silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end

    parameter_block = reconstruct_parameter_block(
        𝓂;
        calibration_original_override = updated_calibration_original,
        parameter_overrides = parameter_overrides,
    )

    parsed_parameters = process_parameter_definitions(parameter_block, 𝓂.constants.post_model_macro)

    𝓂.constants.post_parameters_macro = update_post_parameters_macro(
        𝓂.constants.post_parameters_macro;
        parameters_as_function_of_parameters = parsed_parameters.calib_parameters_no_var,
        ss_calib_list = parsed_parameters.ss_calib_list,
        par_calib_list = parsed_parameters.par_calib_list,
        bounds = parsed_parameters.bounds,
    )

    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
    )

    𝓂.parameter_values = parsed_parameters.parameter_values

    reset_solver_state!(𝓂)

    finalize_model_update!(𝓂; verbose = verbose, silent = silent)

    return nothing
end


"""
    normalize_equation_input(eq)

Normalize a user-provided equation expression: strip line-number nodes and
collapse single-expression `begin ... end` blocks. Accepts `Expr` or
`String` (which is parsed first).
"""
normalize_equation_input(eq::String) = normalize_equation_input(Meta.parse(eq))
function normalize_equation_input(eq::Expr)
    cleaned = rmlines(eq)
    return postwalk(cleaned) do node
        if @capture(node, begin arg_ end)
            arg
        else
            node
        end
    end
end


"""
    expr_contains_dynamic(expr, base_expr, pattern) -> Bool

Check whether `expr` contains a dynamic reference whose base symbol matches
`base_expr`. When `pattern === nothing` any timing counts as a match; when
it is an `Expr` or `Symbol`, match exact equality.
"""
function expr_contains_dynamic(expr, base_expr, pattern)
    base_matches(candidate, base) = candidate == base ||
        string(candidate) == string(base)

    found = Ref(false)
    postwalk(expr) do x
        if pattern === nothing
            if x isa Expr && x.head == :ref
                base_matches(x.args[1], base_expr) && (found[] = true)
            elseif x isa Expr && x.head == :curly
                base_matches(x, base_expr) && (found[] = true)
            elseif x isa Symbol
                base_matches(x, base_expr) && (found[] = true)
            end
        else
            x == pattern && (found[] = true)
        end
        x
    end
    return found[]
end


"""
    find_equation_index(equations::Vector{Expr}, target_eq::Expr) -> Union{Int, Nothing}

Return the 1-based index of `target_eq` inside `equations`, comparing via
the normalized textual form (so whitespace and `{}`/`◖◗` differences are
ignored). Returns `nothing` if no match is found.
"""
function find_equation_index(equations::Vector{Expr}, target_eq::Expr)::Union{Int, Nothing}
    target_str = equation_canonical(target_eq)
    for (i, eq) in enumerate(equations)
        if equation_canonical(eq) == target_str
            return i
        end
    end
    return nothing
end


"""
    equation_canonical(eq) -> String

Return a whitespace- and brace-insensitive canonical string form of `eq`,
used to compare equations for equality.
"""
function equation_canonical(eq)::String
    s = string(eq)
    s = replace(s, "◖" => "{", "◗" => "}")
    s = replace(s, r"\s+" => "")
    return s
end


"""
    revision_entry(action; equation_index, old_equation, new_equation)

Construct a `RevisionEntry` `NamedTuple` with the current timestamp.
"""
revision_entry(action::Symbol;
               equation_index::Union{Int, Nothing} = nothing,
               old_equation::Union{Expr, Nothing} = nothing,
               new_equation::Union{Expr, Nothing} = nothing) =
    (timestamp = Dates.now(),
     action = action,
     equation_index = equation_index,
     old_equation = old_equation,
     new_equation = new_equation)


"""
    parameters_to_dict(parameters) -> Dict{Symbol, Float64}

Convert `ParameterType`-like user input into a `Dict{Symbol, Float64}` used
by `reprocess_calibration_equations!`. Unsupported forms fall back to an
empty dict.
"""
function parameters_to_dict(parameters)::Dict{Symbol, Float64}
    d = Dict{Symbol, Float64}()
    parameters === nothing && return d
    if parameters isa Pair
        k = parameters.first; v = parameters.second
        k_sym = k isa Symbol ? k : Symbol(k)
        d[k_sym] = Float64(v)
    elseif parameters isa AbstractDict
        for (k, v) in parameters
            k_sym = k isa Symbol ? k : Symbol(k)
            d[k_sym] = Float64(v)
        end
    elseif parameters isa Union{Tuple, Vector}
        for p in parameters
            if p isa Pair
                k = p.first; v = p.second
                k_sym = k isa Symbol ? k : Symbol(k)
                d[k_sym] = Float64(v)
            end
        end
    end
    return d
end
