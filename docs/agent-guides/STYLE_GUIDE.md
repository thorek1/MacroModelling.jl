# Code Style Guide for MacroModelling.jl

This document describes the coding conventions and style rules used throughout the MacroModelling.jl codebase.
All new code should follow these guidelines to maintain consistency.

---

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Formatting and Indentation](#formatting-and-indentation)
3. [Function Signatures](#function-signatures)
4. [Type System](#type-system)
5. [Module Organisation](#module-organisation)
6. [Control Flow](#control-flow)
7. [Error Handling](#error-handling)
8. [Documentation](#documentation)
9. [Performance](#performance)
10. [Collections and Arrays](#collections-and-arrays)
11. [Strings and Symbols](#strings-and-symbols)
12. [Logging and Verbosity](#logging-and-verbosity)
13. [Caching](#caching)
14. [Macros](#macros)

---

## Naming Conventions

### Functions

Use **snake_case** for all function names:

```julia
calculate_first_order_solution(...)
get_shock_decomposition(...)
solve_quadratic_matrix_equation(...)
```

Mutating functions must end with `!` per Julia convention:

```julia
solve!(𝓂, ...)
fast_lu!(ws, A)
ensure_lyapunov_doubling_buffers!(ws, n)
```

### Variables

Use **snake_case** for multi-word variable names:

```julia
past_not_future_and_mixed_idx
non_stochastic_steady_state
```

Use **Unicode mathematical symbols** for domain-specific variables to match the underlying mathematics:

```julia
𝓂       # model object
∇₁      # Jacobian
∇₂      # Hessian
𝐒₁      # first-order solution matrix
𝐒₂      # second-order solution matrix
ϵ       # epsilon / shocks
Σʸ₁     # covariance matrix
```

Use **Unicode subscripts and superscripts** for order indices:

```julia
nₑ      # number of exogenous variables
n₋      # number of past variables
n₊      # number of future variables
i₊      # future indices
i₋      # past indices
```

Prefix counts with `n`:

```julia
nVars
nExo
nPresent_only
nMixed
```

### Types and Structs

Use **snake_case** for workspace and internal structs:

```julia
struct second_order_indices ... end
mutable struct qme_workspace{T} ... end
mutable struct sylvester_workspace{G,H} ... end
```

### Constants

Use **SCREAMING_SNAKE_CASE** for constants:

```julia
const DEFAULT_ALGORITHM = :first_order
const DEFAULT_VERBOSE = false
const ANALYTICAL_STEP = 1
const NUMERICAL_STEP = 2
```

Docstring template constants use a `®` suffix:

```julia
const MODEL® = "..."
const ALGORITHM® = "..."
const VERBOSE® = "..."
```

### Module Aliases

Import libraries with **Unicode letter aliases**:

```julia
import LinearAlgebra as ℒ
import LinearSolve as 𝒮
import ForwardDiff as ℱ
import DifferentiationInterface as 𝒟
```

### Type Aliases

Define union types for user-facing inputs:

```julia
const Symbol_input = Union{Symbol, Vector{Symbol}, ...}
const ParameterType = Union{Nothing, Pair{Symbol, Float64}, ...}
```

---

## Formatting and Indentation

### Indentation

Use **4 spaces** for indentation. Never use tabs.

```julia
function foo(x)
    if x > 0
        return x
    else
        return -x
    end
end
```

### Line Length

There is no strict line-length limit. Long lines (200+ characters) are acceptable for complex mathematical expressions and function signatures. Prefer readability over arbitrary wrapping.

### Whitespace

Spaces around binary operators:

```julia
n₋ + 1 + nₑ
A * X * B + C
x == nothing
```

No space before `(` in function calls:

```julia
zeros(T, n, n)
size(A, 1)
push!(vec, val)
```

Space after commas:

```julia
zeros(T, n, n)
solve!(𝓂, parameters = parameters, verbose = verbose)
```

### Blank Lines

No blank lines between closely related one-liner function definitions:

```julia
get_symbols(ex::Symbol) = [ex]
get_symbols(ex::Real) = [ex]
get_symbols(ex::Int) = [ex]
```

Two or more blank lines between major function definitions to visually separate sections.

### Section Headers

Use comment banners to delineate major sections within a file:

```julia
# =========================================================================
# AUXILIARY MATRICES (for perturbation solution)
# =========================================================================
```

### Keyword Argument Alignment

Align keyword arguments vertically, each on its own line, indented to the opening parenthesis:

```julia
function get_shock_decomposition(𝓂::ℳ,
                                data::KeyedArray{Float64};
                                parameters::ParameterType = nothing,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                verbose::Bool = DEFAULT_VERBOSE)
```

---

## Function Signatures

### Type Annotations

Annotate return types on public-facing functions:

```julia
function get_equations(𝓂::ℳ)::Vector{String}
    ...
end
```

Use parametric `where` clauses to constrain type parameters:

```julia
function solve!(A::AbstractMatrix{T},
                B::AbstractMatrix{T}) where {T <: AbstractFloat}
    ...
end
```

### Keyword Arguments

Separate keyword arguments with `;`. Every keyword argument should have a default value, preferably drawn from `DEFAULT_*` constants:

```julia
function get_irf(𝓂::ℳ;
                 parameters::ParameterType = nothing,
                 algorithm::Symbol = DEFAULT_ALGORITHM,
                 verbose::Bool = DEFAULT_VERBOSE,
                 tol::Tolerances = Tolerances())
```

### Short Functions

Write simple functions as one-liners:

```julia
get_symbols(ex::Symbol) = [ex]
noop_state_update(::Float64, ::Float64) = nothing
```

### Multiple Dispatch

Use `Val` dispatch for compile-time-known mode selection:

```julia
filter_data_with_model(𝓂, data, Val(algorithm), Val(filter), ...)
```

Use type dispatch for workspace variants:

```julia
fast_lu!(A::AbstractMatrix{T}) where T = ...
fast_lu!(ws::LUWorkspace, A::AbstractMatrix{T}) where T = ...
```

---

## Type System

### Struct Definitions

Explicitly type all struct fields:

```julia
mutable struct qme_workspace{T <: Real, R <: Real}
    A::Matrix{T}
    B::Matrix{T}
    solved::Bool
    n::Int
end
```

Use `mutable struct` for workspaces and caches that change over time.
Use `struct` for immutable configuration objects.

### Parametric Types

Constrain type parameters to `Real`, `AbstractFloat`, or `Number` as appropriate:

```julia
mutable struct sylvester_workspace{G <: AbstractFloat, H <: Real}
    ...
end
```

---

## Module Organisation

### Import Order

In the main module file, follow this order:

1. `module` declaration
2. `import` statements with Unicode aliases
3. `using` statements (only for packages that should export into scope)
4. Inline utility function definitions
5. Type aliases
6. `include` of source files (in dependency order)
7. `export` statements (grouped by functionality)
8. AD rule includes (at the very end)
9. `end` (module close)

### `import` vs `using`

**Prefer `import` over `using`** to keep the namespace clean:

```julia
# Preferred
import LinearAlgebra as ℒ
import SparseArrays: SparseMatrixCSC, sparse!, spzeros

# Only for packages that must export into scope
using PrecompileTools
using DispatchDoctor
```

### Include Order

Include files in dependency order — structures before functions that use them:

```julia
include("default_options.jl")
include("common_docstrings.jl")
include("structures.jl")
include("solver_parameters.jl")
include("options_and_caches.jl")
include("nsss_solver.jl")
include("macros.jl")
include("get_functions.jl")
# ...subdirectories
include("./algorithms/sylvester.jl")
include("./filter/kalman.jl")
```

### Exports

Provide multiple aliases for discoverability:

```julia
export get_steady_state, get_SS, get_ss,
       get_non_stochastic_steady_state,
       steady_state, SS, SSS, ss, sss
```

---

## Control Flow

### Short-Circuit Returns

Use short-circuit for early returns:

```julia
if !solved return zeros(T, n, n), sol, false end
```

### Ternary Operator

Use ternary for simple inline conditionals:

```julia
verbose ? println("Solving...") : nothing
filter == :kalman ? :kalman : :inversion
```

### Inline `if`

Use single-line `if` for simple branches:

```julia
if opts.verbose println("Quadratic matrix equation solution failed.") end
if solved 𝓂.caches.qme_solution = qme_sol end
```

### `@assert` for Preconditions

```julia
@assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Theoretical mean available only for..."
```

### `for` Loops

Standard range iteration:

```julia
for i in 1:n
    ...
end
```

Reverse iteration with step:

```julia
for n in length(eqs_to_solve)-1:-1:2
    ...
end
```

Destructuring with `enumerate`:

```julia
for (i, x) in enumerate(aux_vars)
    ...
end
```

### `do` Blocks

Use `do` blocks with `postwalk`/`prewalk` for AST manipulation:

```julia
postwalk(expr) do x
    if x isa Expr && x.head == :(=)
        found = true
    end
    return x
end
```

Use `do` blocks with `open` for file I/O:

```julia
open(filepath, "w") do io
    println(io, content)
end
```

### `try/catch`

For cases where failure is expected and should be silently handled, use compact `try/catch`:

```julia
result = try SPyPyC.solve(equation, variable)
         catch
         end
```

For user-facing errors, re-raise with context:

```julia
try
    run(pipeline(...))
catch
    error("Failed to parse the model. ...")
end
```

---

## Error Handling

### Exceptions

Use `throw(ArgumentError(...))` for invalid arguments:

```julia
throw(ArgumentError("invalid argument to LU factorization, info = $info"))
```

### Boolean Solved Flags

Return `(result, solved::Bool)` from solver functions rather than throwing. Callers check the flag:

```julia
sol, solved = calculate_first_order_solution(...)
if !solved
    return zeros(...), sol, false
end
```

### Warnings

Use `@warn` for non-fatal issues:

```julia
@warn "Invalid option `$(x.args[1])` ignored..."
```

Use `@info` with `maxlog` for informational messages that should not repeat:

```julia
@info "Higher order solution algorithms only support the inversion filter." maxlog = maxlog
```

---

## Documentation

### Docstrings

Use `$(SIGNATURES)` from DocStringExtensions for auto-generated signatures.

Structure docstrings with these sections:

```julia
"""
$(SIGNATURES)

Short description of the function.

# Arguments
- `arg1`: description

# Keyword Arguments
- `kwarg1` [default: `value`]: description
$MODEL®
$ALGORITHM®
$VERBOSE®

# Returns
- Description of return value

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    ...
end

@parameters RBC begin
    ...
end

get_equations(RBC)
# output
...
```
"""
```

### Shared Docstring Constants

Define reusable docstring fragments as constants with the `®` suffix and reference them with `$`:

```julia
const MODEL® = """
- `𝓂`: the model object
"""

# In docstring:
"""
# Arguments
\$MODEL®
"""
```

### Comments

Use inline comments to explain non-obvious fields and logic:

```julia
A::Matrix{T}     # n×n copy of A
solved::Bool      # whether QME converged
```

Preserve commented-out alternative approaches for reference.

### Writing Style

- Avoid second-person phrasing ("you") in documentation and docstrings
- Use third person or imperative mood

---

## Performance

### `@inline`

Apply `@inline` to hot-path utility functions:

```julia
@inline function fast_lu!(ws, A::AbstractMatrix{T}) where T
    ...
end
```

### `@views`

Use `@views` to avoid array copies:

```julia
@views sol[:, 1:T.nPast_not_future_and_mixed]
@views [𝐒₁[i₊,:]; ...]
```

### Pre-allocation and Workspaces

All major solvers use pre-allocated workspace structs. Use `ensure_*_buffers!` functions that lazily resize workspaces only when dimensions change:

```julia
function ensure_lyapunov_doubling_buffers!(ws::lyapunov_workspace, n::Int)
    if size(ws.A, 1) != n
        ws.A = zeros(n, n)
        # ...resize all buffers...
    end
end
```

### Type Stability

- Annotate return types on functions
- Use parametric `where` clauses
- Avoid untyped containers in hot paths

### Sparse Matrices

Use `choose_matrix_format` to decide dense vs sparse based on density thresholds.
Clean up near-zero entries with `droptol!`.

### `@ignore_derivatives`

Use `ChainRulesCore.@ignore_derivatives` for code that should be invisible to AD:

```julia
@ignore_derivatives begin
    # cache updates, logging, etc.
end
```

---

## Collections and Arrays

### Broadcasting

Prefer dot syntax for element-wise operations:

```julia
data .- NSSS[obs_idx]
obs_axis .|> Meta.parse .|> replace_indices
solved_vals .= new_values
```

### Comprehensions

Use array comprehensions for constructing new arrays:

```julia
[replace_curly_braces_in_symbols(arg) for arg in expr.args]
```

Use generator expressions inside aggregation functions:

```julia
sum(k * (k + 1) ÷ 2 for k in 1:n)
```

### Pipe Operator

Use `|>` for chaining transformations:

```julia
parse_variables_input_to_index(obs_symbols, 𝓂) |> sort
collect(∂block) |> findnz
```

### `Ref` for Broadcasting Scalars

Wrap non-collection arguments in `Ref` when broadcasting:

```julia
replace_symbols.(expressions, Ref(parameter_dict))
Symbolics.substitute.(x, Ref(back_to_array_dict))
```

### `push!` and `append!`

Use `push!` for single elements, `append!` for extending with another collection:

```julia
push!(b.step_types, ANALYTICAL_STEP)
append!(b.write_indices, write_indices)
```

---

## Strings and Symbols

### Interpolation

Use `$` for string interpolation:

```julia
"invalid argument, info = $info"
```

### Concatenation

Use `*` for string concatenation (Julia convention):

```julia
string(x.args[1]) * "₍ₓ₎"
string(x.args[1]) * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎"
```

### Regex

Use `r"..."` literals, with flags as needed:

```julia
occursin(r"^(x|ex|exo|exogenous){1}$"i, input)
```

### `replace` Chains

Chain `replace` calls for multiple substitutions:

```julia
replace(replace(replace(str, "₍₋₁₎" => "[-1]"), "₍₁₎" => "[1]"), "₍₀₎" => "[0]")
```

---

## Logging and Verbosity

### `verbose::Bool`

Controls solver-internal diagnostics via `println`:

```julia
if opts.verbose println("Quadratic matrix equation solution failed.") end
```

### `silent::Bool`

Controls progress printing for user-facing operations:

```julia
if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end
# ...computation...
if !silent println(round(time() - start_time, digits = 3), " seconds") end
```

### `@info` / `@warn`

Use `@info` with `maxlog` for corrections that should not repeat endlessly:

```julia
@info "Setting filter = :inversion for higher order solution." maxlog = maxlog
```

Use `@warn` for non-fatal warnings:

```julia
@warn "Solution does not have a stochastic steady state."
```

---

## Caching

### Pattern

Use a dedicated `caches` sub-struct with a parallel `outdated` flags struct:

```julia
𝓂.caches.non_stochastic_steady_state = SS_and_pars
𝓂.caches.outdated.non_stochastic_steady_state = solution_error > tol
```

### Check → Recompute → Store → Clear

```julia
if 𝓂.caches.outdated.second_order_solution || parameters_changed
    # ...recompute...
    𝓂.caches.second_order_stochastic_steady_state = result
    𝓂.functions.second_order_state_update = state_update₂
    𝓂.caches.outdated.second_order_solution = false
end
```

### Lazy Allocation

Compute constant values lazily on first use and store in the model struct cache. Subsequent calls must read from the cache.

---

## Macros

### `@model` and `@parameters`

User-facing macros use `begin...end` blocks:

```julia
@model RBC begin
    1 / c[0] = (β / c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
```

### `@stable` Wrapper

Wrap groups of functions in `@stable default_mode = "disable" begin...end` from DispatchDoctor:

```julia
@stable default_mode = "disable" begin

function calculate_first_order_solution(...)
    ...
end

function calculate_second_order_solution(...)
    ...
end

end # dispatch_doctor
```

### AST Manipulation

Use `postwalk`/`prewalk` from MacroTools for expression tree traversal in macro implementations:

```julia
postwalk(expr) do x
    if x isa Expr && x.head == :ref
        # transform variable references
    end
    return x
end
```

---

## Summary of Key Principles

1. **snake_case everywhere** — functions, variables, most struct names
2. **Unicode for mathematics** — match the notation from the underlying papers
3. **`import` over `using`** — keep the namespace clean
4. **Explicit types** — annotate struct fields, return types, and `where` clauses
5. **Pre-allocate workspaces** — avoid allocations in hot loops
6. **Boolean solved flags** — return `(result, solved)` rather than throwing from solvers
7. **Verbose/silent kwargs** — let callers control output
8. **Shared docstring constants** — avoid repeating common parameter documentation
9. **No strict line limit** — readability over wrapping for mathematical code
10. **`@views`, `@inline`, `Ref`** — standard Julia performance patterns
