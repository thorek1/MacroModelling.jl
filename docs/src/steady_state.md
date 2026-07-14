# Steady State

The non stochastic steady state (NSSS) of a DSGE model is the equilibrium point where all variables remain constant over time (in the absence of shocks). Computing the NSSS is a crucial first step before solving the model using perturbation methods.

`MacroModelling.jl` offers an automated way to solving for the NSSS, along with the flexibility to define custom steady state functions when needed.

## Automatic Steady State Solver

The algorithm proceeds through several steps designed to maximize efficiency and robustness:

### Step 1: Eliminate Redundant Variables

The algorithm first identifies and eliminates redundant variables from the system. Variables that are necessary in dynamic equations but redundant in steady state are removed to simplify the problem (e.g. `c` is redundant in `1 / c = beta / c * k ^ alpha + (1 - delta)`).

### Step 2: Partition into Independent Blocks

The reduced system is partitioned into independent blocks that can be solved separately. This block decomposition exploits the sparsity structure of the model equations, allowing smaller subproblems to be solved in sequence rather than tackling the full system at once.

### Step 3: Attempt Symbolic Solution

For each block, the algorithm attempts a full or partial symbolic solution using computer algebra (using `sympy`). When possible, closed-form solutions are obtained, which:

- Provide exact solutions
- Enable faster computation

### Step 4: Create Auxiliary Variables for Domain-Constrained Terms

For terms with domain constraints (e.g., `log(x+y)`, `x^y`), auxiliary variables are created to handle these constraints explicitly. This transformation helps numerical solvers to find solutions while ensuring that numerical solutions respect domain requirements.

### Step 5: Custom Nonlinear Equations Solver

For blocks that cannot be solved symbolically, a custom system of nonlinear equations solver is employed. This solver is a Levenberg-Marquardt (LM) type algorithm with line-search that includes:

- **Box constraints**: Respect the domain constraints of variables as well as user defined bounds from the `@parameters` macro (e.g. `c > 0`, `r < 0.2`, or `1 < π < 1.1`). User defined bounds can be helpful to guide the solver toward plausible values.
- **Domain transformation**: A hyperbolic sine transformation is applied that transforms the geometry of the problem and increases the likelihood of finding a solution
- **Adaptive failure recovery**: Upon failure, the solver optimizes over the LM parameters and starting points to find a solution

### Step 6: Select Optimal Solver Parameters

The algorithm selects solver parameters and starting points that maximise speed for the specific model structure. This adaptive approach ensures efficient computation across diverse model specifications.

### Guiding and Validating the Internal Solver

Additional information can guide the automatic solver toward convergence and validate the result:

- Supply starting values via the `guess` argument of `@parameters`, e.g. `@parameters RBC guess = Dict(:k => 3.0, :c => 1.0) begin ... end`.
- Add bounds directly in the `@parameters` block using inequalities (e.g. `c > 0`, `r < 0.2`, or `1 < π < 1.1`) to restrict the search space and steer the solver toward plausible values.
- After solving, verify that the steady state satisfies all equations by calling `check_residuals`, for example:

  ```julia
  ss = get_steady_state(RBC)
  check_residuals(RBC, ss)
  ```
  
  which returns steady-state equation residuals in absolute value.

## Balanced Growth Path (Models in Levels)

Semistructural and policy models (IRIS / QPM style) are often written directly in **levels**, with non-stationary I(1) variables that share a common balanced growth path (BGP), rather than being hand-stationarized into gaps and growth rates. `MacroModelling.jl` supports such models for first-order analysis: the balanced growth path is **detected automatically**, so fully stationary models are unaffected and require no changes.

When a model contains a trending (unit-root) variable, the steady state is computed by giving every variable both a **level** and an additive per-period **growth** `xᴳ`, and evaluating each steady-state equation at two time origins. This pins the growth rates and any cointegration relationships automatically, while the absolute level of each independent stochastic trend is a free initial condition that is anchored to a particular solution (`0`). For a linear model the dynamics and impulse responses are invariant to this anchor — only the growth rates (slopes) are determined by the model.

As a minimal example, consider a random walk with drift `g` and its first difference `dx`:

```@repl ss_bgp
using MacroModelling

@model RWdrift begin
    x[0]  = x[-1] + g + e[x]
    dx[0] = x[0] - x[-1]
end

@parameters RWdrift begin
    g = 0.02
end

get_steady_state(RWdrift, derivatives = false)
```

The stationary first difference `dx` equals the drift `g`, while the trending level `x` is anchored to `0`. The solved growth rates also drive the level paths: with `levels = true`, impulse responses and simulations of trending variables follow `x_t = anchor + xᴳ · t`, so the level accumulates the deterministic drift while deviations (`levels = false`) do not:

```@repl ss_bgp
get_irf(RWdrift, shocks = :none, periods = 4, levels = true)
```

By default the absolute level of each independent trend is anchored to `0`. To pin a trend to a specific level instead — for example to match a reference steady state or to express a constant trend — add a **steady-state level anchor** `x[ss] = expr` as an equation in the `@model` block (analogous to an IRIS `!!` steady-state override). It constrains the level only; the growth remains determined by the variable's dynamic law:

```@repl ss_bgp
@model RWanchored begin
    x[0]  = x[-1] + g + e[x]
    dx[0] = x[0] - x[-1]
    x[ss] = xbar
end

@parameters RWanchored begin
    g    = 0.02
    xbar = 5.0
end

get_steady_state(RWanchored, derivatives = false)
```

The anchor expression is typically a parameter or constant. Anchors should target the model's independent (fundamental) trends; for a linear model the dynamics are unaffected by the choice of anchor.

Notes and current scope:

- Growth is **additive**. BGP models support first-, second-, and third-order
  solution and moment calculations, including models with forward-looking
  expectations.
- Cointegrated systems are handled automatically: growth-rate identities such as `xᴳ = aᴳ - bᴳ` follow from the level relationships without any additional equations.
- BGP steady-state output has `Steady_state` and `Growth_rate` columns before
  parameter derivatives. Growth rates are the solved per-period additive
  changes; growth-rate derivatives are not included.
- Dynamic models with calibrated persistence parameters on or outside the unit
  circle are also recognized after `@parameters`, including models with
  forward-looking expectations.
- Moment outputs use the stationary first-difference basis for trending
  variables. Such variables are labeled `Delta_x` (for `x`), so covariance,
  variance, standard-deviation, and correlation outputs are finite while
  stationary variables retain their usual labels.

## Custom Steady State Functions

For models where the internal solver fails, or when analytical solutions are available (often faster to compute), a custom steady state function can be provided. There are two primary ways to specify this:

### Method 1: Via the `@parameters` Macro

After defining the model one can specify a custom steady state function and pass it on to the `@parameters` macro. The function should accept a vector of parameter values and return a vector of variables followed by calibration parameters. The input and output needs to follow the correct ordering of parameters and variables. The order of the parameters can be obtained using `get_parameters(m)`, and the order of the output follows `get_variables(m)` and `get_calibrated_parameters(m)`. Practically, one can call the model and parameter macros without defining the custom steady state function, then get the order from the above functions calls and based on this order define the custom steady state function.

```@repl ss
using MacroModelling

# Define the model
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end

# Define a steady state function
function my_ss(parameters)
    # parameters is ordered as: get_parameters(m) (e.g., [:σᶻ, :ρᶻ, :δ, :α, :β])
    σᶻ, ρᶻ, δ, α, β = parameters

    # Compute steady state values
    k_ss = ((1/β - 1 + δ) / α)^(1/(α-1))
    q_ss = k_ss^α
    c_ss = q_ss - δ*k_ss
    z_ss = 0.0
    
    # Return values in variable order: get_variables(m) (e.g., [:c, :k, :q, :z])
    return [c_ss, k_ss, q_ss, z_ss]
end

@parameters RBC steady_state_function = my_ss begin
    σᶻ= 0.01
    ρᶻ= 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end
```

The model can now be used as usual, and the custom steady state function will be called automatically:

```@repl ss
ss = get_steady_state(RBC)
```

One can also use a non-allocating version of the steady state function that modifies a pre-allocated output vector in place. This function should accept two arguments: an output vector and a vector of parameter values. The output vector should be modified in place to contain the steady state values.

```@repl ss
# Define the model
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end

# Define a steady state function
function my_ss_inplace!(ss, parameters)
    # parameters is ordered as: get_parameters(m) (e.g., [:σᶻ, :ρᶻ, :δ, :α, :β])
    σᶻ, ρᶻ, δ, α, β = parameters
    # Compute steady state values
    k_ss = ((1/β - 1 + δ) / α)^(1/(α-1))
    q_ss = k_ss^α
    c_ss = q_ss - δ*k_ss
    z_ss = 0.0

    ss[1] = c_ss
    ss[2] = k_ss
    ss[3] = q_ss
    ss[4] = z_ss
end


@parameters RBC steady_state_function = my_ss_inplace! begin
    σᶻ= 0.01
    ρᶻ= 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_SS(RBC)  # uses the in-place version
```

### Method 2: Via Function Arguments

All functions that accept a `parameters` argument also accept a `steady_state_function` argument:

```@repl ss
# Pass the steady state function to specific function calls
get_irf(RBC, steady_state_function = my_ss)
```

To revert to the internal solver and clear any previously set custom function on the model, pass `nothing`:

```@repl ss
get_std(RBC, steady_state_function = nothing)
```

## When to Use Custom Steady State Functions

Consider a custom steady state function when:

1. **Analytical solution available**: Analytical solutions are more accurate and faster than numerical solutions
2. **Internal solver struggles**: Complex models may have multiple equilibria or convergence issues
3. **Performance is critical**: For estimation with many likelihood evaluations, custom functions can speed up computation
4. **Debugging**: To verify that model equations are correct by comparing against known solutions

The internal solver is robust and works well for most models, so start with the automatic solver and only switch to a custom function if needed.

## Delayed Parameter Declaration

There are cases when one does not want to define all parameter values at the time of model definition. In such cases, one can define a model without parameters (as otherwise defined in the parameter macro) and add them in subsequent function call instead. This is particularly useful if one wants to use parameters from a file, database, or estimation routine. In such cases, one can define the model as follows:

```@repl ss
@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end
```

Then, one can run the parameter macro without specifying parameter values:

```@repl ss
@parameters RBC begin
end
```

Later, one can define the parameter values when needed. For example, to get the steady state one can define the parameter values as a `Dict`:

```@repl ss
ss = get_steady_state(RBC, parameters = Dict(:α => 0.5, :β => 0.95, :δ => 0.02, :ρᶻ => 0.2, :σᶻ => 0.01))
```

The user has the full flexibility to define the parameter values in any way they see fit, and integrate it into their workflow.
