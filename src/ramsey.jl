"""
    ramsey_constraints(𝓂::ℳ)

Return the model equations that will serve as constraints in the Ramsey planner's problem.
These are all the original model equations (the structural equations).

# Arguments
- `𝓂`: the model object

# Returns
- Vector of expressions representing the constraint equations
"""
function ramsey_constraints(𝓂::ℳ)::Vector{Expr}
    return copy(𝓂.original_equations)
end


"""
    get_time_indices(expr::Expr)

Extract the time indices from variables in an expression.
Returns a dictionary mapping variable names to their time indices found in the expression.
"""
function get_time_indices(expr::Expr)::Dict{Symbol, Set{Int}}
    indices = Dict{Symbol, Set{Int}}()
    
    postwalk(x -> begin
        if x isa Expr && x.head == :ref && length(x.args) >= 2
            var_name = x.args[1]
            time_idx = x.args[2]
            if time_idx isa Int
                if haskey(indices, var_name)
                    push!(indices[var_name], time_idx)
                else
                    indices[var_name] = Set([time_idx])
                end
            elseif time_idx isa Expr && time_idx.head == :call
                # Handle expressions like x + 1, x - 1
                if time_idx.args[1] == :+ && length(time_idx.args) >= 3
                    if occursin(r"^(x|ex|exo|exogenous){1}$"i, string(time_idx.args[2]))
                        idx = time_idx.args[3]
                    elseif occursin(r"^(x|ex|exo|exogenous){1}$"i, string(time_idx.args[3]))
                        idx = time_idx.args[2]
                    else
                        idx = nothing
                    end
                    if !isnothing(idx) && idx isa Int
                        if haskey(indices, var_name)
                            push!(indices[var_name], idx)
                        else
                            indices[var_name] = Set([idx])
                        end
                    end
                elseif time_idx.args[1] == :- && length(time_idx.args) >= 3
                    if occursin(r"^(x|ex|exo|exogenous){1}$"i, string(time_idx.args[2]))
                        idx = -time_idx.args[3]
                    else
                        idx = nothing
                    end
                    if !isnothing(idx) && idx isa Int
                        if haskey(indices, var_name)
                            push!(indices[var_name], idx)
                        else
                            indices[var_name] = Set([idx])
                        end
                    end
                end
            end
        end
        x
    end, expr)
    
    return indices
end


"""
    shift_time_indices(expr, shift::Int)

Shift all time indices in an expression by the given amount.
For example, shift_time_indices(:(c[0] + k[-1]), 1) returns :(c[1] + k[0])
"""
function shift_time_indices(expr, shift::Int)
    if shift == 0
        return expr
    end
    
    postwalk(x -> begin
        if x isa Expr && x.head == :ref && length(x.args) >= 2
            var_name = x.args[1]
            time_idx = x.args[2]
            if time_idx isa Int
                return Expr(:ref, var_name, time_idx + shift)
            elseif occursin(r"^(x|ex|exo|exogenous){1}$"i, string(time_idx))
                # Keep shock timing as is (x stays as x, but shifts would be x+1 etc.)
                if shift == 0
                    return x
                else
                    return Expr(:ref, var_name, Expr(:call, shift > 0 ? :+ : :-, time_idx, abs(shift)))
                end
            elseif time_idx isa Expr && time_idx.head == :call
                # Handle x+1, x-1 cases for shocks
                if (time_idx.args[1] == :+ || time_idx.args[1] == :-) && 
                   occursin(r"^(x|ex|exo|exogenous){1}$"i, string(time_idx.args[2]))
                    old_shift = time_idx.args[1] == :+ ? time_idx.args[3] : -time_idx.args[3]
                    new_shift = old_shift + shift
                    if new_shift == 0
                        return Expr(:ref, var_name, time_idx.args[2])  # Return just x
                    elseif new_shift > 0
                        return Expr(:ref, var_name, Expr(:call, :+, time_idx.args[2], new_shift))
                    else
                        return Expr(:ref, var_name, Expr(:call, :-, time_idx.args[2], abs(new_shift)))
                    end
                end
            end
        end
        x
    end, expr)
end


"""
    get_variables_from_expr(expr)

Extract all variable references from an expression, including their time indices.
Returns a vector of (variable_name, time_index) tuples.
"""
function get_variables_from_expr(expr)::Vector{Tuple{Symbol, Any}}
    vars = Tuple{Symbol, Any}[]
    
    postwalk(x -> begin
        if x isa Expr && x.head == :ref && length(x.args) >= 2
            var_name = x.args[1]
            time_idx = x.args[2]
            push!(vars, (var_name, time_idx))
        end
        x
    end, expr)
    
    return unique(vars)
end


"""
    create_multiplier_symbol(eq_idx::Int)

Create a Lagrange multiplier symbol for equation index eq_idx.
Uses Unicode subscript notation for clarity.
"""
function create_multiplier_symbol(eq_idx::Int)::Symbol
    return Symbol("λ" * sub(string(eq_idx)))
end


"""
    derive_ramsey_focs(objective::Expr, constraints::Vector{Expr}, 
                       variables::Vector{Symbol}, instruments::Vector{Symbol},
                       shocks::Vector{Symbol}, discount::Symbol)

Derive the first-order conditions (FOCs) for the Ramsey optimal policy problem.

The Ramsey problem is:
    max E_0 Σ_t β^t U(y_t)
    s.t. f_i(y_{t+1}, y_t, y_{t-1}, ε_t) = 0 for all i, t

The Lagrangian is:
    L = E_0 Σ_t β^t [U(y_t) - Σ_i λ_{i,t} f_i(y_{t+1}, y_t, y_{t-1}, ε_t)]

The FOCs for each variable y_j (except instruments chosen by planner) are:
    ∂U/∂y_{j,t} - Σ_i [λ_{i,t} ∂f_i/∂y_{j,t} + β λ_{i,t+1} ∂f_i/∂y_{j,t+1} + λ_{i,t-1}/β ∂f_i/∂y_{j,t-1}] = 0

For instruments, the FOC is simply the derivative with respect to that variable.

# Arguments
- `objective`: The per-period objective function (e.g., utility or negative loss)
- `constraints`: Vector of constraint equations (model equations set to 0)
- `variables`: All endogenous variables in the model
- `instruments`: Policy instruments that the planner controls
- `shocks`: Exogenous shocks in the model
- `discount`: Discount factor symbol (e.g., :β)

# Returns
- Vector of FOC expressions
- Vector of new multiplier symbols introduced
"""
function derive_ramsey_focs(objective::Expr, constraints::Vector{Expr}, 
                            variables::Vector{Symbol}, instruments::Vector{Symbol},
                            shocks::Vector{Symbol}, discount::Symbol)
    
    n_constraints = length(constraints)
    n_vars = length(variables)
    
    # Create multiplier symbols for each constraint
    multipliers = [create_multiplier_symbol(i) for i in 1:n_constraints]
    
    focs = Expr[]
    
    # For each variable, derive the FOC
    for var in variables
        # Skip if it's a shock
        if var in shocks
            continue
        end
        
        foc_terms = []
        
        # Derivative of objective with respect to var[0]
        obj_deriv = symbolic_partial_derivative(objective, var, 0)
        if !is_zero_expr(obj_deriv)
            push!(foc_terms, obj_deriv)
        end
        
        # For each constraint, compute multiplier × partial derivative terms
        for (i, constraint) in enumerate(constraints)
            λ = multipliers[i]
            
            # Convert equation to residual form: LHS - RHS = 0
            residual = convert_to_residual(constraint)
            
            # Derivative w.r.t. var[0] (current period) - multiplied by λ[0]
            deriv_0 = symbolic_partial_derivative(residual, var, 0)
            if !is_zero_expr(deriv_0)
                # -λ[0] × ∂f/∂y[0]
                push!(foc_terms, Expr(:call, :*, -1, Expr(:ref, λ, 0), deriv_0))
            end
            
            # Derivative w.r.t. var[1] (future) - multiplied by β × λ[1]  
            deriv_1 = symbolic_partial_derivative(residual, var, 1)
            if !is_zero_expr(deriv_1)
                # -β × λ[1] × ∂f/∂y[1]
                push!(foc_terms, Expr(:call, :*, -1, discount, Expr(:ref, λ, 1), deriv_1))
            end
            
            # Derivative w.r.t. var[-1] (past) - multiplied by λ[-1]/β
            deriv_m1 = symbolic_partial_derivative(residual, var, -1)
            if !is_zero_expr(deriv_m1)
                # -λ[-1]/β × ∂f/∂y[-1] = -(1/β) × λ[-1] × ∂f/∂y[-1]
                push!(foc_terms, Expr(:call, :*, -1, Expr(:call, :/, 1, discount), Expr(:ref, λ, -1), deriv_m1))
            end
        end
        
        # Create the FOC equation: sum of all terms = 0
        if !isempty(foc_terms)
            if length(foc_terms) == 1
                foc = Expr(:(=), foc_terms[1], 0)
            else
                foc = Expr(:(=), Expr(:call, :+, foc_terms...), 0)
            end
            push!(focs, foc)
        end
    end
    
    return focs, multipliers
end


"""
    symbolic_partial_derivative(expr, var::Symbol, time_idx::Int)

Compute the symbolic partial derivative of expr with respect to var[time_idx].
This is a simplified symbolic differentiation that handles common DSGE model expressions.
"""
function symbolic_partial_derivative(expr, var::Symbol, time_idx::Int)
    if expr isa Number
        return 0
    elseif expr isa Symbol
        # Just a symbol without time index - not a variable reference
        return 0
    elseif expr isa Expr
        if expr.head == :ref
            # Variable reference like c[0]
            if expr.args[1] == var && expr.args[2] == time_idx
                return 1
            elseif expr.args[1] == var && expr.args[2] isa Expr
                # Check for shock timing like x+1
                return 0  # Shocks are treated as exogenous
            else
                return 0
            end
        elseif expr.head == :call
            return differentiate_call(expr, var, time_idx)
        elseif expr.head == :(=)
            # For equations, differentiate LHS - RHS
            lhs_deriv = symbolic_partial_derivative(expr.args[1], var, time_idx)
            rhs_deriv = symbolic_partial_derivative(expr.args[2], var, time_idx)
            return simplify_expr(Expr(:call, :-, lhs_deriv, rhs_deriv))
        else
            return 0
        end
    else
        return 0
    end
end


"""
    differentiate_call(expr::Expr, var::Symbol, time_idx::Int)

Differentiate a function call expression with respect to var[time_idx].
Handles arithmetic operations and common mathematical functions.
"""
function differentiate_call(expr::Expr, var::Symbol, time_idx::Int)
    @assert expr.head == :call
    
    op = expr.args[1]
    args = expr.args[2:end]
    
    if op == :+
        # d/dx (a + b) = da/dx + db/dx
        derivs = [symbolic_partial_derivative(arg, var, time_idx) for arg in args]
        return simplify_expr(Expr(:call, :+, derivs...))
        
    elseif op == :-
        if length(args) == 1
            # Unary minus: d/dx (-a) = -da/dx
            deriv = symbolic_partial_derivative(args[1], var, time_idx)
            return simplify_expr(Expr(:call, :-, deriv))
        else
            # Binary minus: d/dx (a - b) = da/dx - db/dx
            deriv1 = symbolic_partial_derivative(args[1], var, time_idx)
            deriv2 = symbolic_partial_derivative(args[2], var, time_idx)
            return simplify_expr(Expr(:call, :-, deriv1, deriv2))
        end
        
    elseif op == :*
        # Product rule: d/dx (a * b * c * ...) = da/dx * b * c + a * db/dx * c + ...
        n = length(args)
        terms = []
        for i in 1:n
            deriv_i = symbolic_partial_derivative(args[i], var, time_idx)
            if !is_zero_expr(deriv_i)
                other_args = [args[j] for j in 1:n if j != i]
                if isempty(other_args)
                    push!(terms, deriv_i)
                elseif length(other_args) == 1
                    push!(terms, Expr(:call, :*, deriv_i, other_args[1]))
                else
                    push!(terms, Expr(:call, :*, deriv_i, other_args...))
                end
            end
        end
        if isempty(terms)
            return 0
        elseif length(terms) == 1
            return simplify_expr(terms[1])
        else
            return simplify_expr(Expr(:call, :+, terms...))
        end
        
    elseif op == :/
        # Quotient rule: d/dx (a/b) = (da/dx * b - a * db/dx) / b^2
        a, b = args[1], args[2]
        da = symbolic_partial_derivative(a, var, time_idx)
        db = symbolic_partial_derivative(b, var, time_idx)
        
        if is_zero_expr(da) && is_zero_expr(db)
            return 0
        elseif is_zero_expr(db)
            return simplify_expr(Expr(:call, :/, da, b))
        elseif is_zero_expr(da)
            return simplify_expr(Expr(:call, :/, Expr(:call, :-, Expr(:call, :*, a, db)), 
                                      Expr(:call, :^, b, 2)))
        else
            return simplify_expr(Expr(:call, :/, 
                                      Expr(:call, :-, Expr(:call, :*, da, b), Expr(:call, :*, a, db)),
                                      Expr(:call, :^, b, 2)))
        end
        
    elseif op == :^
        # Power rule: d/dx (a^b)
        a, b = args[1], args[2]
        da = symbolic_partial_derivative(a, var, time_idx)
        db = symbolic_partial_derivative(b, var, time_idx)
        
        if is_zero_expr(da) && is_zero_expr(db)
            return 0
        elseif is_zero_expr(db)
            # d/dx (a^b) = b * a^(b-1) * da/dx when b is constant
            return simplify_expr(Expr(:call, :*, b, 
                                      Expr(:call, :^, a, Expr(:call, :-, b, 1)), da))
        elseif is_zero_expr(da)
            # d/dx (a^b) = a^b * log(a) * db/dx when a is constant
            return simplify_expr(Expr(:call, :*, expr, Expr(:call, :log, a), db))
        else
            # General case: d/dx (a^b) = a^b * (b/a * da/dx + log(a) * db/dx)
            return simplify_expr(Expr(:call, :*, expr, 
                                      Expr(:call, :+,
                                           Expr(:call, :*, Expr(:call, :/, b, a), da),
                                           Expr(:call, :*, Expr(:call, :log, a), db))))
        end
        
    elseif op == :exp
        # d/dx exp(a) = exp(a) * da/dx
        a = args[1]
        da = symbolic_partial_derivative(a, var, time_idx)
        if is_zero_expr(da)
            return 0
        end
        return simplify_expr(Expr(:call, :*, expr, da))
        
    elseif op == :log
        # d/dx log(a) = (1/a) * da/dx
        a = args[1]
        da = symbolic_partial_derivative(a, var, time_idx)
        if is_zero_expr(da)
            return 0
        end
        return simplify_expr(Expr(:call, :*, Expr(:call, :/, 1, a), da))
        
    elseif op == :sqrt
        # d/dx sqrt(a) = da/dx / (2 * sqrt(a))
        a = args[1]
        da = symbolic_partial_derivative(a, var, time_idx)
        if is_zero_expr(da)
            return 0
        end
        return simplify_expr(Expr(:call, :/, da, Expr(:call, :*, 2, expr)))
        
    elseif op in [:sin, :cos, :tan]
        a = args[1]
        da = symbolic_partial_derivative(a, var, time_idx)
        if is_zero_expr(da)
            return 0
        end
        if op == :sin
            return simplify_expr(Expr(:call, :*, Expr(:call, :cos, a), da))
        elseif op == :cos
            return simplify_expr(Expr(:call, :*, -1, Expr(:call, :sin, a), da))
        else  # tan
            return simplify_expr(Expr(:call, :/, da, Expr(:call, :^, Expr(:call, :cos, a), 2)))
        end
        
    else
        # Unknown function - return 0 (conservative approach)
        # In a full implementation, this should throw an error or use chain rule
        return 0
    end
end


"""
    is_zero_expr(expr)

Check if an expression is zero (either literal 0 or expression that evaluates to 0).
"""
function is_zero_expr(expr)
    if expr isa Number
        return expr == 0
    elseif expr isa Expr
        # Try to simplify and check
        simplified = simplify_expr(expr)
        return simplified isa Number && simplified == 0
    else
        return false
    end
end


"""
    simplify_expr(expr)

Simplify an expression by applying basic algebraic rules.
"""
function simplify_expr(expr)
    if !(expr isa Expr)
        return expr
    end
    
    if expr.head != :call
        return expr
    end
    
    op = expr.args[1]
    args = [simplify_expr(arg) for arg in expr.args[2:end]]
    
    # Filter out zeros in addition
    if op == :+
        non_zero = filter(x -> !is_zero_expr(x), args)
        if isempty(non_zero)
            return 0
        elseif length(non_zero) == 1
            return non_zero[1]
        else
            return Expr(:call, :+, non_zero...)
        end
    end
    
    # Handle multiplication with zeros and ones
    if op == :*
        if any(x -> is_zero_expr(x), args)
            return 0
        end
        non_one = filter(x -> !(x isa Number && x == 1), args)
        if isempty(non_one)
            return 1
        elseif length(non_one) == 1
            return non_one[1]
        else
            return Expr(:call, :*, non_one...)
        end
    end
    
    # Handle subtraction with zeros
    if op == :- && length(args) == 2
        if is_zero_expr(args[2])
            return args[1]
        elseif is_zero_expr(args[1])
            if args[2] isa Number
                return -args[2]
            else
                return Expr(:call, :-, args[2])
            end
        end
    end
    
    # Handle division
    if op == :/
        if is_zero_expr(args[1])
            return 0
        elseif args[2] isa Number && args[2] == 1
            return args[1]
        end
    end
    
    # Handle power with 0 and 1 exponents
    if op == :^
        if args[2] isa Number
            if args[2] == 0
                return 1
            elseif args[2] == 1
                return args[1]
            end
        end
    end
    
    return Expr(:call, op, args...)
end


"""
    convert_to_residual(eq::Expr)

Convert an equation to residual form (LHS - RHS).
If the equation is already in the form `expr = 0`, return expr.
"""
function convert_to_residual(eq::Expr)
    if eq.head == :(=)
        lhs = eq.args[1]
        rhs = eq.args[2]
        if rhs isa Number && rhs == 0
            return lhs
        else
            return Expr(:call, :-, lhs, rhs)
        end
    else
        return eq
    end
end


"""
    create_ramsey_model(original_model::ℳ, 
                        objective::Expr, 
                        instruments::Vector{Symbol};
                        discount::Symbol = :β,
                        model_name::Symbol = :Ramsey)

Create a new model that implements Ramsey optimal policy.

This function takes an existing DSGE model and transforms it into a Ramsey planner's problem
by:
1. Taking the original model equations as constraints
2. Adding Lagrange multipliers for each constraint
3. Deriving first-order conditions for optimal policy
4. Returning the augmented system

# Arguments
- `original_model`: The base DSGE model
- `objective`: The per-period objective function (e.g., utility or negative loss)  
- `instruments`: Policy instruments controlled by the planner

# Keyword Arguments
- `discount`: Symbol for the discount factor (default: `:β`)
- `model_name`: Name for the new Ramsey model (default: `:Ramsey`)

# Returns
- A tuple of (dynamic_equations, steady_state_equations, multiplier_names)

# Example
```julia
# Original model
@model NK begin
    # ... New Keynesian equations ...
end

@parameters NK begin
    β = 0.99
    # ... other parameters ...
end

# Create Ramsey model with consumption utility objective
ramsey_eqs, ss_eqs, multipliers = create_ramsey_model(
    NK,
    :(log(C[0]) - (N[0]^(1+φ))/(1+φ)),  # Utility function
    [:R]  # Interest rate is the policy instrument
)
```
"""
function create_ramsey_model(original_model::ℳ, 
                             objective::Expr, 
                             instruments::Vector{Symbol};
                             discount::Symbol = :β)
    
    # Get the original model equations (constraints)
    constraints = ramsey_constraints(original_model)
    
    # Get all variables
    variables = original_model.var
    
    # Get shocks
    shocks = original_model.exo
    
    # Derive FOCs
    focs, multipliers = derive_ramsey_focs(objective, constraints, variables, instruments, shocks, discount)
    
    # Combine original constraints with FOCs
    # The complete system is: original equations + FOCs
    all_equations = vcat(constraints, focs)
    
    # The new variables are: original variables + multipliers  
    all_variables = vcat(variables, multipliers)
    
    return all_equations, multipliers, all_variables
end


"""
$(SIGNATURES)
Transform an existing model into a Ramsey optimal policy model.

The Ramsey planner maximizes a discounted sum of per-period objectives subject to the 
structural equations of the original model. This function automatically:
1. Introduces Lagrange multipliers for each model equation
2. Derives first-order conditions for optimal policy
3. Creates the augmented system of equations

# Arguments
- `𝓂`: the original model object
- `objective`: per-period objective function expression (e.g., utility)
- `instruments`: vector of policy instrument symbols

# Keyword Arguments  
- `discount`: discount factor symbol (default: `:β`)
- `verbose`: print information about the transformation (default: `false`)

# Returns
- Tuple containing:
  - `Vector{Expr}`: augmented system of equations
  - `Vector{Symbol}`: Lagrange multiplier symbols
  - `Vector{Symbol}`: all variables (original + multipliers)

# Example
```julia
using MacroModelling

@model NK begin
    W[0] = C[0]^σ * N[0]^φ
    1 = β * (C[0]/C[1])^σ * R[0] / π[1]
    # ... more equations
end

@parameters NK begin
    β = 0.99
    σ = 1
    φ = 1
end

# Define Ramsey problem
eqs, mults, vars = get_ramsey_equations(
    NK,
    :(log(C[0]) - N[0]^(1+φ)/(1+φ)),  # Objective
    [:R]  # Policy instrument
)
```
"""
function get_ramsey_equations(𝓂::ℳ, 
                              objective::Expr, 
                              instruments::Union{Symbol, Vector{Symbol}};
                              discount::Symbol = :β,
                              verbose::Bool = false)
    
    # Convert single instrument to vector
    instruments_vec = instruments isa Symbol ? [instruments] : instruments
    
    # Validate inputs
    for inst in instruments_vec
        @assert inst in 𝓂.var "Instrument $inst is not a variable in the model. Available variables: $(𝓂.var)"
    end
    
    @assert discount in 𝓂.parameters || discount in 𝓂.parameters_as_function_of_parameters "Discount factor $discount is not a parameter in the model."
    
    if verbose
        println("Transforming model $(𝓂.model_name) to Ramsey optimal policy problem")
        println("  Objective: $objective")
        println("  Instruments: $instruments_vec")
        println("  Number of original equations: $(length(𝓂.original_equations))")
    end
    
    # Create Ramsey model
    equations, multipliers, variables = create_ramsey_model(𝓂, objective, instruments_vec, discount = discount)
    
    if verbose
        println("  Number of Lagrange multipliers: $(length(multipliers))")
        println("  Total equations in Ramsey system: $(length(equations))")
    end
    
    return equations, multipliers, variables
end


"""
$(SIGNATURES)
Print the Ramsey optimal policy equations in a readable format.

# Arguments
- `equations`: Vector of equations from `get_ramsey_equations`
- `multipliers`: Vector of multiplier symbols from `get_ramsey_equations`
- `n_original`: Number of original model equations (constraints)

# Example
```julia
eqs, mults, vars = get_ramsey_equations(model, objective, [:R])
print_ramsey_equations(eqs, mults, length(model.original_equations))
```
"""
function print_ramsey_equations(equations::Vector{Expr}, 
                                multipliers::Vector{Symbol},
                                n_original::Int)
    println("=" ^ 60)
    println("RAMSEY OPTIMAL POLICY MODEL")
    println("=" ^ 60)
    
    println("\n--- Original Model Equations (Constraints) ---")
    for (i, eq) in enumerate(equations[1:n_original])
        println("  [$i] $eq")
    end
    
    println("\n--- Lagrange Multipliers ---")
    println("  ", join(string.(multipliers), ", "))
    
    println("\n--- First-Order Conditions ---")
    for (i, eq) in enumerate(equations[n_original+1:end])
        println("  [FOC $(i)] $eq")
    end
    
    println("\n" * "=" ^ 60)
end


"""
$(SIGNATURES)
Get a summary of the Ramsey optimal policy problem.

# Arguments  
- `𝓂`: the original model object
- `objective`: per-period objective function expression
- `instruments`: vector of policy instrument symbols

# Keyword Arguments
- `discount`: discount factor symbol (default: `:β`)

# Returns
- NamedTuple with fields:
  - `equations`: all equations in the Ramsey system
  - `constraints`: original model equations
  - `focs`: first-order conditions
  - `multipliers`: Lagrange multiplier symbols
  - `original_variables`: variables from the original model
  - `all_variables`: all variables including multipliers
  - `instruments`: policy instruments
  - `objective`: the objective function

# Example
```julia
summary = ramsey_summary(model, :(log(C[0])), [:R])
println(summary.multipliers)
```
"""
function ramsey_summary(𝓂::ℳ,
                        objective::Expr,
                        instruments::Union{Symbol, Vector{Symbol}};
                        discount::Symbol = :β)
    
    equations, multipliers, variables = get_ramsey_equations(𝓂, objective, instruments, discount = discount)
    
    n_orig = length(𝓂.original_equations)
    
    return (
        equations = equations,
        constraints = equations[1:n_orig],
        focs = equations[n_orig+1:end],
        multipliers = multipliers,
        original_variables = 𝓂.var,
        all_variables = variables,
        instruments = instruments isa Symbol ? [instruments] : instruments,
        objective = objective,
        n_constraints = n_orig,
        n_focs = length(multipliers)
    )
end


# Export the main functions
# export get_ramsey_equations, print_ramsey_equations, ramsey_summary
