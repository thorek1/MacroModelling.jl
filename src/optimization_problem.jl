# Optimization Problem Parser
# This module provides functionality to derive FOCs from optimization problems
# similar to gEcon syntax
#
# Main function: derive_focs
#
# Usage:
# ------
# Instead of writing out first-order conditions (FOCs) manually, users can specify
# their model as an optimization problem and let this module derive the FOCs automatically.
#
# Example: Consumer maximizing utility subject to budget and capital accumulation constraints
#
#   controls = [:K_s, :C, :L_s, :I]
#   objective = :(U[0] = log(C[0]) + ψ * log(1 - L_s[0]) + β * U[1])
#   constraints = [
#       :(I[0] + C[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]),
#       :(K_s[0] = (1 - δ) * K_s[-1] + I[0])
#   ]
#
#   focs, multipliers = derive_focs(
#       controls = controls,
#       objective = objective,
#       constraints = constraints,
#       discount_factor = :β,
#       block_name = "consumer"
#   )
#
# This will derive the Euler equation and labor-leisure tradeoff conditions automatically.
#
# See models/gEcon_RBC_example.jl for a complete example.

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for variable manipulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    opt_get_time_index(var_expr)

Extract the time index from a variable expression like `C[0]`, `K[-1]`, etc.
Returns the time index as an integer or symbol.
"""
function opt_get_time_index(var_expr::Expr)
    if var_expr.head == :ref
        return var_expr.args[2]
    end
    error("Expected reference expression like C[0]")
end

"""
    opt_get_var_name(var_expr)

Extract the variable name from a variable expression like `C[0]`, `K[-1]`, etc.
"""
function opt_get_var_name(var_expr::Expr)
    if var_expr.head == :ref
        return var_expr.args[1]
    end
    error("Expected reference expression like C[0]")
end

"""
    opt_shift_all_time_indices(expr, shift)

Shift all time indices in an expression by a given amount.
"""
function opt_shift_all_time_indices(expr::Expr, shift::Int)
    return postwalk(x -> begin
        if x isa Expr && x.head == :ref
            var_name = x.args[1]
            time_idx = x.args[2]
            if time_idx isa Int
                return Expr(:ref, var_name, time_idx + shift)
            end
        end
        return x
    end, expr)
end

function opt_shift_all_time_indices(expr, shift::Int)
    return expr
end

"""
    opt_find_control_occurrences(expr, control)

Find all occurrences of a control variable in an expression with their time indices.
Returns a vector of time indices where the control appears.
"""
function opt_find_control_occurrences(expr, control::Symbol)
    occurrences = Int[]
    postwalk(x -> begin
        if x isa Expr && x.head == :ref && x.args[1] == control
            time_idx = x.args[2]
            if time_idx isa Int
                push!(occurrences, time_idx)
            end
        end
        return x
    end, expr)
    return unique(occurrences)
end

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic differentiation using SymPy
# ─────────────────────────────────────────────────────────────────────────────

"""
    opt_transform_vars_for_sympy(expr, var_mapping)

Transform variable references like K[0] to K_0 for SymPy compatibility.
"""
function opt_transform_vars_for_sympy(expr::Expr, var_mapping::Dict)
    return postwalk(x -> begin
        if x isa Expr && x.head == :ref
            var_name = x.args[1]
            time_idx = x.args[2]
            if time_idx isa Int
                new_sym = time_idx < 0 ? 
                    Symbol(string(var_name) * "__" * string(abs(time_idx))) :
                    Symbol(string(var_name) * "_" * string(time_idx))
                var_mapping[new_sym] = x
                return new_sym
            elseif time_idx isa Symbol && occursin(r"^(ss|stst|steady|steadystate|steady_state)$"i, string(time_idx))
                # Steady state reference
                new_sym = Symbol(string(var_name) * "_ss")
                var_mapping[new_sym] = x
                return new_sym
            end
        end
        return x
    end, expr)
end

function opt_transform_vars_for_sympy(expr, var_mapping::Dict)
    return expr
end

"""
    opt_untransform_vars_from_sympy(expr, var_mapping)

Transform SymPy variable names back to original notation.
"""
function opt_untransform_vars_from_sympy(expr::Expr, var_mapping::Dict)
    return postwalk(x -> begin
        if x isa Symbol && haskey(var_mapping, x)
            return var_mapping[x]
        end
        return x
    end, expr)
end

function opt_untransform_vars_from_sympy(expr::Symbol, var_mapping::Dict)
    return haskey(var_mapping, expr) ? var_mapping[expr] : expr
end

function opt_untransform_vars_from_sympy(expr, var_mapping::Dict)
    return expr
end

"""
    opt_symbolic_differentiate(expr, var_name, time_idx)

Symbolically differentiate an expression with respect to a variable at a specific time.
Uses SymPy for symbolic differentiation.
"""
function opt_symbolic_differentiate(expr::Expr, var_name::Symbol, time_idx::Int)
    # Transform variable references to unique symbols
    var_mapping = Dict{Symbol,Expr}()
    transformed_expr = opt_transform_vars_for_sympy(expr, var_mapping)
    
    # Create the target variable symbol
    target_sym = time_idx < 0 ? 
        Symbol(string(var_name) * "__" * string(abs(time_idx))) :
        Symbol(string(var_name) * "_" * string(time_idx))
    
    # Get all symbols in the expression using MacroModelling's get_symbols
    all_symbols = collect(get_symbols(transformed_expr))
    
    # Filter out non-symbol items (numbers, etc.) 
    # Note: Julia supports Unicode identifiers, so we accept any symbol that starts with a letter/underscore
    valid_symbols = filter(sym -> sym isa Symbol, all_symbols)
    
    # Create SymPy symbols
    for sym in valid_symbols
        sym_value = SPyPyC.symbols(string(sym), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($sym = $sym_value))
    end
    
    # Also ensure target is a symbol
    if target_sym ∉ valid_symbols
        sym_value = SPyPyC.symbols(string(target_sym), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($(target_sym) = $sym_value))
    end
    
    # Evaluate and differentiate
    sympy_expr = Core.eval(SymPyWorkspace, transformed_expr)
    target_sympy = Core.eval(SymPyWorkspace, target_sym)
    
    derivative = SPyPyC.diff(sympy_expr, target_sympy)
    
    # Convert back to Julia expression
    derivative_str = string(derivative)
    if derivative_str == "0"
        return 0
    end
    
    # Parse with error handling
    derivative_expr = try
        Meta.parse(derivative_str)
    catch e
        @warn "Failed to parse derivative expression: $derivative_str"
        return 0
    end
    
    # Transform back to original variable notation
    result = opt_untransform_vars_from_sympy(derivative_expr, var_mapping)
    
    return result
end

function opt_symbolic_differentiate(expr, var_name::Symbol, time_idx::Int)
    return 0  # Non-expression returns 0 derivative
end

# ─────────────────────────────────────────────────────────────────────────────
# Main FOC derivation function
# ─────────────────────────────────────────────────────────────────────────────

"""
    derive_focs(; definitions, controls, objective, constraints, discount_factor=:β, block_name="agent")

Derive first-order conditions (FOCs) from an optimization problem.

# Arguments
- `definitions::Dict{Symbol,Expr}`: Auxiliary definitions to substitute into equations
- `controls::Vector{Symbol}`: Control variables (optimize w.r.t. these)
- `objective::Expr`: Objective function (recursive Bellman form: U[0] = u[0] + β * E[U[1]])
- `constraints::Vector{Expr}`: Vector of constraints (as equations LHS = RHS)
- `discount_factor::Symbol`: Discount factor symbol (default: :β)
- `block_name::String`: Name prefix for Lagrange multipliers (default: "agent")

# Returns
- `focs::Vector{Expr}`: First-order condition equations
- `multipliers::Vector{Symbol}`: Lagrange multiplier symbols created

# Example
```julia
# Consumer problem from gEcon example
definitions = Dict(:u => :((C[0]^μ * (1 - L_s[0])^(1 - μ))^(1 - η) / (1 - η)))
controls = [:K_s, :C, :L_s, :I]
objective = :(U[0] = u[0] + β * U[1])
constraints = [
    :(I[0] + C[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]),
    :(K_s[0] = (1 - δ) * K_s[-1] + I[0])
]

focs, multipliers = derive_focs(
    definitions=definitions,
    controls=controls,
    objective=objective,
    constraints=constraints,
    discount_factor=:β,
    block_name="consumer"
)
```
"""
function derive_focs(; 
    definitions::Dict{Symbol,Expr} = Dict{Symbol,Expr}(),
    controls::Vector{Symbol},
    objective::Expr,
    constraints::Vector{Expr},
    discount_factor::Symbol = :β,
    block_name::String = "agent"
)
    focs = Expr[]
    multipliers = Symbol[]
    
    # Parse objective to extract the instantaneous utility and recursive structure
    # Expected form: U[0] = u[0] + β * E[U[1]] or U[0] = u[0] + β * U[1]
    # For static (firm) problem: π[0] = Y[0] - L_d[0] * W[0] - r[0] * K_d[0]
    
    is_recursive = contains_recursive_term(objective, discount_factor)
    
    # Substitute definitions into objective and constraints
    objective_substituted = substitute_opt_definitions(objective, definitions)
    constraints_substituted = [substitute_opt_definitions(c, definitions) for c in constraints]
    
    # Extract the instantaneous part of the objective
    # For U[0] = u[0] + β * U[1], extract u[0]
    # For π[0] = Y[0] - ..., the whole RHS is the objective to maximize
    if is_recursive
        instant_objective, obj_var = extract_instantaneous_objective(objective_substituted, discount_factor)
    else
        # Static problem - the RHS of the equation is the objective
        if objective_substituted.head == :(=)
            instant_objective = objective_substituted.args[2]
            obj_var = opt_get_var_name(objective_substituted.args[1])
        else
            instant_objective = objective_substituted
            obj_var = nothing
        end
    end
    
    # Create Lagrange multipliers for each constraint
    for (i, constraint) in enumerate(constraints_substituted)
        mult_sym = Symbol("λ_" * block_name * "_" * string(i))
        push!(multipliers, mult_sym)
    end
    
    # Build the Lagrangian (just for reference - we differentiate piece by piece)
    # L = instant_objective + Σ λ_i * (RHS_i - LHS_i)
    
    # For each control variable, derive the FOC
    for control in controls
        foc = derive_single_foc(
            instant_objective, 
            constraints_substituted, 
            multipliers,
            control, 
            discount_factor, 
            is_recursive
        )
        
        if foc !== nothing && foc != 0
            push!(focs, foc)
        end
    end
    
    # Add the constraint equations themselves (rearranged as LHS - RHS = 0 form for @model)
    # The constraints become part of the model equations
    
    return focs, multipliers
end

"""
    contains_recursive_term(expr, discount_factor)

Check if an expression contains a recursive term like β * U[1] or β * E[U[1]].
"""
function contains_recursive_term(expr, discount_factor::Symbol)
    found = false
    postwalk(x -> begin
        if x isa Expr && x.head == :call
            # Look for β * something[1] or discount_factor * something[1]
            if x.args[1] == :* && discount_factor in x.args
                # Check if any of the other args is a variable with [1] index
                for arg in x.args[2:end]
                    if arg isa Expr && arg.head == :ref && arg.args[2] == 1
                        found = true
                    end
                    # Also check for E[...] operator
                    if arg isa Expr && arg.head == :ref && arg.args[1] == :E
                        found = true
                    end
                end
            end
        end
        return x
    end, expr)
    return found
end

"""
    substitute_opt_definitions(expr, definitions)

Substitute definition symbols with their expressions.
Handles the case where definitions use [] notation (e.g., u[] or u[0]).
"""
function substitute_opt_definitions(expr::Expr, definitions::Dict{Symbol,Expr})
    return postwalk(x -> begin
        if x isa Expr && x.head == :ref
            var_name = x.args[1]
            if haskey(definitions, var_name)
                # Return the definition expression
                return definitions[var_name]
            end
        elseif x isa Symbol && haskey(definitions, x)
            return definitions[x]
        end
        return x
    end, expr)
end

function substitute_opt_definitions(expr, definitions::Dict{Symbol,Expr})
    if expr isa Symbol && haskey(definitions, expr)
        return definitions[expr]
    end
    return expr
end

"""
    extract_instantaneous_objective(objective, discount_factor)

Extract the instantaneous objective from a recursive objective.
E.g., from U[0] = u[0] + β * U[1], extract u[0].
E.g., from U[0] = log(C[0]) + ψ * log(1 - L_s[0]) + β * U[1], extract log(C[0]) + ψ * log(1 - L_s[0]).
Returns (instantaneous_objective, objective_variable_name).
"""
function extract_instantaneous_objective(objective::Expr, discount_factor::Symbol)
    if objective.head == :(=)
        obj_var = opt_get_var_name(objective.args[1])
        rhs = objective.args[2]
        
        # Collect all terms without the discount factor
        if rhs isa Expr && rhs.head == :call && rhs.args[1] == :+
            instant_terms = []
            for arg in rhs.args[2:end]
                if !contains_symbol(arg, discount_factor)
                    push!(instant_terms, arg)
                end
            end
            
            if isempty(instant_terms)
                # Whole expression is instant objective (no recursive term found)
                instant_obj = rhs
            elseif length(instant_terms) == 1
                instant_obj = instant_terms[1]
            else
                # Combine multiple terms with +
                instant_obj = Expr(:call, :+, instant_terms...)
            end
            
            return instant_obj, obj_var
        else
            return rhs, obj_var
        end
    end
    
    return objective, nothing
end

"""
    contains_symbol(expr, sym)

Check if an expression contains a specific symbol.
"""
function contains_symbol(expr, sym::Symbol)
    found = false
    postwalk(x -> begin
        if x == sym
            found = true
        end
        return x
    end, expr)
    return found
end

"""
    derive_single_foc(instant_objective, constraints, multipliers, control, discount_factor, is_recursive)

Derive the FOC for a single control variable.
"""
function derive_single_foc(instant_objective, constraints::Vector{Expr}, multipliers::Vector{Symbol}, 
                          control::Symbol, discount_factor::Symbol, is_recursive::Bool)
    # Collect all terms for the FOC
    terms = []
    
    # 1. Derivative of instantaneous objective w.r.t. control[0]
    if 0 in opt_find_control_occurrences(instant_objective, control)
        d_obj = opt_symbolic_differentiate(instant_objective, control, 0)
        if d_obj != 0
            push!(terms, d_obj)
        end
    end
    
    # 2. Derivative of constraints w.r.t. control[0]
    # L = obj + Σ λ_i * (RHS_i - LHS_i)
    # ∂L/∂control = ∂obj/∂control + Σ λ_i * ∂(RHS_i - LHS_i)/∂control
    for (i, constraint) in enumerate(constraints)
        # Convert constraint to RHS - LHS form
        if constraint.head == :(=)
            constraint_expr = Expr(:call, :-, constraint.args[2], constraint.args[1])
        else
            constraint_expr = constraint
        end
        
        # Differentiate w.r.t. control[0]
        if 0 in opt_find_control_occurrences(constraint_expr, control)
            d_constr = opt_symbolic_differentiate(constraint_expr, control, 0)
            if d_constr != 0
                # Add λ_i * d_constr
                λ_term = Expr(:call, :*, Expr(:ref, multipliers[i], 0), d_constr)
                push!(terms, λ_term)
            end
        end
    end
    
    # 3. If recursive and control appears in t-1 (as state variable affecting t+1),
    # add β * E[∂L_{t+1}/∂control_t]
    # This means: for each constraint where control[-1] appears, we need:
    # β * λ_{i,t+1} * ∂(RHS_{i,t+1} - LHS_{i,t+1})/∂control_t
    if is_recursive
        for (i, constraint) in enumerate(constraints)
            if constraint.head == :(=)
                constraint_expr = Expr(:call, :-, constraint.args[2], constraint.args[1])
            else
                constraint_expr = constraint
            end
            
            # Check if control[-1] appears
            if -1 in opt_find_control_occurrences(constraint_expr, control)
                d_constr_lag = opt_symbolic_differentiate(constraint_expr, control, -1)
                if d_constr_lag != 0
                    # Shift to t+1: d_constr_lag evaluated at t+1 means control[-1] -> control[0]
                    d_constr_shifted = opt_shift_all_time_indices(d_constr_lag, 1)
                    # Add β * λ_{i}[1] * d_constr_shifted
                    future_term = Expr(:call, :*, discount_factor, 
                                      Expr(:ref, multipliers[i], 1), 
                                      d_constr_shifted)
                    push!(terms, future_term)
                end
            end
        end
        
        # Also check instantaneous objective for control[-1]
        if -1 in opt_find_control_occurrences(instant_objective, control)
            d_obj_lag = opt_symbolic_differentiate(instant_objective, control, -1)
            if d_obj_lag != 0
                d_obj_shifted = opt_shift_all_time_indices(d_obj_lag, 1)
                future_obj_term = Expr(:call, :*, discount_factor, d_obj_shifted)
                push!(terms, future_obj_term)
            end
        end
    end
    
    # Combine all terms into single FOC expression (= 0)
    if isempty(terms)
        return nothing
    elseif length(terms) == 1
        return terms[1]
    else
        # Combine with + 
        foc = Expr(:call, :+, terms...)
        return foc
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main @optimization_model macro (EXPERIMENTAL - not fully implemented)
# For now, use the derive_focs function directly. See models/gEcon_RBC_example.jl
# ─────────────────────────────────────────────────────────────────────────────

"""
$(SIGNATURES)
**EXPERIMENTAL - Not fully implemented. Use `derive_focs` function directly instead.**

Parses an optimization problem and derives the first-order conditions automatically.

This macro allows users to specify DSGE models in terms of optimization problems 
(similar to gEcon syntax) rather than writing out the FOCs manually.

# Arguments
- `𝓂`: name of the object to be created containing the model information
- `ex`: optimization problem specification

# Blocks
The optimization problem is specified using blocks:

- `agent NAME begin ... end` or `block NAME begin ... end`: Define an optimizing agent
  - `definitions begin ... end`: Auxiliary variable definitions (substituted into other expressions)
  - `controls begin ... end`: Control variables (decision variables)
  - `objective begin ... end`: Objective function (can be recursive)
  - `constraints begin ... end`: Constraints on the optimization problem
  
- `equilibrium begin ... end`: Market clearing/equilibrium conditions (identities)

- `exogenous begin ... end`: Exogenous shock processes
  - `identities begin ... end`: Equations for shock processes
  - `shocks begin ... end`: Shock variables

# Examples
```julia
using MacroModelling

@optimization_model RBC_opt begin
    agent CONSUMER begin
        definitions begin
            u[0] = (C[0]^μ * (1 - L_s[0])^(1 - μ))^(1 - η) / (1 - η)
        end
        
        controls begin
            K_s[0], C[0], L_s[0], I[0]
        end
        
        objective begin
            U[0] = u[0] + β * U[1]
        end
        
        constraints begin
            I[0] + C[0] = π[0] + r[0] * K_s[-1] + W[0] * L_s[0]
            K_s[0] = (1 - δ) * K_s[-1] + I[0]
        end
    end
    
    agent FIRM begin
        controls begin
            K_d[0], L_d[0], Y[0]
        end
        
        objective begin
            π[0] = Y[0] - L_d[0] * W[0] - r[0] * K_d[0]
        end
        
        constraints begin
            Y[0] = Z[0] * K_d[0]^α * L_d[0]^(1 - α)
        end
    end
    
    equilibrium begin
        K_d[0] = K_s[-1]
        L_d[0] = L_s[0]
    end
    
    exogenous begin
        Z[0] = exp(ϕ * log(Z[-1]) + ϵ_Z[x])
        
        shocks begin
            ϵ_Z[x]
        end
    end
end

@parameters RBC_opt begin
    δ = 0.025
    β = 0.99
    η = 2
    μ = 0.3
    α = 0.36
    ϕ = 0.95
end
```

# Returns
- `Nothing`. The macro creates the model `𝓂` with the derived FOC equations in the calling scope.
"""
macro optimization_model(𝓂, ex...)
    # Parse options
    verbose = false
    max_obc_horizon = 40
    
    for exp in ex[1:end-1]
        postwalk(x -> 
            x isa Expr ?
                x.head == :(=) ?  
                    x.args[1] == :verbose && x.args[2] isa Bool ?
                        verbose = x.args[2] :
                    x.args[1] == :max_obc_horizon && x.args[2] isa Int ?
                        max_obc_horizon = x.args[2] :
                    begin
                        @warn "Invalid option `$(x.args[1])` ignored."
                        x
                    end :
                x :
            x,
        exp)
    end
    
    model_block = ex[end]
    
    # Parse the optimization model specification
    all_equations = Expr[]
    all_shocks = Symbol[]
    
    # Walk through the model block and parse each component
    for item in model_block.args
        if item isa Expr
            parsed = parse_optimization_block(item)
            if parsed !== nothing
                append!(all_equations, parsed.equations)
                append!(all_shocks, parsed.shocks)
            end
        end
    end
    
    # Create the model equations block for @model
    equations_block = Expr(:block, all_equations...)
    
    # Generate the @model macro call
    model_name = 𝓂
    
    return quote
        @model $model_name max_obc_horizon = $max_obc_horizon begin
            $(equations_block.args...)
        end
    end
end

# Parsed result structure
struct ParsedBlock
    equations::Vector{Expr}
    shocks::Vector{Symbol}
end

"""
    parse_optimization_block(block_expr)

Parse a single block from the optimization model specification.
"""
function parse_optimization_block(block_expr::Expr)
    if block_expr.head != :call && block_expr.head != :macrocall
        return nothing
    end
    
    block_type = block_expr.args[1]
    
    if block_type in (:agent, :block)
        return parse_agent_block(block_expr)
    elseif block_type == :equilibrium
        return parse_equilibrium_block(block_expr)
    elseif block_type == :exogenous
        return parse_exogenous_block(block_expr)
    end
    
    return nothing
end

"""
    parse_agent_block(block_expr)

Parse an agent/block that contains an optimization problem.
"""
function parse_agent_block(block_expr::Expr)
    # Extract block name and content
    block_name = string(block_expr.args[2])
    content = block_expr.args[3]
    
    # Initialize containers
    definitions = Dict{Symbol,Expr}()
    controls = Symbol[]
    objective = nothing
    constraints = Expr[]
    
    # Parse each section
    for item in content.args
        if item isa Expr && item.head == :call
            section_type = item.args[1]
            section_content = item.args[2]
            
            if section_type == :definitions
                definitions = parse_definitions_section(section_content)
            elseif section_type == :controls
                controls = parse_controls_section(section_content)
            elseif section_type == :objective
                objective = parse_objective_section(section_content)
            elseif section_type == :constraints
                constraints = parse_constraints_section(section_content)
            end
        end
    end
    
    if objective === nothing || isempty(controls)
        return nothing
    end
    
    # Derive FOCs
    focs, multipliers = derive_focs(
        definitions = definitions,
        controls = controls,
        objective = objective,
        constraints = constraints,
        discount_factor = :β,
        block_name = lowercase(block_name)
    )
    
    # Include constraints as equations and FOCs
    equations = Expr[]
    append!(equations, focs)
    append!(equations, constraints)
    
    # Include objective equation if it's a definition (like π[0] = ...)
    if objective.head == :(=) && !contains_recursive_term(objective, :β)
        push!(equations, objective)
    end
    
    return ParsedBlock(equations, Symbol[])
end

"""
Parse definitions section.
"""
function parse_definitions_section(content::Expr)
    definitions = Dict{Symbol,Expr}()
    
    for item in content.args
        if item isa Expr && item.head == :(=)
            if item.args[1] isa Expr && item.args[1].head == :ref
                var_name = item.args[1].args[1]
                definitions[var_name] = item.args[2]
            end
        end
    end
    
    return definitions
end

"""
Parse controls section.
"""
function parse_controls_section(content::Expr)
    controls = Symbol[]
    
    postwalk(x -> begin
        if x isa Expr && x.head == :ref
            push!(controls, x.args[1])
        end
        return x
    end, content)
    
    return unique(controls)
end

"""
Parse objective section.
"""
function parse_objective_section(content::Expr)
    for item in content.args
        if item isa Expr && item.head == :(=)
            return item
        end
    end
    return nothing
end

"""
Parse constraints section.
"""
function parse_constraints_section(content::Expr)
    constraints = Expr[]
    
    for item in content.args
        if item isa Expr && item.head == :(=)
            push!(constraints, item)
        end
    end
    
    return constraints
end

"""
    parse_equilibrium_block(block_expr)

Parse equilibrium (identity) equations.
"""
function parse_equilibrium_block(block_expr::Expr)
    content = block_expr.args[2]
    equations = Expr[]
    
    for item in content.args
        if item isa Expr && item.head == :(=)
            push!(equations, item)
        end
    end
    
    return ParsedBlock(equations, Symbol[])
end

"""
    parse_exogenous_block(block_expr)

Parse exogenous shock processes.
"""
function parse_exogenous_block(block_expr::Expr)
    content = block_expr.args[2]
    equations = Expr[]
    shocks = Symbol[]
    
    for item in content.args
        if item isa Expr
            if item.head == :(=)
                push!(equations, item)
            elseif item.head == :call && item.args[1] == :shocks
                # Parse shocks section
                shocks_content = item.args[2]
                for shock_item in shocks_content.args
                    if shock_item isa Expr && shock_item.head == :ref
                        push!(shocks, shock_item.args[1])
                    end
                end
            end
        end
    end
    
    return ParsedBlock(equations, shocks)
end

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing for @model macro integration
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_optimization_syntax(equations_block)

Preprocess model equations to detect and transform optimization problem syntax.

Supported syntax:
```julia
U[0] = maximise(log(C[0]) + ψ * log(1 - L[0]) + β * U[1],
    controls = [C[0], L[0], K[1]]
) | begin
    C[0] + I[0] = π[0] + r[0] * K[-1] + w[0] * L[0]
    K[0] = (1 - δ) * K[-1] + I[0]
end
```

Also supports `maximize` (American spelling) and `minimise`/`minimize`.

Returns transformed equations block with optimization problems replaced by their FOCs.
"""
function parse_optimization_syntax(equations_block::Expr)
    new_args = Any[]
    
    for arg in equations_block.args
        if arg isa Expr
            transformed = transform_optimization_expr(arg)
            if transformed isa Vector
                append!(new_args, transformed)
            else
                push!(new_args, transformed)
            end
        else
            push!(new_args, arg)
        end
    end
    
    return Expr(:block, new_args...)
end

"""
    transform_optimization_expr(expr)

Transform a single expression, detecting optimization problem syntax.
Returns either the original expression or a vector of FOC equations.
"""
function transform_optimization_expr(expr::Expr)
    # Check for pattern: LHS = maximise/minimize(...) | begin ... end
    # or: LHS = maximise/minimize(..., subject_to = begin ... end)
    
    if expr.head == :(=) && expr.args[2] isa Expr
        rhs = expr.args[2]
        lhs = expr.args[1]
        
        # First, recursively unblock the RHS
        rhs = unblock(rhs)
        
        # Pattern 1: LHS = maximise(...) | begin ... end
        if rhs isa Expr && rhs.head == :call && rhs.args[1] == :|
            opt_call = rhs.args[2]
            constraints_block = rhs.args[3]
            
            if opt_call isa Expr && opt_call.head == :call && 
               opt_call.args[1] in (:maximise, :maximize, :minimise, :minimize)
                return process_optimization_problem(lhs, opt_call, constraints_block)
            end
        end
        
        # Pattern 2: LHS = maximise(..., subject_to = begin ... end)
        if rhs isa Expr && rhs.head == :call && rhs.args[1] in (:maximise, :maximize, :minimise, :minimize)
            # Look for subject_to keyword argument
            subject_to_block = nothing
            for arg in rhs.args
                if arg isa Expr && arg.head == :kw && arg.args[1] == :subject_to
                    subject_to_block = arg.args[2]
                    break
                end
            end
            
            if subject_to_block !== nothing
                return process_optimization_problem(lhs, rhs, subject_to_block)
            end
        end
        
        # Pattern 3: LHS = maximise(...) do begin ... end (do block syntax)
        if rhs isa Expr && rhs.head == :do && rhs.args[1] isa Expr && rhs.args[1].head == :call
            opt_call = rhs.args[1]
            if opt_call.args[1] in (:maximise, :maximize, :minimise, :minimize)
                # Extract constraints from do block
                do_block = rhs.args[2]
                if do_block isa Expr && do_block.head == :(->)
                    constraints_block = do_block.args[2]
                    return process_optimization_problem(lhs, opt_call, constraints_block)
                end
            end
        end
        
        # Not an optimization expression - return with unblocked RHS
        return Expr(:(=), lhs, rhs)
    end
    
    return expr
end

"""
    process_optimization_problem(lhs, opt_call, constraints_block)

Process an optimization problem and return the derived FOC equations.

Arguments:
- lhs: Left-hand side of the objective equation (e.g., U[0])
- opt_call: The maximise/minimize call expression
- constraints_block: Block containing constraint equations

Returns a vector of expressions representing the FOCs and constraints.
"""
function process_optimization_problem(lhs::Expr, opt_call::Expr, constraints_block::Expr)
    # Extract optimization type
    opt_type = opt_call.args[1]
    is_maximization = opt_type in (:maximise, :maximize)
    
    # Extract objective function (first positional argument)
    objective_expr = opt_call.args[2]
    
    # Extract controls from keyword argument
    controls = Symbol[]
    discount_factor = :β  # Default
    
    for arg in opt_call.args[3:end]
        if arg isa Expr && arg.head == :kw
            if arg.args[1] == :controls
                controls_list = arg.args[2]
                if controls_list isa Expr && controls_list.head == :vect
                    for ctrl in controls_list.args
                        if ctrl isa Expr && ctrl.head == :ref
                            push!(controls, ctrl.args[1])
                        elseif ctrl isa Symbol
                            push!(controls, ctrl)
                        end
                    end
                end
            elseif arg.args[1] == :discount_factor
                discount_factor = arg.args[2]
            end
        end
    end
    
    # Extract constraints from the constraints block
    constraints = Expr[]
    if constraints_block isa Expr && constraints_block.head == :block
        for item in constraints_block.args
            if item isa Expr && item.head == :(=)
                push!(constraints, item)
            end
        end
    elseif constraints_block isa Expr && constraints_block.head == :vect
        # Array syntax for constraints
        for item in constraints_block.args
            if item isa Expr && item.head == :(=)
                push!(constraints, item)
            end
        end
    end
    
    # Build the full objective: LHS = objective_expr
    full_objective = Expr(:(=), lhs, objective_expr)
    
    # Generate unique multiplier name prefix based on LHS variable
    if lhs isa Expr && lhs.head == :ref
        block_name = string(lhs.args[1])
    else
        block_name = "opt"
    end
    
    # Derive FOCs
    focs, multipliers = derive_focs(
        controls = controls,
        objective = full_objective,
        constraints = constraints,
        discount_factor = discount_factor,
        block_name = block_name
    )
    
    # For minimization, negate the FOCs from the objective part
    # (The constraint derivatives don't change sign)
    if !is_maximization
        focs = [negate_objective_terms(foc) for foc in focs]
    end
    
    # Build the result: FOCs + constraints
    result = Expr[]
    
    # Add FOCs as equations (FOC = 0 form, converted to LHS = RHS by moving terms)
    # The @model macro expects equations in the form: expr1 = expr2 (implying expr1 - expr2 = 0)
    # Our FOCs are already in the form "expression = 0", so we convert them
    for foc in focs
        if foc isa Expr
            # Convert FOC expression to equation form: foc_expr = 0
            push!(result, Expr(:(=), foc, 0))
        end
    end
    
    # Add constraints as equations (clean up any block wrappers)
    for constraint in constraints
        if constraint isa Expr && constraint.head == :(=)
            # Clean the RHS if it has a block wrapper
            lhs_c = constraint.args[1]
            rhs_c = constraint.args[2]
            
            # Unwrap block expressions
            if rhs_c isa Expr && rhs_c.head == :block
                # Find the actual expression inside the block
                for item in rhs_c.args
                    if item isa Expr
                        rhs_c = item
                        break
                    end
                end
            end
            
            push!(result, Expr(:(=), lhs_c, rhs_c))
        else
            push!(result, constraint)
        end
    end
    
    return result
end

"""
    negate_objective_terms(expr)

For minimization problems, negate the objective derivative terms.
This is a simplified approach - in practice, minimization is the same as
maximizing the negative.
"""
function negate_objective_terms(expr)
    # For now, just return as-is since FOC = 0 structure is the same
    # The derivative signs are already correct from the Lagrangian
    return expr
end

# ─────────────────────────────────────────────────────────────────────────────
# Export functions
# ─────────────────────────────────────────────────────────────────────────────

export derive_focs, parse_optimization_syntax
