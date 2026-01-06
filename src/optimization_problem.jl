# Optimization Problem Parser for @model macro
# 
# This module enables users to specify DSGE models as optimization problems
# instead of writing first-order conditions manually.
#
# Supported syntaxes (since Julia doesn't allow `subject_to` as infix operator):
#
# 1. Using `|` operator:
#    U[0] = maximise(log(C[0]) + β * U[1], controls = [C[0], K[0]]) | begin
#        C[0] + K[0] = Y[0]
#    end
#
# 2. Using `do` block:
#    U[0] = maximise(log(C[0]) + β * U[1], controls = [C[0], K[0]]) do
#        C[0] + K[0] = Y[0]
#    end

using MacroTools: postwalk, unblock

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for symbolic differentiation
# ─────────────────────────────────────────────────────────────────────────────

"""
Extract all variable occurrences with their time indices from an expression.
Returns a Dict mapping variable names to sets of time indices.
"""
function find_all_variables(expr)
    vars = Dict{Symbol, Set{Int}}()
    postwalk(x -> begin
        if x isa Expr && x.head == :ref && x.args[1] isa Symbol && x.args[2] isa Int
            varname = x.args[1]
            time_idx = x.args[2]
            if !haskey(vars, varname)
                vars[varname] = Set{Int}()
            end
            push!(vars[varname], time_idx)
        end
        return x
    end, expr)
    return vars
end

"""
Shift all time indices in an expression by the given amount.
"""
function shift_time_indices(expr::Expr, shift::Int)
    return postwalk(x -> begin
        if x isa Expr && x.head == :ref && x.args[1] isa Symbol && x.args[2] isa Int
            return Expr(:ref, x.args[1], x.args[2] + shift)
        end
        return x
    end, expr)
end
shift_time_indices(x, shift::Int) = x

"""
Transform variables for SymPy: C[0] -> C_0, K[-1] -> K_m1
"""
function transform_for_sympy(expr::Expr)
    mapping = Dict{Symbol, Expr}()
    
    transformed = postwalk(x -> begin
        if x isa Expr && x.head == :ref && x.args[1] isa Symbol && x.args[2] isa Int
            varname = x.args[1]
            time_idx = x.args[2]
            suffix = time_idx >= 0 ? "_$(time_idx)" : "_m$(abs(time_idx))"
            new_sym = Symbol(string(varname) * suffix)
            mapping[new_sym] = x
            return new_sym
        end
        return x
    end, expr)
    
    return transformed, mapping
end

"""
Reverse transformation: C_0 -> C[0]
"""
function untransform_from_sympy(expr, mapping::Dict{Symbol, Expr})
    reverse_mapping = Dict(k => v for (k, v) in mapping)
    
    return postwalk(x -> begin
        if x isa Symbol && haskey(reverse_mapping, x)
            return reverse_mapping[x]
        end
        return x
    end, expr)
end

"""
Differentiate an expression with respect to a variable at a specific time index.
Uses SymPy for symbolic differentiation.
"""
function symbolic_diff(expr, var::Symbol, time_idx::Int)
    # Check if the variable appears in the expression at this time
    vars = find_all_variables(expr)
    if !haskey(vars, var) || !(time_idx in vars[var])
        return 0
    end
    
    # Transform expression for SymPy
    transformed_expr, var_mapping = transform_for_sympy(expr)
    
    # Target variable in SymPy notation
    target_suffix = time_idx >= 0 ? "_$(time_idx)" : "_m$(abs(time_idx))"
    target_sym = Symbol(string(var) * target_suffix)
    
    # Get all symbols in the expression
    all_symbols = collect(get_symbols(transformed_expr))
    
    # Create SymPy symbols
    for sym in all_symbols
        if sym isa Symbol
            sym_value = SPyPyC.symbols(string(sym), real = true, finite = true)
            Core.eval(SymPyWorkspace, :($sym = $sym_value))
        end
    end
    
    # Ensure target is also created
    if target_sym ∉ all_symbols
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
    
    derivative_expr = try
        Meta.parse(derivative_str)
    catch
        return 0
    end
    
    # Transform back to original notation
    return untransform_from_sympy(derivative_expr, var_mapping)
end

# ─────────────────────────────────────────────────────────────────────────────
# FOC Derivation
# ─────────────────────────────────────────────────────────────────────────────

"""
    derive_focs(; controls, objective, constraints, discount_factor=:β, block_name="opt")

Derive first-order conditions from an optimization problem.

# Arguments
- `controls::Vector{Symbol}`: Control variables
- `objective::Expr`: Bellman objective (e.g., `:(U[0] = log(C[0]) + β * U[1])`)
- `constraints::Vector{Expr}`: Constraint equations
- `discount_factor::Symbol`: Discount factor symbol (default `:β`)
- `block_name::String`: Prefix for Lagrange multipliers

# Returns
- `focs::Vector{Expr}`: FOC expressions (to be set = 0)
- `multipliers::Vector{Symbol}`: Lagrange multiplier symbols
"""
function derive_focs(;
    controls::Vector{Symbol},
    objective::Expr,
    constraints::Vector{Expr},
    discount_factor::Symbol = :β,
    block_name::String = "opt"
)
    # Parse objective: U[0] = instant_utility + β * U[1]
    @assert objective.head == :(=) "Objective must be an equation"
    
    instant_obj = objective.args[2]
    
    # Check for recursive structure (β * U[1] term)
    is_recursive = false
    postwalk(x -> begin
        if x isa Expr && x.head == :call && x.args[1] == :*
            if discount_factor in x.args
                for arg in x.args[2:end]
                    if arg isa Expr && arg.head == :ref && arg.args[2] == 1
                        is_recursive = true
                    end
                end
            end
        end
        return x
    end, instant_obj)
    
    # Create Lagrange multipliers
    multipliers = [Symbol("λ_$(block_name)_$i") for i in 1:length(constraints)]
    
    # Convert constraints to (RHS - LHS) form
    constraint_diffs = Expr[]
    for c in constraints
        if c.head == :(=)
            push!(constraint_diffs, Expr(:call, :-, c.args[2], c.args[1]))
        end
    end
    
    # Derive FOC for each control
    focs = Expr[]
    
    for control in controls
        terms = []
        
        # 1. Derivative of instant objective w.r.t. control[0]
        d_obj = symbolic_diff(instant_obj, control, 0)
        if d_obj != 0
            push!(terms, d_obj)
        end
        
        # 2. Derivatives of constraints w.r.t. control[0]
        for (i, cdiff) in enumerate(constraint_diffs)
            d_constr = symbolic_diff(cdiff, control, 0)
            if d_constr != 0
                λ_term = Expr(:call, :*, Expr(:ref, multipliers[i], 0), d_constr)
                push!(terms, λ_term)
            end
        end
        
        # 3. For recursive problems: future effects via state variables
        if is_recursive
            for (i, cdiff) in enumerate(constraint_diffs)
                # Check if control[-1] appears (meaning control is a state)
                d_constr_lag = symbolic_diff(cdiff, control, -1)
                if d_constr_lag != 0
                    # Shift to t+1 and multiply by β * λ[1]
                    shifted = shift_time_indices(d_constr_lag, 1)
                    future_term = Expr(:call, :*, discount_factor, 
                                      Expr(:ref, multipliers[i], 1), shifted)
                    push!(terms, future_term)
                end
            end
        end
        
        # Combine terms into FOC
        if !isempty(terms)
            if length(terms) == 1
                push!(focs, terms[1])
            else
                push!(focs, Expr(:call, :+, terms...))
            end
        end
    end
    
    return focs, multipliers
end

# ─────────────────────────────────────────────────────────────────────────────
# Parsing optimization syntax in @model macro
# ─────────────────────────────────────────────────────────────────────────────

"""
    parse_optimization_syntax(model_block::Expr)

Transform optimization problem syntax into FOC equations.

Detects patterns:
1. `U[0] = maximise(...) | begin ... end`
2. `U[0] = maximise(...) do ... end`
"""
function parse_optimization_syntax(model_block::Expr)
    new_args = Any[]
    
    for arg in model_block.args
        if arg isa Expr
            result = try_parse_optimization(arg)
            if result isa Vector
                append!(new_args, result)
            else
                push!(new_args, result)
            end
        else
            push!(new_args, arg)
        end
    end
    
    return Expr(:block, new_args...)
end

"""
Parse a single expression, checking for optimization syntax.
"""
function try_parse_optimization(expr::Expr)
    # Must be an assignment
    if expr.head != :(=)
        return expr
    end
    
    lhs = expr.args[1]
    rhs = unblock(expr.args[2])
    
    # Pattern 1: maximise(...) | begin ... end
    if rhs isa Expr && rhs.head == :call && rhs.args[1] == :|
        opt_call = rhs.args[2]
        constraints_block = rhs.args[3]  # Don't unblock - keep as block
        
        if opt_call isa Expr && opt_call.head == :call &&
           opt_call.args[1] in (:maximise, :maximize, :minimise, :minimize)
            return process_optimization(lhs, opt_call, constraints_block)
        end
    end
    
    # Pattern 2: maximise(...) do ... end
    if rhs isa Expr && rhs.head == :do
        opt_call = rhs.args[1]
        if opt_call isa Expr && opt_call.head == :call &&
           opt_call.args[1] in (:maximise, :maximize, :minimise, :minimize)
            # Extract block from do syntax
            do_block = rhs.args[2]
            if do_block isa Expr && do_block.head == :(->)
                constraints_block = unblock(do_block.args[2])
                return process_optimization(lhs, opt_call, constraints_block)
            end
        end
    end
    
    # Not an optimization expression - just clean up blocks
    return Expr(:(=), lhs, rhs)
end

"""
Process an optimization problem and return FOC + constraint equations.
"""
function process_optimization(lhs::Expr, opt_call::Expr, constraints_block::Expr)
    # Extract optimization direction
    opt_type = opt_call.args[1]
    is_max = opt_type in (:maximise, :maximize)
    
    # Extract objective (first positional arg)
    objective_expr = opt_call.args[2]
    
    # Extract controls from keyword arg
    controls = Symbol[]
    discount_factor = :β
    
    for arg in opt_call.args[3:end]
        if arg isa Expr && arg.head == :kw
            if arg.args[1] == :controls
                ctrl_list = arg.args[2]
                if ctrl_list isa Expr && ctrl_list.head == :vect
                    for c in ctrl_list.args
                        if c isa Expr && c.head == :ref
                            push!(controls, c.args[1])
                        elseif c isa Symbol
                            push!(controls, c)
                        end
                    end
                end
            elseif arg.args[1] == :discount_factor
                discount_factor = arg.args[2]
            end
        end
    end
    
    # Extract constraints - handle both block and single equation
    constraints = Expr[]
    if constraints_block.head == :block
        for item in constraints_block.args
            if item isa Expr && item.head == :(=)
                clean_rhs = unblock(item.args[2])
                push!(constraints, Expr(:(=), item.args[1], clean_rhs))
            end
        end
    elseif constraints_block.head == :(=)
        # Single constraint (block was unwrapped)
        clean_rhs = unblock(constraints_block.args[2])
        push!(constraints, Expr(:(=), constraints_block.args[1], clean_rhs))
    end
    
    # Build full objective equation
    full_objective = Expr(:(=), lhs, objective_expr)
    
    # Generate block name from LHS variable
    block_name = lhs isa Expr && lhs.head == :ref ? string(lhs.args[1]) : "opt"
    
    # Derive FOCs
    focs, multipliers = derive_focs(
        controls = controls,
        objective = full_objective,
        constraints = constraints,
        discount_factor = discount_factor,
        block_name = block_name
    )
    
    # Build result equations
    result = Expr[]
    
    # FOCs (= 0)
    for foc in focs
        push!(result, Expr(:(=), foc, 0))
    end
    
    # Constraints
    for c in constraints
        push!(result, c)
    end
    
    return result
end

export derive_focs, parse_optimization_syntax
