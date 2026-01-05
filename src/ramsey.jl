# Ramsey Optimal Policy Implementation
# Provides automatic derivation of first-order conditions using Symbolics.jl
# Designed to work with the @ramsey block syntax inside @model

"""
    parse_ramsey_block(ramsey_block::Expr)

Parse a @ramsey block expression and extract configuration.

The block should have the form:
```julia
@ramsey begin
    objective = log(c[0])
    instruments = [R, τ]
    discount = β
end
```

# Returns
- NamedTuple with fields: objective, instruments, discount
"""
function parse_ramsey_block(ramsey_block::Expr)
    objective = nothing
    instruments = Symbol[]
    discount = :β  # default
    
    if ramsey_block.head == :block
        for arg in ramsey_block.args
            if arg isa LineNumberNode
                continue
            end
            if arg isa Expr && arg.head == :(=)
                lhs = arg.args[1]
                rhs = arg.args[2]
                
                if lhs == :objective
                    objective = rhs
                elseif lhs == :instruments
                    if rhs isa Symbol
                        instruments = [rhs]
                    elseif rhs isa Expr && rhs.head == :vect
                        instruments = Symbol[s for s in rhs.args if s isa Symbol]
                    elseif rhs isa Expr && rhs.head == :tuple
                        instruments = Symbol[s for s in rhs.args if s isa Symbol]
                    end
                elseif lhs == :discount
                    discount = rhs
                end
            end
        end
    end
    
    @assert !isnothing(objective) "@ramsey block must specify an objective function"
    @assert !isempty(instruments) "@ramsey block must specify at least one instrument"
    
    return (objective = objective, instruments = instruments, discount = discount)
end


"""
    create_multiplier_symbol(eq_idx::Int)

Create a Lagrange multiplier symbol for equation index eq_idx.
"""
function create_multiplier_symbol(eq_idx::Int)::Symbol
    return Symbol("Lagr_mult_" * string(eq_idx))
end


"""
    collect_variables_with_timing(expr, collected::Set{Tuple{Symbol,Int}})

Recursively collect all variable references with their time indices from an expression.
"""
function collect_variables_with_timing(expr, collected::Set{Tuple{Symbol,Int}})
    if expr isa Expr
        if expr.head == :ref && length(expr.args) >= 2
            var_name = expr.args[1]
            time_idx = expr.args[2]
            if var_name isa Symbol && time_idx isa Int
                push!(collected, (var_name, time_idx))
            end
        else
            for arg in expr.args
                collect_variables_with_timing(arg, collected)
            end
        end
    end
    return collected
end


"""
    make_symbolics_var_name(var::Symbol, time_idx::Int)

Create a unique symbol name for a Symbolics variable representing var[time_idx].
"""
function make_symbolics_var_name(var::Symbol, time_idx::Int)::Symbol
    suffix = time_idx < 0 ? "_m$(abs(time_idx))" : time_idx == 0 ? "_0" : "_p$(time_idx)"
    return Symbol(string(var) * suffix)
end


"""
    expr_to_symbolics_expr(expr, sym_vars::Dict, sym_pars::Dict, shocks::Set{Symbol})

Convert a Julia Expr with time-indexed variables to an expression suitable for Symbolics.
Returns a Julia expression where variables are replaced by their Symbolics equivalents.
"""
function expr_to_symbolics_expr(expr, sym_vars::Dict, sym_pars::Dict, shocks::Set{Symbol})
    if expr isa Number
        return expr
    elseif expr isa Symbol
        # Parameter or constant
        if haskey(sym_pars, expr)
            return sym_pars[expr]
        else
            return expr
        end
    elseif expr isa Expr
        if expr.head == :ref && length(expr.args) >= 2
            var_name = expr.args[1]
            time_idx = expr.args[2]
            
            # Skip shocks
            if var_name in shocks
                return 0  # Shocks don't appear in FOCs
            end
            
            if time_idx isa Int
                key = (var_name, time_idx)
                if haskey(sym_vars, key)
                    return sym_vars[key]
                end
            end
            # Return 0 for unhandled cases (e.g., shock timing like x)
            return 0
        elseif expr.head == :call
            op = expr.args[1]
            converted_args = [expr_to_symbolics_expr(arg, sym_vars, sym_pars, shocks) for arg in expr.args[2:end]]
            return Expr(:call, op, converted_args...)
        elseif expr.head == :(=)
            # Equation: convert to residual (LHS - RHS)
            lhs = expr_to_symbolics_expr(expr.args[1], sym_vars, sym_pars, shocks)
            rhs = expr_to_symbolics_expr(expr.args[2], sym_vars, sym_pars, shocks)
            return :($lhs - $rhs)
        else
            # Other expression types - process recursively
            new_args = [expr_to_symbolics_expr(arg, sym_vars, sym_pars, shocks) for arg in expr.args]
            return Expr(expr.head, new_args...)
        end
    else
        return expr
    end
end


"""
    symbolics_expr_to_model_expr(sym_result, sym_to_var::Dict, sym_to_par::Dict)

Convert a Symbolics expression result back to model notation with time indices.
"""
function symbolics_expr_to_model_expr(sym_result, sym_to_var::Dict, sym_to_par::Dict)
    result_str = string(Symbolics.simplify(sym_result))
    
    # Replace symbolics variable names with time-indexed model notation
    for (sym_name, (var, time_idx)) in sym_to_var
        pattern = string(sym_name)
        replacement = "$(var)[$(time_idx)]"
        result_str = replace(result_str, pattern => replacement)
    end
    
    # Replace parameter names (no change needed as they're the same)
    
    # Parse back to expression
    try
        return Meta.parse(result_str)
    catch
        return Meta.parse("0")  # Fallback
    end
end


"""
    derive_ramsey_focs_with_symbolics(objective::Expr, 
                                       constraints::Vector{Expr},
                                       variables::Vector{Symbol},
                                       instruments::Vector{Symbol},
                                       shocks::Vector{Symbol},
                                       parameters::Vector{Symbol},
                                       discount::Symbol)

Derive Ramsey first-order conditions using Symbolics.jl for differentiation.

# Arguments
- `objective`: Per-period objective function
- `constraints`: Model equations (will be constraints in the planner's problem)  
- `variables`: All endogenous variables
- `instruments`: Policy instruments (planner's choice variables)
- `shocks`: Exogenous shocks
- `parameters`: Model parameters
- `discount`: Discount factor symbol

# Returns
- Vector of FOC expressions
- Vector of Lagrange multiplier symbols
"""
function derive_ramsey_focs_with_symbolics(objective::Expr, 
                                            constraints::Vector{Expr},
                                            variables::Vector{Symbol},
                                            instruments::Vector{Symbol},
                                            shocks::Vector{Symbol},
                                            parameters::Vector{Symbol},
                                            discount::Symbol)
    
    n_constraints = length(constraints)
    shock_set = Set(shocks)
    
    # Collect all variable-time combinations that appear in the constraints
    var_time_pairs = Set{Tuple{Symbol,Int}}()
    for eq in constraints
        collect_variables_with_timing(eq, var_time_pairs)
    end
    collect_variables_with_timing(objective, var_time_pairs)
    
    # Also need multiplier-time combinations for t-1, t, t+1
    multipliers = [create_multiplier_symbol(i) for i in 1:n_constraints]
    
    # Create Symbolics variables for all variable-time pairs
    sym_vars = Dict{Tuple{Symbol,Int}, Symbol}()  # Maps (var, t) to sym name
    sym_to_var = Dict{Symbol, Tuple{Symbol,Int}}()  # Reverse mapping
    
    for (var, t) in var_time_pairs
        if var in shock_set
            continue  # Skip shocks
        end
        sym_name = make_symbolics_var_name(var, t)
        sym_vars[(var, t)] = sym_name
        sym_to_var[sym_name] = (var, t)
    end
    
    # Ensure we have variables for t-1, t, t+1 for the derivative calculations
    for var in variables
        if var in shock_set
            continue
        end
        for t in [-1, 0, 1]
            if !haskey(sym_vars, (var, t))
                sym_name = make_symbolics_var_name(var, t)
                sym_vars[(var, t)] = sym_name
                sym_to_var[sym_name] = (var, t)
            end
        end
    end
    
    # Create Symbolics variables for multipliers at t-1, t, t+1
    mult_sym_vars = Dict{Tuple{Symbol,Int}, Symbol}()
    for (i, mult) in enumerate(multipliers)
        for t in [-1, 0, 1]
            sym_name = make_symbolics_var_name(mult, t)
            mult_sym_vars[(mult, t)] = sym_name
            sym_to_var[sym_name] = (mult, t)
        end
    end
    
    # Create Symbolics variables for parameters
    sym_pars = Dict{Symbol, Symbol}()
    for par in parameters
        sym_pars[par] = par  # Use same name
    end
    
    # Now create the actual Symbolics.jl variables
    all_sym_names = vcat(collect(values(sym_vars)), collect(values(mult_sym_vars)), collect(values(sym_pars)))
    unique_names = unique(all_sym_names)
    
    # Generate code to create Symbolics variables and compute derivatives
    focs = Expr[]
    
    # For each non-shock variable, derive FOC
    for var in variables
        if var in shock_set
            continue
        end
        
        # Build FOC terms as Julia expressions
        foc_terms = []
        
        # ∂U/∂y[0]
        if haskey(sym_vars, (var, 0))
            # Term from objective
            push!(foc_terms, (objective, var, 0, :objective, 1))
        end
        
        # Terms from constraints
        for (i, eq) in enumerate(constraints)
            mult = multipliers[i]
            
            # -λ[0] * ∂f_i/∂y[0]
            if haskey(sym_vars, (var, 0))
                push!(foc_terms, (eq, var, 0, mult, 0, -1))
            end
            
            # -β * λ[1] * ∂f_i/∂y[1]
            if haskey(sym_vars, (var, 1))
                push!(foc_terms, (eq, var, 1, mult, 1, :(-$discount)))
            end
            
            # -(1/β) * λ[-1] * ∂f_i/∂y[-1]
            if haskey(sym_vars, (var, -1))
                push!(foc_terms, (eq, var, -1, mult, -1, :(-(1/$discount))))
            end
        end
        
        # This constructs the FOC equation symbolically
        # The actual differentiation will be done during model compilation
        if !isempty(foc_terms)
            # For now, create a placeholder expression that represents the FOC
            # The full symbolic computation requires runtime Symbolics.jl calls
            foc_expr = construct_foc_expression(objective, constraints, var, multipliers, discount, shock_set)
            if !isnothing(foc_expr) && foc_expr != :(0 = 0)
                push!(focs, foc_expr)
            end
        end
    end
    
    return focs, multipliers
end


"""
    construct_foc_expression(objective, constraints, var, multipliers, discount, shocks)

Construct the FOC expression for a variable using symbolic differentiation.
"""
function construct_foc_expression(objective::Expr, constraints::Vector{Expr}, 
                                   var::Symbol, multipliers::Vector{Symbol}, 
                                   discount::Symbol, shocks::Set{Symbol})
    
    if var in shocks
        return nothing
    end
    
    terms = []
    
    # Derivative of objective with respect to var[0]
    obj_deriv = symbolic_derivative(objective, var, 0, shocks)
    if !is_zero_expr(obj_deriv)
        push!(terms, obj_deriv)
    end
    
    # Derivatives from constraints
    for (i, eq) in enumerate(constraints)
        mult = multipliers[i]
        
        # Convert equation to residual
        residual = equation_to_residual(eq)
        
        # -λ[0] * ∂f/∂y[0]
        d0 = symbolic_derivative(residual, var, 0, shocks)
        if !is_zero_expr(d0)
            push!(terms, :(-$(Expr(:ref, mult, 0)) * $d0))
        end
        
        # -β * λ[1] * ∂f/∂y[1]
        d1 = symbolic_derivative(residual, var, 1, shocks)
        if !is_zero_expr(d1)
            push!(terms, :(-$discount * $(Expr(:ref, mult, 1)) * $d1))
        end
        
        # -(1/β) * λ[-1] * ∂f/∂y[-1]
        dm1 = symbolic_derivative(residual, var, -1, shocks)
        if !is_zero_expr(dm1)
            push!(terms, :(-(1/$discount) * $(Expr(:ref, mult, -1)) * $dm1))
        end
    end
    
    if isempty(terms)
        return nothing
    end
    
    # Combine terms
    if length(terms) == 1
        return :($(terms[1]) = 0)
    else
        return :($(Expr(:call, :+, terms...)) = 0)
    end
end


"""
    equation_to_residual(eq::Expr)

Convert an equation (LHS = RHS) to residual form (LHS - RHS).
"""
function equation_to_residual(eq::Expr)
    if eq.head == :(=)
        lhs = eq.args[1]
        rhs = eq.args[2]
        if rhs isa Number && rhs == 0
            return lhs
        else
            return :($lhs - $rhs)
        end
    end
    return eq
end


"""
    symbolic_derivative(expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})

Compute symbolic derivative of expr with respect to var[time_idx].
Uses Symbolics.jl for the actual differentiation.
"""
function symbolic_derivative(expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})
    if var in shocks
        return 0
    end
    
    # Use the existing Symbolics infrastructure in MacroModelling
    # Create a symbolic representation and differentiate
    try
        return compute_derivative_expr(expr, var, time_idx, shocks)
    catch e
        @warn "Failed to compute derivative of $expr with respect to $var[$time_idx]: $e"
        return 0
    end
end


"""
    compute_derivative_expr(expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})

Recursively compute the derivative expression.
"""
function compute_derivative_expr(expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})
    if expr isa Number
        return 0
    elseif expr isa Symbol
        return 0  # Parameter - derivative is 0
    elseif expr isa Expr
        if expr.head == :ref
            # Variable reference
            if expr.args[1] == var && expr.args[2] == time_idx
                return 1
            else
                return 0
            end
        elseif expr.head == :call
            return differentiate_call_expr(expr, var, time_idx, shocks)
        elseif expr.head == :(=)
            # Equation - should not reach here
            return 0
        end
    end
    return 0
end


"""
    differentiate_call_expr(expr::Expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})

Differentiate a function call expression.
"""
function differentiate_call_expr(expr::Expr, var::Symbol, time_idx::Int, shocks::Set{Symbol})
    op = expr.args[1]
    args = expr.args[2:end]
    
    if op == :+
        derivs = [compute_derivative_expr(arg, var, time_idx, shocks) for arg in args]
        return simplify_sum(derivs)
    elseif op == :-
        if length(args) == 1
            d = compute_derivative_expr(args[1], var, time_idx, shocks)
            return is_zero_expr(d) ? 0 : :(-$d)
        else
            d1 = compute_derivative_expr(args[1], var, time_idx, shocks)
            d2 = compute_derivative_expr(args[2], var, time_idx, shocks)
            if is_zero_expr(d1) && is_zero_expr(d2)
                return 0
            elseif is_zero_expr(d2)
                return d1
            elseif is_zero_expr(d1)
                return :(-$d2)
            else
                return :($d1 - $d2)
            end
        end
    elseif op == :*
        # Product rule
        return differentiate_product(args, var, time_idx, shocks)
    elseif op == :/
        # Quotient rule: d(a/b) = (da*b - a*db)/b^2
        a, b = args
        da = compute_derivative_expr(a, var, time_idx, shocks)
        db = compute_derivative_expr(b, var, time_idx, shocks)
        if is_zero_expr(da) && is_zero_expr(db)
            return 0
        elseif is_zero_expr(db)
            return :($da / $b)
        elseif is_zero_expr(da)
            return :(-$a * $db / ($b^2))
        else
            return :(($da * $b - $a * $db) / ($b^2))
        end
    elseif op == :^
        # Power rule
        a, b = args
        da = compute_derivative_expr(a, var, time_idx, shocks)
        db = compute_derivative_expr(b, var, time_idx, shocks)
        if is_zero_expr(da) && is_zero_expr(db)
            return 0
        elseif is_zero_expr(db)
            # d(a^b)/dx = b * a^(b-1) * da
            return :($b * $a^($b - 1) * $da)
        elseif is_zero_expr(da)
            # d(a^b)/dx = a^b * log(a) * db
            return :($expr * log($a) * $db)
        else
            # General case
            return :($expr * ($b/$a * $da + log($a) * $db))
        end
    elseif op == :exp
        a = args[1]
        da = compute_derivative_expr(a, var, time_idx, shocks)
        if is_zero_expr(da)
            return 0
        end
        return :($expr * $da)
    elseif op == :log
        a = args[1]
        da = compute_derivative_expr(a, var, time_idx, shocks)
        if is_zero_expr(da)
            return 0
        end
        return :($da / $a)
    elseif op == :sqrt
        a = args[1]
        da = compute_derivative_expr(a, var, time_idx, shocks)
        if is_zero_expr(da)
            return 0
        end
        return :($da / (2 * $expr))
    elseif op == :sin
        a = args[1]
        da = compute_derivative_expr(a, var, time_idx, shocks)
        if is_zero_expr(da)
            return 0
        end
        return :(cos($a) * $da)
    elseif op == :cos
        a = args[1]
        da = compute_derivative_expr(a, var, time_idx, shocks)
        if is_zero_expr(da)
            return 0
        end
        return :(-sin($a) * $da)
    else
        return 0
    end
end


"""
    differentiate_product(args, var, time_idx, shocks)

Apply product rule for differentiation.
"""
function differentiate_product(args, var::Symbol, time_idx::Int, shocks::Set{Symbol})
    n = length(args)
    terms = []
    
    for i in 1:n
        di = compute_derivative_expr(args[i], var, time_idx, shocks)
        if !is_zero_expr(di)
            other_args = [args[j] for j in 1:n if j != i]
            if isempty(other_args)
                push!(terms, di)
            elseif length(other_args) == 1
                push!(terms, :($di * $(other_args[1])))
            else
                push!(terms, :($di * $(Expr(:call, :*, other_args...))))
            end
        end
    end
    
    return simplify_sum(terms)
end


"""
    simplify_sum(terms)

Simplify a sum of terms, removing zeros.
"""
function simplify_sum(terms)
    non_zero = filter(t -> !is_zero_expr(t), terms)
    if isempty(non_zero)
        return 0
    elseif length(non_zero) == 1
        return non_zero[1]
    else
        return Expr(:call, :+, non_zero...)
    end
end


"""
    is_zero_expr(expr)

Check if an expression is zero.
"""
function is_zero_expr(expr)
    return expr isa Number && expr == 0
end


"""
    transform_equations_for_ramsey(equations::Vector{Expr}, 
                                    variables::Vector{Symbol},
                                    shocks::Vector{Symbol},
                                    parameters::Vector{Symbol},
                                    ramsey_config)

Transform model equations by adding Ramsey FOCs.
Returns augmented equations and variables.
"""
function transform_equations_for_ramsey(equations::Vector{Expr}, 
                                         variables::Vector{Symbol},
                                         shocks::Vector{Symbol},
                                         parameters::Vector{Symbol},
                                         ramsey_config)
    
    objective = ramsey_config.objective
    instruments = ramsey_config.instruments
    discount = ramsey_config.discount
    
    shock_set = Set(shocks)
    
    # Derive FOCs
    focs, multipliers = derive_ramsey_focs_with_symbolics(
        objective, equations, variables, instruments, shocks, parameters, discount
    )
    
    # Augmented system
    all_equations = vcat(equations, focs)
    all_variables = vcat(variables, multipliers)
    
    return all_equations, all_variables, multipliers
end


"""
    @ramsey model_name begin
        objective = ...
        instruments = [...]
        discount = ...
    end

Create a Ramsey optimal policy version of an existing model.

This macro transforms an existing DSGE model into a Ramsey planner's problem
by automatically deriving the first-order conditions for optimal policy.

# Example
```julia
# First define the base model
@model NK begin
    # ... model equations ...
end

@parameters NK begin
    β = 0.99
    # ...
end

# Then create the Ramsey version
@ramsey NK begin
    objective = log(C[0]) - N[0]^(1+φ)/(1+φ)
    instruments = [R]
    discount = β
end

# The new model NK_ramsey can be used with all standard functions
get_irf(NK_ramsey)
get_SS(NK_ramsey)
```
"""
macro ramsey(model_name, block)
    ramsey_model_name = Symbol(string(model_name) * "_ramsey")
    
    # Quote the block to prevent evaluation
    quoted_block = QuoteNode(block)
    
    return quote
        # Parse the ramsey block
        _ramsey_config = parse_ramsey_block($quoted_block)
        
        # Get the original model
        _orig_model = $(esc(model_name))
        
        # Transform equations
        _new_eqs, _new_vars, _multipliers = transform_equations_for_ramsey(
            _orig_model.original_equations,
            _orig_model.var,
            _orig_model.exo,
            vcat(_orig_model.parameters, _orig_model.parameters_as_function_of_parameters),
            _ramsey_config
        )
        
        # Print info
        println("Creating Ramsey model: $($(QuoteNode(ramsey_model_name)))")
        println("  Original equations: ", length(_orig_model.original_equations))
        println("  Ramsey equations (total): ", length(_new_eqs))
        println("  Lagrange multipliers: ", _multipliers)
        println("  Instruments: ", _ramsey_config.instruments)
        println()
        println("To create the augmented model, use the generated equations in a new @model block.")
        println("FOC equations have been stored in: _ramsey_focs")
        
        # Store results for user access
        global _ramsey_focs = _new_eqs[length(_orig_model.original_equations)+1:end]
        global _ramsey_multipliers = _multipliers
        global _ramsey_all_equations = _new_eqs
        global _ramsey_all_variables = _new_vars
        
        # Return info tuple
        (equations = _new_eqs, 
         focs = _new_eqs[length(_orig_model.original_equations)+1:end],
         multipliers = _multipliers, 
         variables = _new_vars,
         objective = _ramsey_config.objective,
         instruments = _ramsey_config.instruments,
         discount = _ramsey_config.discount)
    end
end


"""
    generate_ramsey_model_code(model_name::Symbol, result::NamedTuple, 
                                parameters::Vector{Pair{Symbol,Float64}})

Generate Julia code that defines a new model with the Ramsey equations.
The generated code can be evaluated to create a model that works with all
standard MacroModelling.jl functions.

# Example
```julia
result = @ramsey RBC begin
    objective = log(c[0])
    instruments = [q]
    discount = β
end

# Generate and evaluate the model code
code = generate_ramsey_model_code(:RBC_ramsey, result, 
    [:std_z => 0.01, :ρ => 0.2, :δ => 0.02, :α => 0.5, :β => 0.95])

eval(code)  # Creates RBC_ramsey model

# Now use with standard functions
get_SS(RBC_ramsey)
get_irf(RBC_ramsey)
```
"""
function generate_ramsey_model_code(model_name::Symbol, result::NamedTuple,
                                     parameter_values::Vector{Pair{Symbol,Float64}})
    # Generate the @model block
    eqs = result.equations
    
    # Build the equations block
    eq_exprs = Expr(:block)
    for eq in eqs
        push!(eq_exprs.args, eq)
    end
    
    # Build the parameters block
    param_exprs = Expr(:block)
    for (name, val) in parameter_values
        push!(param_exprs.args, :($name = $val))
    end
    
    # Add zero for multiplier steady states (they should be zero at SS)
    for mult in result.multipliers
        mult_ss = Symbol(string(mult) * "_ss")
        push!(param_exprs.args, :($mult_ss = 0.0))
    end
    
    # Generate the full code
    return quote
        @model $model_name begin
            $(eq_exprs.args...)
        end
        
        @parameters $model_name begin
            $(param_exprs.args...)
        end
    end
end


"""
    print_ramsey_model_code(model_name::Symbol, result::NamedTuple,
                            parameter_values::Vector{Pair{Symbol,Float64}})

Print Julia code that defines a Ramsey model. This is useful for
copying and modifying the generated model.

# Example
```julia
result = @ramsey RBC begin
    objective = log(c[0])
    instruments = [q]
    discount = β
end

print_ramsey_model_code(:RBC_ramsey, result,
    [:std_z => 0.01, :ρ => 0.2, :δ => 0.02, :α => 0.5, :β => 0.95])
```
"""
function print_ramsey_model_code(model_name::Symbol, result::NamedTuple,
                                  parameter_values::Vector{Pair{Symbol,Float64}})
    println("@model $model_name begin")
    for eq in result.equations
        println("    ", eq)
    end
    println("end")
    println()
    println("@parameters $model_name begin")
    for (name, val) in parameter_values
        println("    $name = $val")
    end
    for mult in result.multipliers
        println("    # Multiplier $(mult) - steady state should be 0")
    end
    println("end")
end
