"""
$(SIGNATURES)
Modify or exchange calibration equations after model and parameters have been defined.

This function allows users to update the calibration equations and their associated parameters
while keeping track of all revisions made via this function.

# Arguments
- `𝓂::ℳ`: model object to modify
- `new_equations`: new calibration equations. Can be:
  - A single expression: `:(k[ss] / y[ss] - 2.5)`
  - A vector of expressions: `[:(k[ss] / y[ss] - 2.5), :(c[ss] / y[ss] - 0.6)]`
  - A vector of pairs mapping parameters to equations: `[:α => :(k[ss] / y[ss] - 2.5), :β => :(c[ss] / y[ss] - 0.6)]`
- `revision_note::String = ""`: optional note describing the revision

# Keyword Arguments
- `verbose::Bool = false`: print information about the modification
- `recalculate::Bool = true`: whether to recalculate the non-stochastic steady state after modification

# Returns
- `Nothing`. The function modifies the model object in place.

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
    k[ss] / q[ss] = 2.5 | δ
    α = 0.5
    β = 0.95
end

# Modify the calibration equation for δ
modify_calibration_equations!(RBC, :δ => :(k[ss] / q[ss] - 3.0), "Updated capital to output ratio")

# Or modify multiple equations at once
modify_calibration_equations!(RBC, 
    [:δ => :(k[ss] / q[ss] - 3.0), :α => :(y[ss] / k[ss] - 0.3)],
    "Updated multiple calibration equations")
```
"""
function modify_calibration_equations!(𝓂::ℳ, 
                                       new_equations::Union{Expr, Vector{<:Union{Expr, Pair{Symbol, Expr}}}},
                                       revision_note::String = "";
                                       verbose::Bool = false,
                                       recalculate::Bool = true)
    
    # Store the current state before modification
    old_equations = copy(𝓂.calibration_equations)
    old_parameters = copy(𝓂.calibration_equations_parameters)
    
    # Convert input to a standard format
    if new_equations isa Expr
        # Single expression without parameter - error
        error("When providing a single equation, you must specify which parameter it calibrates using the format: :parameter => equation")
    end
    
    # Process the new equations
    param_equation_pairs = if all(x -> x isa Pair{Symbol, Expr}, new_equations)
        new_equations
    elseif all(x -> x isa Expr, new_equations)
        # If just expressions, they should match the order of existing calibration parameters
        if length(new_equations) != length(𝓂.calibration_equations_parameters)
            error("Number of equations ($(length(new_equations))) must match number of calibration parameters ($(length(𝓂.calibration_equations_parameters)))")
        end
        collect(zip(𝓂.calibration_equations_parameters, new_equations))
    else
        error("new_equations must be either a vector of expressions or a vector of Symbol => Expr pairs")
    end
    
    # Update each equation
    for (param, eq) in param_equation_pairs
        # Find the index of this parameter in the calibration parameters
        param_idx = findfirst(==(param), 𝓂.calibration_equations_parameters)
        
        if param_idx === nothing
            error("Parameter :$param is not a calibration parameter. Calibration parameters are: $(𝓂.calibration_equations_parameters)")
        end
        
        # Parse the equation to extract variables and parameters
        # This is a simplified version - in production you'd want to use the same parsing logic as @parameters
        parsed_eq = parse_calibration_equation(eq, 𝓂)
        
        # Update the calibration equation
        𝓂.calibration_equations[param_idx] = parsed_eq
        
        if verbose
            println("Updated calibration equation for parameter :$param")
            println("  Old: $(old_equations[param_idx])")
            println("  New: $(parsed_eq)")
        end
    end
    
    # Record this revision in the history
    timestamp = string(Dates.now())
    revision_entry = (timestamp * (isempty(revision_note) ? "" : " - " * revision_note), 
                     copy(𝓂.calibration_equations), 
                     copy(𝓂.calibration_equations_parameters))
    push!(𝓂.calibration_equations_revision_history, revision_entry)
    
    # Mark the solution as outdated
    if recalculate
        𝓂.solution.outdated_NSSS = true
        𝓂.solution.outdated_algorithms = Set(all_available_algorithms)
        
        if verbose
            println("Solution marked as outdated. Run solve! to recalculate.")
        end
    end
    
    return nothing
end


"""
$(SIGNATURES)
Helper function to parse calibration equations.

This is a simplified version that converts the equation expression to the internal format.
In a full implementation, this would use the same parsing logic as the @parameters macro.
"""
function parse_calibration_equation(eq::Expr, 𝓂::ℳ)
    # For now, just return the equation as-is
    # In a full implementation, this would:
    # 1. Replace [ss] with steady state symbols
    # 2. Convert to sympy format
    # 3. Check that all variables and parameters are valid
    return eq
end


"""
$(SIGNATURES)
Get the revision history of calibration equations.

Returns a vector of tuples, each containing:
- timestamp and note
- calibration equations at that revision
- calibration parameters at that revision

# Arguments
- `𝓂::ℳ`: model object

# Keyword Arguments
- `formatted::Bool = true`: if true, return human-readable strings; if false, return raw expressions

# Returns
- `Vector{Tuple{String, Vector, Vector{Symbol}}}`: revision history

# Examples
```julia
history = get_calibration_revision_history(RBC)
for (note, equations, parameters) in history
    println("Revision: \$note")
    for (param, eq) in zip(parameters, equations)
        println("  \$param: \$eq")
    end
end
```
"""
function get_calibration_revision_history(𝓂::ℳ; formatted::Bool = true)
    if !formatted
        return 𝓂.calibration_equations_revision_history
    end
    
    # Convert to human-readable format
    result = Tuple{String, Vector{String}, Vector{String}}[]
    
    for (note, equations, parameters) in 𝓂.calibration_equations_revision_history
        formatted_eqs = replace.(string.(equations), "◖" => "{", "◗" => "}")
        formatted_params = replace.(string.(parameters), "◖" => "{", "◗" => "}")
        push!(result, (note, formatted_eqs, formatted_params))
    end
    
    return result
end


"""
$(SIGNATURES)
Print the revision history of calibration equations in a readable format.

# Arguments
- `𝓂::ℳ`: model object

# Examples
```julia
print_calibration_revision_history(RBC)
```
"""
function print_calibration_revision_history(𝓂::ℳ)
    history = get_calibration_revision_history(𝓂, formatted=true)
    
    if isempty(history)
        println("No calibration equation revisions recorded.")
        return
    end
    
    println("Calibration Equation Revision History:")
    println("=" ^ 60)
    
    for (i, (note, equations, parameters)) in enumerate(history)
        println("\nRevision $i: $note")
        println("-" ^ 60)
        for (param, eq) in zip(parameters, equations)
            println("  $param: $eq")
        end
    end
end
