"""
$(SIGNATURES)
Modify or exchange calibration equations after model and parameters have been defined.

This function allows users to update the calibration equations and their associated parameters
while keeping track of all revisions made via this function.

**Note**: This function records the changes for tracking purposes, but modifying calibration 
equations after the model has been set up requires re-running the `@parameters` macro with 
the new equations to properly update the steady state solver. This function is primarily 
useful for documentation and tracking purposes.

# Arguments
- `𝓂::ℳ`: model object to modify
- `param_equation_pairs`: A vector of pairs mapping parameters to equations: 
  `[:α => :(k[ss] / y[ss] - 2.5), :β => :(c[ss] / y[ss] - 0.6)]`
- `revision_note::String = ""`: optional note describing the revision

# Keyword Arguments
- `verbose::Bool = false`: print information about the modification

# Returns
- `Nothing`. The function records the revision in the model object.

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

# Track a modification to the calibration equation for δ
# (Note: To actually apply this change, you need to re-run @parameters with the new equation)
modify_calibration_equations!(RBC, 
    [:δ => :(k[ss] / q[ss] - 3.0)],
    "Updated capital to output ratio target from 2.5 to 3.0")

# View the revision history
print_calibration_revision_history(RBC)
```
"""
function modify_calibration_equations!(𝓂::ℳ, 
                                       param_equation_pairs::Vector{<:Pair{Symbol, <:Any}},
                                       revision_note::String = "";
                                       verbose::Bool = false)
    
    # Validate input
    if isempty(param_equation_pairs)
        error("param_equation_pairs cannot be empty")
    end
    
    # Store equations for the revision history
    documented_equations = Expr[]
    documented_parameters = Symbol[]
    
    # Process each pair
    for (param, eq) in param_equation_pairs
        # Check if parameter exists in calibration parameters
        if param ∉ 𝓂.calibration_equations_parameters
            error("Parameter :$param is not a calibration parameter. Calibration parameters are: $(𝓂.calibration_equations_parameters)")
        end
        
        # Convert equation to Expr if needed
        eq_expr = eq isa Expr ? eq : :($(eq))
        
        push!(documented_equations, eq_expr)
        push!(documented_parameters, param)
        
        if verbose
            println("Documented revision for parameter :$param")
            println("  New target: $(eq_expr)")
        end
    end
    
    # Record this revision in the history
    timestamp = string(Dates.now())
    revision_entry = (timestamp * (isempty(revision_note) ? "" : " - " * revision_note), 
                     documented_equations, 
                     documented_parameters)
    push!(𝓂.calibration_equations_revision_history, revision_entry)
    
    if verbose
        println("\nRevision recorded. To apply these changes, re-run the @parameters macro with the new calibration equations.")
    end
    
    return nothing
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
