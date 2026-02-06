@stable default_mode = "disable" begin

get_symbols(ex::Symbol) = [ex]

get_symbols(ex::Real) = [ex]

get_symbols(ex::Int) = [ex]

function get_symbols(ex::Expr)
    par = Set{Symbol}()
    postwalk(x ->   
    x isa Expr ? 
        x.head == :(=) ?
            for i in x.args
                i isa Symbol ? 
                    push!(par,i) :
                x
            end :
        x.head == :call ? 
            for i in 2:length(x.args)
                x.args[i] isa Symbol ? 
                    push!(par,x.args[i]) : 
                x
            end : 
        x : 
    x, ex)
    return par
end

end # dispatch_doctor

"""
    replace_curly_braces_in_symbols(expr) -> Union{Expr, Symbol, Any}

Recursively traverse expression tree and convert ◖/◗ to proper curly brace syntax.
Transforms symbols like `Symbol("eps◖δ◗")` into `Expr(:curly, :eps, :δ)`.
Handles multiple curly braces like `a◖b◗◖c◗` → `Expr(:curly, Expr(:curly, :a, :b), :c)`.
"""
function replace_curly_braces_in_symbols(expr)
    if expr isa Symbol
        str = string(expr)
        if occursin("◖", str) && occursin("◗", str)
            # Process all ◖...◗ pairs iteratively from left to right
            result = nothing
            remaining = str
            
            while occursin("◖", remaining)
                # Match the pattern: base◖content◗rest
                m = match(r"^([^◖]*)◖([^◗]+)◗(.*)$", remaining)
                if m === nothing
                    break
                end
                
                base_str = m.captures[1]
                content = Symbol(m.captures[2])
                rest = m.captures[3]
                
                if result === nothing && !isempty(base_str)
                    # First iteration with a base
                    result = Expr(:curly, Symbol(base_str), content)
                elseif result === nothing
                    # First iteration without base (shouldn't happen normally)
                    result = content
                else
                    # Subsequent iterations: nest the curly expression
                    result = Expr(:curly, result, content)
                end
                
                remaining = rest
            end
            
            return result === nothing ? expr : result
        end
        return expr
    elseif expr isa Expr
        return Expr(expr.head, [replace_curly_braces_in_symbols(arg) for arg in expr.args]...)
    else
        return expr
    end
end

"""
    replace_dynamic_symbols(expr) -> Union{Expr, Symbol, Any}

Replace timing subscripts (₍₋₁₎, ₍₀₎, ₍₁₎, ₍ₓ₎) with bracket notation and convert ◖/◗ to curly braces.
Transforms symbols like `Symbol("z◖TFP◗₍₀₎")` into `Expr(:ref, Expr(:curly, :z, :TFP), 0)`.
"""
function replace_dynamic_symbols(expr)
    if expr isa Symbol
        str = string(expr)
        # First replace timing subscripts
        str = replace(replace(replace(replace(str, "₍₋₁₎" => "[-1]"), "₍₁₎" => "[1]"), "₍₀₎" => "[0]"), "₍ₓ₎" => "[x]")
        # Parse to handle timing indices, then apply curly brace conversion
        parsed = Meta.parse(str)
        return replace_curly_braces_in_symbols(parsed)
    elseif expr isa Expr
        return Expr(expr.head, [replace_dynamic_symbols(arg) for arg in expr.args]...)
    else
        return expr
    end
end

"""
    parse_filter_term(term::Union{Symbol, String}) -> (Symbol, Union{Expr, Nothing})

Parse a filter term into (base_symbol, pattern_expr).
- `:k` or `"k"` → `(:k, nothing)` matches variable at any timing
- `"k[-1]"` → `(:k, :(k[-1]))` matches exact timing
- `"eps[x]"` → `(:eps, :(eps[x]))` matches shock at exact timing
"""
function parse_filter_term(term::Union{Symbol, String})
    term_str = replace(string(term), "{" => "◖", "}" => "◗")
    m = match(r"^(.+)\[(.+)\]$", term_str)
    m === nothing && return (Symbol(term_str), nothing)
    return (Symbol(m.captures[1]), Meta.parse(term_str))
end

"""
    parse_dynamic_filter_term(term::Union{Symbol, String}) -> (Any, Union{Expr, Nothing})

Parse a filter term into a base expression and an optional timing pattern that
match the dynamic equation representation (bracket timing and curly braces).
"""
function parse_dynamic_filter_term(term::Union{Symbol, String})
    term_str = string(term)
    m = match(r"^(.+)\[(.+)\]$", term_str)
    if m === nothing
        base_expr = replace_curly_braces_in_symbols(Meta.parse(term_str))
        return (base_expr, nothing)
    end

    pattern_expr = replace_curly_braces_in_symbols(Meta.parse(term_str))
    base_expr = pattern_expr isa Expr && pattern_expr.head == :ref ? pattern_expr.args[1] : pattern_expr
    return (base_expr, pattern_expr)
end

"""
Helper function to normalize equation strings for comparison.
Removes whitespace variations and normalizes special characters.
"""
function normalize_equation_string(s::String)::String
    # Replace special characters used internally
    s = replace(s, "◖" => "{", "◗" => "}")
    # Remove all whitespace for comparison
    s = replace(s, r"\s+" => "")
    return s
end


"""
Normalize an equation expression by removing line-number nodes and collapsing
single-expression blocks introduced by parsing.
"""
function normalize_equation_expr(eq::Expr)
    cleaned_eq = rmlines(eq)

    normalized_eq = postwalk(cleaned_eq) do node
        if @capture(node, begin arg_ end)
            arg
        else
            node
        end
    end

    return normalized_eq
end


normalize_equation_input(eq::String) = normalize_equation_expr(Meta.parse(eq))
normalize_equation_input(eq::Expr) = normalize_equation_expr(eq)

@stable default_mode = "disable" begin

"""
    reset_solver_state!(𝓂::ℳ)

Reset the solver-related state of the model struct in preparation for reinitializing
the steady-state solver. This clears caches, NSSS solve blocks, and marks functions
as needing recompilation.

Used by `update_equations!`, `update_calibration_equations!`, and `add_calibration_equation!`
to ensure consistent state reset before regenerating solver functions.
"""
function reset_solver_state!(𝓂::ℳ)
    𝓂.caches = Caches()
    𝓂.NSSS.solve_blocks_in_place = ss_solve_block[]
    𝓂.NSSS.dependencies = nothing
    𝓂.functions.functions_written = false
    return nothing
end


"""
    expr_contains(expr, sym::Symbol, pattern) -> Bool

Check if `expr` contains `sym` matching `pattern` (nothing = any timing).
"""
function expr_contains(expr, sym::Symbol, pattern)
    found = Ref(false)
    postwalk(expr) do x
        if pattern === nothing
            # Match symbol anywhere (as ref base or standalone)
            if x === sym || (x isa Expr && x.head == :ref && x.args[1] === sym)
                found[] = true
            end
        else
            # Match exact expression pattern
            x == pattern && (found[] = true)
        end
        x
    end
    found[]
end

"""
    expr_contains_dynamic(expr, base_expr, pattern) -> Bool

Check if `expr` contains a base expression or exact timing pattern in dynamic equations.
This is aware of auxiliary variables with lead/lag superscripts.
"""
function expr_contains_dynamic(expr, base_expr, pattern)
    function base_matches(candidate, base)
        candidate == base && return true

        base_str = string(base)
        cand_str = string(candidate)

        cand_str == base_str && return true

        if startswith(cand_str, base_str)
            rest_index = nextind(cand_str, 1, length(base_str))
            if rest_index <= lastindex(cand_str)
                rest = cand_str[rest_index:end]
                return startswith(rest, "ᴸ⁽") && endswith(rest, "⁾")
            end
        end

        return false
    end

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
$(SIGNATURES)
Return the equations of the model. In case programmatic model writing was used this function returns the parsed equations (see loop over shocks in `Examples`).

# Arguments
- $MODEL®

# Keyword Arguments
- `filter` [Default: `nothing`, Type: `Union{Symbol, String, Nothing}`]: filter equations by variable name. Specify a variable name (e.g., `:k` or `"k"`) to return only equations containing that variable. Optionally include timing (e.g., `"k[-1]"` or `"eps[x]"`) to match exact timing.

# Returns
- `Vector{Expr}` of the parsed equations as expressions.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_equations(RBC)
# output
7-element Vector{Expr}:
 :(1 / c[0] = (β / c[1]) * (α * exp(z{TFP}[1]) * k[0] ^ (α - 1) + (1 - exp(z{δ}[1]) * δ)))
 :(c[0] + k[0] = (1 - exp(z{δ}[0]) * δ) * k[-1] + q[0])
 :(q[0] = exp(z{TFP}[0]) * k[-1] ^ α)
 :(z{TFP}[0] = ρ{TFP} * z{TFP}[-1] + σ{TFP} * (eps{TFP}[x] + eps_news{TFP}[x - 1]))
 :(z{δ}[0] = ρ{δ} * z{δ}[-1] + σ{δ} * (eps{δ}[x] + eps_news{δ}[x - 1]))
 :(Δc_share[0] = log(c[0] / q[0]) - log(c[-1] / q[-1]))
 :(Δk_4q[0] = log(k[0]) - log(k[-4]))
```
"""
function get_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{Expr}
    # Replace ◖/◗ with {/} in symbols within expression tree
    exprs = replace_curly_braces_in_symbols.(𝓂.equations.original)
    
    if filter === nothing
        return exprs
    end
    
    sym, pattern = parse_filter_term(filter)
    return [expr for (expr, orig) in zip(exprs, 𝓂.equations.original) if expr_contains(orig, sym, pattern)]
end


"""
$(SIGNATURES)
Write the current model equations and parameters to a Julia file in `@model`/`@parameters` syntax.

This is useful for exporting a model after multiple updates so the current equation set can be inspected
or versioned outside the running session.

# Arguments
- $MODEL®

# Keyword Arguments
- `path::AbstractString = ""`: Output path. Defaults to `<model_name>.jl` in the current directory.
- `overwrite::Bool = false`: Overwrite an existing file if `true`.

# Returns
- `String`: The path of the written file.
"""
function write_julia_model_file(𝓂::ℳ; path::AbstractString = "", overwrite::Bool = false)::String
    output_path = isempty(path) ? 𝓂.model_name * ".jl" : path

    if isfile(output_path) && !overwrite
        error("File already exists: $output_path. Pass overwrite = true to replace it.")
    end

    parameter_block = reconstruct_parameter_block(𝓂)

    open(output_path, "w") do io
        println(io, "using MacroModelling\n")
        println(io, "@model ", 𝓂.model_name, " begin")
        for eq in 𝓂.equations.original
            println(io, "    ", replace(string(eq), "◖" => "{", "◗" => "}"))
        end
        println(io, "end\n")

        println(io, "@parameters ", 𝓂.model_name, " begin")
        for line in parameter_block.args
            line isa LineNumberNode && continue
            println(io, "    ", replace(string(line), "◖" => "{", "◗" => "}"))
        end
        println(io, "end")
    end

    return output_path
end


"""
$(SIGNATURES)
Return the non-stochastic steady state (NSSS) equations of the model. The difference to the equations as they were written in the `@model` block is that exogenous shocks are set to `0`, time subscripts are eliminated (e.g. `c[-1]` becomes `c`), trivial simplifications are carried out (e.g. `log(k) - log(k) = 0`), and auxiliary variables are added for expressions that cannot become negative. 

Auxiliary variables facilitate the solution of the NSSS problem. The package substitutes expressions which cannot become negative with auxiliary variables and adds another equation to the system of equations determining the NSSS. For example, `log(c/q)` cannot be negative and `c/q` is substituted by an auxiliary variable `➕₁` and an additional equation is added: `➕₁ = c / q`.

Note that the output assumes the equations are equal to 0. As in, `-z{δ} * ρ{δ} + z{δ}` implies `-z{δ} * ρ{δ} + z{δ} = 0` and therefore: `z{δ} * ρ{δ} = z{δ}`.

# Arguments
- $MODEL®

# Keyword Arguments
- `filter` [Default: `nothing`, Type: `Union{Symbol, String, Nothing}`]: filter equations by variable name. Specify a variable name (e.g., `:k` or `"k"`) to return only equations containing that variable. Time subscripts are ignored for steady state equations.

# Returns
- `Vector{Expr}` of the NSSS equations as expressions.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_steady_state_equations(RBC)
# output
9-element Vector{Expr}:
 :((-β * ((k ^ (α - 1) * α * exp(z{TFP}) - δ * exp(z{δ})) + 1)) / c + 1 / c)
 :(((c - k * (-δ * exp(z{δ}) + 1)) + k) - q)
 :(-(k ^ α) * exp(z{TFP}) + q)
 :(-z{TFP} * ρ{TFP} + z{TFP})
 :(-z{δ} * ρ{δ} + z{δ})
 :(➕₁ - c / q)
 :(➕₂ - c / q)
 :((Δc_share - log(➕₁)) + log(➕₂))
 :(Δk_4q - 0)
```
"""
function get_steady_state_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{Expr}
    # Replace ◖/◗ with {/} in symbols within expression tree
    exprs = replace_curly_braces_in_symbols.(𝓂.equations.steady_state_aux)
    
    if filter === nothing
        return exprs
    end
    
    sym, pattern = parse_filter_term(filter)
    
    # Warn if time subscript provided
    if pattern !== nothing
        @info "Time subscript in filter will be ignored for steady state equations. Equations containing the variable '$(sym)' will be returned regardless of timing."
    end
    
    # Always ignore timing for steady state equations (no time subscripts in SS)
    return [expr for (expr, orig) in zip(exprs, 𝓂.equations.steady_state_aux) if expr_contains(orig, sym, nothing)]
end


"""
$(SIGNATURES)
Return the augmented system of equations describing the model dynamics. Augmented means that, when variables have leads or lags with absolute value larger than 1, or exogenous shocks have leads or lags, auxiliary equations containing lead/lag variables are added. The augmented system contains only variables in the present `[0]`, future `[1]`, or past `[-1]`. For example, `Δk_4q[0] = log(k[0]) - log(k[-3])` contains `k[-3]`. Introducing two auxiliary variables (`kᴸ⁽⁻¹⁾` and `kᴸ⁽⁻²⁾`, where `ᴸ` denotes the lead/lag operator) and augmenting the system with `kᴸ⁽⁻²⁾[0] = kᴸ⁽⁻¹⁾[-1]` and `kᴸ⁽⁻¹⁾[0] = k[-1]` ensures that all timing indices have absolute value at most 1: `Δk_4q[0] - (log(k[0]) - log(kᴸ⁽⁻²⁾[-1]))`.

In case programmatic model writing was used this function returns the parsed equations (see loop over shocks in example).

Note that the output assumes the equations are equal to 0. As in, `kᴸ⁽⁻¹⁾[0] - k[-1]` implies `kᴸ⁽⁻¹⁾[0] - k[-1] = 0` and therefore: `kᴸ⁽⁻¹⁾[0] = k[-1]`.

# Arguments
- $MODEL®

# Keyword Arguments
- `filter` [Default: `nothing`, Type: `Union{Symbol, String, Nothing}`]: filter equations by variable name. Specify a variable name (e.g., `:k` or `"k"`) to return only equations containing that variable. Optionally include timing (e.g., `"k[-1]"` or `"eps[x]"`) to match exact timing.

# Returns
- `Vector{Expr}` of the dynamic model equations as expressions.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_dynamic_equations(RBC)
# output
12-element Vector{Expr}:
 :(1 / c[0] - (β / c[1]) * (α * exp(z{TFP}[1]) * k[0] ^ (α - 1) + (1 - exp(z{δ}[1]) * δ)))
 :((c[0] + k[0]) - ((1 - exp(z{δ}[0]) * δ) * k[-1] + q[0]))
 :(q[0] - exp(z{TFP}[0]) * k[-1] ^ α)
 :(eps_news{TFP}[0] - eps_news{TFP}[x])
 :(z{TFP}[0] - (ρ{TFP} * z{TFP}[-1] + σ{TFP} * (eps{TFP}[x] + eps_news{TFP}[-1])))
 :(eps_news{δ}[0] - eps_news{δ}[x])
 :(z{δ}[0] - (ρ{δ} * z{δ}[-1] + σ{δ} * (eps{δ}[x] + eps_news{δ}[-1])))
 :(Δc_share[0] - (log(c[0] / q[0]) - log(c[-1] / q[-1])))
 :(kᴸ⁽⁻³⁾[0] - kᴸ⁽⁻²⁾[-1])
 :(kᴸ⁽⁻²⁾[0] - kᴸ⁽⁻¹⁾[-1])
 :(kᴸ⁽⁻¹⁾[0] - k[-1])
 :(Δk_4q[0] - (log(k[0]) - log(kᴸ⁽⁻³⁾[-1])))
```
"""
function get_dynamic_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{Expr}
    exprs = replace_dynamic_symbols.(𝓂.equations.dynamic)
    
    if filter === nothing
        return exprs
    end
    
    # Parse filter term (uses user-friendly format with [-1], [0], etc.)
    base_expr, pattern = parse_dynamic_filter_term(filter)

    return [expr for expr in exprs if expr_contains_dynamic(expr, base_expr, pattern)]
end


"""
$(SIGNATURES)
Return the solve counters struct for the model.

# Arguments
- $MODEL®
"""
function get_solution_counts(𝓂::ℳ)::SolveCounters
    return 𝓂.counters
end


"""
    reprocess_model_equations!(𝓂, updated_original_equations; verbose, silent)

Internal function that reprocesses model equations after modifications.
Called by `update_equations!`, `add_equation!`, and `remove_equation!`.

This handles the common pattern of:
1. Processing the updated equation block
2. Updating constants, workspaces, equations
3. Resetting solver state
4. Processing parameter definitions
5. Setting up and solving the steady state
"""
function reprocess_model_equations!(𝓂::ℳ, 
                                    updated_original_equations::Vector{Expr};
                                    parameters::ParameterType = nothing,
                                    verbose::Bool = false,
                                    silent::Bool = true)
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end
    
    updated_block = Expr(:block, updated_original_equations...)
    parameter_block = reconstruct_parameter_block(𝓂)

    T, equations_struct = process_model_equations(
        updated_block,
        𝓂.constants.post_model_macro.max_obc_horizon,
        𝓂.constants.post_parameters_macro.precompile,
    )

    𝓂.constants = Constants(T)
    𝓂.workspaces = Workspaces()
    𝓂.equations = equations_struct

    reset_solver_state!(𝓂)
    
    parsed_parameters = process_parameter_definitions(
        parameter_block,
        𝓂.constants.post_model_macro
    )
    
    𝓂.constants.post_parameters_macro = update_post_parameters_macro(
        𝓂.constants.post_parameters_macro,
        parameters_as_function_of_parameters = parsed_parameters.calib_parameters_no_var,
        ss_calib_list = parsed_parameters.ss_calib_list,
        par_calib_list = parsed_parameters.par_calib_list,
        bounds = parsed_parameters.bounds
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
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters, parameter_overrides, verbose, silent)

Internal function that reprocesses calibration equations after modifications.
Called by `update_calibration_equations!`, `add_calibration_equation!`, and `remove_calibration_equation!`.

This handles the common pattern of:
1. Reconstructing the parameter block with updated calibration
2. Processing parameter definitions
3. Updating all calibration-related model fields
4. Resetting solver state
5. Setting up and solving the steady state
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

    parsed_parameters = process_parameter_definitions(
        parameter_block,
        𝓂.constants.post_model_macro,
    )

    𝓂.constants.post_parameters_macro = post_parameters_macro(
        parsed_parameters.calib_parameters_no_var,
        𝓂.constants.post_parameters_macro.precompile,
        𝓂.constants.post_parameters_macro.simplify,
        𝓂.constants.post_parameters_macro.symbolic,
        𝓂.constants.post_parameters_macro.guess,
        parsed_parameters.ss_calib_list,
        parsed_parameters.par_calib_list,
        parsed_parameters.bounds,
        𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
        𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime,
    )

    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
        var_axis = Symbol[],
        calib_axis = Symbol[],
    )
    
    𝓂.parameter_values = parsed_parameters.parameter_values

    reset_solver_state!(𝓂)

    finalize_model_update!(𝓂; verbose = verbose, silent = silent)
    
    return nothing
end


"""
    finalize_model_update!(𝓂; verbose, silent)

Internal function that finalizes a model update by setting up and solving the steady state.
Called by both `reprocess_model_equations!` and `reprocess_calibration_equations!`.
"""
function finalize_model_update!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    has_missing_parameters = !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
    missing_params = 𝓂.constants.post_complete_parameters.missing_parameters
    
    if !isnothing(𝓂.functions.NSSS_custom)
        write_ss_check_function!(𝓂)
    else
        if !has_missing_parameters
            set_up_steady_state_solver!(
                𝓂,
                verbose = verbose,
                silent = silent,
                avoid_solve = !𝓂.constants.post_parameters_macro.simplify,
                symbolic = 𝓂.constants.post_parameters_macro.symbolic,
            )
        end
    end

    if !has_missing_parameters
        opts = merge_calculation_options(verbose = verbose)
        solve_steady_state!(
            𝓂,
            opts,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
            𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime,
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
$(SIGNATURES)
Print the solve counters for the model in a human-readable format.

# Arguments
- $MODEL®
"""
function print_solution_counts(𝓂::ℳ)::Nothing
    counts = get_solution_counts(𝓂)

    println("Solve counters",
            "\n Steady state",
            "\n  Total:      ", counts.ss_solves_total,
            "\n  Failed:     ", counts.ss_solves_failed,
            "\n  Estimation",
            "\n   Total:     ", counts.ss_solves_total_estimation,
            "\n   Failed:    ", counts.ss_solves_failed_estimation,
            "\n First order",
            "\n  Total:      ", counts.first_order_solves_total,
            "\n  Failed:     ", counts.first_order_solves_failed,
            "\n  Estimation",
            "\n   Total:     ", counts.first_order_solves_total_estimation,
            "\n   Failed:    ", counts.first_order_solves_failed_estimation,
            "\n Second order",
            "\n  Total:      ", counts.second_order_solves_total,
            "\n  Failed:     ", counts.second_order_solves_failed,
            "\n  Estimation",
            "\n   Total:     ", counts.second_order_solves_total_estimation,
            "\n   Failed:    ", counts.second_order_solves_failed_estimation,
            "\n Third order",
            "\n  Total:      ", counts.third_order_solves_total,
            "\n  Failed:     ", counts.third_order_solves_failed,
            "\n  Estimation",
            "\n   Total:     ", counts.third_order_solves_total_estimation,
            "\n   Failed:    ", counts.third_order_solves_failed_estimation)

    return nothing
end


"""
$(SIGNATURES)
Return the calibration equations declared in the `@parameters` block. Calibration equations are additional equations which are part of the non-stochastic steady state problem. The additional equation is matched with a calibated parameter which is part of the equations declared in the `@model` block and can be retrieved with: `get_calibrated_parameters`

In case programmatic model writing was used this function returns the parsed equations (see loop over shocks in example).

Note that the output assumes the equations are equal to 0. As in, `k / (q * 4) - capital_to_output` implies `k / (q * 4) - capital_to_output = 0` and therefore: `k / (q * 4) = capital_to_output`.

# Arguments
- $MODEL®

# Keyword Arguments
- `filter` [Default: `nothing`, Type: `Union{Symbol, String, Nothing}`]: filter equations by variable name. Specify a variable name (e.g., `:k` or `"k"`) to return only equations containing that variable. Time subscripts (except `[ss]`) are ignored for calibration equations.

# Returns
- `Vector{Expr}` of the calibration equations as expressions (including the `|` calibrated parameter).

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_calibration_equations(RBC)
# output
1-element Vector{Expr}:
 :(k[ss] / (4 * q[ss]) = capital_to_output | δ)
```
"""
function get_calibration_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{Expr}
    parameter_block = reconstruct_parameter_block(𝓂)

    calibration_original = Expr[]
    for line in parameter_block.args
        line isa LineNumberNode && continue
        line isa Expr || continue
        extract_calibrated_parameter(line) === nothing && continue
        push!(calibration_original, line)
    end

    # Replace ◖/◗ with {/} in symbols within expression tree
    exprs = replace_curly_braces_in_symbols.(calibration_original)
    
    if filter === nothing
        return exprs
    end

    sym, pattern = parse_filter_term(filter)
    
    # Warn if time subscript provided (other than [ss] which is valid for calibration)
    if pattern !== nothing
        pattern_str = string(pattern)
        if !occursin("[ss]", pattern_str)
            @info "Time subscript in filter will be ignored for calibration equations. Equations containing the variable '$(sym)' will be returned regardless of timing."
        end
    end
    
    # Always ignore timing for calibration equations
    return [expr for (expr, orig) in zip(exprs, calibration_original) if expr_contains(orig, sym, nothing)]
end


"""
$(SIGNATURES)
Returns the parameters (and optionally the values) which have an impact on the model dynamics but do not depend on other parameters and are not determined by calibration equations. 

In case programmatic model writing was used this function returns the parsed parameters (see `σ` in `Examples`).

# Arguments
- $MODEL®
# Keyword Arguments
- `values` [Default: `false`, Type: `Bool`]: return the values together with the parameter names.

# Returns
- `Vector{String}` of the parameters or `Vector{Pair{String, Float64}}` of parameters and values if `values` is set to `true`.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_parameters(RBC)
# output
7-element Vector{String}:
 "σ{TFP}"
 "σ{δ}"
 "ρ{TFP}"
 "ρ{δ}"
 "capital_to_output"
 "alpha"
 "β"
```
"""
function get_parameters(𝓂::ℳ; values::Bool = false)::Union{Vector{Pair{String, Float64}},Vector{String}}
    if values
        return replace.(string.(𝓂.constants.post_complete_parameters.parameters), "◖" => "{", "◗" => "}") .=> 𝓂.parameter_values
    else
        return replace.(string.(𝓂.constants.post_complete_parameters.parameters), "◖" => "{", "◗" => "}")# |> sort
    end
end


"""
$(SIGNATURES)
Returns the parameters (and optionally the values) which are determined by a calibration equation. 

# Arguments
- $MODEL®
# Keyword Arguments
- `values` [Default: `false`, Type: `Bool`]: return the values together with the parameter names.

# Returns
- `Vector{String}` of the calibrated parameters or `Vector{Pair{String, Float64}}` of the calibrated parameters and values if `values` is set to `true`.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_calibrated_parameters(RBC)
# output
1-element Vector{String}:
 "δ"
```
"""
function get_calibrated_parameters(𝓂::ℳ; values::Bool = false)::Union{Vector{Pair{String, Float64}},Vector{String}}
    if values
        return replace.(string.(𝓂.equations.calibration_parameters), "◖" => "{", "◗" => "}") .=> 𝓂.caches.non_stochastic_steady_state[𝓂.constants.post_model_macro.nVars + 1:end]
    else
        return replace.(string.(𝓂.equations.calibration_parameters), "◖" => "{", "◗" => "}")# |> sort
    end
end


"""
$(SIGNATURES)
Returns the parameters which are required by the model but have not been assigned values in the `@parameters` block. These parameters must be provided via the `parameters` keyword argument in functions like `get_irf`, `get_SS`, `simulate`, etc. before the model can be solved.

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the missing parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC_incomplete begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_incomplete begin
    std_z = 0.01
    ρ = 0.2
    # Note: α, β, δ are not defined
end

get_missing_parameters(RBC_incomplete)
# output
3-element Vector{String}:
 "α"
 "β"
 "δ"
```
"""
function get_missing_parameters(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.constants.post_complete_parameters.missing_parameters), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Returns whether the model has missing parameters that need to be provided before solving.

# Arguments
- $MODEL®

# Returns
- `Bool` indicating whether the model has missing parameters.

# Examples
```jldoctest
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
end

has_missing_parameters(RBC)
# output
true
```
"""
function has_missing_parameters(𝓂::ℳ)::Bool
    !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
end


"""
$(SIGNATURES)
Returns the parameters contained in the model equations. Note that these parameters might be determined by other parameters or calibration equations defined in the `@parameters` block.

In case programmatic model writing was used this function returns the parsed parameters (see `σ` in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_parameters_in_equations(RBC)
# output
7-element Vector{String}:
 "α"
 "β"
 "δ"
 "ρ{TFP}"
 "ρ{δ}"
 "σ{TFP}"
 "σ{δ}"
```
"""
function get_parameters_in_equations(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.constants.post_model_macro.parameters_in_equations), "◖" => "{", "◗" => "}")# |> sort
end


"""
$(SIGNATURES)
Returns the parameters which are defined by other parameters which are not necessarily used in the equations of the model (see `α` in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_parameters_defined_by_parameters(RBC)
# output
1-element Vector{String}:
 "α"
```
"""
function get_parameters_defined_by_parameters(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters), "◖" => "{", "◗" => "}")# |> sort
end


"""
$(SIGNATURES)
Returns the parameters which define other parameters in the `@parameters` block which are not necessarily used in the equations of the model (see `alpha` in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_parameters_defining_parameters(RBC)
# output
1-element Vector{String}:
 "alpha"
```
"""
function get_parameters_defining_parameters(𝓂::ℳ)::Vector{String}
    replace.(string.(setdiff(𝓂.constants.post_complete_parameters.parameters, 𝓂.equations.calibration_parameters, 𝓂.constants.post_model_macro.parameters_in_equations, 𝓂.equations.calibration_parameters, 𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters, reduce(union, 𝓂.constants.post_parameters_macro.par_calib_list, init = []))), "◖" => "{", "◗" => "}")# |> sort
end


"""
$(SIGNATURES)
Returns the parameters used in calibration equations which are not used in the equations of the model (see `capital_to_output` in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_calibration_equation_parameters(RBC)
# output
1-element Vector{String}:
 "capital_to_output"
```
"""
function get_calibration_equation_parameters(𝓂::ℳ)::Vector{String}
    reduce(union, 𝓂.constants.post_parameters_macro.par_calib_list, init = []) |> collect |> sort  .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Returns the variables of the model without timing subscripts and not including auxiliary variables.

In case programmatic model writing was used this function returns the parsed variables (see `z` in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the variables.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_variables(RBC)
# output
7-element Vector{String}:
 "c"
 "k"
 "q"
 "z{TFP}"
 "z{δ}"
 "Δc_share"
 "Δk_4q"
```
"""
function get_variables(𝓂::ℳ)::Vector{String}
    setdiff(reduce(union,get_symbols.(𝓂.equations.steady_state_aux), init = []), union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars)) |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Returns the auxiliary variables, without timing subscripts, added to the non-stochastic steady state problem because certain expression cannot be negative (e.g. given `log(c/q)` an auxiliary variable is created for `c/q`).

See `get_steady_state_equations` for more details on the auxiliary variables and equations.

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the auxiliary parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_nonnegativity_auxiliary_variables(RBC)
# output
2-element Vector{String}:
 "➕₁"
 "➕₂"
```
"""
function get_nonnegativity_auxiliary_variables(𝓂::ℳ)::Vector{String}
    𝓂.constants.post_model_macro.➕_vars |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Returns the auxiliary variables, without timing subscripts, part of the augmented system of equations describing the model dynamics. Augmented means that, in case of variables with leads or lags larger than 1, or exogenous shocks with leads or lags, the system is augemented by auxiliary variables containing variables or shocks in lead or lag. Because the original equations included variables with leads or lags certain expression cannot be negative (e.g. given `log(c/q)` an auxiliary variable is created for `c/q`).

See `get_dynamic_equations` for more details on the auxiliary variables and equations.

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the auxiliary parameters.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_dynamic_auxiliary_variables(RBC)
# output
3-element Vector{String}:
 "kᴸ⁽⁻²⁾"
 "kᴸ⁽⁻³⁾"
 "kᴸ⁽⁻¹⁾"
```
"""
function get_dynamic_auxiliary_variables(𝓂::ℳ)::Vector{String}
    𝓂.constants.post_model_macro.aux |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end



"""
$(SIGNATURES)
Returns the exogenous shocks.

In case programmatic model writing was used this function returns the parsed variables (see `eps` in example).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the exogenous shocks.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_shocks(RBC)
# output
4-element Vector{String}:
 "eps_news{TFP}"
 "eps_news{δ}"
 "eps{TFP}"
 "eps{δ}"
```
"""
function get_shocks(𝓂::ℳ)::Vector{String}
    𝓂.constants.post_model_macro.exo |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end




"""
$(SIGNATURES)
Returns the state variables of the model. State variables occur in the past and not in the future or occur in all three: past, present, and future.

In case programmatic model writing was used this function returns the parsed variables (see `z` in example).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the state variables.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_state_variables(RBC)
# output
10-element Vector{String}:
 "c"
 "eps_news{TFP}"
 "eps_news{δ}"
 "k"
 "kᴸ⁽⁻²⁾"
 "kᴸ⁽⁻³⁾"
 "kᴸ⁽⁻¹⁾"
 "q"
 "z{TFP}"
 "z{δ}"
```
"""
function get_state_variables(𝓂::ℳ)::Vector{String}
    𝓂.constants.post_model_macro.past_not_future_and_mixed |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end



"""
$(SIGNATURES)
Returns the jump variables of the model. Jump variables occur in the future and not in the past or occur in all three: past, present, and future.

In case programmatic model writing was used this function returns the parsed variables (see `z` in example).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the jump variables.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z{TFP}[1]) * k[0]^(α - 1) + (1 - exp(z{δ}[1]) * δ))
    c[0] + k[0] = (1 - exp(z{δ}[0])δ) * k[-1] + q[0]
    q[0] = exp(z{TFP}[0]) * k[-1]^α
    for shock in [TFP, δ]
        z{shock}[0] = ρ{shock} * z{shock}[-1] + σ{shock} * (eps{shock}[x] + eps_news{shock}[x-1])
    end
    Δc_share[0] = log(c[0]/q[0]) - log(c[-1]/q[-1])
    Δk_4q[0] = log(k[0]) - log(k[-4])
end

@parameters RBC begin
    σ = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

get_jump_variables(RBC)
# output
3-element Vector{String}:
 "c"
 "z{TFP}"
 "z{δ}"
```
"""
function get_jump_variables(𝓂::ℳ)::Vector{String}
    𝓂.constants.post_model_macro.future_not_past_and_mixed |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Modify or replace a model equation. This function updates the model equations and automatically re-parses, re-solves, and recomputes derivatives.

# Arguments
- $MODEL®
- `old_equation_or_index::Union{Int, Expr, String}`: Either the index of the equation to modify (1-based), or the old equation itself (as `Expr` or `String`) to be matched and replaced. Use `get_equations` to see current equations.
- `new_equation::Union{Expr, String}`: The new equation to replace the old one. Can be an `Expr` or a `String` that will be parsed.

# Keyword Arguments
- $PARAMETERS®
- `verbose` [Default: `false`, Type: `Bool`]: Print detailed information about the update process.
- `silent` [Default: `true`, Type: `Bool`]: Suppress all output during reprocessing.

# Returns
- `Nothing`. The model is modified in place and re-solved.

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

# Get current equations
get_equations(RBC)

# Update the third equation by index
update_equations!(RBC, 3, :(q[0] = exp(z[0]) * k[-1]^α + 0.01))

# Or update by specifying the old equation
update_equations!(RBC, :(q[0] = exp(z[0]) * k[-1]^α + 0.01), :(q[0] = exp(z[0]) * k[-1]^α + 0.02))

# The model is automatically re-solved - verify with:
get_steady_state(RBC)

# View revision history
get_revision_history(RBC)
```
"""
function update_equations!(𝓂::ℳ, 
                          old_equation_or_index::Union{Int, Expr, String}, 
                          new_equation::Union{Expr, String}; 
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    # Determine equation index
    if old_equation_or_index isa Int
        equation_index = old_equation_or_index
        n_equations = length(𝓂.equations.original)
        @assert 1 <= equation_index <= n_equations "Equation index must be between 1 and $n_equations. Use `get_equations(model)` to see current equations."
    else
        # Parse string to Expr if needed
        old_eq_to_match = normalize_equation_input(old_equation_or_index)
        
        # Find the equation index by matching
        equation_index = find_equation_index(𝓂.equations.original, old_eq_to_match)
        @assert equation_index !== nothing "Could not find equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}")). Use `get_equations(model)` to see current equations."
    end
    
    # Parse string to Expr if needed
    parsed_new_equation = normalize_equation_input(new_equation)
    
    # Store old equation for revision history
    old_equation = 𝓂.equations.original[equation_index]
    
    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :update_equation,
        equation_index = equation_index,
        old_equation = old_equation,
        new_equation = parsed_new_equation
    )
    push!(𝓂.revision_history, revision_entry)
    
    # Update the original equation
    updated_original_equations = copy(𝓂.equations.original)
    updated_original_equations[equation_index] = parsed_new_equation
    
    if verbose
        println("Updated equation $equation_index:")
        println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
        println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
    end
    
    # Re-process the model equations using common function
    reprocess_model_equations!(𝓂, updated_original_equations; 
                                parameters = parameters,
                                verbose = verbose, 
                                silent = silent)
    
    return nothing
end

"""
$(SIGNATURES)
Alias for [`update_equations!`](@ref).
"""
replace_equations! = update_equations!


# Type aliases for equation input types
const EquationInput = Union{Expr, String}
const EquationOrIndex = Union{Int, Expr, String}

"""
$(SIGNATURES)
Update multiple model equations at once. The model is only re-solved once after all updates.

# Arguments
- $MODEL®
- `updates::Union{Vector, Tuple}`: A collection of update specifications. Each element can be:
  - `(index::Int, new_equation)`: Update equation at index with new_equation
  - `(old_equation, new_equation)`: Match and replace old_equation with new_equation
  - A `Pair` like `3 => :(new_eq)` or `:(old_eq) => :(new_eq)`

# Keyword Arguments  
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = true`: Suppress output during model re-solving

# Examples
```julia
# Update multiple equations by index
update_equations!(model, [
    (3, :(q[0] = exp(z[0]) * k[-1]^α * 1.1)),
    (4, :(z[0] = 0.3 * z[-1] + std_z * eps_z[x]))
])

# Using Pair syntax  
update_equations!(model, [
    3 => :(q[0] = exp(z[0]) * k[-1]^α * 1.1),
    :(z[0] = ρ * z[-1] + std_z * eps_z[x]) => :(z[0] = 0.3 * z[-1] + std_z * eps_z[x])
])
```
"""
function update_equations!(𝓂::ℳ, 
                          updates::Union{Vector, Tuple};
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    isempty(updates) && return nothing
    
    # Process updates to collect all equation changes
    n_equations = length(𝓂.equations.original)
    updated_original_equations = copy(𝓂.equations.original)
    
    for update in updates
        # Handle different input formats
        old_eq_or_idx, new_eq = if update isa Pair
            update.first, update.second
        elseif update isa Tuple && length(update) == 2
            update[1], update[2]
        else
            error("Invalid update format. Each update must be a tuple (old_eq_or_index, new_eq) or a Pair.")
        end
        
        # Determine equation index
        if old_eq_or_idx isa Int
            equation_index = old_eq_or_idx
            @assert 1 <= equation_index <= n_equations "Equation index must be between 1 and $n_equations."
        else
            old_eq_to_match = normalize_equation_input(old_eq_or_idx)
            equation_index = find_equation_index(updated_original_equations, old_eq_to_match)
            @assert equation_index !== nothing "Could not find equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}"))"
        end
        
        # Parse new equation
        parsed_new_eq = normalize_equation_input(new_eq)
        
        # Store old equation for revision history
        old_equation = updated_original_equations[equation_index]
        
        # Record the revision
        revision_entry = (
            timestamp = Dates.now(),
            action = :update_equation,
            equation_index = equation_index,
            old_equation = old_equation,
            new_equation = parsed_new_eq
        )
        push!(𝓂.revision_history, revision_entry)
        
        # Update the equation
        updated_original_equations[equation_index] = parsed_new_eq
        
        if verbose
            println("Updated equation $equation_index:")
            println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
            println("  New: ", replace(string(parsed_new_eq), "◖" => "{", "◗" => "}"))
        end
    end
    
    # Re-process the model equations once using common function
    reprocess_model_equations!(𝓂, updated_original_equations; 
                                parameters = parameters, 
                                verbose = verbose, 
                                silent = silent)
    
    return nothing
end


"""
$(SIGNATURES)
Add a new equation to the model.

This function adds a new equation to an existing model and automatically re-solves it.
The model structure (variables, parameters, shocks) is updated to reflect any new symbols
introduced by the equation.

# Arguments
- $MODEL®
- `new_equation::Union{Expr, String}`: The new equation to add. Can be either a Julia expression
  using the `:(...)` syntax or a string that will be parsed.

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information about the update process
- `silent::Bool = true`: Suppress output during model re-solving

# Returns
- `nothing`

# Examples
```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

# Add a new equation for investment
add_equation!(RBC, :(i[0] = k[0] - (1 - δ) * k[-1]))

# The model now has 5 equations instead of 4
get_equations(RBC)
```
"""
function add_equation!(𝓂::ℳ, 
                       new_equation::Union{Expr, String};
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    # Parse string to Expr if needed
    parsed_new_equation = normalize_equation_input(new_equation)

    # Validate the equation has proper structure (should be an assignment or equality)
    if !(parsed_new_equation isa Expr && parsed_new_equation.head in (:(=), :call))
        error("Invalid equation format. Equations should be in the form `lhs = rhs` or use comparison operators.")
    end

    if !silent
        println("\nAdding equation:")
        println("  Equation: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        println()
    end

    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :add_equation,
        equation_index = length(𝓂.equations.original) + 1,
        old_equation = nothing,
        new_equation = parsed_new_equation
    )
    push!(𝓂.revision_history, revision_entry)

    if verbose
        println("Adding equation $(length(𝓂.equations.original) + 1):")
        println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
    end

    # Update the original equations list with the new equation
    updated_original_equations = copy(𝓂.equations.original)
    push!(updated_original_equations, parsed_new_equation)

    # Re-process the model equations using common function
    reprocess_model_equations!(𝓂, updated_original_equations; 
                                parameters = parameters,
                                verbose = verbose, 
                                silent = silent)
    
    return nothing
end


"""
$(SIGNATURES)
Add multiple new equations to the model at once. The model is only re-solved once after all additions.

# Arguments
- $MODEL®
- `new_equations::Union{Vector, Tuple}`: A collection of equations to add. Each element can be
  an `Expr` or a `String` that will be parsed.

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = true`: Suppress output during model re-solving

# Examples
```julia
# Add multiple equations at once
add_equation!(model, [
    :(i[0] = k[0] - (1 - δ) * k[-1]),
    :(y[0] = q[0])
])
```
"""
function add_equation!(𝓂::ℳ, 
                       new_equations::Union{Vector, Tuple};
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    isempty(new_equations) && return nothing
    
    updated_original_equations = copy(𝓂.equations.original)
    
    for new_equation in new_equations
        # Parse string to Expr if needed
        parsed_new_equation = normalize_equation_input(new_equation)

        # Validate the equation has proper structure
        if !(parsed_new_equation isa Expr && parsed_new_equation.head in (:(=), :call))
            error("Invalid equation format. Equations should be in the form `lhs = rhs` or use comparison operators.")
        end

        if !silent
            println("\nAdding equation:")
            println("  Equation: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        end

        # Record the revision
        revision_entry = (
            timestamp = Dates.now(),
            action = :add_equation,
            equation_index = length(updated_original_equations) + 1,
            old_equation = nothing,
            new_equation = parsed_new_equation
        )
        push!(𝓂.revision_history, revision_entry)

        if verbose
            println("Adding equation $(length(updated_original_equations) + 1):")
            println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        end

        push!(updated_original_equations, parsed_new_equation)
    end

    # Re-process the model equations using common function
    reprocess_model_equations!(𝓂, updated_original_equations; 
                                parameters = parameters,
                                verbose = verbose, 
                                silent = silent)
    
    return nothing
end


"""
$(SIGNATURES)
Remove an equation from the model.

This function removes an equation from an existing model by index or by matching the equation expression,
and automatically re-solves the model. The model structure is updated accordingly.

# Arguments
- $MODEL®
- `equation_or_index::Union{Int, Expr, String}`: Either the 1-based index of the equation to remove,
  or the equation expression to match and remove.

# Keyword Arguments
- `verbose::Bool = false`: Print detailed information about the removal process
- `silent::Bool = true`: Suppress output during model re-solving

# Returns
- `nothing`

# Examples
```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
    i[0] = k[0] - (1 - δ) * k[-1]  # Extra equation to remove
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Remove by index
remove_equation!(RBC, 5)

# Or remove by matching the equation
remove_equation!(RBC, :(i[0] = k[0] - (1 - δ) * k[-1]))
```
"""
function remove_equation!(𝓂::ℳ, 
                          equation_or_index::Union{Int, Expr, String};
                          verbose::Bool = false,
                          silent::Bool = true)
    n_equations = length(𝓂.equations.original)
    
    # Determine equation index
    if equation_or_index isa Int
        equation_index = equation_or_index
        @assert 1 <= equation_index <= n_equations "Equation index must be between 1 and $n_equations. Use `get_equations(model)` to see current equations."
    else
        # Parse string to Expr if needed
        eq_to_match = normalize_equation_input(equation_or_index)
        
        # Find the equation index by matching
        equation_index = find_equation_index(𝓂.equations.original, eq_to_match)
        @assert equation_index !== nothing "Could not find equation matching: $(replace(string(eq_to_match), "◖" => "{", "◗" => "}")). Use `get_equations(model)` to see current equations."
    end
    
    # Ensure we have at least one equation remaining
    @assert n_equations > 1 "Cannot remove the last equation from the model."
    
    # Store the equation being removed for revision history
    removed_equation = 𝓂.equations.original[equation_index]
    
    if !silent
        println("\nRemoving equation $equation_index:")
        println("  Equation: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
        println()
    end

    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :remove_equation,
        equation_index = equation_index,
        old_equation = removed_equation,
        new_equation = nothing
    )
    push!(𝓂.revision_history, revision_entry)

    if verbose
        println("Removing equation $equation_index:")
        println("  Removed: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
    end

    # Create updated equations list without the removed equation
    updated_original_equations = copy(𝓂.equations.original)
    deleteat!(updated_original_equations, equation_index)

    # Re-process the model equations using common function
    reprocess_model_equations!(𝓂, updated_original_equations; verbose = verbose, silent = silent)
    
    return nothing
end


"""
$(SIGNATURES)
Remove multiple equations from the model at once. The model is only re-solved once after all removals.

# Arguments
- $MODEL®
- `equations_or_indices::Union{Vector, Tuple}`: A collection of equations to remove. Each element can be:
  - An `Int` index of the equation to remove
  - An `Expr` or `String` to match and remove

Note: When removing by index, indices are applied to the original equation list. If removing
multiple equations by index, provide them in descending order to avoid index shifting issues,
or use equation matching instead.

# Keyword Arguments
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = true`: Suppress output during model re-solving

# Examples
```julia
# Remove multiple equations by index (in descending order)
remove_equation!(model, [5, 4])

# Remove by matching equations
remove_equation!(model, [
    :(i[0] = k[0] - (1 - δ) * k[-1]),
    :(y[0] = q[0])
])
```
"""
function remove_equation!(𝓂::ℳ, 
                          equations_or_indices::Union{Vector, Tuple};
                          verbose::Bool = false,
                          silent::Bool = true)
    isempty(equations_or_indices) && return nothing
    
    updated_original_equations = copy(𝓂.equations.original)
    indices_to_remove = Int[]
    
    # First pass: collect all indices to remove
    for eq_or_idx in equations_or_indices
        if eq_or_idx isa Int
            equation_index = eq_or_idx
            @assert 1 <= equation_index <= length(𝓂.equations.original) "Equation index must be between 1 and $(length(𝓂.equations.original))."
            push!(indices_to_remove, equation_index)
        else
            eq_to_match = normalize_equation_input(eq_or_idx)
            equation_index = find_equation_index(𝓂.equations.original, eq_to_match)
            @assert equation_index !== nothing "Could not find equation matching: $(replace(string(eq_to_match), "◖" => "{", "◗" => "}"))"
            push!(indices_to_remove, equation_index)
        end
    end
    
    # Validate we're not removing all equations
    @assert length(𝓂.equations.original) - length(unique(indices_to_remove)) >= 1 "Cannot remove all equations from the model."
    
    # Sort indices in descending order to remove from end first
    sort!(indices_to_remove, rev=true)
    
    # Remove duplicates
    unique!(indices_to_remove)
    
    # Record revisions and remove equations
    for equation_index in indices_to_remove
        removed_equation = updated_original_equations[equation_index]
        
        if !silent
            println("\nRemoving equation $equation_index:")
            println("  Equation: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
        end

        revision_entry = (
            timestamp = Dates.now(),
            action = :remove_equation,
            equation_index = equation_index,
            old_equation = removed_equation,
            new_equation = nothing
        )
        push!(𝓂.revision_history, revision_entry)

        if verbose
            println("Removing equation $equation_index:")
            println("  Removed: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
        end

        deleteat!(updated_original_equations, equation_index)
    end

    # Re-process the model equations using common function
    reprocess_model_equations!(𝓂, updated_original_equations; verbose = verbose, silent = silent)
    
    return nothing
end


"""
Helper function to find the index of an equation by matching its expression.
Returns the index if found, or nothing if not found.
"""
function find_equation_index(equations::Vector{Expr}, target_eq::Expr)::Union{Int, Nothing}
    # Normalize the target equation for comparison
    target_str = normalize_equation_string(string(target_eq))
    
    for (i, eq) in enumerate(equations)
        eq_str = normalize_equation_string(string(eq))
        if eq_str == target_str
            return i
        end
    end
    
    return nothing
end



"""
$(SIGNATURES)
Modify or replace a calibration equation. This function updates the calibration equations and automatically re-parses, re-solves, and recomputes derivatives.

# Arguments
- $MODEL®
- `old_equation_or_index::Union{Int, Expr, String}`: Either the index of the calibration equation to modify (1-based), or the old equation itself (as `Expr` or `String`) to be matched and replaced. Use `get_calibration_equations` to see current calibration equations.
- `new_equation::Union{Expr, String}`: The new calibration equation to replace the old one. Can be an `Expr` or a `String` that will be parsed. The equation should be in the form `lhs = rhs` (e.g., `k[ss] / (4 * q[ss]) = 1.5`). When provided as `lhs = rhs | param`, the `param` portion determines which parameter is calibrated.

# Keyword Arguments
- $PARAMETERS®
- `verbose` [Default: `false`, Type: `Bool`]: Print detailed information about the update process.
- `silent` [Default: `true`, Type: `Bool`]: Suppress all output during reprocessing.

# Returns
- `Nothing`. The model is modified in place and re-solved.

# Examples
```julia
using MacroModelling

@model RBC_calibrated begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_calibrated begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / (4 * q[ss]) = 1.5 | α
    β = 0.95
end

# Get current calibration equations
get_calibration_equations(RBC_calibrated)

# Update the calibration equation by index
update_calibration_equations!(RBC_calibrated, 1, :(k[ss] / (4 * q[ss]) = 2.0 | α))

# The model is automatically re-solved - verify with:
get_steady_state(RBC_calibrated)

# View revision history
get_revision_history(RBC_calibrated)
```
"""
function update_calibration_equations!(𝓂::ℳ, 
                                       old_equation_or_index::Union{Int, Expr, String}, 
                                       new_equation::Union{Expr, String};
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    # Parse string to Expr if needed
    parsed_new_equation = normalize_equation_input(new_equation)

    # Determine equation index
    n_equations = length(𝓂.equations.calibration_original)
    @assert n_equations > 0 "Model has no calibration equations."

    if old_equation_or_index isa Int
        equation_index = old_equation_or_index
        @assert 1 <= equation_index <= n_equations "Calibration equation index must be between 1 and $n_equations. Use `get_calibration_equations(model)` to see current calibration equations."
    else
        old_eq_to_match = normalize_equation_input(old_equation_or_index)
        equation_index = find_equation_index(𝓂.equations.calibration_original, old_eq_to_match)
        @assert equation_index !== nothing "Could not find calibration equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}")). Use `get_calibration_equations(model)` to see current calibration equations."
    end

    # Store old equation for revision history
    old_equation = 𝓂.equations.calibration_original[equation_index]
    
    # Validate that the calibrated parameter (if present) is used in model equations
    new_calib_param = extract_calibrated_parameter(parsed_new_equation)
    if new_calib_param !== nothing
        model_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
        if new_calib_param ∉ model_params
            error("Cannot calibrate parameter `$new_calib_param`: it is not used in any model equation. " *
                  "Calibrated parameters must appear in the model equations. " *
                  "Available parameters: $(sort(collect(model_params)))")
        end
    end

    # Detect parameter switching and inform user
    old_calib_param = extract_calibrated_parameter(old_equation)
    old_calib_params = Set{Symbol}(𝓂.equations.calibration_parameters)
    new_calib_params = old_calib_param === nothing ? Set{Symbol}() : Set{Symbol}([old_calib_param])
    if new_calib_param !== nothing
        new_calib_params = Set{Symbol}([new_calib_param])
    end
    
    # Parameters becoming calibrated (were fixed, now calibrated)
    params_becoming_calibrated = setdiff(new_calib_params, old_calib_params)
    # Parameters no longer calibrated (were calibrated, now fixed)
    params_no_longer_calibrated = setdiff(old_calib_params, new_calib_params)
    
    if !silent && (!isempty(params_becoming_calibrated) || !isempty(params_no_longer_calibrated))
        println("\nCalibration parameter changes:")
        
        # Report parameters that are now calibrated (were fixed before)
        for param in params_becoming_calibrated
            # Find the previous fixed value
            param_idx = findfirst(==(param), 𝓂.constants.post_complete_parameters.parameters)
            if param_idx !== nothing
                old_value = 𝓂.parameter_values[param_idx]
                if !isnan(old_value)
                    println("  → `$param` is now CALIBRATED (was fixed at $old_value)")
                else
                    println("  → `$param` is now CALIBRATED")
                end
            else
                println("  → `$param` is now CALIBRATED")
            end
        end
        
        # Report parameters that are now fixed (were calibrated before)
        for param in params_no_longer_calibrated
            # Get the calibrated value from NSSS cache
            n_vars = 𝓂.constants.post_model_macro.nVars
            calib_param_list = 𝓂.equations.calibration_parameters
            idx = findfirst(==(param), calib_param_list)
            if idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state)
                ss_value = 𝓂.caches.non_stochastic_steady_state[n_vars + idx]
                println("  → `$param` is now FIXED at $ss_value (was calibrated)")
            else
                println("  → `$param` is now FIXED (was calibrated)")
            end
        end
        println()
    end

    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :update_calibration_equation,
        equation_index = equation_index,
        old_equation = old_equation,
        new_equation = parsed_new_equation
    )
    push!(𝓂.revision_history, revision_entry)

    if verbose
        println("Updated calibration equation $equation_index:")
        println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
        println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
    end

    # Reconstruct parameter block with updated calibration equation
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    updated_calibration_original[equation_index] = parsed_new_equation

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters = parameters, verbose = verbose, silent = silent)

    return nothing
end

"""
$(SIGNATURES)
Alias for [`update_calibration_equations!`](@ref).
"""
replace_calibration_equations! = update_calibration_equations!


"""
$(SIGNATURES)
Update multiple calibration equations at once. The model is only re-solved once after all updates.

# Arguments
- $MODEL®
- `updates::Union{Vector, Tuple}`: A collection of update specifications. Each element can be:
  - `(index::Int, new_equation)`: Update calibration equation at index with new_equation
  - `(old_equation, new_equation)`: Match and replace old_equation with new_equation
  - A `Pair` like `1 => :(new_eq)` or `:(old_eq) => :(new_eq)`

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = true`: Suppress output during model re-solving

# Examples
```julia
# Update multiple calibration equations
update_calibration_equations!(model, [
    (1, :(k[ss] / (4 * y[ss]) = 1.6 | α)),
    (2, :(c[ss] / y[ss] = 0.7 | β))
])
```
"""
function update_calibration_equations!(𝓂::ℳ, 
                                       updates::Union{Vector, Tuple};
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    isempty(updates) && return nothing
    
    n_equations = length(𝓂.equations.calibration_original)
    @assert n_equations > 0 "Model has no calibration equations."
    
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    
    for update in updates
        # Handle different input formats
        old_eq_or_idx, new_eq = if update isa Pair
            update.first, update.second
        elseif update isa Tuple && length(update) == 2
            update[1], update[2]
        else
            error("Invalid update format. Each update must be a tuple (old_eq_or_index, new_eq) or a Pair.")
        end
        
        # Parse new equation
        parsed_new_equation = normalize_equation_input(new_eq)
        
        # Determine equation index
        if old_eq_or_idx isa Int
            equation_index = old_eq_or_idx
            @assert 1 <= equation_index <= n_equations "Calibration equation index must be between 1 and $n_equations."
        else
            old_eq_to_match = normalize_equation_input(old_eq_or_idx)
            equation_index = find_equation_index(updated_calibration_original, old_eq_to_match)
            @assert equation_index !== nothing "Could not find calibration equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}"))"
        end
        
        # Validate calibrated parameter
        new_calib_param = extract_calibrated_parameter(parsed_new_equation)
        if new_calib_param !== nothing
            model_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
            if new_calib_param ∉ model_params
                error("Cannot calibrate parameter `$new_calib_param`: it is not used in any model equation.")
            end
        end
        
        # Store old equation for revision history
        old_equation = updated_calibration_original[equation_index]
        
        # Record the revision
        revision_entry = (
            timestamp = Dates.now(),
            action = :update_calibration_equation,
            equation_index = equation_index,
            old_equation = old_equation,
            new_equation = parsed_new_equation
        )
        push!(𝓂.revision_history, revision_entry)

        if verbose
            println("Updated calibration equation $equation_index:")
            println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
            println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        end
        
        updated_calibration_original[equation_index] = parsed_new_equation
    end

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters = parameters, verbose = verbose, silent = silent)

    return nothing
end


"""
$(SIGNATURES)
Add a new calibration equation to the model.

This function adds a calibration equation to a model, converting a fixed parameter to a calibrated one.
The calibration equation specifies a steady-state relationship that determines the parameter value.

# Arguments
- $MODEL®
- `new_equation::Union{Expr, String}`: The new calibration equation to add. Must use the format 
  `:(lhs = rhs | parameter)` where `parameter` is the parameter to be calibrated.

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information about the update process
- `silent::Bool = false`: Suppress all output including warnings

# Returns
- `nothing`

# Examples
```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
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

# Add a calibration equation - α will be calibrated to match a capital-output ratio
add_calibration_equation!(RBC, :(k[ss] / (4 * q[ss]) = 1.5 | α))
```
"""
function add_calibration_equation!(𝓂::ℳ, 
                                   new_equation::Union{Expr, String};
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = false)
    # Parse string to Expr if needed
    parsed_new_equation = normalize_equation_input(new_equation)

    # Validate that the calibrated parameter is specified
    new_calib_param = extract_calibrated_parameter(parsed_new_equation)
    if new_calib_param === nothing
        error("Calibration equation must specify a parameter to calibrate using the `| parameter` syntax. " *
              "Example: `:(k[ss] / (4 * y[ss]) = 1.5 | alpha)`")
    end

    # Validate that the calibrated parameter is used in model equations
    model_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
    if new_calib_param ∉ model_params
        error("Cannot calibrate parameter `$new_calib_param`: it is not used in any model equation. " *
              "Calibrated parameters must appear in the model equations. " *
              "Available parameters: $(sort(collect(model_params)))")
    end

    # Check if parameter is already calibrated
    if new_calib_param ∈ 𝓂.equations.calibration_parameters
        error("Parameter `$new_calib_param` is already calibrated. " *
              "Use `update_calibration_equations!` to modify existing calibration equations.")
    end

    # Get the previous fixed value for user notification
    param_idx = findfirst(==(new_calib_param), 𝓂.constants.post_complete_parameters.parameters)
    old_value = param_idx !== nothing ? 𝓂.parameter_values[param_idx] : NaN

    if !silent
        println("\nAdding calibration equation:")
        println("  Equation: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        if !isnan(old_value)
            println("  → `$new_calib_param` is now CALIBRATED (was fixed at $old_value)")
        else
            println("  → `$new_calib_param` is now CALIBRATED")
        end
        println()
    end

    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :add_calibration_equation,
        equation_index = length(𝓂.equations.calibration_original) + 1,
        old_equation = nothing,
        new_equation = parsed_new_equation
    )
    push!(𝓂.revision_history, revision_entry)

    if verbose
        println("Added calibration equation $(length(𝓂.equations.calibration_original) + 1):")
        println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
    end

    # Reconstruct parameter block with new calibration equation appended
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    push!(updated_calibration_original, parsed_new_equation)

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters = parameters, verbose = verbose, silent = silent)

    return nothing
end


"""
$(SIGNATURES)
Add multiple calibration equations to the model at once. The model is only re-solved once after all additions.

# Arguments
- $MODEL®
- `new_equations::Union{Vector, Tuple}`: A collection of calibration equations to add. Each equation 
  must use the format `:(lhs = rhs | parameter)`.

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = false`: Suppress output during model re-solving

# Examples
```julia
# Add multiple calibration equations
add_calibration_equation!(model, [
    :(k[ss] / (4 * y[ss]) = 1.5 | α),
    :(c[ss] / y[ss] = 0.7 | β)
])
```
"""
function add_calibration_equation!(𝓂::ℳ, 
                                   new_equations::Union{Vector, Tuple};
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = false)
    isempty(new_equations) && return nothing
    
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    
    for new_equation in new_equations
        # Parse string to Expr if needed
        parsed_new_equation = normalize_equation_input(new_equation)

        # Validate that the calibrated parameter is specified
        new_calib_param = extract_calibrated_parameter(parsed_new_equation)
        if new_calib_param === nothing
            error("Calibration equation must specify a parameter to calibrate using the `| parameter` syntax. " *
                  "Example: `:(k[ss] / (4 * y[ss]) = 1.5 | alpha)`")
        end

        # Validate that the calibrated parameter is used in model equations
        model_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
        if new_calib_param ∉ model_params
            error("Cannot calibrate parameter `$new_calib_param`: it is not used in any model equation. " *
                  "Available parameters: $(sort(collect(model_params)))")
        end

        # Check if parameter is already calibrated (including previous additions in this batch)
        existing_calib_params = Set{Symbol}(𝓂.equations.calibration_parameters)
        for eq in updated_calibration_original
            p = extract_calibrated_parameter(eq)
            p !== nothing && push!(existing_calib_params, p)
        end
        if new_calib_param ∈ existing_calib_params
            error("Parameter `$new_calib_param` is already calibrated. " *
                  "Use `update_calibration_equations!` to modify existing calibration equations.")
        end

        if !silent
            println("\nAdding calibration equation:")
            println("  Equation: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
            println("  → `$new_calib_param` is now CALIBRATED")
        end

        # Record the revision
        revision_entry = (
            timestamp = Dates.now(),
            action = :add_calibration_equation,
            equation_index = length(updated_calibration_original) + 1,
            old_equation = nothing,
            new_equation = parsed_new_equation
        )
        push!(𝓂.revision_history, revision_entry)

        if verbose
            println("Added calibration equation $(length(updated_calibration_original) + 1):")
            println("  New: ", replace(string(parsed_new_equation), "◖" => "{", "◗" => "}"))
        end

        push!(updated_calibration_original, parsed_new_equation)
    end

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; parameters = parameters, verbose = verbose, silent = silent)

    return nothing
end


"""
$(SIGNATURES)
Remove a calibration equation from the model.

This function removes a calibration equation from the model by index or by matching the equation expression.
The previously calibrated parameter becomes a regular (fixed) parameter, and the model is automatically re-solved.

# Arguments
- $MODEL®
- `equation_or_index::Union{Int, Expr, String}`: Either the 1-based index of the calibration equation to remove,
  or the calibration equation expression to match and remove.

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information about the removal process
- `silent::Bool = false`: Suppress all output including warnings

# Returns
- `nothing`

# Examples
```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    α = 0.5
    β = 0.95
    k[ss] / q[ss] = 10 | δ  # Calibration equation
end

# Remove by index - δ becomes a fixed parameter
remove_calibration_equation!(RBC, 1, parameters = :δ => 0.025)

# Or remove by matching (partial match on calibrated parameter)
remove_calibration_equation!(RBC, 1)  # Uses the current calibrated value
```
"""
function remove_calibration_equation!(𝓂::ℳ, 
                                      equation_or_index::Union{Int, Expr, String};
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = false)
    n_calib_equations = length(𝓂.equations.calibration_original)
    
    @assert n_calib_equations > 0 "Model has no calibration equations to remove."
    
    # Determine equation index
    if equation_or_index isa Int
        equation_index = equation_or_index
        @assert 1 <= equation_index <= n_calib_equations "Calibration equation index must be between 1 and $n_calib_equations. Use `get_calibration_equations(model)` to see current calibration equations."
    else
        # Parse string to Expr if needed
        eq_to_match = normalize_equation_input(equation_or_index)
        
        # Find the equation index by matching
        equation_index = find_equation_index(𝓂.equations.calibration_original, eq_to_match)
        @assert equation_index !== nothing "Could not find calibration equation matching: $(replace(string(eq_to_match), "◖" => "{", "◗" => "}")). Use `get_calibration_equations(model)` to see current calibration equations."
    end
    
    # Store the equation being removed
    removed_equation = 𝓂.equations.calibration_original[equation_index]
    
    # Get the calibrated parameter name
    calib_param = 𝓂.equations.calibration_parameters[equation_index]
    
    new_value = nothing
    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
        param_idx = findfirst(==(calib_param), 𝓂.constants.post_complete_parameters.parameters)
        if param_idx !== nothing
            candidate = 𝓂.parameter_values[param_idx]
            if !isnan(candidate)
                new_value = candidate
            end
        end
    end

    if new_value === nothing
        # Determine the value for the now-fixed parameter from current calibrated solution
        param_idx = findfirst(==(calib_param), 𝓂.constants.post_complete_parameters.SS_and_pars_names)
        if param_idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state) && param_idx <= length(𝓂.caches.non_stochastic_steady_state)
            new_value = 𝓂.caches.non_stochastic_steady_state[param_idx]
        else
            # Fallback: look up in the NSSS cache using calibration parameter index
            n_vars = 𝓂.constants.post_model_macro.nVars
            calib_param_list = 𝓂.equations.calibration_parameters
            calib_idx = findfirst(==(calib_param), calib_param_list)
            if calib_idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state) && (n_vars + calib_idx) <= length(𝓂.caches.non_stochastic_steady_state)
                new_value = 𝓂.caches.non_stochastic_steady_state[n_vars + calib_idx]
            else
                error("Could not determine current value for calibrated parameter `$calib_param`. Provide a value via `parameters`.")
            end
        end
    end

    if !silent
        println("\nRemoving calibration equation $equation_index:")
        println("  Equation: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
        println("  → `$calib_param` is now FIXED at $new_value (was calibrated)")
        println()
    end

    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :remove_calibration_equation,
        equation_index = equation_index,
        old_equation = removed_equation,
        new_equation = nothing
    )
    push!(𝓂.revision_history, revision_entry)

    if verbose
        println("Removing calibration equation $equation_index:")
        println("  Removed: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
        println("  Parameter $calib_param fixed at: $new_value")
    end

    # Create updated calibration equations list without the removed equation
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    deleteat!(updated_calibration_original, equation_index)

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; 
                                     parameters = parameters,
                                     parameter_overrides = Dict{Symbol, Float64}(calib_param => Float64(new_value)),
                                     verbose = verbose, silent = silent)

    return nothing
end


"""
$(SIGNATURES)
Remove multiple calibration equations from the model at once. The model is only re-solved once after all removals.

# Arguments
- $MODEL®
- `equations_or_indices::Union{Vector, Tuple}`: A collection of calibration equations to remove. Each element can be:
  - An `Int` index of the calibration equation to remove
  - An `Expr` or `String` to match and remove

# Keyword Arguments
- $PARAMETERS®
- `verbose::Bool = false`: Print detailed information
- `silent::Bool = false`: Suppress output during model re-solving

# Examples
```julia
# Remove multiple calibration equations by index
remove_calibration_equation!(model, [1, 2], parameters = Dict(:α => 0.33, :β => 0.99))

# Remove by matching equations
remove_calibration_equation!(model, [
    :(k[ss] / (4 * y[ss]) = 1.5 | α),
    :(c[ss] / y[ss] = 0.7 | β)
])
```
"""
function remove_calibration_equation!(𝓂::ℳ, 
                                      equations_or_indices::Union{Vector, Tuple};
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = false)
    isempty(equations_or_indices) && return nothing
    
    n_calib_equations = length(𝓂.equations.calibration_original)
    @assert n_calib_equations > 0 "Model has no calibration equations to remove."
    
    updated_calibration_original = copy(𝓂.equations.calibration_original)
    indices_to_remove = Int[]
    parameter_overrides = Dict{Symbol, Float64}()

    if parameters !== nothing
        write_parameters_input!(𝓂, parameters, verbose = verbose)
    end
    
    # First pass: collect all indices to remove
    for eq_or_idx in equations_or_indices
        if eq_or_idx isa Int
            equation_index = eq_or_idx
            @assert 1 <= equation_index <= n_calib_equations "Calibration equation index must be between 1 and $n_calib_equations."
            push!(indices_to_remove, equation_index)
        else
            eq_to_match = normalize_equation_input(eq_or_idx)
            equation_index = find_equation_index(𝓂.equations.calibration_original, eq_to_match)
            @assert equation_index !== nothing "Could not find calibration equation matching: $(replace(string(eq_to_match), "◖" => "{", "◗" => "}"))"
            push!(indices_to_remove, equation_index)
        end
    end
    
    # Sort indices in descending order for proper removal
    sort!(indices_to_remove, rev=true)
    unique!(indices_to_remove)
    
    # Process each removal and determine new values
    for equation_index in indices_to_remove
        removed_equation = 𝓂.equations.calibration_original[equation_index]
        calib_param = 𝓂.equations.calibration_parameters[equation_index]
        
        # Determine new value for the parameter
        new_val = nothing
        param_idx = findfirst(==(calib_param), 𝓂.constants.post_complete_parameters.parameters)
        if param_idx !== nothing
            candidate = 𝓂.parameter_values[param_idx]
            if !isnan(candidate)
                new_val = candidate
            end
        end

        if new_val === nothing
            # Get current calibrated value - first try SS_and_pars_names
            param_idx = findfirst(==(calib_param), 𝓂.constants.post_complete_parameters.SS_and_pars_names)
            if param_idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state)
                new_val = 𝓂.caches.non_stochastic_steady_state[param_idx]
            else
                # Fallback: look up in the NSSS cache using calibration parameter index
                n_vars = 𝓂.constants.post_model_macro.nVars
                calib_param_list = 𝓂.equations.calibration_parameters
                calib_idx = findfirst(==(calib_param), calib_param_list)
                if calib_idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state) && (n_vars + calib_idx) <= length(𝓂.caches.non_stochastic_steady_state)
                    new_val = 𝓂.caches.non_stochastic_steady_state[n_vars + calib_idx]
                else
                    error("Could not determine current value for calibrated parameter `$calib_param`. Provide a value via `parameters`.")
                end
            end
        end
        
        parameter_overrides[calib_param] = Float64(new_val)
        
        if !silent
            println("\nRemoving calibration equation $equation_index:")
            println("  Equation: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
            println("  → `$calib_param` is now FIXED at $new_val")
        end

        revision_entry = (
            timestamp = Dates.now(),
            action = :remove_calibration_equation,
            equation_index = equation_index,
            old_equation = removed_equation,
            new_equation = nothing
        )
        push!(𝓂.revision_history, revision_entry)

        if verbose
            println("Removing calibration equation $equation_index:")
            println("  Removed: ", replace(string(removed_equation), "◖" => "{", "◗" => "}"))
            println("  Parameter $calib_param fixed at: $new_val")
        end

        # Find the current index in the updated list
        current_idx = findfirst(eq -> eq == removed_equation, updated_calibration_original)
        if current_idx !== nothing
            deleteat!(updated_calibration_original, current_idx)
        end
    end

    # Re-process the calibration equations using common function
    reprocess_calibration_equations!(𝓂, updated_calibration_original; 
                                     parameters = parameters,
                                     parameter_overrides = parameter_overrides,
                                     verbose = verbose, silent = silent)

    return nothing
end


"""
$(SIGNATURES)
Get the revision history of the model, showing all modifications made via `update_equations!` and `update_calibration_equations!`.

# Arguments
- $MODEL®

# Returns
- `Vector` of named tuples containing the revision history. Each entry has:
  - `timestamp`: When the revision was made
  - `action`: The type of action (`:update_equation` or `:update_calibration_equation`)
  - `equation_index`: The index of the equation that was modified
  - `old_equation`: The equation before modification
  - `new_equation`: The equation after modification

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

# Make some modifications
update_equations!(RBC, 3, :(q[0] = exp(z[0]) * k[-1]^α + 0.01))

# View the revision history
get_revision_history(RBC)
```
"""
function get_revision_history(𝓂::ℳ)
    if isempty(𝓂.revision_history)
        println("No revisions have been made to this model.")
        return 𝓂.revision_history
    end
    
    println("Revision history for model: ", 𝓂.model_name)
    println("=" ^ 60)
    
    for (i, revision) in enumerate(𝓂.revision_history)
        println("\nRevision $i:")
        println("  Timestamp: ", revision.timestamp)
        println("  Action: ", revision.action)
        if revision.equation_index !== nothing
            println("  Equation index: ", revision.equation_index)
        end
        if revision.old_equation !== nothing
            println("  Old equation: ", replace(string(revision.old_equation), "◖" => "{", "◗" => "}"))
        end
        if revision.new_equation !== nothing
            println("  New equation: ", replace(string(revision.new_equation), "◖" => "{", "◗" => "}"))
        end
    end
    
    return 𝓂.revision_history
end

end # dispatch_doctor