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
                
                remaining = something(rest, "")
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

@stable default_mode = "disable" begin

"""
    expr_contains(expr, sym::Symbol, pattern) -> Bool

Check if `expr` contains `sym` matching `pattern` (nothing = any timing).
"""
function expr_contains(expr, sym::Symbol, pattern)
    normalize_repr(x) = replace(string(x), "◖" => "{", "◗" => "}")
    sym_str = normalize_repr(sym)
    pattern_str = pattern === nothing ? "" : normalize_repr(pattern)

    found = Ref(false)
    postwalk(expr) do x
        if pattern === nothing
            # Match symbol anywhere (as ref base or standalone)
            if normalize_repr(x) == sym_str ||
               (x isa Expr && x.head == :ref && normalize_repr(x.args[1]) == sym_str)
                found[] = true
            end
        else
            # Match exact expression pattern
            normalize_repr(x) == pattern_str && (found[] = true)
        end
        x
    end
    found[]
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
    sym, pattern = parse_filter_term(filter)
    
    return [expr for expr in exprs if expr_contains(expr, sym, pattern)]
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
- `Vector{Expr}` of the calibration equations as expressions.

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
 :(k / (q * 4) - capital_to_output)
```
"""
function get_calibration_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{Expr}
    # Replace ◖/◗ with {/} in symbols within expression tree
    exprs = replace_curly_braces_in_symbols.(𝓂.equations.calibration)
    
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
    return [expr for (expr, orig) in zip(exprs, 𝓂.equations.calibration) if expr_contains(orig, sym, nothing)]
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

end # dispatch_doctor