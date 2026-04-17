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
        get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)
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


# =========================================================================
# Equation modification API
# =========================================================================

"""
    normalize_equation_input(eq)

Normalize a user-provided equation expression: strip line-number nodes and
collapse single-expression `begin ... end` blocks. Accepts `Expr` or
`String` (which is parsed first).
"""
normalize_equation_input(eq::String) = normalize_equation_input(Meta.parse(eq))
function normalize_equation_input(eq::Expr)
    cleaned = rmlines(eq)
    return postwalk(cleaned) do node
        if @capture(node, begin arg_ end)
            arg
        else
            node
        end
    end
end


"""
    expr_contains_dynamic(expr, base_expr, pattern) -> Bool

Check whether `expr` contains a dynamic reference whose base symbol matches
`base_expr`. When `pattern === nothing` any timing counts as a match; when
it is an `Expr` or `Symbol`, match exact equality.
"""
function expr_contains_dynamic(expr, base_expr, pattern)
    base_matches(candidate, base) = candidate == base ||
        string(candidate) == string(base)

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
    find_equation_index(equations::Vector{Expr}, target_eq::Expr) -> Union{Int, Nothing}

Return the 1-based index of `target_eq` inside `equations`, comparing via
the normalized textual form (so whitespace and `{}`/`◖◗` differences are
ignored). Returns `nothing` if no match is found.
"""
function find_equation_index(equations::Vector{Expr}, target_eq::Expr)::Union{Int, Nothing}
    target_str = _equation_canonical(target_eq)
    for (i, eq) in enumerate(equations)
        if _equation_canonical(eq) == target_str
            return i
        end
    end
    return nothing
end

function _equation_canonical(eq)::String
    s = string(eq)
    s = replace(s, "◖" => "{", "◗" => "}")
    s = replace(s, r"\s+" => "")
    return s
end

_revision_entry(action::Symbol;
                equation_index::Union{Int, Nothing} = nothing,
                old_equation::Union{Expr, Nothing} = nothing,
                new_equation::Union{Expr, Nothing} = nothing) =
    (timestamp = Dates.now(),
     action = action,
     equation_index = equation_index,
     old_equation = old_equation,
     new_equation = new_equation)


"""
$(SIGNATURES)
Return the recorded history of equation modifications for the model. Each
entry is a `NamedTuple` containing the `timestamp`, `action`,
`equation_index`, `old_equation`, and `new_equation` fields. The list is
append-only and ordered chronologically.
"""
function get_revision_history(𝓂::ℳ)::Vector{RevisionEntry}
    return copy(𝓂.revision_history)
end


"""
$(SIGNATURES)
Write the current model equations and parameter block to a Julia source
file that re-creates the model when `include`d.

# Keyword arguments
- `overwrite::Bool = false`: replace an existing file if `true`.
"""
function write_julia_model_file(𝓂::ℳ, filepath::String; overwrite::Bool = false)::String
    if isfile(filepath) && !overwrite
        error("File already exists: $filepath. Pass overwrite = true to replace it.")
    end

    parameter_block = reconstruct_parameter_block(𝓂)
    model_name = string(𝓂.model_name)

    open(filepath, "w") do io
        println(io, "using MacroModelling\n")
        println(io, "@model ", model_name, " begin")
        for eq in 𝓂.equations.original
            println(io, "    ", replace(string(eq), "◖" => "{", "◗" => "}"))
        end
        println(io, "end\n")

        println(io, "@parameters ", model_name, " begin")
        for line in parameter_block.args
            line isa LineNumberNode && continue
            println(io, "    ", replace(string(line), "◖" => "{", "◗" => "}"))
        end
        println(io, "end")
    end

    return filepath
end


# ------------------------------------------------------------------------
# update / add / remove equations
# ------------------------------------------------------------------------

const _EquationInput = Union{Expr, String}
const _EquationOrIndex = Union{Int, Expr, String}

"""
$(SIGNATURES)
Replace an existing model equation with a new one.

The first argument selects which equation to update: pass either the 1-based
index, the old equation `Expr`, or the equation as a `String`. The second
argument is the new equation (as `Expr` or `String`).

After the update, the revision history is appended, caches are invalidated
and the non-stochastic steady state is resolved.
"""
function update_equations!(𝓂::ℳ,
                           old_equation_or_index::_EquationOrIndex,
                           new_equation::_EquationInput;
                           parameters::ParameterType = nothing,
                           verbose::Bool = false,
                           silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr

    originals = copy(𝓂.equations.original)
    idx::Int = if old_equation_or_index isa Int
        @assert 1 <= old_equation_or_index <= length(originals) "Equation index $(old_equation_or_index) out of bounds (1:$(length(originals)))."
        old_equation_or_index
    else
        target = normalize_equation_input(old_equation_or_index)::Expr
        found = find_equation_index(originals, target)
        @assert found !== nothing "Equation not found in model: $(target)"
        found
    end

    old_eq = originals[idx]
    originals[idx] = new_eq

    push!(𝓂.revision_history, _revision_entry(:update_equation;
        equation_index = idx, old_equation = old_eq, new_equation = new_eq))

    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function update_equations!(𝓂::ℳ,
                           updates::Union{Vector, Tuple};
                           parameters::ParameterType = nothing,
                           verbose::Bool = false,
                           silent::Bool = true)
    originals = copy(𝓂.equations.original)
    history_entries = RevisionEntry[]
    for upd in updates
        @assert upd isa Union{Tuple, Pair} && length(upd) == 2 "Each update entry must be a (old_or_index, new_equation) pair/tuple."
        first_el = upd isa Pair ? upd.first : upd[1]
        second_el = upd isa Pair ? upd.second : upd[2]
        new_eq = normalize_equation_input(second_el)::Expr
        idx::Int = if first_el isa Int
            @assert 1 <= first_el <= length(originals) "Equation index $(first_el) out of bounds."
            first_el
        else
            target = normalize_equation_input(first_el)::Expr
            found = find_equation_index(originals, target)
            @assert found !== nothing "Equation not found in model: $(target)"
            found
        end
        old_eq = originals[idx]
        originals[idx] = new_eq
        push!(history_entries, _revision_entry(:update_equation;
            equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    end
    append!(𝓂.revision_history, history_entries)
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Append a new equation to the model and rebuild caches / solver.
"""
function add_equation!(𝓂::ℳ,
                       new_equation::_EquationInput;
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    originals = copy(𝓂.equations.original)
    push!(originals, new_eq)
    push!(𝓂.revision_history, _revision_entry(:add_equation;
        equation_index = length(originals), old_equation = nothing, new_equation = new_eq))
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function add_equation!(𝓂::ℳ,
                       new_equations::Union{Vector, Tuple};
                       parameters::ParameterType = nothing,
                       verbose::Bool = false,
                       silent::Bool = true)
    originals = copy(𝓂.equations.original)
    entries = RevisionEntry[]
    for ne in new_equations
        new_eq = normalize_equation_input(ne)::Expr
        push!(originals, new_eq)
        push!(entries, _revision_entry(:add_equation;
            equation_index = length(originals), old_equation = nothing, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Remove an equation from the model by index, `Expr`, or `String` match.
"""
function remove_equation!(𝓂::ℳ,
                          equation_or_index::_EquationOrIndex;
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    originals = copy(𝓂.equations.original)
    @assert length(originals) > 1 "Cannot remove the last equation from the model."
    idx::Int = if equation_or_index isa Int
        @assert 1 <= equation_or_index <= length(originals) "Equation index $(equation_or_index) out of bounds."
        equation_or_index
    else
        target = normalize_equation_input(equation_or_index)::Expr
        found = find_equation_index(originals, target)
        @assert found !== nothing "Equation not found in model: $(target)"
        found
    end
    old_eq = originals[idx]
    deleteat!(originals, idx)
    push!(𝓂.revision_history, _revision_entry(:remove_equation;
        equation_index = idx, old_equation = old_eq, new_equation = nothing))
    reprocess_model_equations!(𝓂, originals; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function remove_equation!(𝓂::ℳ,
                          removals::Union{Vector, Tuple};
                          parameters::ParameterType = nothing,
                          verbose::Bool = false,
                          silent::Bool = true)
    originals = copy(𝓂.equations.original)
    # Resolve all indices against the original list, then delete in descending order
    indices = Int[]
    old_eqs = Expr[]
    for item in removals
        @assert length(originals) - length(indices) > 1 "Cannot remove the last equation from the model."
        idx::Int = if item isa Int
            @assert 1 <= item <= length(originals) "Equation index $(item) out of bounds."
            item
        else
            target = normalize_equation_input(item)::Expr
            found = find_equation_index(originals, target)
            @assert found !== nothing "Equation not found in model: $(target)"
            found
        end
        push!(indices, idx)
        push!(old_eqs, originals[idx])
    end
    order = sortperm(indices, rev = true)
    updated = copy(originals)
    for i in order
        deleteat!(updated, indices[i])
    end
    entries = RevisionEntry[]
    for (i, idx) in enumerate(indices)
        push!(entries, _revision_entry(:remove_equation;
            equation_index = idx, old_equation = old_eqs[i], new_equation = nothing))
    end
    append!(𝓂.revision_history, entries)
    reprocess_model_equations!(𝓂, updated; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


# ------------------------------------------------------------------------
# calibration variants
# ------------------------------------------------------------------------

"""
$(SIGNATURES)
Replace an existing calibration equation.
"""
function update_calibration_equations!(𝓂::ℳ,
                                       old_equation_or_index::_EquationOrIndex,
                                       new_equation::_EquationInput;
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    @assert extract_calibrated_parameter(new_eq) !== nothing "Calibration equation must contain `| param` syntax."

    calib_orig = copy(𝓂.equations.calibration_original)
    idx::Int = if old_equation_or_index isa Int
        @assert 1 <= old_equation_or_index <= length(calib_orig) "Calibration index $(old_equation_or_index) out of bounds."
        old_equation_or_index
    else
        target = normalize_equation_input(old_equation_or_index)::Expr
        found = find_equation_index(calib_orig, target)
        @assert found !== nothing "Calibration equation not found: $(target)"
        found
    end

    new_param = extract_calibrated_parameter(new_eq)
    known_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
    union!(known_params, 𝓂.constants.post_complete_parameters.parameters)
    union!(known_params, 𝓂.equations.calibration_parameters)
    new_param !== nothing && !(new_param in known_params) &&
        error("Parameter `$(new_param)` is not part of the model.")

    old_eq = calib_orig[idx]
    calib_orig[idx] = new_eq
    push!(𝓂.revision_history, _revision_entry(:update_calibration_equation;
        equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function update_calibration_equations!(𝓂::ℳ,
                                       updates::Union{Vector, Tuple};
                                       parameters::ParameterType = nothing,
                                       verbose::Bool = false,
                                       silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    entries = RevisionEntry[]
    for upd in updates
        first_el = upd isa Pair ? upd.first : upd[1]
        second_el = upd isa Pair ? upd.second : upd[2]
        new_eq = normalize_equation_input(second_el)::Expr
        @assert extract_calibrated_parameter(new_eq) !== nothing "Calibration equation must contain `| param` syntax."
        idx::Int = if first_el isa Int
            @assert 1 <= first_el <= length(calib_orig) "Calibration index $(first_el) out of bounds."
            first_el
        else
            target = normalize_equation_input(first_el)::Expr
            found = find_equation_index(calib_orig, target)
            @assert found !== nothing "Calibration equation not found: $(target)"
            found
        end
        old_eq = calib_orig[idx]
        calib_orig[idx] = new_eq
        push!(entries, _revision_entry(:update_calibration_equation;
            equation_index = idx, old_equation = old_eq, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Add a new calibration equation (`lhs = rhs | param` syntax) to the model.
"""
function add_calibration_equation!(𝓂::ℳ,
                                   new_equation::_EquationInput;
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = true)
    new_eq = normalize_equation_input(new_equation)::Expr
    new_param = extract_calibrated_parameter(new_eq)
    new_param === nothing && error("Calibration equation must contain `| param` syntax.")
    new_param in 𝓂.equations.calibration_parameters &&
        error("Parameter `$(new_param)` is already calibrated.")
    known_params = Set{Symbol}(𝓂.constants.post_model_macro.parameters_in_equations)
    union!(known_params, 𝓂.constants.post_complete_parameters.parameters)
    !(new_param in known_params) && error("Parameter `$(new_param)` is not part of the model.")

    calib_orig = copy(𝓂.equations.calibration_original)
    push!(calib_orig, new_eq)
    push!(𝓂.revision_history, _revision_entry(:add_calibration_equation;
        equation_index = length(calib_orig), old_equation = nothing, new_equation = new_eq))
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end

function add_calibration_equation!(𝓂::ℳ,
                                   new_equations::Union{Vector, Tuple};
                                   parameters::ParameterType = nothing,
                                   verbose::Bool = false,
                                   silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    entries = RevisionEntry[]
    for ne in new_equations
        new_eq = normalize_equation_input(ne)::Expr
        new_param = extract_calibrated_parameter(new_eq)
        new_param === nothing && error("Calibration equation must contain `| param` syntax.")
        push!(calib_orig, new_eq)
        push!(entries, _revision_entry(:add_calibration_equation;
            equation_index = length(calib_orig), old_equation = nothing, new_equation = new_eq))
    end
    append!(𝓂.revision_history, entries)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = parameters,
        verbose = verbose, silent = silent)
    return nothing
end


"""
$(SIGNATURES)
Remove a calibration equation. Use the `parameters` keyword to supply a
value for the parameter that is now fixed (defaults to the current NSSS
value of that parameter).
"""
function remove_calibration_equation!(𝓂::ℳ,
                                      equation_or_index::_EquationOrIndex;
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    @assert !isempty(calib_orig) "No calibration equations to remove."
    idx::Int = if equation_or_index isa Int
        @assert 1 <= equation_or_index <= length(calib_orig) "Calibration index $(equation_or_index) out of bounds."
        equation_or_index
    else
        target = normalize_equation_input(equation_or_index)::Expr
        found = find_equation_index(calib_orig, target)
        @assert found !== nothing "Calibration equation not found: $(target)"
        found
    end
    old_eq = calib_orig[idx]
    deleteat!(calib_orig, idx)
    push!(𝓂.revision_history, _revision_entry(:remove_calibration_equation;
        equation_index = idx, old_equation = old_eq, new_equation = nothing))

    param_overrides = _parameters_to_dict(parameters)
    reprocess_calibration_equations!(𝓂, calib_orig; parameters = nothing,
        parameter_overrides = param_overrides, verbose = verbose, silent = silent)
    return nothing
end

function remove_calibration_equation!(𝓂::ℳ,
                                      removals::Union{Vector, Tuple};
                                      parameters::ParameterType = nothing,
                                      verbose::Bool = false,
                                      silent::Bool = true)
    calib_orig = copy(𝓂.equations.calibration_original)
    @assert !isempty(calib_orig) "No calibration equations to remove."
    indices = Int[]
    old_eqs = Expr[]
    for item in removals
        idx::Int = if item isa Int
            @assert 1 <= item <= length(calib_orig) "Calibration index $(item) out of bounds."
            item
        else
            target = normalize_equation_input(item)::Expr
            found = find_equation_index(calib_orig, target)
            @assert found !== nothing "Calibration equation not found: $(target)"
            found
        end
        push!(indices, idx)
        push!(old_eqs, calib_orig[idx])
    end
    updated = copy(calib_orig)
    for i in sort(indices, rev = true)
        deleteat!(updated, i)
    end
    entries = RevisionEntry[]
    for (i, idx) in enumerate(indices)
        push!(entries, _revision_entry(:remove_calibration_equation;
            equation_index = idx, old_equation = old_eqs[i], new_equation = nothing))
    end
    append!(𝓂.revision_history, entries)

    param_overrides = _parameters_to_dict(parameters)
    reprocess_calibration_equations!(𝓂, updated; parameters = nothing,
        parameter_overrides = param_overrides, verbose = verbose, silent = silent)
    return nothing
end

# Convert ParameterType-like user input into a Dict{Symbol, Float64} used by
# reprocess_calibration_equations!. Unsupported forms fall back to an empty dict.
function _parameters_to_dict(parameters)::Dict{Symbol, Float64}
    d = Dict{Symbol, Float64}()
    parameters === nothing && return d
    if parameters isa Pair
        k = parameters.first; v = parameters.second
        k_sym = k isa Symbol ? k : Symbol(k)
        d[k_sym] = Float64(v)
    elseif parameters isa AbstractDict
        for (k, v) in parameters
            k_sym = k isa Symbol ? k : Symbol(k)
            d[k_sym] = Float64(v)
        end
    elseif parameters isa Union{Tuple, Vector}
        for p in parameters
            if p isa Pair
                k = p.first; v = p.second
                k_sym = k isa Symbol ? k : Symbol(k)
                d[k_sym] = Float64(v)
            end
        end
    end
    return d
end

const replace_equations! = update_equations!
const replace_calibration_equations! = update_calibration_equations!
