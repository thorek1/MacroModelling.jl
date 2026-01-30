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
$(SIGNATURES)
Return the equations of the model. In case programmatic model writing was used this function returns the parsed equations (see loop over shocks in `Examples`).

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the parsed equations. 

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
7-element Vector{String}:
 "1 / c[0] = (β / c[1]) * (α * ex" ⋯ 25 bytes ⋯ " - 1) + (1 - exp(z{δ}[1]) * δ))"
 "c[0] + k[0] = (1 - exp(z{δ}[0]) * δ) * k[-1] + q[0]"
 "q[0] = exp(z{TFP}[0]) * k[-1] ^ α"
 "z{TFP}[0] = ρ{TFP} * z{TFP}[-1]" ⋯ 18 bytes ⋯ "TFP}[x] + eps_news{TFP}[x - 1])"
 "z{δ}[0] = ρ{δ} * z{δ}[-1] + σ{δ} * (eps{δ}[x] + eps_news{δ}[x - 1])"
 "Δc_share[0] = log(c[0] / q[0]) - log(c[-1] / q[-1])"
 "Δk_4q[0] = log(k[0]) - log(k[-4])"
```
"""
function get_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{String}
    equations = replace.(string.(𝓂.equations.original), "◖" => "{", "◗" => "}")
    filter === nothing && return equations
    
    sym, pattern = parse_filter_term(filter)
    return [eq for (eq, expr) in zip(equations, 𝓂.equations.original) if expr_contains(expr, sym, pattern)]
end


"""
$(SIGNATURES)
Return the non-stochastic steady state (NSSS) equations of the model. The difference to the equations as they were written in the `@model` block is that exogenous shocks are set to `0`, time subscripts are eliminated (e.g. `c[-1]` becomes `c`), trivial simplifications are carried out (e.g. `log(k) - log(k) = 0`), and auxiliary variables are added for expressions that cannot become negative. 

Auxiliary variables facilitate the solution of the NSSS problem. The package substitutes expressions which cannot become negative with auxiliary variables and adds another equation to the system of equations determining the NSSS. For example, `log(c/q)` cannot be negative and `c/q` is substituted by an auxiliary variable `➕₁` and an additional equation is added: `➕₁ = c / q`.

Note that the output assumes the equations are equal to 0. As in, `-z{δ} * ρ{δ} + z{δ}` implies `-z{δ} * ρ{δ} + z{δ} = 0` and therefore: `z{δ} * ρ{δ} = z{δ}`.

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the NSSS equations. 

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
9-element Vector{String}:
 "(-β * ((k ^ (α - 1) * α * exp(z{TFP}) - δ * exp(z{δ})) + 1)) / c + 1 / c"
 "((c - k * (-δ * exp(z{δ}) + 1)) + k) - q"
 "-(k ^ α) * exp(z{TFP}) + q"
 "-z{TFP} * ρ{TFP} + z{TFP}"
 "-z{δ} * ρ{δ} + z{δ}"
 "➕₁ - c / q"
 "➕₂ - c / q"
 "(Δc_share - log(➕₁)) + log(➕₂)"
 "Δk_4q - 0"
```
"""
function get_steady_state_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{String}
    equations = replace.(string.(𝓂.equations.steady_state_aux), "◖" => "{", "◗" => "}")
    filter === nothing && return equations
    
    sym, pattern = parse_filter_term(filter)
    
    # Warn if time subscript provided
    if pattern !== nothing
        @info "Time subscript in filter will be ignored for steady state equations. Equations containing the variable '$(sym)' will be returned regardless of timing."
    end
    
    # Always ignore timing for steady state equations (no time subscripts in SS)
    return [eq for (eq, expr) in zip(equations, 𝓂.equations.steady_state_aux) if expr_contains(expr, sym, nothing)]
end


"""
$(SIGNATURES)
Return the augmented system of equations describing the model dynamics. Augmented means that, when variables have leads or lags with absolute value larger than 1, or exogenous shocks have leads or lags, auxiliary equations containing lead/lag variables are added. The augmented system contains only variables in the present `[0]`, future `[1]`, or past `[-1]`. For example, `Δk_4q[0] = log(k[0]) - log(k[-3])` contains `k[-3]`. Introducing two auxiliary variables (`kᴸ⁽⁻¹⁾` and `kᴸ⁽⁻²⁾`, where `ᴸ` denotes the lead/lag operator) and augmenting the system with `kᴸ⁽⁻²⁾[0] = kᴸ⁽⁻¹⁾[-1]` and `kᴸ⁽⁻¹⁾[0] = k[-1]` ensures that all timing indices have absolute value at most 1: `Δk_4q[0] - (log(k[0]) - log(kᴸ⁽⁻²⁾[-1]))`.

In case programmatic model writing was used this function returns the parsed equations (see loop over shocks in example).

Note that the output assumes the equations are equal to 0. As in, `kᴸ⁽⁻¹⁾[0] - k[-1]` implies `kᴸ⁽⁻¹⁾[0] - k[-1] = 0` and therefore: `kᴸ⁽⁻¹⁾[0] = k[-1]`.

# Arguments
- $MODEL®

# Returns
- `Vector{String}` of the dynamic model equations. 

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
12-element Vector{String}:
 "1 / c[0] - (β / c[1]) * (α * ex" ⋯ 25 bytes ⋯ " - 1) + (1 - exp(z{δ}[1]) * δ))"
 "(c[0] + k[0]) - ((1 - exp(z{δ}[0]) * δ) * k[-1] + q[0])"
 "q[0] - exp(z{TFP}[0]) * k[-1] ^ α"
 "eps_news{TFP}[0] - eps_news{TFP}[x]"
 "z{TFP}[0] - (ρ{TFP} * z{TFP}[-1] + σ{TFP} * (eps{TFP}[x] + eps_news{TFP}[-1]))"
 "eps_news{δ}[0] - eps_news{δ}[x]"
 "z{δ}[0] - (ρ{δ} * z{δ}[-1] + σ{δ} * (eps{δ}[x] + eps_news{δ}[-1]))"
 "Δc_share[0] - (log(c[0] / q[0]) - log(c[-1] / q[-1]))"
 "kᴸ⁽⁻³⁾[0] - kᴸ⁽⁻²⁾[-1]"
 "kᴸ⁽⁻²⁾[0] - kᴸ⁽⁻¹⁾[-1]"
 "kᴸ⁽⁻¹⁾[0] - k[-1]"
 "Δk_4q[0] - (log(k[0]) - log(kᴸ⁽⁻³⁾[-1]))"
```
"""
function get_dynamic_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{String}
    # Transform equations to user-friendly format
    equations = replace.(string.(𝓂.equations.dynamic), "◖" => "{", "◗" => "}", "₍₋₁₎" => "[-1]", "₍₁₎" => "[1]", "₍₀₎" => "[0]", "₍ₓ₎" => "[x]")
    filter === nothing && return equations
    
    # Transform internal expressions to user-friendly format for matching
    transformed_exprs = [Meta.parse(replace(string(expr), "₍₋₁₎" => "[-1]", "₍₁₎" => "[1]", "₍₀₎" => "[0]", "₍ₓ₎" => "[x]")) for expr in 𝓂.equations.dynamic]
    
    # Parse filter term (uses user-friendly format with [-1], [0], etc.)
    sym, pattern = parse_filter_term(filter)
    
    return [eq for (eq, expr) in zip(equations, transformed_exprs) if expr_contains(expr, sym, pattern)]
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

# Returns
- `Vector{String}` of the calibration equations. 

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
1-element Vector{String}:
 "k / (q * 4) - capital_to_output"
```
"""
function get_calibration_equations(𝓂::ℳ; filter::Union{Symbol, String, Nothing} = nothing)::Vector{String}
    equations = replace.(string.(𝓂.equations.calibration), "◖" => "{", "◗" => "}")
    filter === nothing && return equations

    sym, pattern = parse_filter_term(filter)
    
    # Warn if time subscript provided (other than [ss] which is valid for calibration)
    if pattern !== nothing
        pattern_str = string(pattern)
        if !occursin("[ss]", pattern_str)
            @info "Time subscript in filter will be ignored for calibration equations. Equations containing the variable '$(sym)' will be returned regardless of timing."
        end
    end
    
    # Always ignore timing for calibration equations
    return [eq for (eq, expr) in zip(equations, 𝓂.equations.calibration) if expr_contains(expr, sym, nothing)]
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
- `verbose` [Default: `false`, Type: `Bool`]: Print detailed information about the update process.
- `silent` [Default: `false`, Type: `Bool`]: Suppress all output during reprocessing.

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
                          verbose::Bool = false,
                          silent::Bool = false)
    # Determine equation index
    if old_equation_or_index isa Int
        equation_index = old_equation_or_index
        n_equations = length(𝓂.equations.original)
        @assert 1 <= equation_index <= n_equations "Equation index must be between 1 and $n_equations. Use `get_equations(model)` to see current equations."
    else
        # Parse string to Expr if needed
        old_eq_to_match = old_equation_or_index isa String ? Meta.parse(old_equation_or_index) : old_equation_or_index
        
        # Find the equation index by matching
        equation_index = find_equation_index(𝓂.equations.original, old_eq_to_match)
        @assert equation_index !== nothing "Could not find equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}")). Use `get_equations(model)` to see current equations."
    end
    
    # Parse string to Expr if needed
    if new_equation isa String
        new_equation = Meta.parse(new_equation)
    end
    
    # Store old equation for revision history
    old_equation = 𝓂.equations.original[equation_index]
    
    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :update_equation,
        equation_index = equation_index,
        old_equation = old_equation,
        new_equation = new_equation
    )
    push!(𝓂.revision_history, revision_entry)
    
    # Update the original equation
    updated_original_equations = copy(𝓂.equations.original)
    updated_original_equations[equation_index] = new_equation
    
    if verbose
        println("Updated equation $equation_index:")
        println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
        println("  New: ", replace(string(new_equation), "◖" => "{", "◗" => "}"))
    end
    
    # Re-process the model equations and combine with existing calibration parts
    updated_block = Expr(:block, updated_original_equations...)
    T, equations_struct = process_model_equations(
        updated_block,
        𝓂.constants.post_model_macro.max_obc_horizon,
        𝓂.constants.post_parameters_macro.precompile,
    )

    equations_struct.calibration = 𝓂.equations.calibration
    equations_struct.calibration_no_var = 𝓂.equations.calibration_no_var
    equations_struct.calibration_parameters = 𝓂.equations.calibration_parameters

    𝓂.constants.post_model_macro = T
    𝓂.equations = equations_struct

    𝓂.caches.outdated.non_stochastic_steady_state = true
    𝓂.caches.outdated.jacobian = true
    𝓂.caches.outdated.hessian = true
    𝓂.caches.outdated.third_order_derivatives = true
    𝓂.caches.outdated.first_order_solution = true
    𝓂.caches.outdated.second_order_solution = true
    𝓂.caches.outdated.pruned_second_order_solution = true
    𝓂.caches.outdated.third_order_solution = true
    𝓂.caches.outdated.pruned_third_order_solution = true
    𝓂.functions.functions_written = false

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
                symbolic = false,
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
$(SIGNATURES)
Modify or replace a calibration equation. This function updates the calibration equations and automatically re-parses, re-solves, and recomputes derivatives.

# Arguments
- $MODEL®
- `old_equation_or_index::Union{Int, Expr, String}`: Either the index of the calibration equation to modify (1-based), or the old equation itself (as `Expr` or `String`) to be matched and replaced. Use `get_calibration_equations` to see current calibration equations.
- `new_equation::Union{Expr, String}`: The new calibration equation to replace the old one. Can be an `Expr` or a `String` that will be parsed. The equation should be in the form `lhs = rhs` (e.g., `k[ss] / (4 * q[ss]) = 1.5`).
- `calibrated_parameter::Union{Symbol, String, Nothing}` [Optional]: The parameter to calibrate. If `nothing`, keeps the existing calibrated parameter.

# Keyword Arguments
- `verbose` [Default: `false`, Type: `Bool`]: Print detailed information about the update process.
- `silent` [Default: `false`, Type: `Bool`]: Suppress all output during reprocessing.

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
update_calibration_equations!(RBC_calibrated, 1, :(k[ss] / (4 * q[ss]) = 2.0))

# The model is automatically re-solved - verify with:
get_steady_state(RBC_calibrated)

# View revision history
get_revision_history(RBC_calibrated)
```
"""
function update_calibration_equations!(𝓂::ℳ, 
                                       old_equation_or_index::Union{Int, Expr, String}, 
                                       new_equation::Union{Expr, String},
                                       calibrated_parameter::Union{Symbol, String, Nothing} = nothing;
                                       verbose::Bool = false,
                                       silent::Bool = false)
    # Validate and determine equation index
    n_equations = length(𝓂.equations.calibration)
    @assert n_equations > 0 "Model has no calibration equations."
    
    if old_equation_or_index isa Int
        equation_index = old_equation_or_index
        @assert 1 <= equation_index <= n_equations "Calibration equation index must be between 1 and $n_equations. Use `get_calibration_equations(model)` to see current calibration equations."
    else
        # Parse string to Expr if needed
        old_eq_to_match = old_equation_or_index isa String ? Meta.parse(old_equation_or_index) : old_equation_or_index
        
        # Find the equation index by matching
        equation_index = find_equation_index(𝓂.equations.calibration, old_eq_to_match)
        @assert equation_index !== nothing "Could not find calibration equation matching: $(replace(string(old_eq_to_match), "◖" => "{", "◗" => "}")). Use `get_calibration_equations(model)` to see current calibration equations."
    end
    
    # Parse string to Expr if needed
    if new_equation isa String
        new_equation = Meta.parse(new_equation)
    end
    
    # Convert calibrated_parameter to Symbol if String
    if calibrated_parameter isa String
        calibrated_parameter = Symbol(calibrated_parameter)
    end
    
    # Store old equation for revision history
    old_equation = 𝓂.equations.calibration[equation_index]
    
    # Record the revision
    revision_entry = (
        timestamp = Dates.now(),
        action = :update_calibration_equation,
        equation_index = equation_index,
        old_equation = old_equation,
        new_equation = new_equation
    )
    push!(𝓂.revision_history, revision_entry)
    
    # Parse the new equation to the internal format (convert = to -, remove SS refs)
    parsed_equation = postwalk(x -> 
        x isa Expr ? 
            x.head == :(=) ? 
                Expr(:call, :(-), x.args[1], x.args[2]) :
            x.head == :ref ?
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i, string(x.args[2])) ?
                    x.args[1] : 
                x :
            x.head == :call && x.args[1] == :* && length(x.args) >= 3 && x.args[2] isa Int && !(x.args[3] isa Int) ?
                Expr(:call, :*, x.args[3:end]..., x.args[2]) :
            unblock(x) :
        x,
    new_equation)
    
    # Update the calibration equation
    updated_calibration_equations = copy(𝓂.equations.calibration)
    updated_calibration_equations[equation_index] = unblock(parsed_equation)
    
    # Update the calibrated parameter if provided
    if calibrated_parameter !== nothing
        old_param = 𝓂.equations.calibration_parameters[equation_index]
        updated_calibration_parameters = copy(𝓂.equations.calibration_parameters)
        updated_calibration_parameters[equation_index] = calibrated_parameter
        𝓂.equations.calibration_parameters = updated_calibration_parameters
        if verbose
            println("Updated calibrated parameter: $old_param -> $calibrated_parameter")
        end
    end

    𝓂.equations.calibration = updated_calibration_equations

    # Rebuild ss_calib_list and par_calib_list from updated calibration equations
    model_vars = Set(𝓂.constants.post_model_macro.var)
    ss_calib_list = Vector{Set{Symbol}}(undef, length(updated_calibration_equations))
    par_calib_list = Vector{Set{Symbol}}(undef, length(updated_calibration_equations))
    for (i, eq) in enumerate(updated_calibration_equations)
        syms = Set{Symbol}()
        for s in get_symbols(eq)
            s isa Symbol && push!(syms, s)
        end
        ss_tmp = intersect(syms, model_vars)
        par_tmp = setdiff(syms, ss_tmp)
        ss_calib_list[i] = ss_tmp
        par_calib_list[i] = par_tmp
    end

    𝓂.constants.post_parameters_macro = post_parameters_macro(
        𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters,
        𝓂.constants.post_parameters_macro.precompile,
        𝓂.constants.post_parameters_macro.simplify,
        𝓂.constants.post_parameters_macro.guess,
        ss_calib_list,
        par_calib_list,
        𝓂.constants.post_parameters_macro.bounds,
        𝓂.constants.post_parameters_macro.ss_solver_parameters_algorithm,
        𝓂.constants.post_parameters_macro.ss_solver_parameters_maxtime,
    )
    
    if verbose
        println("Updated calibration equation $equation_index:")
        println("  Old: ", replace(string(old_equation), "◖" => "{", "◗" => "}"))
        println("  New: ", replace(string(parsed_equation), "◖" => "{", "◗" => "}"))
    end
    
    𝓂.caches.outdated.non_stochastic_steady_state = true
    𝓂.caches.outdated.jacobian = true
    𝓂.caches.outdated.hessian = true
    𝓂.caches.outdated.third_order_derivatives = true
    𝓂.caches.outdated.first_order_solution = true
    𝓂.caches.outdated.second_order_solution = true
    𝓂.caches.outdated.pruned_second_order_solution = true
    𝓂.caches.outdated.third_order_solution = true
    𝓂.caches.outdated.pruned_third_order_solution = true
    𝓂.functions.functions_written = false

    if !isempty(𝓂.constants.post_complete_parameters.missing_parameters)
        if !silent
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(𝓂.constants.post_complete_parameters.missing_parameters). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined."
        end
        return nothing
    end

    set_up_steady_state_solver!(𝓂, verbose = verbose, silent = silent, avoid_solve = !𝓂.constants.post_parameters_macro.simplify, symbolic = false)
    opts = merge_calculation_options(verbose = verbose)
    solve_steady_state!(𝓂, opts, :ESCH, 120.0, silent = silent)
    write_symbolic_derivatives!(𝓂; perturbation_order = 1, silent = silent)
    𝓂.functions.functions_written = true
    
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