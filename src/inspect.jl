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

# Keyword Arguments
- `filter::Union{Symbol, String, Nothing} = nothing`: Optional filter to return only equations that contain a specific symbol.
    Use `:k`/"k" to match any reference to `k`, or `"k[0]"` to target specific timing.

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
get_equations(RBC, filter = :k)
get_equations(RBC, filter = "k[0]")
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
- `filter::Union{Symbol, String, Nothing} = nothing`: Optional filter to return only equations that contain a specific symbol.
    Time subscripts in the filter are ignored for steady state equations.

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

# Keyword Arguments
- `filter::Union{Symbol, String, Nothing} = nothing`: Optional filter to return only equations that contain a specific symbol.
    Use `:k`/"k" to match any reference to `k`, or `"k[0]"` to target specific timing.

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

# Keyword Arguments
- `filter::Union{Symbol, String, Nothing} = nothing`: Optional filter to return only equations that contain a specific symbol.
    Time subscripts (other than `[ss]`) are ignored for calibration equations.

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
                          verbose::Bool = false,
                          silent::Bool = true)
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

    parameter_block = reconstruct_parameter_block(𝓂)

    T, equations_struct = process_model_equations(
        updated_block,
        𝓂.constants.post_model_macro.max_obc_horizon,
        𝓂.constants.post_parameters_macro.precompile,
    )

    𝓂.constants = Constants(T)

    𝓂.workspaces = Workspaces()

    𝓂.caches = Caches()
    
    𝓂.equations = equations_struct

    # Reset NSSS solve blocks (will be rebuilt by set_up_steady_state_solver!)
    𝓂.NSSS.solve_blocks_in_place = ss_solve_block[]
    𝓂.NSSS.dependencies = nothing
    
    NSSS_custom = 𝓂.functions.NSSS_custom

    𝓂.functions = Functions()

    𝓂.functions.NSSS_custom = NSSS_custom
    
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
    
    # Update equations struct with calibration fields
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
Alias for [`update_equations!`](@ref).
"""
replace_equations! = update_equations!


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
- `new_equation::Union{Expr, String}`: The new calibration equation to replace the old one. Can be an `Expr` or a `String` that will be parsed. The equation should be in the form `lhs = rhs` (e.g., `k[ss] / (4 * q[ss]) = 1.5`). When provided as `lhs = rhs | param`, the `param` portion is accepted and ignored in this stub.

# Keyword Arguments
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
                                       verbose::Bool = false,
                                       silent::Bool = true)
    # Parse string to Expr if needed
    parsed_new_equation = new_equation isa String ? Meta.parse(new_equation) : new_equation

    # Determine equation index
    n_equations = length(𝓂.equations.calibration_original)
    @assert n_equations > 0 "Model has no calibration equations."

    if old_equation_or_index isa Int
        equation_index = old_equation_or_index
        @assert 1 <= equation_index <= n_equations "Calibration equation index must be between 1 and $n_equations. Use `get_calibration_equations(model)` to see current calibration equations."
    else
        old_eq_to_match = old_equation_or_index isa String ? Meta.parse(old_equation_or_index) : old_equation_or_index
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

    parameter_block = reconstruct_parameter_block(
        𝓂;
        calibration_original_override = updated_calibration_original,
    )

    parsed_parameters = process_parameter_definitions(
        parameter_block,
        𝓂.constants.post_model_macro,
    )

    # Update post_parameters_macro (keeping existing options like guess, simplify, etc.)
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

    # Update equations struct with calibration fields
    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    # Update post_complete_parameters 
    # Reset axis fields to empty so ensure_name_display_constants! will recompute them
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
        var_axis = Symbol[],  # Reset so it gets recomputed
        calib_axis = Symbol[],  # Reset so it gets recomputed with new calibration_parameters
    )
    
    # Update parameter values
    𝓂.parameter_values = parsed_parameters.parameter_values

    # Reinitialize caches (fresh empty state, all marked outdated)
    𝓂.caches = Caches()

    # Reset NSSS solve blocks (will be rebuilt by set_up_steady_state_solver!)
    𝓂.NSSS.solve_blocks_in_place = ss_solve_block[]
    𝓂.NSSS.dependencies = nothing

    # Mark functions as needing recompilation
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
Alias for [`update_calibration_equations!`](@ref).
"""
replace_calibration_equations! = update_calibration_equations!


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
                                   verbose::Bool = false,
                                   silent::Bool = false)
    # Parse string to Expr if needed
    parsed_new_equation = new_equation isa String ? Meta.parse(new_equation) : new_equation

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

    parameter_block = reconstruct_parameter_block(
        𝓂;
        calibration_original_override = updated_calibration_original,
    )

    parsed_parameters = process_parameter_definitions(
        parameter_block,
        𝓂.constants.post_model_macro,
    )

    # Update post_parameters_macro (keeping existing options like guess, simplify, etc.)
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

    # Update equations struct with calibration fields
    𝓂.equations.calibration = parsed_parameters.equations.calibration
    𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
    𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters
    𝓂.equations.calibration_original = parsed_parameters.equations.calibration_original

    # Update post_complete_parameters 
    # Reset axis fields to empty so ensure_name_display_constants! will recompute them
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = parsed_parameters.parameters,
        missing_parameters = parsed_parameters.missing_parameters,
        var_axis = Symbol[],  # Reset so it gets recomputed
        calib_axis = Symbol[],  # Reset so it gets recomputed with new calibration_parameters
    )
    
    # Update parameter values
    𝓂.parameter_values = parsed_parameters.parameter_values

    # Reinitialize caches (fresh empty state, all marked outdated)
    𝓂.caches = Caches()

    # Reset NSSS solve blocks (will be rebuilt by set_up_steady_state_solver!)
    𝓂.NSSS.solve_blocks_in_place = ss_solve_block[]
    𝓂.NSSS.dependencies = nothing

    # Mark functions as needing recompilation
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