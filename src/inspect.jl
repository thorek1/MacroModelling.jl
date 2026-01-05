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
function get_equations(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.original_equations), "◖" => "{", "◗" => "}")
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
function get_steady_state_equations(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.ss_aux_equations), "◖" => "{", "◗" => "}")
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
function get_dynamic_equations(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.dyn_equations), "◖" => "{", "◗" => "}", "₍₋₁₎" => "[-1]", "₍₁₎" => "[1]", "₍₀₎" => "[0]", "₍ₓ₎" => "[x]")
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
function get_calibration_equations(𝓂::ℳ)::Vector{String}
    replace.(string.(𝓂.calibration_equations), "◖" => "{", "◗" => "}")
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
        return replace.(string.(𝓂.parameters), "◖" => "{", "◗" => "}") .=> 𝓂.parameter_values
    else
        return replace.(string.(𝓂.parameters), "◖" => "{", "◗" => "}")# |> sort
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
        return replace.(string.(𝓂.calibration_equations_parameters), "◖" => "{", "◗" => "}") .=> 𝓂.solution.non_stochastic_steady_state[𝓂.timings.nVars + 1:end]
    else
        return replace.(string.(𝓂.calibration_equations_parameters), "◖" => "{", "◗" => "}")# |> sort
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
    replace.(string.(𝓂.missing_parameters), "◖" => "{", "◗" => "}")
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
    !isempty(𝓂.missing_parameters)
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
    replace.(string.(𝓂.parameters_in_equations), "◖" => "{", "◗" => "}")# |> sort
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
    replace.(string.(𝓂.parameters_as_function_of_parameters), "◖" => "{", "◗" => "}")# |> sort
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
    replace.(string.(setdiff(𝓂.parameters, 𝓂.calibration_equations_parameters, 𝓂.parameters_in_equations, 𝓂.calibration_equations_parameters, 𝓂.parameters_as_function_of_parameters, reduce(union, 𝓂.par_calib_list, init = []))), "◖" => "{", "◗" => "}")# |> sort
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
    reduce(union, 𝓂.par_calib_list, init = []) |> collect |> sort  .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations), init = []), union(𝓂.parameters_in_equations,𝓂.➕_vars)) |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    𝓂.➕_vars |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    𝓂.aux |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    𝓂.exo |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    𝓂.timings.past_not_future_and_mixed |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
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
    𝓂.timings.future_not_past_and_mixed |> collect |> sort .|> x -> replace.(string.(x), "◖" => "{", "◗" => "}")
end


"""
$(SIGNATURES)
Returns information about the balanced growth path configuration of the model.

# Arguments
- $MODEL®

# Returns
- `NamedTuple` with the following fields:
  - `has_balanced_growth`: `Bool` indicating whether balanced growth path handling is enabled
  - `trend_vars`: `Dict{Symbol, Union{Symbol, Expr}}` mapping trend variables to their growth factors
  - `deflators`: `Dict{Symbol, Symbol}` mapping variables to their deflators (trend variables)
  - `detrended_vars`: `Vector{Symbol}` list of variables that are detrended

# Examples
```julia
using MacroModelling

@model RBC_growth deflator = Dict(:y => :A, :k => :A, :c => :A) begin
    # ... model equations ...
end

@parameters RBC_growth trend_var = Dict(:A => :γ) begin
    γ = 1.02  # 2% growth rate
    # ... other parameters ...
end

get_balanced_growth_path_info(RBC_growth)
```
"""
function get_balanced_growth_path_info(𝓂::ℳ)
    bg = 𝓂.balanced_growth
    return (
        has_balanced_growth = !isempty(bg.deflators) || !isempty(bg.trend_vars),
        trend_vars = bg.trend_vars,
        deflators = bg.deflators,
        detrended_vars = collect(bg.detrended_vars)
    )
end


"""
$(SIGNATURES)
Returns `true` if the model has balanced growth path handling enabled, `false` otherwise.

A model has balanced growth path handling enabled if either deflators have been specified
in the `@model` macro or trend variables have been specified in the `@parameters` macro.

# Arguments
- $MODEL®

# Returns
- `Bool`

# Examples
```julia
using MacroModelling

@model RBC begin
    # standard RBC equations without growth
end

@parameters RBC begin
    # parameters
end

has_balanced_growth(RBC)  # returns false
```
"""
function has_balanced_growth(𝓂::ℳ)::Bool
    return !isempty(𝓂.balanced_growth.deflators) || !isempty(𝓂.balanced_growth.trend_vars)
end

end # dispatch_doctor