@stable default_mode = "disable" begin


# ── Occasionally Binding Constraints (OBC) ───────────────────────────────────
#
# Self-contained OBC functions extracted from MacroModelling.jl.
# Struct definitions remain in structures.jl; default constants in default_options.jl.


# ── Parsing & transformation ─────────────────────────────────────────────────

check_for_dynamic_variables(ex::Int) = false
check_for_dynamic_variables(ex::Float64) = false
check_for_dynamic_variables(ex::Symbol) = occursin(r"₍₁₎|₍₀₎|₍₋₁₎",string(ex))

function check_for_dynamic_variables(ex::Expr)
    dynamic_indicator = Bool[]

    postwalk(x -> 
        x isa Expr ?
            x.head == :ref ? 
                occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i,string(x.args[2])) ?
                    x :
                begin
                    push!(dynamic_indicator,true)
                    x
                end :
            x :
        x,
    ex)

    any(dynamic_indicator)
end


function transform_expression(expr::Expr)
    # Dictionary to store the transformations for reversing
    reverse_transformations = Dict{Symbol, Expr}()

    # Counter for generating unique placeholders
    unique_counter = Ref(0)

    # Step 1: Replace min/max calls and record their original form
    function replace_min_max(expr)
        if expr isa Expr && expr.head == :call && (expr.args[1] == :min || expr.args[1] == :max)
            # Replace min/max functions with a placeholder
            # placeholder = Symbol("minimal__P", unique_counter[])
            placeholder = :minmax__P
            unique_counter[] += 1

            # Store the original min/max call for reversal
            reverse_transformations[placeholder] = expr

            return placeholder
        else
            return expr
        end
    end

    # Step 2: Transform :ref fields in the rest of the expression
    function transform_ref_fields(expr)
        if expr isa Expr && expr.head == :ref && isa(expr.args[1], Symbol)
            # Handle :ref expressions
            if isa(expr.args[2], Number) || isa(expr.args[2], Symbol)           
                if expr.args[2] < 0
                    new_symbol = Symbol(expr.args[1], "__", abs(expr.args[2]))
                else
                    new_symbol = Symbol(expr.args[1], "_", expr.args[2])
                end
            else
                # Generate a unique placeholder for complex :ref
                unique_counter[] += 1
                placeholder = Symbol("__placeholder", unique_counter[])
                new_symbol = placeholder
            end

            # Record the reverse transformation
            reverse_transformations[new_symbol] = expr

            return new_symbol
        else
            return expr
        end
    end


    # Replace equality sign with minus
    function replace_equality_with_minus(expr)
        if expr isa Expr && expr.head == :(=)
            return Expr(:call, :-, expr.args...)
        else
            return expr
        end
    end

    # Apply transformations
    expr = postwalk(replace_min_max, expr)
    expr = postwalk(transform_ref_fields, expr)
    transformed_expr = postwalk(replace_equality_with_minus, expr)
    
    return transformed_expr, reverse_transformations
end


function reverse_transformation(transformed_expr::Expr, reverse_dict::Dict{Symbol, Expr})
    # Function to replace the transformed symbols with their original form
    function revert_symbol(expr)
        if expr isa Symbol && haskey(reverse_dict, expr)
            return reverse_dict[expr]
        else
            return expr
        end
    end

    # Revert the expression using postwalk
    reverted_expr = postwalk(revert_symbol, transformed_expr)

    return reverted_expr
end


function check_for_minmax(expr)
    contains_minmax = Bool[]

    postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:max,:min] ?
                            begin
                                push!(contains_minmax,true)
                                x
                            end :
                        x :
                    x :
                x,
    expr)

    any(contains_minmax)
end


function transform_obc(ex::Expr; avoid_solve::Bool = false)
    transformed_expr, reverse_dict = transform_expression(ex)

    for symbs in get_symbols(transformed_expr)
        sym_value = SPyPyC.symbols(string(symbs), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($symbs = $sym_value))
    end

    eq = Core.eval(SymPyWorkspace, transformed_expr)

    if avoid_solve || count_ops(Meta.parse(string(eq))) > 15
        soll = nothing
    else
        soll = solve_symbolically(eq, Core.eval(SymPyWorkspace, :minmax__P))
    end

    if !isempty(soll)
        sorted_minmax = Expr(:call, reverse_dict[:minmax__P].args[1], :($(reverse_dict[:minmax__P].args[2]) - $(Meta.parse(string(soll[1])))),  :($(reverse_dict[:minmax__P].args[3]) - $(Meta.parse(string(soll[1])))))
        return reverse_transformation(sorted_minmax, reverse_dict)
    else
        @error "Occasionally binding constraint not well-defined. See documentation for examples."
    end
end


# try: run optim only if there is a violation / capture case with small shocks and set them to zero
function parse_occasionally_binding_constraints(equations_block; max_obc_horizon::Int = 40, avoid_solve::Bool = false)
    # precision_factor = 1e  #factor to force the optimiser to have non-relevatn shocks at zero

    eqs = []
    obc_shocks = Expr[]

    for arg in equations_block.args
        if isa(arg,Expr)
            if check_for_minmax(arg)
                arg_trans = transform_obc(arg)
            else
                arg_trans = arg
            end

            eq = postwalk(x -> 
                    x isa Expr ?
                        x.head == :call ? 
                            x.args[1] == :max ?
                                begin

                                    obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                                    obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                                    if !(x.args[2] isa Symbol) && check_for_dynamic_variables(x.args[2])
                                        push!(eqs, :($obc_vars_left = $(x.args[2])))
                                    else
                                        obc_vars_left = x.args[2]
                                    end

                                    if !(x.args[3] isa Symbol) && check_for_dynamic_variables(x.args[3])
                                        push!(eqs, :($obc_vars_right = $(x.args[3])))
                                    else
                                        obc_vars_right = x.args[3]
                                    end

                                    obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                                    push!(eqs, :($obc_inequality = $(Expr(x.head, x.args[1], obc_vars_left, obc_vars_right))))

                                    obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                                    push!(obc_shocks, obc_shock)

                                    :($obc_inequality - $obc_shock)
                                end :
                            x.args[1] == :min ?
                                begin
                                    obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                                    obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                                    if !(x.args[2] isa Symbol) && check_for_dynamic_variables(x.args[2])
                                        push!(eqs, :($obc_vars_left = $(x.args[2])))
                                    else
                                        obc_vars_left = x.args[2]
                                    end

                                    if !(x.args[3] isa Symbol) && check_for_dynamic_variables(x.args[3])
                                        push!(eqs, :($obc_vars_right = $(x.args[3])))
                                    else
                                        obc_vars_right = x.args[3]
                                    end

                                    obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                                    push!(eqs, :($obc_inequality = $(Expr(x.head, x.args[1], obc_vars_left, obc_vars_right))))

                                    obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                                    push!(obc_shocks, obc_shock)

                                    :($obc_inequality - $obc_shock)
                                end :
                            x :
                        x :
                    x,
            arg_trans)

            push!(eqs, eq)
        end
    end

    for obc in obc_shocks
        # push!(eqs, :($(obc) = $(Expr(:ref, obc.args[1], -1)) * 0.3 + $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(max_obc_horizon)) * "⁾"), 0))))
        push!(eqs, :($(obc) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(max_obc_horizon)) * "⁾"), 0))))

        push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻⁰⁾"), 0)) = activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon)) * "⁾"), :x))))

        for i in 1:max_obc_horizon
            push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i)) * "⁾"), 0)) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i-1)) * "⁾"), -1)) + activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon-i)) * "⁾"), :x))))
        end
    end

    return Expr(:block, eqs...)
end


function write_obc_violation_equations(𝓂)
    eqs = Expr[]
    for (i,eq) in enumerate(𝓂.equations.dynamic)
        if check_for_minmax(eq)
            minmax_fixed_eqs = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        length(x.args) == 3 ?
                            x.args[3] isa Expr ?
                                x.args[3].args[1] ∈ [:Min, :min, :Max, :max] ?
                                    begin
                                        plchldr = Symbol(replace(string(x.args[2]), "₍₀₎" => ""))

                                        ineq_plchldr_1 = x.args[3].args[2] isa Symbol ? Symbol(replace(string(x.args[3].args[2]), "₍₀₎" => "")) : x.args[3].args[2]

                                        arg1 = x.args[3].args[2]
                                        arg2 = x.args[3].args[3]

                                        dyn_1 = check_for_dynamic_variables(x.args[3].args[2])
                                        dyn_2 = check_for_dynamic_variables(x.args[3].args[3])

                                        cond1 = Expr[]
                                        cond2 = Expr[]

                                        maximisation = contains(string(plchldr), "⁺")
                                        
                                        # if dyn_1
                                        #     if maximisation
                                        #         push!(cond1, :(push!(constraint_values, $(x.args[3].args[2]))))
                                        #         # push!(cond2, :(push!(constraint_values, $(x.args[3].args[2]))))
                                        #     else
                                        #         push!(cond1, :(push!(constraint_values, -$(x.args[3].args[2]))))
                                        #         # push!(cond2, :(push!(constraint_values, -$(x.args[3].args[2])))) # RBC
                                        #     end
                                        # end

                                        # if dyn_2
                                        #     if maximisation
                                        #         push!(cond1, :(push!(constraint_values, $(x.args[3].args[3]))))
                                        #         # push!(cond2, :(push!(constraint_values, $(x.args[3].args[3])))) # testmax
                                        #     else
                                        #         push!(cond1, :(push!(constraint_values, -$(x.args[3].args[3]))))
                                        #         # push!(cond2, :(push!(constraint_values, -$(x.args[3].args[3])))) # RBC
                                        #     end
                                        # end


                                        if maximisation
                                            push!(cond1, :(push!(constraint_values, [sum($(x.args[3].args[2]) .* $(x.args[3].args[3]))])))
                                            push!(cond1, :(push!(constraint_values, $(x.args[3].args[2]))))
                                            push!(cond1, :(push!(constraint_values, $(x.args[3].args[3]))))
                                            # push!(cond1, :(push!(constraint_values, max.($(x.args[3].args[2]), $(x.args[3].args[3])))))
                                        else
                                            push!(cond1, :(push!(constraint_values, [sum($(x.args[3].args[2]) .* $(x.args[3].args[3]))])))
                                            push!(cond1, :(push!(constraint_values, -$(x.args[3].args[2]))))
                                            push!(cond1, :(push!(constraint_values, -$(x.args[3].args[3]))))
                                            # push!(cond1, :(push!(constraint_values, min.($(x.args[3].args[2]), $(x.args[3].args[3])))))
                                        end

                                        # if maximisation
                                        #     push!(cond1, :(push!(shock_sign_indicators, true)))
                                        #     # push!(cond2, :(push!(shock_sign_indicators, true)))
                                        # else
                                        #     push!(cond1, :(push!(shock_sign_indicators, false)))
                                        #     # push!(cond2, :(push!(shock_sign_indicators, false)))
                                        # end

                                        # :(if isapprox($plchldr, $ineq_plchldr_1, atol = 1e-12)
                                        #     $(Expr(:block, cond1...))
                                        # else
                                        #     $(Expr(:block, cond2...))
                                        # end)
                                        :($(Expr(:block, cond1...)))
                                    end :
                                x :
                            x :
                        x :
                    x :
                x,
            eq)

            push!(eqs, minmax_fixed_eqs)
        end
    end

    return eqs
end


# ── OBC flag processing ──────────────────────────────────────────────────────

function process_ignore_obc_flag(shocks,
                                 ignore_obc::Bool,
                                 𝓂::ℳ; 
                                 maxlog::Int = DEFAULT_MAXLOG)
    stochastic_model = length(𝓂.constants.post_model_macro.exo) > 0
    obc_model = length(𝓂.equations.obc_violation) > 0

    obc_shocks_included = false

    if stochastic_model && obc_model
        if shocks isa Matrix{Float64}
            obc_indices = contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")
            if any(obc_indices)
                obc_shocks_included = sum(abs2, shocks[obc_indices, :]) > 1e-10
            end
        elseif shocks isa KeyedArray{Float64}
            shock_axis = collect(axiskeys(shocks, 1))
            shock_axis = shock_axis isa Vector{String} ? shock_axis .|> Meta.parse .|> replace_indices : shock_axis

            obc_shocks = 𝓂.constants.post_model_macro.exo[contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")]
            relevant_shocks = intersect(obc_shocks, shock_axis)

            if !isempty(relevant_shocks)
                obc_shocks_included = sum(abs2, shocks(relevant_shocks, :)) > 1e-10
            end
        else
            shock_idx = parse_shocks_input_to_index(shocks, 𝓂.constants)

            selected_shocks = if (shock_idx isa Vector) || (shock_idx isa UnitRange)
                length(shock_idx) > 0 ? 𝓂.constants.post_model_macro.exo[shock_idx] : Symbol[]
            else
                [𝓂.constants.post_model_macro.exo[shock_idx]]
            end

            obc_shocks = 𝓂.constants.post_model_macro.exo[contains.(string.(𝓂.constants.post_model_macro.exo), "ᵒᵇᶜ")]
            obc_shocks_included = !isempty(intersect(selected_shocks, obc_shocks))
        end
    end

    ignore_obc_flag = ignore_obc

    if ignore_obc_flag && !obc_model
        @info "`ignore_obc = true` has no effect because $(𝓂.model_name) has no occasionally binding constraints. Setting `ignore_obc = false`." maxlog = maxlog
        ignore_obc_flag = false
    end

    if ignore_obc_flag && obc_shocks_included
        @warn "`ignore_obc = true` cannot be applied because shocks affecting occasionally binding constraints are included. Enforcing the constraints instead and setting `ignore_obc = false`." maxlog = maxlog
        ignore_obc_flag = false
    end

    occasionally_binding_constraints = obc_model && !ignore_obc_flag

    return ignore_obc_flag, occasionally_binding_constraints, obc_shocks_included
end


# ── OBC violation function setup ─────────────────────────────────────────────

function set_up_obc_violation_function!(𝓂)
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎$")))

    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))

    # write indices in auxiliary objects
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎")))

    dyn_var_present = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_present_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

    SS_and_pars_names = ms.SS_and_pars_names

    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)

    alll = []
    for (i,var) in enumerate(present_varss)
        if !(match(r"^χᵒᵇᶜ", string(var)) === nothing)
            push!(alll,:($var = Y[$(dyn_var_present_idx[i]),1:max(periods, 1)]))
        end
    end

    calc_obc_violation = :(function calculate_obc_violation(x, p)
        state, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values = p

        T = 𝓂.constants.post_model_macro

        Y = zeros(typeof(x[1]), T.nVars, periods+1)

        shock_values = convert(typeof(x), shock_values)

        shock_values[contains.(string.(T.exo),"ᵒᵇᶜ")] .= x

        zero_shock = zero(shock_values)

        if algorithm ∈ [:pruned_second_order, :pruned_third_order]
            states = state_update(state, shock_values)
            Y[:,1] = sum(states)
        else
            Y[:,1] = state_update(state, shock_values)
        end

        for t in 1:periods
            if algorithm ∈ [:pruned_second_order, :pruned_third_order]
                states = state_update(states, zero_shock)
                Y[:,t+1] = sum(states)
            else
                Y[:,t+1] = state_update(Y[:,t], zero_shock)
            end
        end

        Y .+= reference_steady_state[1:T.nVars]

        $(alll...)

        constraint_values = Vector[]

        $(𝓂.equations.obc_violation...)

        return vcat(constraint_values...)
    end)

    𝓂.functions.obc_violation = @RuntimeGeneratedFunction(calc_obc_violation)

    # ── Extract OBC constraint metadata for the analytical Jacobian ──
    # Build mapping: χᵒᵇᶜ variable name (without ₍₀₎) → Y row index
    chi_row_map = Dict{String, Int}()
    for (i, var) in enumerate(present_varss)
        vstr = string(var)
        if startswith(vstr, "χᵒᵇᶜ")
            name = replace(vstr, "₍₀₎" => "")
            chi_row_map[name] = dyn_var_present_idx[i]
        end
    end

    # Pair left/right χᵒᵇᶜ variables by constraint key
    left_vars  = Dict{String, String}()
    right_vars = Dict{String, String}()
    for name in keys(chi_row_map)
        if endswith(name, "ˡ")
            key = name[1:prevind(name, lastindex(name))]
            left_vars[key] = name
        elseif endswith(name, "ʳ")
            key = name[1:prevind(name, lastindex(name))]
            right_vars[key] = name
        end
    end

    obc_info = Tuple{Int, Int, Float64}[]
    for key in sort(collect(keys(left_vars)))
        if haskey(right_vars, key)
            left_idx  = chi_row_map[left_vars[key]]
            right_idx = chi_row_map[right_vars[key]]
            sign = contains(key, "⁺") ? 1.0 : -1.0   # max → +1, min → −1
            push!(obc_info, (left_idx, right_idx, sign))
        end
    end
    𝓂.functions.obc_constraint_info = obc_info

    return nothing
end


# ── NLopt objective & constraint callbacks ───────────────────────────────────

function obc_objective_optim_fun(X::Vector{S}, grad::Vector{S})::S where S
    if length(grad) > 0
        grad .= 2 .* X
    end
    
    sum(abs2, X)
end

function obc_constraint_optim_fun(res::Vector{S}, X::Vector{S}, jac::Matrix{S}, p) where S
    𝓂 = p[4]

    res .= 𝓂.functions.obc_violation(X, p)

    if length(jac) > 0
        compute_obc_analytical_jacobian!(jac, X, p)
    end

	return nothing
end


# ── Analytical OBC Jacobian ──────────────────────────────────────────────────
#
# The OBC constraint vector has, per constraint, three blocks:
#   1.  [sum(a .* b)]           (1 element — complementary slackness)
#   2.  sign * a                (P elements — left argument)
#   3.  sign * b                (P elements — right argument)
# where a = Y[left_row, 1:P], b = Y[right_row, 1:P], and
#       sign = +1 for max, −1 for min.
#
# Y is the forward path simulated through state_update, which is a known
# function of the perturbation solution matrices.  dY/dx is therefore
# computed analytically (exactly for all algorithm orders).

function compute_obc_analytical_jacobian!(jac::Matrix{S}, X::Vector{S}, p) where S
    state, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values = p
    T    = 𝓂.constants.post_model_macro
    nv   = T.nVars
    past_idx = T.past_not_future_and_mixed_idx
    n_past   = T.nPast_not_future_and_mixed
    n_x  = length(X)
    P    = max(periods, 1)

    obc_idx    = findall(contains.(string.(T.exo), "ᵒᵇᶜ"))
    shock_vals = copy(shock_values)
    shock_vals[obc_idx] .= X
    n_shocks   = length(shock_vals)
    zero_shock = zero(shock_vals)

    Ŝ₁ = 𝓂.caches.first_order_obc_solution_matrix

    Y    = zeros(S, nv, periods + 1)
    dYdx = zeros(S, nv, n_x, periods + 1)

    if algorithm == :first_order
        obc_dYdx_first_order!(Y, dYdx, state, shock_vals, zero_shock,
                               past_idx, n_past, obc_idx, Ŝ₁, periods)

    elseif algorithm ∈ [:second_order, :third_order]
        obc_dYdx_nonpruned_higher!(Y, dYdx, state, shock_vals, zero_shock,
                                    past_idx, n_past, n_shocks, obc_idx,
                                    Ŝ₁, 𝓂, algorithm, periods)

    elseif algorithm ∈ [:pruned_second_order, :pruned_third_order]
        obc_dYdx_pruned!(Y, dYdx, state, shock_vals, zero_shock,
                          past_idx, n_past, n_shocks, obc_idx,
                          Ŝ₁, 𝓂, algorithm, periods)
    end

    Y .+= @view reference_steady_state[1:nv]

    fill_obc_constraint_jacobian!(jac, Y, dYdx,
                                   𝓂.functions.obc_constraint_info, n_x, P)
    return nothing
end


# ── First-order: purely linear propagation ───────────────────────────────────
function obc_dYdx_first_order!(Y, dYdx, state, shock_vals, zero_shock,
                                past_idx, n_past, obc_idx, Ŝ₁, periods)
    A = @view Ŝ₁[:, 1:n_past]
    Y[:, 1] = Ŝ₁ * [state[past_idx]; shock_vals]
    dYdx[:, :, 1] .= @view Ŝ₁[:, n_past .+ obc_idx]
    for t in 1:periods
        Y[:, t+1] = A * Y[past_idx, t]
        dYdx[:, :, t+1] = A * dYdx[past_idx, :, t]
    end
end


# ── Non-pruned second / third order ─────────────────────────────────────────
function obc_dYdx_nonpruned_higher!(Y, dYdx, state, shock_vals, zero_shock,
                                     past_idx, n_past, n_shocks, obc_idx,
                                     Ŝ₁, 𝓂, algorithm, periods)
    S = eltype(Y)
    nv = size(Y, 1)
    n_x = size(dYdx, 2)
    𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
    Ŝ₁̂ = [Ŝ₁[:, 1:n_past] zeros(S, nv) Ŝ₁[:, n_past+1:end]]
    n_aug = n_past + 1 + n_shocks

    has_third = algorithm == :third_order
    𝐒₃ = has_third ? 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃ : nothing

    # ── t = 0 ──
    aug = [state[past_idx]; one(S); shock_vals]
    kron_aug = ℒ.kron(aug, aug)
    Y[:, 1] = Ŝ₁̂ * aug + 𝐒₂ * kron_aug / 2
    if has_third;  Y[:, 1] += 𝐒₃ * ℒ.kron(kron_aug, aug) / 6;  end

    d_aug = zeros(S, n_aug)
    for j in 1:n_x
        fill!(d_aug, zero(S))
        d_aug[n_past + 1 + obc_idx[j]] = one(S)
        dYdx[:, j, 1] = Ŝ₁̂ * d_aug +
                         𝐒₂ * (ℒ.kron(d_aug, aug) + ℒ.kron(aug, d_aug)) / 2
        if has_third
            dYdx[:, j, 1] += 𝐒₃ * (ℒ.kron(ℒ.kron(d_aug, aug), aug) +
                                    ℒ.kron(ℒ.kron(aug, d_aug), aug) +
                                    ℒ.kron(kron_aug, d_aug)) / 6
        end
    end

    # ── t > 0 ──
    d_aug_t = zeros(S, n_aug)
    for t in 1:periods
        aug_t    = [Y[past_idx, t]; one(S); zeros(S, n_shocks)]
        kron_aug_t = ℒ.kron(aug_t, aug_t)
        Y[:, t+1] = Ŝ₁̂ * aug_t + 𝐒₂ * kron_aug_t / 2
        if has_third;  Y[:, t+1] += 𝐒₃ * ℒ.kron(kron_aug_t, aug_t) / 6;  end

        for j in 1:n_x
            fill!(d_aug_t, zero(S))
            d_aug_t[1:n_past] .= @view dYdx[past_idx, j, t]
            dYdx[:, j, t+1] = Ŝ₁̂ * d_aug_t +
                              𝐒₂ * (ℒ.kron(d_aug_t, aug_t) + ℒ.kron(aug_t, d_aug_t)) / 2
            if has_third
                dYdx[:, j, t+1] += 𝐒₃ * (ℒ.kron(ℒ.kron(d_aug_t, aug_t), aug_t) +
                                         ℒ.kron(ℒ.kron(aug_t, d_aug_t), aug_t) +
                                         ℒ.kron(kron_aug_t, d_aug_t)) / 6
            end
        end
    end
end


# ── Pruned second / third order ─────────────────────────────────────────────
function obc_dYdx_pruned!(Y, dYdx, state, shock_vals, zero_shock,
                           past_idx, n_past, n_shocks, obc_idx,
                           Ŝ₁, 𝓂, algorithm, periods)
    S = eltype(Y)
    nv = size(Y, 1)
    n_x = size(dYdx, 2)
    𝐒₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
    Ŝ₁̂ = [Ŝ₁[:, 1:n_past] zeros(S, nv) Ŝ₁[:, n_past+1:end]]
    n_aug = n_past + 1 + n_shocks

    has_third = algorithm == :pruned_third_order
    𝐒₃ = has_third ? 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃ : nothing

    # Component vectors
    y₁ = state isa AbstractVector{<:AbstractVector} ? state[1] : state
    y₂ = state isa AbstractVector{<:AbstractVector} ? state[2] : zeros(S, nv)
    y₃ = (has_third && state isa AbstractVector{<:AbstractVector} && length(state) >= 3) ?
         state[3] : zeros(S, nv)

    dy₁dx = zeros(S, nv, n_x)
    dy₂dx = zeros(S, nv, n_x)
    dy₃dx = zeros(S, nv, n_x)

    d_aug = zeros(S, n_aug)

    # ── t = 0 ──
    aug₁ = [y₁[past_idx]; one(S); shock_vals]
    y₁_new = Ŝ₁̂ * aug₁

    aug₂ = [y₂[past_idx]; zero(S); zeros(S, n_shocks)]
    kron_aug₁ = ℒ.kron(aug₁, aug₁)
    y₂_new = Ŝ₁̂ * aug₂ + 𝐒₂ * kron_aug₁ / 2

    for j in 1:n_x
        fill!(d_aug, zero(S))
        d_aug[n_past + 1 + obc_idx[j]] = one(S)
        dy₁dx[:, j] = Ŝ₁̂ * d_aug
        # dy₂ only depends on aug₁ perturbation (aug₂ initial is independent of x)
        dy₂dx[:, j] = 𝐒₂ * (ℒ.kron(d_aug, aug₁) + ℒ.kron(aug₁, d_aug)) / 2
    end

    if has_third
        aug₁̂ = [y₁[past_idx]; zero(S); shock_vals]
        aug₃ = [y₃[past_idx]; zero(S); zeros(S, n_shocks)]
        y₃_new = Ŝ₁̂ * aug₃ + 𝐒₂ * ℒ.kron(aug₁̂, aug₂) + 𝐒₃ * ℒ.kron(kron_aug₁, aug₁) / 6

        for j in 1:n_x
            fill!(d_aug, zero(S))
            d_aug[n_past + 1 + obc_idx[j]] = one(S)
            d_aug₁̂ = copy(d_aug);  d_aug₁̂[n_past + 1] = zero(S)  # hat: zero for the "1" slot
            dy₃dx[:, j] = 𝐒₂ * (ℒ.kron(d_aug₁̂, aug₂) + ℒ.kron(aug₁̂, zeros(S, n_aug))) +
                          𝐒₃ * (ℒ.kron(ℒ.kron(d_aug, aug₁), aug₁) +
                                 ℒ.kron(ℒ.kron(aug₁, d_aug), aug₁) +
                                 ℒ.kron(kron_aug₁, d_aug)) / 6
        end
        y₃ = y₃_new
    end

    y₁ = y₁_new
    y₂ = y₂_new
    Y[:, 1] = y₁ + y₂
    dYdx[:, :, 1] = dy₁dx + dy₂dx
    if has_third; Y[:, 1] += y₃; dYdx[:, :, 1] += dy₃dx; end

    # ── t > 0 ──
    d_aug_t = zeros(S, n_aug)
    for t in 1:periods
        aug₁_t = [y₁[past_idx]; one(S); zeros(S, n_shocks)]
        kron_aug₁_t = ℒ.kron(aug₁_t, aug₁_t)

        y₁_new = Ŝ₁̂ * aug₁_t
        aug₂_t = [y₂[past_idx]; zero(S); zeros(S, n_shocks)]
        y₂_new = Ŝ₁̂ * aug₂_t + 𝐒₂ * kron_aug₁_t / 2

        dy₁dx_new = zeros(S, nv, n_x)
        dy₂dx_new = zeros(S, nv, n_x)

        for j in 1:n_x
            fill!(d_aug_t, zero(S))
            d_aug_t[1:n_past] .= @view dy₁dx[past_idx, j]
            dy₁dx_new[:, j] = Ŝ₁̂ * d_aug_t

            d_aug₂_t = zeros(S, n_aug)
            d_aug₂_t[1:n_past] .= @view dy₂dx[past_idx, j]
            dy₂dx_new[:, j] = Ŝ₁̂ * d_aug₂_t +
                              𝐒₂ * (ℒ.kron(d_aug_t, aug₁_t) + ℒ.kron(aug₁_t, d_aug_t)) / 2
        end

        if has_third
            aug₁̂_t = [y₁[past_idx]; zero(S); zeros(S, n_shocks)]
            aug₃_t = [y₃[past_idx]; zero(S); zeros(S, n_shocks)]
            y₃_new = Ŝ₁̂ * aug₃_t + 𝐒₂ * ℒ.kron(aug₁̂_t, aug₂_t) + 𝐒₃ * ℒ.kron(kron_aug₁_t, aug₁_t) / 6

            dy₃dx_new = zeros(S, nv, n_x)
            for j in 1:n_x
                fill!(d_aug_t, zero(S))
                d_aug_t[1:n_past] .= @view dy₁dx[past_idx, j]
                d_aug₁̂_t = copy(d_aug_t);  d_aug₁̂_t[n_past + 1] = zero(S)

                d_aug₂_t = zeros(S, n_aug)
                d_aug₂_t[1:n_past] .= @view dy₂dx[past_idx, j]

                d_aug₃_t = zeros(S, n_aug)
                d_aug₃_t[1:n_past] .= @view dy₃dx[past_idx, j]

                dy₃dx_new[:, j] = Ŝ₁̂ * d_aug₃_t +
                                  𝐒₂ * (ℒ.kron(d_aug₁̂_t, aug₂_t) + ℒ.kron(aug₁̂_t, d_aug₂_t)) +
                                  𝐒₃ * (ℒ.kron(ℒ.kron(d_aug_t, aug₁_t), aug₁_t) +
                                         ℒ.kron(ℒ.kron(aug₁_t, d_aug_t), aug₁_t) +
                                         ℒ.kron(kron_aug₁_t, d_aug_t)) / 6
            end
            y₃ = y₃_new
            dy₃dx .= dy₃dx_new
        end

        y₁ = y₁_new
        y₂ = y₂_new
        dy₁dx .= dy₁dx_new
        dy₂dx .= dy₂dx_new

        Y[:, t+1] = y₁ + y₂
        dYdx[:, :, t+1] = dy₁dx + dy₂dx
        if has_third; Y[:, t+1] += y₃; dYdx[:, :, t+1] += dy₃dx; end
    end
end


# ── Fill NLopt Jacobian from dY/dx and constraint structure ──────────────────
function fill_obc_constraint_jacobian!(jac, Y, dYdx, constraint_info, n_x, P)
    row_offset = 0
    for (left_idx, right_idx, sign) in constraint_info
        # Complementary-slackness scalar: sum(Y[left,1:P] .* Y[right,1:P])
        for j in 1:n_x
            val = zero(eltype(jac))
            for t in 1:P
                val += dYdx[left_idx, j, t] * Y[right_idx, t] +
                       Y[left_idx, t] * dYdx[right_idx, j, t]
            end
            jac[j, row_offset + 1] = val
        end

        # Left argument: sign * Y[left, 1:P]
        for j in 1:n_x
            for t in 1:P
                jac[j, row_offset + 1 + t] = sign * dYdx[left_idx, j, t]
            end
        end

        # Right argument: sign * Y[right, 1:P]
        for j in 1:n_x
            for t in 1:P
                jac[j, row_offset + 1 + P + t] = sign * dYdx[right_idx, j, t]
            end
        end

        row_offset += 1 + 2 * P
    end
end


# ── First-order OBC solution ─────────────────────────────────────────────────

function calculate_first_order_obc_solution!(𝓂::ℳ, constants, opts::CalculationOptions)
    # Cache hit: return if valid for current parameters
    if cache_valid_for_parameters(𝓂.caches.valid_for.first_order_obc_solution, 𝓂.parameter_values) &&
       !isempty(𝓂.caches.first_order_obc_solution_matrix)
        return nothing
    end

    write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 1, verbose = false)

    ∇̂₁ = calculate_jacobian(𝓂.parameter_values, 𝓂.caches.non_stochastic_steady_state, 𝓂.caches, 𝓂.functions.jacobian, 𝓂.workspaces, caching = false)

    Ŝ₁, qme_sol, solved = calculate_first_order_solution(∇̂₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution,
                                                        caching = false)

    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    write_parameters_input!(𝓂, :activeᵒᵇᶜshocks => 0, verbose = false)

    # Cache write + stamp
    𝓂.caches.first_order_obc_solution_matrix = Ŝ₁
    𝓂.caches.valid_for.first_order_obc_solution = Float64.(𝓂.parameter_values)

    return nothing
end


# ── OBC state update (per-period NLopt solver) ───────────────────────────────

function obc_state_update(present_states::S, present_shocks::Vector{R}, state_update::F, 𝓂::ℳ, algorithm::Symbol) where {S, R <: Float64, F}
    unconditional_forecast_horizon = 𝓂.constants.post_model_macro.max_obc_horizon

    reference_ss = 𝓂.caches.non_stochastic_steady_state

    obc_shock_idx = contains.(string.(𝓂.constants.post_model_macro.exo),"ᵒᵇᶜ")

    periods_per_shock = 𝓂.constants.post_model_macro.max_obc_horizon + 1

    num_shocks = sum(obc_shock_idx) ÷ periods_per_shock

    p = (present_states, state_update, reference_ss, 𝓂, algorithm, unconditional_forecast_horizon, present_shocks)

    constraints_violated = any(𝓂.functions.obc_violation(zeros(num_shocks*periods_per_shock), p) .> eps(Float32))::Bool

    if constraints_violated
        opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks*periods_per_shock)

        opt.min_objective = obc_objective_optim_fun

        opt.xtol_abs = eps(Float32)
        opt.ftol_abs = eps(Float32)
        opt.maxeval = 500

        upper_bounds = fill(eps(), num_shocks * (1 + 2 * max(unconditional_forecast_horizon, 1)))

        NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p), upper_bounds)

        (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks*periods_per_shock))

        present_shocks[contains.(string.(𝓂.constants.post_model_macro.exo),"ᵒᵇᶜ")] .= x

        constraints_violated = any(𝓂.functions.obc_violation(x, p) .> eps(Float32))::Bool

        solved = !constraints_violated
    else
        solved = true
    end

    present_states = state_update(present_states, present_shocks)::S

    return present_states, present_shocks, solved
end


end # @stable
