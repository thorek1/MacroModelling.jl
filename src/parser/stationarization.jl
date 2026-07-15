struct stationarization_metadata
    trend_drivers::Vector{Symbol}
    trending_variables::Vector{Symbol}
    growth_variables::Vector{Symbol}
    growth_exponents::Dict{Symbol, Vector{Float64}}
    original_equations::Vector{Expr}
    stationary_equations::Vector{Expr}
end

stationary_growth_symbol(name::Symbol) = Symbol(string(name) * "ᴳ")

function stationarization_equation_sides(eq::Expr)
    if eq.head == :(=)
        return eq.args[1], eq.args[2]
    elseif eq.head == :call && eq.args[1] == :(=)
        return eq.args[2], eq.args[3]
    end
    return nothing
end

function timed_reference(node, name::Symbol, timing)
    node isa Expr &&
    node.head == :ref &&
    node.args[1] == name &&
    length(node.args) == 2 &&
    node.args[2] == timing
end

function contains_timed_reference(expr, name::Symbol, timing)
    found = Ref(false)
    postwalk(expr) do node
        found[] |= timed_reference(node, name, timing)
        node
    end
    found[]
end

function additive_terms(expr, sign::Float64 = 1.0)
    if expr isa Expr && expr.head == :call && expr.args[1] == :+
        return vcat((additive_terms(arg, sign) for arg in expr.args[2:end])...)
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :-
        if length(expr.args) == 2
            return additive_terms(expr.args[2], -sign)
        end
        out = additive_terms(expr.args[2], sign)
        for arg in expr.args[3:end]
            append!(out, additive_terms(arg, -sign))
        end
        return out
    end
    return [(sign, expr)]
end

function exact_additive_unit_root(eq::Expr, name::Symbol)
    sides = stationarization_equation_sides(eq)
    sides === nothing && return false
    lhs, rhs = sides
    timed_reference(lhs, name, 0) || return false
    terms = additive_terms(rhs)
    has_lag = any(sign == 1.0 && timed_reference(term, name, -1)
                  for (sign, term) in terms)
    has_lag || return false
    other_reference = Ref(false)
    postwalk(rhs) do node
        if node isa Expr && node.head == :ref && node.args[1] == name &&
           node.args[2] isa Int && node.args[2] != -1
            other_reference[] = true
        end
        node
    end
    other_reference[] && return false
    true
end

function multiplicative_factor(expr, name::Symbol)
    if timed_reference(expr, name, -1)
        return 1
    end
    if expr isa Expr && expr.head == :call && expr.args[1] == :*
        matches = findall(arg -> timed_reference(arg, name, -1), expr.args[2:end])
        length(matches) == 1 || return nothing
        for (index, arg) in enumerate(expr.args[2:end])
            index == matches[1] && continue
            contains_timed_reference(arg, name, -1) && return nothing
            contains_timed_reference(arg, name, 0) && return nothing
        end
        other = [arg for (index, arg) in enumerate(expr.args[2:end]) if index != matches[1]]
        return isempty(other) ? 1 : foldl((a, b) -> Expr(:call, :*, a, b), other)
    end
    return nothing
end

function ratio_subexpression(expr, name::Symbol)
    found = Ref(false)
    postwalk(expr) do node
        if node isa Expr && node.head == :call && node.args[1] == :/ &&
           length(node.args) == 3 &&
           timed_reference(node.args[2], name, 0) &&
           timed_reference(node.args[3], name, -1)
            found[] = true
        end
        node
    end
    found[]
end

function is_trend_driver(equations::Vector{Expr}, name::Symbol)
    any(eq -> ratio_subexpression(eq, name), equations) && return true
    any(equation -> begin
            sides = stationarization_equation_sides(equation)
            sides === nothing && return false
            lhs, rhs = sides
            if !(timed_reference(lhs, name, 0) &&
                 contains_timed_reference(rhs, name, -1))
                return false
            end
            factor = multiplicative_factor(rhs, name)
            factor === nothing && return false
            true
        end, equations)
end

function growth_add(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64}, scale::Float64 = 1.0)
    result = copy(a)
    for (name, value) in b
        result[name] = get(result, name, 0.0) + scale * value
        abs(result[name]) < 1e-12 && delete!(result, name)
    end
    result
end

function growth_equal(a::Dict{Symbol, Float64}, b::Dict{Symbol, Float64})
    growth_add(a, b, -1.0)
end

function evaluate_growth_parameter(expr, parameter_values::Dict{Symbol, Float64})
    expr isa Real && return Float64(expr)
    expr isa Symbol && return get(parameter_values, expr, nothing)
    expr isa Expr && expr.head == :call || return nothing
    op = expr.args[1]
    args = [evaluate_growth_parameter(arg, parameter_values) for arg in expr.args[2:end]]
    any(isnothing, args) && return nothing
    op == :+ && return sum(args)
    op == :- && return length(args) == 1 ? -args[1] : args[1] - sum(args[2:end])
    op == :* && return prod(args)
    op == :/ && return args[1] / args[2]
    op == :^ && return args[1] ^ args[2]
    op == :exp && return exp(args[1])
    op == :log && return log(args[1])
    op == :sqrt && return sqrt(args[1])
    nothing
end

function growth_form(expr, variables::Set{Symbol}, parameter_values::Dict{Symbol, Float64})
    if expr isa Real
        return Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
    elseif expr isa Symbol
        return expr ∈ variables ? Dict(expr => 1.0) : Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
    elseif expr isa Expr && expr.head == :ref
        timing = length(expr.args) == 2 ? expr.args[2] : nothing
        if timing ∈ (:x, :ex, :exo, :exogenous, :ss)
            return Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
        end
        name = expr.args[1]
        return name ∈ variables ? Dict(name => 1.0) : Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
    elseif !(expr isa Expr)
        return Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
    end

    expr.head == :call || throw(ArgumentError("Unsupported expression in balanced-growth restrictions: $(expr)"))
    op = expr.args[1]
    args = expr.args[2:end]

    if op == :+
        isempty(args) && return Dict{Symbol, Float64}(), Dict{Symbol, Float64}[]
        forms = [growth_form(arg, variables, parameter_values) for arg in args]
        restrictions = reduce(vcat, (item[2] for item in forms); init = Dict{Symbol, Float64}[])
        first_form = forms[1][1]
        for item in forms[2:end]
            push!(restrictions, growth_equal(first_form, item[1]))
        end
        return first_form, restrictions
    elseif op == :-
        if length(args) == 1
            form, restrictions = growth_form(args[1], variables, parameter_values)
            return growth_add(Dict{Symbol, Float64}(), form, -1.0), restrictions
        end
        left, left_restrictions = growth_form(args[1], variables, parameter_values)
        right_forms = [growth_form(arg, variables, parameter_values) for arg in args[2:end]]
        restrictions = vcat(left_restrictions, reduce(vcat, (item[2] for item in right_forms); init = Dict{Symbol, Float64}[]))
        for item in right_forms
            push!(restrictions, growth_equal(left, item[1]))
        end
        return left, restrictions
    elseif op == :*
        result = Dict{Symbol, Float64}()
        restrictions = Dict{Symbol, Float64}[]
        for arg in args
            form, local_restrictions = growth_form(arg, variables, parameter_values)
            result = growth_add(result, form)
            append!(restrictions, local_restrictions)
        end
        return result, restrictions
    elseif op == :/
        length(args) == 2 || throw(ArgumentError("Only binary division is supported in balanced-growth restrictions: $(expr)"))
        left, left_restrictions = growth_form(args[1], variables, parameter_values)
        right, right_restrictions = growth_form(args[2], variables, parameter_values)
        return growth_add(left, right, -1.0), vcat(left_restrictions, right_restrictions)
    elseif op == :^
        length(args) == 2 || throw(ArgumentError("Only binary powers are supported in balanced-growth restrictions: $(expr)"))
        base, base_restrictions = growth_form(args[1], variables, parameter_values)
        exponent, exponent_restrictions = growth_form(args[2], variables, parameter_values)
        isempty(exponent) || throw(ArgumentError("A non-stationary exponent is not supported by symbolic stationarization: $(expr)"))
        numeric_exponent = evaluate_growth_parameter(args[2], parameter_values)
        numeric_exponent === nothing && throw(ArgumentError("Power exponents must be numeric or parameter-resolved before stationarization: $(expr)"))
        return growth_add(Dict{Symbol, Float64}(), base, numeric_exponent),
               vcat(base_restrictions, exponent_restrictions)
    elseif op == :sqrt
        form, restrictions = growth_form(args[1], variables, parameter_values)
        return growth_add(Dict{Symbol, Float64}(), form, 0.5), restrictions
    elseif op ∈ (:exp, :log)
        form, restrictions = growth_form(args[1], variables, parameter_values)
        push!(restrictions, form)
        return Dict{Symbol, Float64}(), restrictions
    end

    throw(ArgumentError("Unsupported function in balanced-growth restrictions: $(op)"))
end

function expression_symbols(expr)
    symbols = Set{Symbol}()
    postwalk(expr) do node
        if node isa Expr && node.head == :ref &&
           length(node.args) == 2 &&
           node.args[2] ∉ (:x, :ex, :exo, :exogenous, :ss)
            push!(symbols, node.args[1])
        end
        node
    end
    symbols
end

function collect_growth_restrictions(equations::Vector{Expr},
                                     variables::Set{Symbol},
                                     parameter_values::Dict{Symbol, Float64})
    restrictions = Dict{Symbol, Float64}[]
    for equation in equations
        sides = stationarization_equation_sides(equation)
        sides === nothing && continue
        lhs, rhs = sides
        lhs_form, lhs_restrictions = growth_form(lhs, variables, parameter_values)
        rhs_form, rhs_restrictions = growth_form(rhs, variables, parameter_values)
        append!(restrictions, lhs_restrictions)
        append!(restrictions, rhs_restrictions)
        push!(restrictions, growth_equal(lhs_form, rhs_form))
    end
    filter!(!isempty, restrictions)
    unique(restrictions)
end

function restriction_matrix(restrictions, variables)
    result = zeros(Float64, length(restrictions), length(variables))
    for (row, restriction) in enumerate(restrictions)
        for (column, variable) in enumerate(variables)
            result[row, column] = get(restriction, variable, 0.0)
        end
    end
    result
end

function growth_coefficient_matrix(equations::Vector{Expr},
                                   variables::Vector{Symbol},
                                   drivers::Vector{Symbol},
                                   parameter_values::Dict{Symbol, Float64})
    restrictions = collect_growth_restrictions(equations, Set(variables), parameter_values)
    rows = Dict{Symbol, Float64}[]
    append!(rows, restrictions)
    mentioned = reduce(union, (Set(keys(row)) for row in restrictions); init = Set{Symbol}())
    for variable in setdiff(variables, drivers, mentioned)
        push!(rows, Dict(variable => 1.0))
    end
    for driver in drivers
        push!(rows, Dict(driver => 1.0))
    end

    matrix = restriction_matrix(rows, variables)
    ℒ.rank(matrix) == length(variables) ||
        throw(ArgumentError("Balanced-growth restrictions are rank deficient; cannot identify all trend rates. Variables: $(variables)"))

    n_drivers = length(drivers)
    coefficients = zeros(Float64, length(variables), n_drivers)
    for driver_index in 1:n_drivers
        rhs = zeros(Float64, size(matrix, 1))
        rhs[end - n_drivers + driver_index] = 1.0
        solution = matrix \ rhs
        ℒ.norm(matrix * solution - rhs) < 1e-7 ||
            throw(ArgumentError("Balanced-growth restrictions are inconsistent for trend driver $(drivers[driver_index])."))
        coefficients[:, driver_index] .= solution
    end
    coefficients
end

function growth_factor_expression(name::Symbol,
                                  timing::Int,
                                  drivers::Vector{Symbol},
                                  coefficients::Dict{Symbol, Vector{Float64}})
    exponents = get(coefficients, name, zeros(length(drivers)))
    factors = Any[]
    if timing > 0
        append!(factors, (Expr(:ref, stationary_growth_symbol(driver), shift)
                          for shift in 1:timing for (index, driver) in enumerate(drivers)
                          if abs(exponents[index]) > 1e-10))
        factor_exponents = [exponents[index] for shift in 1:timing for (index, driver) in enumerate(drivers)
                            if abs(exponents[index]) > 1e-10]
    elseif timing < 0
        append!(factors, (Expr(:ref, stationary_growth_symbol(driver), shift)
                          for shift in (timing + 1):0 for (index, driver) in enumerate(drivers)
                          if abs(exponents[index]) > 1e-10))
        factor_exponents = [-exponents[index] for shift in (timing + 1):0 for (index, driver) in enumerate(drivers)
                            if abs(exponents[index]) > 1e-10]
    else
        return 1
    end

    result = 1
    factor_index = 1
    for factor in factors
        exponent = factor_exponents[factor_index]
        result = exponent == 1.0 ? Expr(:call, :*, result, factor) :
                 Expr(:call, :*, result, Expr(:call, :^, factor, exponent))
        factor_index += 1
    end
    result == 1 ? 1 : result
end

function driver_growth_factor_expression(name::Symbol, timing::Int)
    growth_name = stationary_growth_symbol(name)
    timing == 0 && return 1

    factors = if timing > 0
        [Expr(:ref, growth_name, shift) for shift in 1:timing]
    else
        [Expr(:ref, growth_name, shift) for shift in (timing + 1):0]
    end

    result = 1
    for factor in factors
        result = timing > 0 ?
            Expr(:call, :*, result, factor) :
            Expr(:call, :/, result, factor)
    end
    result
end

function stationarize_expression(expr,
                                 drivers::Vector{Symbol},
                                 coefficients::Dict{Symbol, Vector{Float64}})
    postwalk(expr) do node
        if node isa Expr && node.head == :ref && length(node.args) == 2 &&
           node.args[2] isa Int
            name = node.args[1]
            timing = node.args[2]
            if name ∈ drivers
                driver_growth_factor_expression(name, timing)
            elseif name ∈ keys(coefficients)
                factor = growth_factor_expression(name, timing, drivers, coefficients)
                factor == 1 ? node : Expr(:call, :*, node, factor)
            else
                node
            end
        else
            node
        end
    end
end

function driver_growth_equation(eq::Expr, name::Symbol)
    sides = stationarization_equation_sides(eq)
    sides === nothing && return nothing
    lhs, rhs = sides
    growth_ref = Expr(:ref, stationary_growth_symbol(name), 0)

    if timed_reference(lhs, name, 0)
        factor = multiplicative_factor(rhs, name)
        factor === nothing && return nothing
        return Expr(:(=), growth_ref, factor)
    end

    if lhs isa Expr && lhs.head == :call && lhs.args[1] == :/ &&
       length(lhs.args) == 3 &&
       timed_reference(lhs.args[2], name, 0) &&
       timed_reference(lhs.args[3], name, -1)
        return Expr(:(=), growth_ref, rhs)
    end

    nothing
end

function stationarization_candidates(equations::Vector{Expr})
    variables = sort(collect(reduce(union, (expression_symbols(eq) for eq in equations); init = Set{Symbol}())))
    drivers = [name for name in variables if is_trend_driver(equations, name)]
    additive = [name for name in variables if any(eq -> exact_additive_unit_root(eq, name), equations)]
    variables, drivers, additive
end

function build_stationarization_metadata(raw_equations::Vector{Expr},
                                         parameter_values::Dict{Symbol, Float64} = Dict{Symbol, Float64}())
    variables, drivers, additive = stationarization_candidates(raw_equations)
    isempty(drivers) && !isempty(additive) &&
        throw(ArgumentError("Pure additive BGP variables $(additive) are not supported by symbolic stationarization; write their trend as a positive multiplicative growth factor."))
    isempty(drivers) && return nothing, raw_equations

    coefficients_array = growth_coefficient_matrix(raw_equations, variables, drivers, parameter_values)
    coefficients = Dict(name => vec(coefficients_array[index, :]) for (index, name) in enumerate(variables))
    trending = [name for name in variables if any(abs.(get(coefficients, name, zeros(length(drivers)))) .> 1e-10)]
    stationary_equations = Expr[]
    for equation in raw_equations
        growth_equation = nothing
        for driver in drivers
            growth_equation = driver_growth_equation(equation, driver)
            growth_equation === nothing || break
        end
        push!(stationary_equations,
              growth_equation === nothing ?
              stationarize_expression(equation, drivers, coefficients) :
              growth_equation)
    end
    append!(stationary_equations,
            [Expr(:(=), Expr(:ref, driver, 0), 1) for driver in drivers])

    metadata = stationarization_metadata(
        drivers,
        trending,
        stationary_growth_symbol.(drivers),
        coefficients,
        copy(raw_equations),
        copy(stationary_equations),
    )
    metadata, stationary_equations
end

function stationarize_model!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    raw_equations = copy(𝓂.equations.original)
    parameter_values = Dict{Symbol, Float64}(
        𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values,
    )
    metadata, stationary_equations = build_stationarization_metadata(raw_equations, parameter_values)
    𝓂.equations.stationarization = metadata
    metadata === nothing && return false

    old_equations = 𝓂.equations
    old_constants = 𝓂.constants
    T, equations_struct, constants, workspaces = process_model_equations(
        Expr(:block, stationary_equations...),
        old_constants.post_model_macro.max_obc_horizon,
        old_constants.post_parameters_macro.precompile,
        allow_single_variable_equations = true,
    )

    equations_struct.original = raw_equations
    equations_struct.ss_anchors = old_equations.ss_anchors
    equations_struct.stationarization = metadata
    equations_struct.calibration = old_equations.calibration
    equations_struct.calibration_no_var = old_equations.calibration_no_var
    equations_struct.calibration_parameters = old_equations.calibration_parameters
    equations_struct.calibration_original = old_equations.calibration_original

    𝓂.constants = constants
    𝓂.constants.post_parameters_macro = old_constants.post_parameters_macro
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        parameters = old_constants.post_complete_parameters.parameters,
        missing_parameters = old_constants.post_complete_parameters.missing_parameters,
    )
    𝓂.equations = equations_struct
    𝓂.workspaces = workspaces
    reset_solver_state!(𝓂)
    verbose && !silent && println("Stationarized model with trend drivers: $(metadata.trend_drivers)")
    true
end
