const BGPExponent = Union{Float64, Int, Symbol, Expr}

struct stationarization_metadata
    trend_drivers::Vector{Symbol}
    trending_variables::Vector{Symbol}
    growth_variables::Vector{Symbol}
    growth_exponents::Dict{Symbol, Vector{Float64}}
    growth_exponent_expressions::Dict{Symbol, Vector{BGPExponent}}
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
    return Tuple{Float64, Any}[(sign, expr)]
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

function evaluate_growth_parameter(expr,
                                  parameter_values::AbstractVector{<:Real},
                                  parameter_indices::Dict{Symbol, Int})
    expr isa Real && return Float64(expr)
    if expr isa Symbol
        index = get(parameter_indices, expr, 0)
        return index == 0 ? nothing : parameter_values[index]
    end
    expr isa Expr && expr.head == :call || return nothing
    op = expr.args[1]
    args = [evaluate_growth_parameter(arg, parameter_values, parameter_indices)
            for arg in expr.args[2:end]]
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

function symbolic_parameter_expression(expr)::Union{Nothing, BGPExponent}
    expr isa Integer && return Int(expr)
    expr isa AbstractFloat && return Float64(expr)
    expr isa Symbol && return expr
    expr isa Expr && expr.head == :call || return nothing
    args = BGPExponent[]
    for arg in expr.args[2:end]
        value = symbolic_parameter_expression(arg)
        value === nothing && return nothing
        push!(args, value)
    end
    Expr(:call, expr.args[1], args...)
end

symbolic_is_zero(value) = value isa Number && iszero(value)
symbolic_is_one(value) = value isa Number && isone(value)

function symbolic_add(left::BGPExponent, right::BGPExponent)
    symbolic_is_zero(left) && return right
    symbolic_is_zero(right) && return left
    left isa Number && right isa Number && return Float64(left + right)
    Expr(:call, :+, left, right)
end

function symbolic_subtract(left::BGPExponent, right::BGPExponent)
    symbolic_is_zero(right) && return left
    left isa Number && right isa Number && return Float64(left - right)
    Expr(:call, :-, left, right)
end

function symbolic_multiply(left::BGPExponent, right::BGPExponent)
    (symbolic_is_zero(left) || symbolic_is_zero(right)) && return 0.0
    symbolic_is_one(left) && return right
    symbolic_is_one(right) && return left
    left isa Number && right isa Number && return Float64(left * right)
    Expr(:call, :*, left, right)
end

function symbolic_divide(left::BGPExponent, right::BGPExponent)
    symbolic_is_zero(left) && return 0.0
    symbolic_is_one(right) && return left
    left isa Number && right isa Number && return Float64(left / right)
    Expr(:call, :/, left, right)
end

function symbolic_growth_add(a::Dict{Symbol, BGPExponent},
                             b::Dict{Symbol, BGPExponent},
                             scale::BGPExponent = 1.0)
    result = copy(a)
    for (name, value) in b
        term = symbolic_multiply(scale, value)
        updated = symbolic_add(get(result, name, 0.0), term)
        symbolic_is_zero(updated) ? delete!(result, name) : (result[name] = updated)
    end
    result
end

function symbolic_growth_equal(a::Dict{Symbol, BGPExponent},
                               b::Dict{Symbol, BGPExponent})
    symbolic_growth_add(a, b, -1.0)
end

function symbolic_growth_form(expr, variables::Set{Symbol})
    if expr isa Real
        return Dict{Symbol, BGPExponent}(), Dict{Symbol, BGPExponent}[]
    elseif expr isa Symbol
        return expr ∈ variables ? Dict{Symbol, BGPExponent}(expr => 1.0) : Dict{Symbol, BGPExponent}(),
               Dict{Symbol, BGPExponent}[]
    elseif expr isa Expr && expr.head == :ref
        timing = length(expr.args) == 2 ? expr.args[2] : nothing
        if timing ∈ (:x, :ex, :exo, :exogenous, :ss)
            return Dict{Symbol, BGPExponent}(), Dict{Symbol, BGPExponent}[]
        end
        name = expr.args[1]
        return name ∈ variables ? Dict{Symbol, BGPExponent}(name => 1.0) : Dict{Symbol, BGPExponent}(),
               Dict{Symbol, BGPExponent}[]
    elseif !(expr isa Expr)
        return Dict{Symbol, BGPExponent}(), Dict{Symbol, BGPExponent}[]
    end

    expr.head == :call || throw(ArgumentError("Unsupported expression in balanced-growth restrictions: $(expr)"))
    op = expr.args[1]
    args = expr.args[2:end]

    if op == :+
        isempty(args) && return Dict{Symbol, BGPExponent}(), Dict{Symbol, BGPExponent}[]
        forms = [symbolic_growth_form(arg, variables) for arg in args]
        restrictions = reduce(vcat, (item[2] for item in forms); init = Dict{Symbol, BGPExponent}[])
        first_form = forms[1][1]
        for item in forms[2:end]
            push!(restrictions, symbolic_growth_equal(first_form, item[1]))
        end
        return first_form, restrictions
    elseif op == :-
        if length(args) == 1
            form, restrictions = symbolic_growth_form(args[1], variables)
            return symbolic_growth_add(Dict{Symbol, BGPExponent}(), form, -1.0), restrictions
        end
        left, left_restrictions = symbolic_growth_form(args[1], variables)
        right_forms = [symbolic_growth_form(arg, variables) for arg in args[2:end]]
        restrictions = vcat(left_restrictions,
                            reduce(vcat, (item[2] for item in right_forms);
                                   init = Dict{Symbol, BGPExponent}[]))
        for item in right_forms
            push!(restrictions, symbolic_growth_equal(left, item[1]))
        end
        return left, restrictions
    elseif op == :*
        result = Dict{Symbol, BGPExponent}()
        restrictions = Dict{Symbol, BGPExponent}[]
        for arg in args
            form, local_restrictions = symbolic_growth_form(arg, variables)
            result = symbolic_growth_add(result, form)
            append!(restrictions, local_restrictions)
        end
        return result, restrictions
    elseif op == :/
        length(args) == 2 || throw(ArgumentError("Only binary division is supported in balanced-growth restrictions: $(expr)"))
        left, left_restrictions = symbolic_growth_form(args[1], variables)
        right, right_restrictions = symbolic_growth_form(args[2], variables)
        return symbolic_growth_add(left, right, -1.0),
               vcat(left_restrictions, right_restrictions)
    elseif op == :^
        length(args) == 2 || throw(ArgumentError("Only binary powers are supported in balanced-growth restrictions: $(expr)"))
        base, base_restrictions = symbolic_growth_form(args[1], variables)
        exponent, exponent_restrictions = symbolic_growth_form(args[2], variables)
        isempty(exponent) || throw(ArgumentError("A non-stationary exponent is not supported by symbolic stationarization: $(expr)"))
        exponent_expression = symbolic_parameter_expression(args[2])
        exponent_expression === nothing &&
            throw(ArgumentError("Power exponents must be numeric or parameter-resolved before stationarization: $(expr)"))
        return symbolic_growth_add(Dict{Symbol, BGPExponent}(), base, exponent_expression),
               vcat(base_restrictions, exponent_restrictions)
    elseif op == :sqrt
        form, restrictions = symbolic_growth_form(args[1], variables)
        return symbolic_growth_add(Dict{Symbol, BGPExponent}(), form, 0.5), restrictions
    elseif op ∈ (:exp, :log)
        form, restrictions = symbolic_growth_form(args[1], variables)
        push!(restrictions, form)
        return Dict{Symbol, BGPExponent}(), restrictions
    end

    throw(ArgumentError("Unsupported function in balanced-growth restrictions: $(op)"))
end

function collect_symbolic_growth_restrictions(equations::Vector{Expr},
                                              variables::Set{Symbol})
    restrictions = Dict{Symbol, BGPExponent}[]
    for equation in equations
        sides = stationarization_equation_sides(equation)
        sides === nothing && continue
        lhs, rhs = sides
        lhs_form, lhs_restrictions = symbolic_growth_form(lhs, variables)
        rhs_form, rhs_restrictions = symbolic_growth_form(rhs, variables)
        append!(restrictions, lhs_restrictions)
        append!(restrictions, rhs_restrictions)
        push!(restrictions, symbolic_growth_equal(lhs_form, rhs_form))
    end
    filter!(!isempty, restrictions)
    unique(restrictions)
end

function symbolic_restriction_matrix(restrictions, variables)
    result = Matrix{BGPExponent}(undef, length(restrictions), length(variables))
    fill!(result, 0.0)
    for (row, restriction) in enumerate(restrictions)
        for (column, variable) in enumerate(variables)
            result[row, column] = get(restriction, variable, 0.0)
        end
    end
    result
end

function symbolic_matrix_solve(matrix::Matrix{BGPExponent},
                               rhs::Vector{BGPExponent},
                               parameter_values::Dict{Symbol, Float64})
    n = size(matrix, 2)
    augmented = hcat(copy(matrix), reshape(copy(rhs), :, 1))
    for column in 1:n
        pivot = nothing
        for row in column:size(augmented, 1)
            value = evaluate_growth_parameter(augmented[row, column], parameter_values)
            if value !== nothing && isfinite(value) && abs(value) > 1e-10
                pivot = row
                break
            end
        end
        pivot === nothing &&
            throw(ArgumentError("Balanced-growth restrictions are rank deficient at the current parameter values."))
        if pivot != column
            pivot_row = copy(augmented[column, :])
            augmented[column, :] = augmented[pivot, :]
            augmented[pivot, :] = pivot_row
        end

        for row in (column + 1):size(augmented, 1)
            factor = symbolic_divide(augmented[row, column], augmented[column, column])
            for index in column:(n + 1)
                augmented[row, index] = symbolic_subtract(
                    augmented[row, index],
                    symbolic_multiply(factor, augmented[column, index]),
                )
            end
        end
    end

    solution = BGPExponent[0.0 for _ in 1:n]
    for row in n:-1:1
        value = augmented[row, n + 1]
        for column in (row + 1):n
            value = symbolic_subtract(value,
                                      symbolic_multiply(augmented[row, column],
                                                        solution[column]))
        end
        solution[row] = symbolic_divide(value, augmented[row, row])
    end
    solution
end

function symbolic_growth_coefficient_matrix(equations::Vector{Expr},
                                            variables::Vector{Symbol},
                                            drivers::Vector{Symbol},
                                            parameter_values::Dict{Symbol, Float64})
    restrictions = collect_symbolic_growth_restrictions(equations, Set(variables))
    rows = Dict{Symbol, BGPExponent}[]
    append!(rows, restrictions)
    mentioned = reduce(union, (Set(keys(row)) for row in restrictions); init = Set{Symbol}())
    for variable in setdiff(variables, drivers, mentioned)
        push!(rows, Dict{Symbol, BGPExponent}(variable => 1.0))
    end
    for driver in drivers
        push!(rows, Dict{Symbol, BGPExponent}(driver => 1.0))
    end

    matrix = symbolic_restriction_matrix(rows, variables)
    numeric_matrix = map(value -> something(evaluate_growth_parameter(value, parameter_values), NaN),
                         matrix)
    all(isfinite, numeric_matrix) ||
        throw(ArgumentError("Power exponents must be numeric or parameter-resolved before stationarization."))
    ℒ.rank(numeric_matrix) == length(variables) ||
        throw(ArgumentError("Balanced-growth restrictions are rank deficient; cannot identify all trend rates. Variables: $(variables)"))

    coefficients = Matrix{BGPExponent}(undef, length(variables), length(drivers))
    for driver_index in eachindex(drivers)
        rhs = BGPExponent[0.0 for _ in axes(matrix, 1)]
        rhs[end - length(drivers) + driver_index] = 1.0
        coefficients[:, driver_index] = symbolic_matrix_solve(matrix, rhs, parameter_values)
        numeric_solution = map(value -> something(evaluate_growth_parameter(value, parameter_values), NaN),
                               coefficients[:, driver_index])
        numeric_rhs = zeros(size(matrix, 1))
        numeric_rhs[end - length(drivers) + driver_index] = 1.0
        ℒ.norm(numeric_matrix * numeric_solution - numeric_rhs) < 1e-7 ||
            throw(ArgumentError("Balanced-growth restrictions are inconsistent for trend driver $(drivers[driver_index])."))
    end
    coefficients
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
                                  coefficients::Dict{Symbol, Vector{BGPExponent}})
    exponents = get(coefficients, name, zeros(length(drivers)))
    factors = Any[]
    if timing > 0
        append!(factors, (Expr(:ref, stationary_growth_symbol(driver), shift)
                          for shift in 1:timing for (index, driver) in enumerate(drivers)
                          if !(exponents[index] isa Number) || abs(exponents[index]) > 1e-10))
        factor_exponents = [exponents[index] for shift in 1:timing for (index, driver) in enumerate(drivers)
                            if !(exponents[index] isa Number) || abs(exponents[index]) > 1e-10]
    elseif timing < 0
        append!(factors, (Expr(:ref, stationary_growth_symbol(driver), shift)
                          for shift in (timing + 1):0 for (index, driver) in enumerate(drivers)
                          if !(exponents[index] isa Number) || abs(exponents[index]) > 1e-10))
        factor_exponents = [exponents[index] isa Number ? -exponents[index] :
                            Expr(:call, :-, exponents[index])
                            for shift in (timing + 1):0 for (index, driver) in enumerate(drivers)
                            if !(exponents[index] isa Number) || abs(exponents[index]) > 1e-10]
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
                                 coefficients::Dict{Symbol, Vector{BGPExponent}})
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

const BGP_CANDIDATE_RATIO = UInt8(1)
const BGP_CANDIDATE_MULTIPLICATIVE = UInt8(2)

function candidate_growth_factor(equations::Vector{Expr}, name::Symbol)
    for equation in equations
        sides = stationarization_equation_sides(equation)
        sides === nothing && continue
        lhs, rhs = sides

        if lhs isa Expr && lhs.head == :call && lhs.args[1] == :/ &&
           length(lhs.args) == 3 &&
           timed_reference(lhs.args[2], name, 0) &&
           timed_reference(lhs.args[3], name, -1)
            return rhs, BGP_CANDIDATE_RATIO
        end

        if timed_reference(lhs, name, 0)
            factor = multiplicative_factor(rhs, name)
            factor === nothing || return factor, BGP_CANDIDATE_MULTIPLICATIVE
        end
    end
    return nothing, UInt8(0)
end

function bare_parameter_symbols(expr, parameter_set::Set{Symbol})
    result = Set{Symbol}()
    postwalk(expr) do node
        node isa Symbol && node ∈ parameter_set && push!(result, node)
        node
    end
    result
end

function timed_variable_symbols(expr)
    result = Set{Symbol}()
    postwalk(expr) do node
        if node isa Expr && node.head == :ref && length(node.args) == 2 &&
           node.args[2] isa Int
            push!(result, node.args[1])
        end
        node
    end
    result
end

function equation_lhs_variable(equation::Expr)
    sides = stationarization_equation_sides(equation)
    sides === nothing && return nothing
    lhs = sides[1]
    lhs isa Expr && lhs.head == :ref && length(lhs.args) == 2 &&
        lhs.args[2] == 0 && return lhs.args[1]
    nothing
end

function candidate_parameter_dependencies(equations::Vector{Expr},
                                          factor,
                                          parameter_set::Set{Symbol})
    dependencies = bare_parameter_symbols(factor, parameter_set)
    pending = collect(timed_variable_symbols(factor))
    visited = Set{Symbol}()

    while !isempty(pending)
        variable = pop!(pending)
        variable ∈ visited && continue
        push!(visited, variable)

        for equation in equations
            equation_lhs_variable(equation) == variable || continue
            union!(dependencies, bare_parameter_symbols(equation, parameter_set))
            append!(pending, timed_variable_symbols(equation))
        end
    end

    dependencies
end

function classify_bgp_candidates(drivers::Vector{Symbol},
                                 factors::Vector{Union{Nothing, Real, Symbol, Expr}},
                                 candidate_kinds::Vector{UInt8},
                                 candidate_has_timed_variables::BitVector,
                                 parameter_values::Vector{Float64},
                                 parameter_indices::Dict{Symbol, Int})
    active_drivers = Symbol[]

    for (index, driver) in enumerate(drivers)
        factor = factors[index]
        kind = candidate_kinds[index]
        factor === nothing && continue

        active = if kind == BGP_CANDIDATE_RATIO || candidate_has_timed_variables[index]
            true
        else
            value = evaluate_growth_parameter(factor, parameter_values, parameter_indices)
            value === nothing || !isfinite(value) ? true : abs(value) >= 1
        end

        active && push!(active_drivers, driver)
    end

    mode = isempty(active_drivers) ? BGP_STATIONARY_MODE : BGP_ACTIVE_MODE
    mode, active_drivers
end

function build_bgp_detection_metadata(raw_equations::Vector{Expr},
                                       parameters::Vector{Symbol},
                                       parameter_values::Vector{Float64})
    _, drivers, additive = stationarization_candidates(raw_equations)
    candidate_kinds = fill(UInt8(0), length(drivers))
    candidate_factors = Vector{Union{Nothing, Real, Symbol, Expr}}(undef, length(drivers))
    candidate_has_timed_variables = falses(length(drivers))
    parameter_set = Set(parameters)
    parameter_indices = Dict(parameter => index for (index, parameter) in enumerate(parameters))
    trigger_set = Set{Symbol}()

    for (index, driver) in enumerate(drivers)
        factor, kind = candidate_growth_factor(raw_equations, driver)
        candidate_kinds[index] = kind
        candidate_factors[index] = factor
        factor === nothing && continue
        candidate_has_timed_variables[index] = !isempty(timed_variable_symbols(factor))
        union!(trigger_set,
               candidate_parameter_dependencies(raw_equations, factor, parameter_set))
    end

    trigger_parameters = sort!(collect(trigger_set), by = parameter -> findfirst(==(parameter), parameters))
    trigger_indices = Int[]
    for parameter in trigger_parameters
        index = findfirst(==(parameter), parameters)
        index === nothing || push!(trigger_indices, index)
    end
    trigger_values = parameter_values[trigger_indices]

    mode, active_drivers = classify_bgp_candidates(
        drivers,
        candidate_factors,
        candidate_kinds,
        candidate_has_timed_variables,
        parameter_values,
        parameter_indices,
    )
    !isempty(additive) && (mode = BGP_UNSUPPORTED_MODE)

    bgp_detection_metadata(
        drivers,
        active_drivers,
        candidate_kinds,
        candidate_factors,
        candidate_has_timed_variables,
        additive,
        trigger_parameters,
        trigger_indices,
        copy(trigger_values),
        parameter_indices,
        mode,
    )
end

function refresh_bgp_detection!(𝓂::ℳ)
    profile = 𝓂.equations.bgp_detection
    profile === nothing && return false

    values = 𝓂.parameter_values
    changed = length(profile.trigger_values) != length(profile.trigger_indices)
    if !changed
        @inbounds for index in eachindex(profile.trigger_indices)
            changed |= values[profile.trigger_indices[index]] != profile.trigger_values[index]
        end
    end
    changed || return false

    old_mode = profile.mode
    old_active_drivers = profile.active_drivers
    mode, active_drivers = classify_bgp_candidates(
        profile.candidate_drivers,
        profile.candidate_factors,
        profile.candidate_kinds,
        profile.candidate_has_timed_variables,
        values,
        profile.parameter_indices,
    )
    new_mode = !isempty(profile.additive_candidates) ?
                BGP_UNSUPPORTED_MODE : mode
    new_mode == BGP_UNSUPPORTED_MODE &&
        throw(ArgumentError("The updated parameter values imply an unsupported balanced-growth path."))
    profile.active_drivers = active_drivers
    profile.trigger_values .= values[profile.trigger_indices]
    profile.mode = new_mode

    old_mode != new_mode || old_active_drivers != active_drivers
end

function build_stationarization_metadata(raw_equations::Vector{Expr},
                                         parameter_values::Dict{Symbol, Float64} = Dict{Symbol, Float64}();
                                         drivers_override::Union{Nothing, Vector{Symbol}} = nothing)
    variables, drivers, additive = stationarization_candidates(raw_equations)
    !isempty(additive) &&
        throw(ArgumentError("Additive BGP variables $(additive) are not supported by symbolic stationarization; write their trend as a positive multiplicative growth factor."))
    drivers = drivers_override === nothing ? drivers : drivers_override
    isempty(drivers) && return nothing, raw_equations

    coefficient_expressions = symbolic_growth_coefficient_matrix(
        raw_equations,
        variables,
        drivers,
        parameter_values,
    )
    coefficients = Dict(
        name => Float64[
            something(evaluate_growth_parameter(value, parameter_values), NaN)
            for value in coefficient_expressions[index, :]
        ]
        for (index, name) in enumerate(variables)
    )
    coefficient_expressions_by_name = Dict(
        name => BGPExponent[value for value in coefficient_expressions[index, :]]
        for (index, name) in enumerate(variables)
    )
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
              stationarize_expression(equation, drivers, coefficient_expressions_by_name) :
              growth_equation)
    end
    append!(stationary_equations,
            [Expr(:(=), Expr(:ref, driver, 0), 1) for driver in drivers])

    metadata = stationarization_metadata(
        drivers,
        trending,
        stationary_growth_symbol.(drivers),
        coefficients,
        coefficient_expressions_by_name,
        copy(raw_equations),
        copy(stationary_equations),
    )
    metadata, stationary_equations
end

function stationarize_model!(𝓂::ℳ; verbose::Bool = false, silent::Bool = true)
    raw_equations = copy(𝓂.equations.original)
    profile = build_bgp_detection_metadata(
        raw_equations,
        𝓂.constants.post_complete_parameters.parameters,
        𝓂.parameter_values,
    )
    𝓂.equations.bgp_detection = profile
    profile.mode == BGP_UNSUPPORTED_MODE &&
        throw(ArgumentError("The model contains an additive or unsupported balanced-growth path."))
    if profile.mode == BGP_STATIONARY_MODE
        𝓂.equations.stationarization = nothing
        return false
    end

    parameter_values = Dict{Symbol, Float64}(
        𝓂.constants.post_complete_parameters.parameters .=> 𝓂.parameter_values,
    )
    metadata, stationary_equations = build_stationarization_metadata(
        raw_equations,
        parameter_values;
        drivers_override = profile.active_drivers,
    )
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
    equations_struct.bgp_detection = profile
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

function restore_raw_model!(𝓂::ℳ)
    raw_equations = copy(𝓂.equations.original)
    old_equations = 𝓂.equations
    old_constants = 𝓂.constants
    T, equations_struct, constants, workspaces = process_model_equations(
        Expr(:block, raw_equations...),
        old_constants.post_model_macro.max_obc_horizon,
        old_constants.post_parameters_macro.precompile,
    )

    equations_struct.ss_anchors = old_equations.ss_anchors
    equations_struct.bgp_detection = old_equations.bgp_detection
    equations_struct.stationarization = nothing
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
    return nothing
end

function refresh_bgp_numeric_state!(𝓂::ℳ)
    metadata = 𝓂.equations.stationarization
    metadata === nothing && return false
    profile = 𝓂.equations.bgp_detection
    profile === nothing && return false
    for (name, expressions) in metadata.growth_exponent_expressions
        exponents = get!(metadata.growth_exponents, name) do
            zeros(Float64, length(expressions))
        end
        length(exponents) == length(expressions) || resize!(exponents, length(expressions))
        for index in eachindex(expressions)
            value = evaluate_growth_parameter(
                expressions[index],
                𝓂.parameter_values,
                profile.parameter_indices,
            )
            value === nothing || (exponents[index] = value)
        end
    end
    true
end

function refresh_bgp_representation!(𝓂::ℳ)
    mode_changed = refresh_bgp_detection!(𝓂)
    profile = 𝓂.equations.bgp_detection
    profile !== nothing && profile.mode == BGP_ACTIVE_MODE &&
        refresh_bgp_numeric_state!(𝓂)
    mode_changed || return false

    if profile.mode == BGP_ACTIVE_MODE
        stationarize_model!(𝓂)
    elseif profile.mode == BGP_STATIONARY_MODE
        restore_raw_model!(𝓂)
    else
        throw(ArgumentError("Cannot activate an unsupported balanced-growth representation."))
    end
    true
end
