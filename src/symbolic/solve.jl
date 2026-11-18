# Closed-form solution of equations for one variable.
#
# The approach follows the Rust implementation: an equation is solved when the
# variable can be isolated by inverting the operations on the path from the
# root to its single occurrence (`isolate_obc_target` in
# `dsge-core/src/builder/obc.rs`), or when the equation is linear in the
# variable (`solve_nsss_numeric_linear_identity` in
# `dsge-core/src/nsss/simplify.rs`). Two further reductions widen the set of
# equations that reach those two cases: common powers of the variable are
# divided out, and denominators that contain the variable are multiplied out.
# Equations of other shapes are left to the numerical steady-state solver.

"""Inverse functions used to isolate a variable: `f(x) = r` gives `x = inverse(r)`."""
const SYMBOLIC_INVERSE_FUNCTIONS = Dict{Symbol, Symbol}(
    :exp => :log, :log => :exp,
    :exp2 => :log2, :log2 => :exp2,
    :exp10 => :log10, :log10 => :exp10,
    :tanh => :atanh, :atanh => :tanh,
    :sinh => :asinh, :asinh => :sinh,
    :normcdf => :norminvcdf, :pnorm => :norminvcdf,
    :norminvcdf => :normcdf, :norminv => :normcdf, :qnorm => :normcdf,
    :erfc => :erfcinv, :erfcinv => :erfc,
)

"""Return the power of `variable` that `factor` is (`variable ^ p` gives `p`), or `nothing`."""
function variable_power(factor::SymbolicNode, variable::Symbol)::Union{Nothing, SymbolicNode}
    factor.kind == SYMBOLIC_SYMBOL && factor.name == variable && return symbolic_one()
    if factor.kind == SYMBOLIC_POWER && factor.args[1].kind == SYMBOLIC_SYMBOL && factor.args[1].name == variable &&
       factor.args[2].kind == SYMBOLIC_NUMBER
        return factor.args[2]
    end
    return nothing
end

"""Return the exponent `p` (free of the variable) if `factor` is `variable ^ p`, or `nothing`."""
function variable_power_general(factor::SymbolicNode, variable::Symbol)::Union{Nothing, SymbolicNode}
    factor.kind == SYMBOLIC_SYMBOL && factor.name == variable && return symbolic_one()
    if factor.kind == SYMBOLIC_POWER && factor.args[1].kind == SYMBOLIC_SYMBOL && factor.args[1].name == variable &&
       !contains_symbol(factor.args[2], variable)
        return factor.args[2]
    end
    return nothing
end

factors_of(node::SymbolicNode) = node.kind == SYMBOLIC_PRODUCT ? node.args : SymbolicNode[node]
terms_of(node::SymbolicNode) = node.kind == SYMBOLIC_SUM ? node.args : SymbolicNode[node]

"""
Divide out the variable where it enters as a common multiplicative power, i.e.
`x ^ p * g(x) = 0` becomes `g(x) = 0`. Factors free of the variable are
dropped from a product, as a computer algebra system does when it assumes them
to be nonzero.
"""
function remove_common_variable_factor(eq::SymbolicNode, variable::Symbol)::SymbolicNode
    if eq.kind == SYMBOLIC_PRODUCT
        kept = SymbolicNode[f for f in eq.args if variable_power(f, variable) === nothing && contains_symbol(f, variable)]
        isempty(kept) && return symbolic_one()
        return symbolic_product(kept)
    elseif eq.kind == SYMBOLIC_SUM
        minimum_power = nothing
        for t in eq.args
            p = nothing
            for f in factors_of(t)
                fp = variable_power(f, variable)
                fp === nothing || (p = fp)
            end
            p === nothing && return eq
            if minimum_power === nothing || p.value < minimum_power.value
                minimum_power = p
            end
        end
        minimum_power === nothing && return eq
        divisor = symbolic_power(symbolic_symbol(variable), symbolic_negate(minimum_power))
        return symbolic_sum([symbolic_product(t, divisor) for t in eq.args])
    end
    if variable_power(eq, variable) !== nothing
        return symbolic_one()
    end
    return eq
end

"""Multiply a sum by every denominator of its terms that contains the variable."""
function clear_variable_denominators(eq::SymbolicNode, variable::Symbol)::SymbolicNode
    eq.kind == SYMBOLIC_SUM || return eq
    denominators = SymbolicNode[]
    for t in eq.args, f in factors_of(t)
        if f.kind == SYMBOLIC_POWER && f.args[2].kind == SYMBOLIC_NUMBER && f.args[2].value < 0 &&
           f.args[1].kind != SYMBOLIC_SYMBOL && contains_symbol(f.args[1], variable)
            d = symbolic_power(f.args[1], symbolic_negate(f.args[2]))
            d ∈ denominators || push!(denominators, d)
        end
    end
    isempty(denominators) && return eq
    return symbolic_sum([expand_variable_products(symbolic_product(vcat(t, denominators)), variable) for t in eq.args])
end

"""Distribute products over sums that contain the variable."""
function expand_variable_products(node::SymbolicNode, variable::Symbol)::SymbolicNode
    if node.kind == SYMBOLIC_SUM
        return symbolic_sum([expand_variable_products(t, variable) for t in node.args])
    elseif node.kind == SYMBOLIC_PRODUCT
        expanded = SymbolicNode[symbolic_one()]
        for f in node.args
            if f.kind == SYMBOLIC_SUM && contains_symbol(f, variable)
                terms = [expand_variable_products(t, variable) for t in f.args]
                expanded = SymbolicNode[symbolic_product(e, t) for e in expanded for t in terms]
            else
                expanded = SymbolicNode[symbolic_product(e, f) for e in expanded]
            end
        end
        return symbolic_sum(expanded)
    elseif node.kind == SYMBOLIC_POWER && is_exact_integer(node.args[2]) && 1 < node.args[2].value <= 4 &&
           node.args[1].kind == SYMBOLIC_SUM && contains_symbol(node.args[1], variable)
        return expand_variable_products(symbolic_product(fill(node.args[1], Int(node.args[2].value))), variable)
    end
    return node
end

"""
Solve `node = rhs` for the single occurrence of `variable` in `node` by
inverting the operations on the path to it.
"""
function isolate_variable(node::SymbolicNode, rhs::SymbolicNode, variable::Symbol, positive::Bool)::Union{Nothing, SymbolicNode}
    node.kind == SYMBOLIC_SYMBOL && node.name == variable && return rhs

    if node.kind == SYMBOLIC_SUM
        with = SymbolicNode[t for t in node.args if contains_symbol(t, variable)]
        length(with) == 1 || return nothing
        without = SymbolicNode[t for t in node.args if !contains_symbol(t, variable)]
        return isolate_variable(with[1], symbolic_difference(rhs, symbolic_sum(without)), variable, positive)
    elseif node.kind == SYMBOLIC_PRODUCT
        with = SymbolicNode[f for f in node.args if contains_symbol(f, variable)]
        length(with) == 1 || return nothing
        without = SymbolicNode[f for f in node.args if !contains_symbol(f, variable)]
        return isolate_variable(with[1], symbolic_quotient(rhs, symbolic_product(without)), variable, positive)
    elseif node.kind == SYMBOLIC_POWER
        base, exponent = node.args
        base_has = contains_symbol(base, variable)
        exponent_has = contains_symbol(exponent, variable)
        if base_has && !exponent_has
            # an even power has two real roots; keep only the principal one when the base is the positive variable
            if is_integer_valued(exponent) && iseven(Int(exponent.value)) && !(positive && base.kind == SYMBOLIC_SYMBOL)
                return nothing
            end
            return isolate_variable(base, symbolic_power(rhs, symbolic_power(exponent, symbolic_minus_one())), variable, positive)
        elseif exponent_has && !base_has
            return isolate_variable(exponent, symbolic_quotient(symbolic_call(:log, rhs), symbolic_call(:log, base)), variable, positive)
        end
        return nothing
    elseif node.kind == SYMBOLIC_CALL && length(node.args) == 1 && haskey(SYMBOLIC_INVERSE_FUNCTIONS, node.name)
        return isolate_variable(node.args[1], symbolic_call(SYMBOLIC_INVERSE_FUNCTIONS[node.name], rhs), variable, positive)
    end
    return nothing
end

"""
Solve a sum of terms of the form `c * variable ^ p` (with `c` and `p` free of
the variable) when it holds exactly two distinct powers.
"""
function solve_power_terms(eq::SymbolicNode, variable::Symbol, positive::Bool)::Union{Nothing, SymbolicNode}
    groups = Dict{SymbolicNode, Vector{SymbolicNode}}()
    order = SymbolicNode[]
    for t in terms_of(eq)
        power = symbolic_zero()
        rest = SymbolicNode[]
        for f in factors_of(t)
            p = variable_power_general(f, variable)
            if p !== nothing
                power = p
            elseif contains_symbol(f, variable)
                return nothing
            else
                push!(rest, f)
            end
        end
        if !haskey(groups, power)
            groups[power] = SymbolicNode[]
            push!(order, power)
        end
        push!(groups[power], symbolic_product(rest))
    end

    length(order) == 2 || return nothing
    p, q = order[1], order[2]
    if p.kind == SYMBOLIC_NUMBER && q.kind == SYMBOLIC_NUMBER && p.value < q.value
        p, q = q, p
    end
    a = symbolic_sum(groups[p])
    b = symbolic_sum(groups[q])
    (is_symbolic_zero(a) || is_symbolic_zero(b)) && return nothing
    difference = symbolic_difference(p, q)
    ratio = symbolic_negate(symbolic_quotient(b, a))
    is_symbolic_one(difference) && return ratio
    if is_integer_valued(difference) && iseven(Int(difference.value)) && !positive
        return nothing
    end
    return symbolic_power(ratio, symbolic_power(difference, symbolic_minus_one()))
end

"""
    solve_for_variable(eq, variable; positive = false)

Return the closed-form solution of `eq = 0` for `variable`, or `nothing` if
none is found. `positive` states that the variable is known to be positive,
which selects the principal root of even powers.
"""
function solve_for_variable(eq::SymbolicNode, variable::Symbol; positive::Bool = false)::Union{Nothing, SymbolicNode}
    contains_symbol(eq, variable) || return nothing

    # a single term `c * x ^ p` with p > 0 has the root x = 0
    if eq.kind != SYMBOLIC_SUM
        variable_factors = SymbolicNode[f for f in factors_of(eq) if contains_symbol(f, variable)]
        powers = [variable_power(f, variable) for f in variable_factors]
        if all(p -> p !== nothing && p.kind == SYMBOLIC_NUMBER && p.value > 0, powers)
            return symbolic_zero()
        end
    end

    reduced = remove_common_variable_factor(eq, variable)
    contains_symbol(reduced, variable) || return nothing

    for candidate in (reduced, clear_variable_denominators(reduced, variable))
        solution = isolate_variable(candidate, symbolic_zero(), variable, positive)
        if solution === nothing
            expanded = expand_variable_products(candidate, variable)
            reduced_expanded = remove_common_variable_factor(expanded, variable)
            contains_symbol(reduced_expanded, variable) || return nothing
            solution = isolate_variable(reduced_expanded, symbolic_zero(), variable, positive)
            if solution === nothing
                solution = solve_power_terms(reduced_expanded, variable, positive)
            end
        end
        if solution !== nothing && !contains_symbol(solution, variable)
            return solution
        end
    end
    return nothing
end

"""
    variable_is_redundant(eq, variable)

Return `true` if `variable` does not occur in `eq = 0`, or occurs only as a
common multiplicative power so that the equation does not determine it.
"""
function variable_is_redundant(eq::SymbolicNode, variable::Symbol)::Bool
    contains_symbol(eq, variable) || return true
    return !contains_symbol(remove_common_variable_factor(eq, variable), variable)
end

"""
    solve_system_for_variables(equations, variables; positive = Set{Symbol}())

Solve a square system by successive elimination: an equation with a unique
closed-form solution for one of the remaining variables is solved, the result
is substituted into the other equations, and the solutions are completed by
back substitution. Returns a dictionary from variables to solutions free of
the solved variables, or `nothing`.
"""
function solve_system_for_variables(equations::Vector{SymbolicNode}, variables::Vector{Symbol};
                                    positive::Set{Symbol} = Set{Symbol}())::Union{Nothing, Dict{Symbol, SymbolicNode}}
    remaining_equations = copy(equations)
    remaining_variables = copy(variables)
    eliminated = Tuple{Symbol, SymbolicNode}[]

    while !isempty(remaining_variables)
        found = false
        for (i, eq) in enumerate(remaining_equations), v in remaining_variables
            solution = solve_for_variable(eq, v, positive = v ∈ positive)
            solution === nothing && continue
            push!(eliminated, (v, solution))
            deleteat!(remaining_equations, i)
            filter!(x -> x != v, remaining_variables)
            replacement = Dict{Symbol, SymbolicNode}(v => solution)
            remaining_equations = SymbolicNode[substitute_symbolic(e, replacement) for e in remaining_equations]
            found = true
            break
        end
        found || return nothing
    end

    solutions = Dict{Symbol, SymbolicNode}()
    for (v, solution) in reverse(eliminated)
        solutions[v] = substitute_symbolic(solution, solutions)
    end
    for s in values(solutions), v in variables
        contains_symbol(s, v) && return nothing
    end
    return solutions
end
