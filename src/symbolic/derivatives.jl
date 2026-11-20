# Symbolic differentiation of `SymbolicNode` expressions.
#
# Follows the differentiator of the Rust implementation
# (`dsge-core/src/builder/optimization.rs`): a memoized recursive walk over the
# shared expression graph, where every result is built through the canonical
# constructors so that derivatives come out simplified.

"""
Partial derivative of the call `name(args...)` with respect to its `i`-th argument.
Returns `nothing` for functions without a known derivative.
"""
function symbolic_partial_derivative(name::Symbol, args::Vector{SymbolicNode}, i::Int)::Union{Nothing, SymbolicNode}
    a = args[1]
    call(f, x...) = symbolic_call(f, SymbolicNode[x...])
    number(x) = symbolic_number(x)
    square(x) = symbolic_power(x, number(2))
    reciprocal(x) = symbolic_power(x, number(-1))
    node = symbolic_call(name, args)

    if length(args) == 1
        name == :exp && return node
        name == :log && return reciprocal(a)
        name == :log2 && return reciprocal(symbolic_product(a, number(log(2))))
        name == :log10 && return reciprocal(symbolic_product(a, number(log(10))))
        name == :log1p && return reciprocal(symbolic_sum(number(1), a))
        name == :exp2 && return symbolic_product(node, number(log(2)))
        name == :exp10 && return symbolic_product(node, number(log(10)))
        name == :sin && return call(:cos, a)
        name == :cos && return symbolic_negate(call(:sin, a))
        name == :tan && return symbolic_sum(number(1), square(node))
        name == :asin && return symbolic_power(symbolic_difference(number(1), square(a)), number(-1 // 2))
        name == :acos && return symbolic_negate(symbolic_power(symbolic_difference(number(1), square(a)), number(-1 // 2)))
        name == :atan && return reciprocal(symbolic_sum(number(1), square(a)))
        name == :sinh && return call(:cosh, a)
        name == :cosh && return call(:sinh, a)
        name == :tanh && return symbolic_difference(number(1), square(node))
        name == :asinh && return symbolic_power(symbolic_sum(square(a), number(1)), number(-1 // 2))
        name == :acosh && return symbolic_power(symbolic_difference(square(a), number(1)), number(-1 // 2))
        name == :atanh && return reciprocal(symbolic_difference(number(1), square(a)))
        name == :abs && return call(:sign, a)
        name == :abs2 && return symbolic_product(number(2), a)
        name == :sign && return symbolic_zero()
        name == :erf && return symbolic_product(number(2 / sqrt(π)), call(:exp, symbolic_negate(square(a))))
        name == :erfc && return symbolic_product(number(-2 / sqrt(π)), call(:exp, symbolic_negate(square(a))))
        name == :erfcinv && return symbolic_product(number(-sqrt(π) / 2), call(:exp, square(node)))
        name ∈ (:norminvcdf, :norminv, :qnorm) && return reciprocal(call(:normpdf, call(:norminvcdf, a)))
        name ∈ (:normcdf, :pnorm) && return call(:normpdf, a)
        name ∈ (:normpdf, :dnorm) && return symbolic_negate(symbolic_product(a, call(:normpdf, a)))
        name == :normlogpdf && return symbolic_negate(a)
    elseif length(args) == 2 && name ∈ (:max, :min)
        b = args[2]
        condition = name == :max ? call(:>=, a, b) : call(:<=, a, b)
        return i == 1 ? call(:ifelse, condition, number(1), number(0)) : call(:ifelse, condition, number(0), number(1))
    end

    name ∈ SYMBOLIC_COMPARISONS && return symbolic_zero()
    return nothing
end

mutable struct SymbolicDifferentiator
    variable::Symbol
    memo::IdDict{SymbolicNode, SymbolicNode}
end

SymbolicDifferentiator(variable::Symbol) = SymbolicDifferentiator(variable, IdDict{SymbolicNode, SymbolicNode}())

function differentiate(d::SymbolicDifferentiator, node::SymbolicNode)::SymbolicNode
    node.kind == SYMBOLIC_NUMBER && return symbolic_zero()
    node.kind == SYMBOLIC_SYMBOL && return node.name == d.variable ? symbolic_one() : symbolic_zero()
    haskey(d.memo, node) && return d.memo[node]

    result = if node.kind == SYMBOLIC_SUM
        symbolic_sum(SymbolicNode[differentiate(d, t) for t in node.args])
    elseif node.kind == SYMBOLIC_PRODUCT
        terms = SymbolicNode[]
        for (i, f) in enumerate(node.args)
            df = differentiate(d, f)
            is_symbolic_zero(df) && continue
            others = SymbolicNode[node.args[j] for j in eachindex(node.args) if j != i]
            push!(terms, symbolic_product(vcat(df, others)))
        end
        symbolic_sum(terms)
    elseif node.kind == SYMBOLIC_POWER
        base, exponent = node.args
        dbase = differentiate(d, base)
        dexponent = differentiate(d, exponent)
        terms = SymbolicNode[]
        if !is_symbolic_zero(dbase)
            push!(terms, symbolic_product(exponent, symbolic_power(base, symbolic_difference(exponent, symbolic_one())), dbase))
        end
        if !is_symbolic_zero(dexponent)
            push!(terms, symbolic_product(node, symbolic_call(:log, SymbolicNode[base]), dexponent))
        end
        symbolic_sum(terms)
    else
        if node.name == :ifelse && length(node.args) == 3
            symbolic_call(:ifelse, SymbolicNode[node.args[1], differentiate(d, node.args[2]), differentiate(d, node.args[3])])
        else
            terms = SymbolicNode[]
            for (i, a) in enumerate(node.args)
                da = differentiate(d, a)
                is_symbolic_zero(da) && continue
                partial = symbolic_partial_derivative(node.name, node.args, i)
                partial === nothing && throw(ArgumentError("no symbolic derivative is available for function $(node.name) in $(node)"))
                push!(terms, symbolic_product(partial, da))
            end
            symbolic_sum(terms)
        end
    end

    d.memo[node] = result
    return result
end

"""
    differentiate(node, variable)

Return the derivative of `node` with respect to the symbol `variable`.
"""
differentiate(node::SymbolicNode, variable::Symbol) = differentiate(SymbolicDifferentiator(variable), node)

"""
    symbolic_sparse_jacobian(expressions, variables)

Return the Jacobian of `expressions` with respect to `variables` as a
`SparseMatrixCSC{SymbolicNode}` holding only the entries that are not
identically zero.
"""
function symbolic_sparse_jacobian(expressions::AbstractVector{SymbolicNode}, variables::AbstractVector{Symbol})::SparseMatrixCSC{SymbolicNode, Int}
    column_of = Dict{Symbol, Int}(v => j for (j, v) in enumerate(variables))
    differentiators = Dict{Symbol, SymbolicDifferentiator}()

    rows = Int[]
    cols = Int[]
    vals = SymbolicNode[]

    for (i, ex) in enumerate(expressions)
        present = Int[column_of[s] for s in free_symbols(ex) if haskey(column_of, s)]
        sort!(present)
        for j in present
            v = variables[j]
            d = get!(() -> SymbolicDifferentiator(v), differentiators, v)
            dex = differentiate(d, ex)
            is_symbolic_zero(dex) && continue
            push!(rows, i)
            push!(cols, j)
            push!(vals, dex)
        end
    end

    return sparse(rows, cols, vals, length(expressions), length(variables), +)
end

Base.:+(a::SymbolicNode, b::SymbolicNode) = symbolic_sum(a, b)
