# Compilation of symbolic expressions into in-place Julia functions.
#
# Replaces `Symbolics.build_function`. Shared subexpressions (nodes reached
# from more than one parent in the expression graph) are computed once into
# local temporaries, which is the common subexpression elimination that the
# interned node graph provides directly. Functions with a restricted real
# domain are called through NaN-returning versions, so that a steady state
# solver probing outside the domain sees `NaN` instead of a `DomainError`.

nan_safe_log(x::Real) = x < zero(x) ? x * NaN : log(x)
nan_safe_log2(x::Real) = x < zero(x) ? x * NaN : log2(x)
nan_safe_log10(x::Real) = x < zero(x) ? x * NaN : log10(x)
nan_safe_log1p(x::Real) = x < -one(x) ? x * NaN : log1p(x)
nan_safe_sqrt(x::Real) = x < zero(x) ? x * NaN : sqrt(x)
nan_safe_asin(x::Real) = abs(x) > one(x) ? x * NaN : asin(x)
nan_safe_acos(x::Real) = abs(x) > one(x) ? x * NaN : acos(x)
nan_safe_acosh(x::Real) = x < one(x) ? x * NaN : acosh(x)
nan_safe_atanh(x::Real) = abs(x) > one(x) ? x * NaN : atanh(x)
nan_safe_pow(x::Real, y::Integer) = x ^ y
nan_safe_pow(x::Real, y::Real) = x < zero(x) && !isinteger(y) ? (x + y) * NaN : x ^ y

const NAN_SAFE_FUNCTIONS = Dict{Symbol, Symbol}(
    :log => :nan_safe_log, :log2 => :nan_safe_log2, :log10 => :nan_safe_log10, :log1p => :nan_safe_log1p,
    :sqrt => :nan_safe_sqrt, :asin => :nan_safe_asin, :acos => :nan_safe_acos,
    :acosh => :nan_safe_acosh, :atanh => :nan_safe_atanh,
)

mutable struct SymbolicCodeWriter
    leaves::Dict{Symbol, Tuple{Int, Int}}
    use_counts::IdDict{SymbolicNode, Int}
    emitted::IdDict{SymbolicNode, Any}
    statements::Vector{Any}
    loaded_leaves::Dict{Symbol, Symbol}
    temporary_count::Int
end

function count_symbolic_uses!(counts::IdDict{SymbolicNode, Int}, node::SymbolicNode)
    stack = SymbolicNode[node]
    while !isempty(stack)
        n = pop!(stack)
        previous = get(counts, n, 0)
        counts[n] = previous + 1
        previous == 0 || continue
        n.kind ∈ (SYMBOLIC_NUMBER, SYMBOLIC_SYMBOL) && continue
        append!(stack, n.args)
    end
    return counts
end

function code_number(n::SymbolicNode)
    if n.exact
        denominator(n.rational) == 1 && return numerator(n.rational)
        return Float64(n.rational)
    end
    return n.value
end

function code_leaf(w::SymbolicCodeWriter, name::Symbol)
    haskey(w.loaded_leaves, name) && return w.loaded_leaves[name]
    haskey(w.leaves, name) || return name
    argument, index = w.leaves[name]
    local_name = Symbol("leaf_", length(w.loaded_leaves) + 1)
    push!(w.statements, :($local_name = $(Symbol("input_", argument))[$index]))
    w.loaded_leaves[name] = local_name
    return local_name
end

function code_signed_term(w::SymbolicCodeWriter, term::SymbolicNode)
    if term.kind == SYMBOLIC_NUMBER && term.value < 0
        return true, code_number(symbolic_negate(term))
    end
    if term.kind == SYMBOLIC_PRODUCT && is_symbolic_number(term.args[1]) && term.args[1].value < 0
        positive = symbolic_negate(term)
        return true, code_expression(w, positive)
    end
    return false, code_expression(w, term)
end

function code_node(w::SymbolicCodeWriter, node::SymbolicNode)
    if node.kind == SYMBOLIC_SUM
        expr = nothing
        for term in node.args
            negative, term_expr = code_signed_term(w, term)
            if expr === nothing
                expr = negative ? Expr(:call, :-, term_expr) : term_expr
            elseif negative
                expr = Expr(:call, :-, expr, term_expr)
            else
                expr = Expr(:call, :+, expr, term_expr)
            end
        end
        return expr
    elseif node.kind == SYMBOLIC_PRODUCT
        numerator_factors = Any[]
        denominator_factors = Any[]
        for f in node.args
            if f.kind == SYMBOLIC_NUMBER
                pushfirst!(numerator_factors, code_number(f))
            elseif f.kind == SYMBOLIC_POWER && is_exact_integer(f.args[2]) && f.args[2].value < 0
                denominator = symbolic_power(f.args[1], symbolic_negate(f.args[2]))
                push!(denominator_factors, code_expression(w, denominator))
            else
                push!(numerator_factors, code_expression(w, f))
            end
        end
        numerator_expr = isempty(numerator_factors) ? 1 :
                         length(numerator_factors) == 1 ? numerator_factors[1] :
                         Expr(:call, :*, numerator_factors...)
        isempty(denominator_factors) && return numerator_expr
        denominator_expr = length(denominator_factors) == 1 ? denominator_factors[1] : Expr(:call, :*, denominator_factors...)
        return Expr(:call, :/, numerator_expr, denominator_expr)
    elseif node.kind == SYMBOLIC_POWER
        base, exponent = node.args
        base_expr = code_expression(w, base)
        if is_exact_integer(exponent)
            return Expr(:call, :^, base_expr, Int(exponent.rational.num))
        elseif exponent.kind == SYMBOLIC_NUMBER && exponent.exact && exponent.rational == 1 // 2
            return Expr(:call, :nan_safe_sqrt, base_expr)
        end
        return Expr(:call, :nan_safe_pow, base_expr, code_expression(w, exponent))
    end

    args = Any[code_expression(w, a) for a in node.args]
    node.name ∈ (:&&, :||) && return Expr(node.name, args...)
    return Expr(:call, get(NAN_SAFE_FUNCTIONS, node.name, node.name), args...)
end

function code_expression(w::SymbolicCodeWriter, node::SymbolicNode)
    node.kind == SYMBOLIC_NUMBER && return code_number(node)
    node.kind == SYMBOLIC_SYMBOL && return code_leaf(w, node.name)
    haskey(w.emitted, node) && return w.emitted[node]
    expr = code_node(w, node)
    if get(w.use_counts, node, 0) > 1
        w.temporary_count += 1
        name = Symbol("shared_", w.temporary_count)
        push!(w.statements, :($name = $expr))
        w.emitted[node] = name
        return name
    end
    w.emitted[node] = expr
    return expr
end

function symbolic_code_writer(outputs, arguments::Vector{Vector{Symbol}})
    leaves = Dict{Symbol, Tuple{Int, Int}}()
    for (a, names) in enumerate(arguments), (i, name) in enumerate(names)
        leaves[name] = (a, i)
    end
    counts = IdDict{SymbolicNode, Int}()
    for o in outputs
        count_symbolic_uses!(counts, o)
    end
    return SymbolicCodeWriter(leaves, counts, IdDict{SymbolicNode, Any}(), Any[], Dict{Symbol, Symbol}(), 0)
end

function compile_symbolic_function(w::SymbolicCodeWriter, n_arguments::Int)
    inputs = [Symbol("input_", a) for a in 1:n_arguments]
    body = Expr(:block, w.statements..., :(return nothing))
    func = Expr(:function, Expr(:tuple, :out, inputs...), Expr(:block, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(0), body)))
    return @RuntimeGeneratedFunction(func)
end

"""
    build_symbolic_function(outputs, arguments)

Compile `outputs` (a vector, a dense matrix or a sparse matrix of symbolic
nodes) into an in-place function `f!(out, input_1, input_2, ...)`. Each entry
of `arguments` lists the symbols that are read from the corresponding input
vector, by position. A dense matrix is zeroed and then filled at its nonzero
entries; a sparse matrix output writes the stored values of `out`, which must
share the sparsity pattern of `outputs`.
"""
function build_symbolic_function(outputs::AbstractVector{SymbolicNode}, arguments::Vector{Vector{Symbol}})
    w = symbolic_code_writer(outputs, arguments)
    for (i, o) in enumerate(outputs)
        push!(w.statements, :(out[$i] = $(code_expression(w, o))))
    end
    return compile_symbolic_function(w, length(arguments))
end

function build_symbolic_function(outputs::SparseMatrixCSC{SymbolicNode}, arguments::Vector{Vector{Symbol}})
    values = nonzeros(outputs)
    w = symbolic_code_writer(values, arguments)
    push!(w.statements, :(stored_values = SparseArrays.nonzeros(out)))
    for (k, o) in enumerate(values)
        push!(w.statements, :(stored_values[$k] = $(code_expression(w, o))))
    end
    return compile_symbolic_function(w, length(arguments))
end

function build_symbolic_function(outputs::AbstractMatrix{SymbolicNode}, arguments::Vector{Vector{Symbol}})
    entries = Tuple{Int, Int, SymbolicNode}[(i, j, outputs[i, j]) for j in axes(outputs, 2) for i in axes(outputs, 1) if !is_symbolic_zero(outputs[i, j])]
    w = symbolic_code_writer([e[3] for e in entries], arguments)
    push!(w.statements, :(fill!(out, zero(eltype(out)))))
    for (i, j, o) in entries
        push!(w.statements, :(out[$i, $j] = $(code_expression(w, o))))
    end
    return compile_symbolic_function(w, length(arguments))
end

"""Return the placeholder symbols `prefix_1, …, prefix_n` used to address input vectors."""
symbolic_placeholders(prefix::AbstractString, n::Int) = Symbol[Symbol(prefix, "_", i) for i in 1:n]

"""Convert Julia expressions to symbolic nodes after renaming symbols with `placeholders`."""
function to_symbolic_with_placeholders(exprs, placeholders::AbstractDict{Symbol})
    return SymbolicNode[to_symbolic(e isa Number ? e : replace_symbols(e, placeholders)) for e in exprs]
end
