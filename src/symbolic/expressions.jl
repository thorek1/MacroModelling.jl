# Native symbolic expressions.
#
# This replaces the SymPy (computer algebra) and Symbolics.jl (derivatives and
# code generation) layers. The design follows the expression tree of the Rust
# implementation (`dsge-core/src/model.rs`): nodes are immutable after
# construction, structurally shared and interned, and every constructor returns
# a canonical, locally simplified node. Canonical forms:
#
# - a sum is a flat list of non-numeric terms (each with its numeric
#   coefficient folded into a product) followed by an optional numeric constant;
#   like terms are combined;
# - a product is an optional leading numeric coefficient and a flat list of
#   factors; equal bases are combined by adding their exponents;
# - a quotient `a / b` is `a * b ^ -1`, a difference `a - b` is `a + (-1) * b`
#   and `sqrt(a)` is `a ^ (1 // 2)`;
# - numbers are exact rationals (from integer literals) or `Float64`.
#
# Nodes are converted from and to Julia `Expr` at the boundaries, so the rest of
# the package keeps working with `Expr`.

const SYMBOLIC_NUMBER   = 0x01
const SYMBOLIC_SYMBOL   = 0x02
const SYMBOLIC_SUM      = 0x03
const SYMBOLIC_PRODUCT  = 0x04
const SYMBOLIC_POWER    = 0x05
const SYMBOLIC_CALL     = 0x06

mutable struct SymbolicNode
    kind::UInt8
    exact::Bool
    rational::Rational{Int}
    value::Float64
    name::Symbol
    args::Vector{SymbolicNode}
    hash::UInt
    order_name::String
end

const SymbolicExpression = Union{Expr, Symbol, Number}

const SYMBOLIC_NODE_TABLE = Dict{UInt, Vector{SymbolicNode}}()
const SYMBOLIC_NAME_HASHES = Dict{Symbol, UInt}()
const SYMBOLIC_TABLE_LOCK = ReentrantLock()

const NO_SYMBOLIC_ARGS = SymbolicNode[]

Base.hash(node::SymbolicNode, h::UInt) = hash(node.hash, h)

function Base.:(==)(a::SymbolicNode, b::SymbolicNode)
    a === b && return true
    a.hash == b.hash || return false
    a.kind == b.kind || return false
    a.exact == b.exact || return false
    a.name == b.name || return false
    if a.kind == SYMBOLIC_NUMBER
        return a.exact ? a.rational == b.rational : isequal(a.value, b.value)
    end
    length(a.args) == length(b.args) || return false
    for i in eachindex(a.args)
        a.args[i] == b.args[i] || return false
    end
    return true
end

Base.isequal(a::SymbolicNode, b::SymbolicNode) = a == b

"""
    empty_symbolic_table!()

Release all interned symbolic nodes. Equality and hashing of nodes are
structural, so nodes created before the reset remain valid.
"""
function empty_symbolic_table!()
    lock(SYMBOLIC_TABLE_LOCK) do
        empty!(SYMBOLIC_NODE_TABLE)
    end
    return nothing
end

function symbolic_name_hash(name::Symbol)::UInt
    lock(SYMBOLIC_TABLE_LOCK) do
        get!(() -> hash(String(name)), SYMBOLIC_NAME_HASHES, name)
    end
end

function intern_symbolic_node(kind::UInt8, exact::Bool, rational::Rational{Int}, value::Float64, name::Symbol, args::Vector{SymbolicNode})::SymbolicNode
    h = hash(kind, hash(exact))
    if kind == SYMBOLIC_NUMBER
        h = exact ? hash(rational.num, hash(rational.den, h)) : hash(value, h)
    end
    if name !== Symbol("")
        h = hash(symbolic_name_hash(name), h)
    end
    for a in args
        h = hash(a.hash, h)
    end

    order_name = if kind == SYMBOLIC_SYMBOL
        String(name)
    elseif kind == SYMBOLIC_NUMBER
        ""
    else
        best = ""
        for a in args
            if !isempty(a.order_name) && (isempty(best) || a.order_name < best)
                best = a.order_name
            end
        end
        isempty(best) && kind == SYMBOLIC_CALL ? String(name) : best
    end

    candidate = SymbolicNode(kind, exact, rational, value, name, args, h, order_name)

    lock(SYMBOLIC_TABLE_LOCK) do
        bucket = get!(() -> SymbolicNode[], SYMBOLIC_NODE_TABLE, h)
        for existing in bucket
            existing == candidate && return existing
        end
        push!(bucket, candidate)
        return candidate
    end
end

# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------

function symbolic_number(x::Rational{Int})::SymbolicNode
    intern_symbolic_node(SYMBOLIC_NUMBER, true, x, Float64(x), Symbol(""), NO_SYMBOLIC_ARGS)
end

symbolic_number(x::Integer) = typemin(Int) < x <= typemax(Int) ? symbolic_number(Rational{Int}(Int(x))) : symbolic_number(Float64(x))

function symbolic_number(x::Rational)
    n, d = numerator(x), denominator(x)
    if typemin(Int) < n <= typemax(Int) && 0 < d <= typemax(Int)
        return symbolic_number(Int(n) // Int(d))
    end
    return symbolic_number(Float64(x))
end

function symbolic_number(x::Float64)::SymbolicNode
    intern_symbolic_node(SYMBOLIC_NUMBER, false, 0 // 1, x, Symbol(""), NO_SYMBOLIC_ARGS)
end

symbolic_number(x::AbstractFloat) = symbolic_number(Float64(x))
symbolic_number(x::Irrational) = symbolic_number(Float64(x))
symbolic_number(x::Bool) = symbolic_number(Int(x))

symbolic_zero() = symbolic_number(0 // 1)
symbolic_one() = symbolic_number(1 // 1)
symbolic_minus_one() = symbolic_number(-1 // 1)

is_symbolic_number(n::SymbolicNode) = n.kind == SYMBOLIC_NUMBER
is_symbolic_zero(n::SymbolicNode) = n.kind == SYMBOLIC_NUMBER && n.value == 0
is_symbolic_one(n::SymbolicNode) = n.kind == SYMBOLIC_NUMBER && n.value == 1
is_exact_integer(n::SymbolicNode) = n.kind == SYMBOLIC_NUMBER && n.exact && denominator(n.rational) == 1
is_integer_valued(n::SymbolicNode) = n.kind == SYMBOLIC_NUMBER && isinteger(n.value)

Base.zero(::Type{SymbolicNode}) = symbolic_zero()
Base.zero(::SymbolicNode) = symbolic_zero()
Base.iszero(n::SymbolicNode) = is_symbolic_zero(n)

function checked_rational(num::Int128, den::Int128)::Union{Nothing, Rational{Int}}
    if den < 0
        num, den = -num, -den
    end
    g = gcd(num, den)
    if g > 1
        num ÷= g
        den ÷= g
    end
    if typemin(Int) < num <= typemax(Int) && 0 < den <= typemax(Int)
        return Int(num) // Int(den)
    end
    return nothing
end

function add_numbers(a::SymbolicNode, b::SymbolicNode)::SymbolicNode
    if a.exact && b.exact
        r = checked_rational(Int128(a.rational.num) * b.rational.den + Int128(b.rational.num) * a.rational.den,
                             Int128(a.rational.den) * b.rational.den)
        r === nothing || return symbolic_number(r)
    end
    return symbolic_number(a.value + b.value)
end

function multiply_numbers(a::SymbolicNode, b::SymbolicNode)::SymbolicNode
    if a.exact && b.exact
        r = checked_rational(Int128(a.rational.num) * b.rational.num, Int128(a.rational.den) * b.rational.den)
        r === nothing || return symbolic_number(r)
    end
    return symbolic_number(a.value * b.value)
end

"""Return the folded power of two numbers, or `nothing` if it is not a finite real number."""
function power_numbers(base::SymbolicNode, exponent::SymbolicNode)::Union{Nothing, SymbolicNode}
    if base.value == 0 && exponent.value < 0
        return nothing
    end
    if base.exact && is_exact_integer(exponent)
        e = exponent.rational.num
        if abs(e) <= 64
            num = big(base.rational.num) ^ abs(e)
            den = big(base.rational.den) ^ abs(e)
            if e < 0
                num, den = den, num
            end
            if den < 0
                num, den = -num, -den
            end
            if typemin(Int) < num <= typemax(Int) && 0 < den <= typemax(Int)
                return symbolic_number(Int(num) // Int(den))
            end
        end
    end
    if base.value < 0 && !isinteger(exponent.value)
        return nothing
    end
    if base.exact && exponent.exact && !isinteger(exponent.value)
        # keep exact surds such as 2 ^ (1 // 2) symbolic, like a computer algebra system
        return nothing
    end
    result = base.value ^ exponent.value
    isfinite(result) || return nothing
    return symbolic_number(result)
end

# ----------------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------------

symbolic_symbol(name::Symbol)::SymbolicNode = intern_symbolic_node(SYMBOLIC_SYMBOL, false, 0 // 1, 0.0, name, NO_SYMBOLIC_ARGS)

function symbolic_isless(a::SymbolicNode, b::SymbolicNode)
    if a.order_name != b.order_name
        return a.order_name < b.order_name
    end
    if a.kind != b.kind
        return a.kind < b.kind
    end
    if length(a.args) != length(b.args)
        return length(a.args) < length(b.args)
    end
    return a.hash < b.hash
end

"""Split a node into its numeric coefficient and the remaining non-numeric part."""
function split_coefficient(term::SymbolicNode)::Tuple{SymbolicNode, SymbolicNode}
    if term.kind == SYMBOLIC_PRODUCT && is_symbolic_number(term.args[1])
        coefficient = term.args[1]
        rest = term.args[2:end]
        if length(rest) == 1
            return coefficient, rest[1]
        end
        return coefficient, intern_symbolic_node(SYMBOLIC_PRODUCT, false, 0 // 1, 0.0, Symbol(""), rest)
    end
    return symbolic_one(), term
end

function symbolic_sum(terms::AbstractVector{SymbolicNode})::SymbolicNode
    constant = symbolic_zero()
    coefficients = Dict{SymbolicNode, SymbolicNode}()
    order = SymbolicNode[]

    stack = collect(terms)
    flat = SymbolicNode[]
    while !isempty(stack)
        t = popfirst!(stack)
        if t.kind == SYMBOLIC_SUM
            prepend!(stack, t.args)
        else
            push!(flat, t)
        end
    end

    for t in flat
        if t.kind == SYMBOLIC_NUMBER
            constant = add_numbers(constant, t)
            continue
        end
        c, core = split_coefficient(t)
        if haskey(coefficients, core)
            coefficients[core] = add_numbers(coefficients[core], c)
        else
            coefficients[core] = c
            push!(order, core)
        end
    end

    result_terms = SymbolicNode[]
    for core in order
        c = coefficients[core]
        is_symbolic_zero(c) && continue
        push!(result_terms, attach_coefficient(c, core))
    end
    sort!(result_terms, lt = symbolic_isless)

    if isempty(result_terms)
        return constant
    end
    is_symbolic_zero(constant) || push!(result_terms, constant)
    length(result_terms) == 1 && return result_terms[1]
    return intern_symbolic_node(SYMBOLIC_SUM, false, 0 // 1, 0.0, Symbol(""), result_terms)
end

symbolic_sum(terms::SymbolicNode...) = symbolic_sum(SymbolicNode[terms...])

function attach_coefficient(c::SymbolicNode, core::SymbolicNode)::SymbolicNode
    is_symbolic_one(c) && c.exact && return core
    if core.kind == SYMBOLIC_PRODUCT
        return intern_symbolic_node(SYMBOLIC_PRODUCT, false, 0 // 1, 0.0, Symbol(""), vcat(c, core.args))
    end
    return intern_symbolic_node(SYMBOLIC_PRODUCT, false, 0 // 1, 0.0, Symbol(""), SymbolicNode[c, core])
end

function symbolic_product(factors::AbstractVector{SymbolicNode})::SymbolicNode
    coefficient = symbolic_one()
    exponents = Dict{SymbolicNode, Vector{SymbolicNode}}()
    order = SymbolicNode[]

    stack = collect(factors)
    while !isempty(stack)
        f = popfirst!(stack)
        if f.kind == SYMBOLIC_PRODUCT
            prepend!(stack, f.args)
        elseif f.kind == SYMBOLIC_NUMBER
            coefficient = multiply_numbers(coefficient, f)
        else
            base, exponent = f.kind == SYMBOLIC_POWER ? (f.args[1], f.args[2]) : (f, symbolic_one())
            if haskey(exponents, base)
                push!(exponents[base], exponent)
            else
                exponents[base] = SymbolicNode[exponent]
                push!(order, base)
            end
        end
    end

    if is_symbolic_zero(coefficient)
        return coefficient
    end

    result_factors = SymbolicNode[]
    needs_flattening = false
    for base in order
        exponent = length(exponents[base]) == 1 ? exponents[base][1] : symbolic_sum(exponents[base])
        is_symbolic_zero(exponent) && continue
        f = symbolic_power(base, exponent)
        if f.kind == SYMBOLIC_NUMBER || f.kind == SYMBOLIC_PRODUCT
            needs_flattening = true
        end
        push!(result_factors, f)
    end

    if needs_flattening
        return symbolic_product(vcat(coefficient, result_factors))
    end

    sort!(result_factors, lt = symbolic_isless)

    if isempty(result_factors)
        return coefficient
    end

    if length(result_factors) == 1
        single = result_factors[1]
        is_symbolic_one(coefficient) && coefficient.exact && return single
        if single.kind == SYMBOLIC_SUM
            # distribute a numeric coefficient over a sum, as computer algebra systems do
            return symbolic_sum([symbolic_product(SymbolicNode[coefficient, t]) for t in single.args])
        end
    end

    if is_symbolic_one(coefficient) && coefficient.exact
        return intern_symbolic_node(SYMBOLIC_PRODUCT, false, 0 // 1, 0.0, Symbol(""), result_factors)
    end
    return intern_symbolic_node(SYMBOLIC_PRODUCT, false, 0 // 1, 0.0, Symbol(""), vcat(coefficient, result_factors))
end

symbolic_product(factors::SymbolicNode...) = symbolic_product(SymbolicNode[factors...])

function symbolic_power(base::SymbolicNode, exponent::SymbolicNode)::SymbolicNode
    if exponent.kind == SYMBOLIC_NUMBER
        exponent.value == 0 && return symbolic_one()
        is_symbolic_one(exponent) && exponent.exact && return base
        if is_symbolic_one(exponent) && base.kind == SYMBOLIC_NUMBER
            return base
        end
    end
    if base.kind == SYMBOLIC_NUMBER
        if base.value == 1 && base.exact
            return symbolic_one()
        end
        if exponent.kind == SYMBOLIC_NUMBER
            folded = power_numbers(base, exponent)
            folded === nothing || return folded
        end
    end
    if is_exact_integer(exponent)
        if base.kind == SYMBOLIC_POWER
            return symbolic_power(base.args[1], symbolic_product(base.args[2], exponent))
        elseif base.kind == SYMBOLIC_PRODUCT
            return symbolic_product([symbolic_power(f, exponent) for f in base.args])
        end
    end
    if base.kind == SYMBOLIC_CALL && base.name == :exp && length(base.args) == 1
        # (exp(a))^b = exp(a * b) holds for real a
        return symbolic_call(:exp, SymbolicNode[symbolic_product(base.args[1], exponent)])
    end
    return intern_symbolic_node(SYMBOLIC_POWER, false, 0 // 1, 0.0, Symbol(""), SymbolicNode[base, exponent])
end

symbolic_power(base::SymbolicNode, exponent::Number) = symbolic_power(base, symbolic_number(exponent))

symbolic_negate(a::SymbolicNode) = symbolic_product(symbolic_minus_one(), a)
symbolic_difference(a::SymbolicNode, b::SymbolicNode) = symbolic_sum(a, symbolic_negate(b))
symbolic_quotient(a::SymbolicNode, b::SymbolicNode) = symbolic_product(a, symbolic_power(b, symbolic_minus_one()))

const SYMBOLIC_FUNCTION_ALIASES = Dict{Symbol, Symbol}(:Max => :max, :Min => :min)

"""Functions whose numeric evaluation is defined for every finite real argument."""
const SYMBOLIC_TOTAL_FUNCTIONS = Set{Symbol}([:exp, :exp2, :exp10, :sin, :cos, :tan, :atan, :sinh, :cosh, :tanh, :asinh,
                                               :abs, :erfc, :erf, :normcdf, :pnorm, :normpdf, :dnorm, :normlogpdf, :sign,
                                               :max, :min, :abs2])

function symbolic_function_domain_ok(name::Symbol, args::Vector{Float64})::Bool
    name ∈ SYMBOLIC_TOTAL_FUNCTIONS && return true
    length(args) == 1 || return false
    x = args[1]
    name ∈ (:log, :log2, :log10) && return x > 0
    name == :log1p && return x > -1
    name ∈ (:asin, :acos) && return -1 <= x <= 1
    name == :acosh && return x >= 1
    name == :atanh && return -1 < x < 1
    name == :erfcinv && return 0 < x < 2
    name ∈ (:norminvcdf, :norminv, :qnorm) && return 0 < x < 1
    return false
end

function symbolic_numeric_function(name::Symbol)
    name == :erfc && return erfc
    name == :erfcinv && return erfcinv
    name == :erf && return SpecialFunctions.erf
    isdefined(@__MODULE__, name) || return nothing
    f = getfield(@__MODULE__, name)
    return f isa Function ? f : nothing
end

const SYMBOLIC_COMPARISONS = Set{Symbol}([:<, :>, :<=, :>=, :(==), :!=, :≤, :≥, :≠])

function symbolic_call(name::Symbol, args::Vector{SymbolicNode})::SymbolicNode
    name = get(SYMBOLIC_FUNCTION_ALIASES, name, name)
    n = length(args)

    if name == :sqrt && n == 1
        return symbolic_power(args[1], symbolic_number(1 // 2))
    elseif name == :log && n == 2
        return symbolic_quotient(symbolic_call(:log, SymbolicNode[args[2]]), symbolic_call(:log, SymbolicNode[args[1]]))
    elseif name == :exp && n == 1
        a = args[1]
        is_symbolic_zero(a) && return symbolic_one()
        if a.kind == SYMBOLIC_CALL && a.name == :log && length(a.args) == 1
            return a.args[1]
        end
    elseif name == :log && n == 1
        a = args[1]
        is_symbolic_one(a) && return symbolic_zero()
        if a.kind == SYMBOLIC_CALL && a.name == :exp && length(a.args) == 1
            return a.args[1]
        end
    elseif name == :ifelse && n == 3
        c = args[1]
        if c.kind == SYMBOLIC_NUMBER
            return c.value != 0 ? args[2] : args[3]
        end
        args[2] == args[3] && return args[2]
    end

    if n > 0 && all(is_symbolic_number, args)
        values = Float64[a.value for a in args]
        if name ∈ SYMBOLIC_COMPARISONS && n == 2
            f = name == :≤ ? (<=) : name == :≥ ? (>=) : name == :≠ ? (!=) : getfield(Base, name)
            return symbolic_number(Int(f(values[1], values[2])))
        end
        if symbolic_function_domain_ok(name, values)
            f = symbolic_numeric_function(name)
            if f !== nothing
                result = f(values...)
                if result isa Real && isfinite(result)
                    if all(a -> a.exact, args) && name ∈ (:abs, :max, :min, :sign)
                        exact_values = [a.rational for a in args]
                        return symbolic_number(name == :abs ? abs(exact_values[1]) :
                                               name == :max ? maximum(exact_values) :
                                               name == :min ? minimum(exact_values) :
                                               sign(exact_values[1]))
                    end
                    return symbolic_number(Float64(result))
                end
            end
        end
    end

    return intern_symbolic_node(SYMBOLIC_CALL, false, 0 // 1, 0.0, name, args)
end

symbolic_call(name::Symbol, args::SymbolicNode...) = symbolic_call(name, SymbolicNode[args...])

"""Rebuild `node` with new children through the canonical constructors."""
function rebuild_symbolic_node(node::SymbolicNode, args::Vector{SymbolicNode})::SymbolicNode
    node.kind == SYMBOLIC_SUM && return symbolic_sum(args)
    node.kind == SYMBOLIC_PRODUCT && return symbolic_product(args)
    node.kind == SYMBOLIC_POWER && return symbolic_power(args[1], args[2])
    node.kind == SYMBOLIC_CALL && return symbolic_call(node.name, args)
    return node
end

# ----------------------------------------------------------------------------
# Conversion from Julia expressions
# ----------------------------------------------------------------------------

to_symbolic(x::SymbolicNode) = x
to_symbolic(x::Number) = symbolic_number(x)
to_symbolic(x::Symbol) = symbolic_symbol(x)

function to_symbolic(ex::Expr)::SymbolicNode
    if ex.head == :block
        args = filter(a -> !(a isa LineNumberNode), ex.args)
        length(args) == 1 || throw(ArgumentError("cannot convert block with $(length(args)) expressions to a symbolic expression"))
        return to_symbolic(args[1])
    elseif ex.head == :(=)
        return symbolic_difference(to_symbolic(ex.args[1]), to_symbolic(ex.args[2]))
    elseif ex.head == :if && length(ex.args) == 3
        return symbolic_call(:ifelse, SymbolicNode[to_symbolic(a) for a in ex.args])
    elseif ex.head == :comparison && length(ex.args) == 3
        return symbolic_call(ex.args[2], SymbolicNode[to_symbolic(ex.args[1]), to_symbolic(ex.args[3])])
    elseif ex.head ∈ (:&&, :||)
        return symbolic_call(ex.head, SymbolicNode[to_symbolic(a) for a in ex.args])
    elseif ex.head != :call
        throw(ArgumentError("cannot convert expression with head $(ex.head) to a symbolic expression: $ex"))
    end

    op = ex.args[1]
    if op isa Expr && op.head == :. && length(op.args) == 2 && op.args[2] isa QuoteNode
        op = op.args[2].value
    end
    op isa Symbol || throw(ArgumentError("cannot convert call to $(op) to a symbolic expression"))
    args = SymbolicNode[to_symbolic(a) for a in ex.args[2:end]]

    if op == :+
        return symbolic_sum(args)
    elseif op == :-
        length(args) == 1 && return symbolic_negate(args[1])
        return symbolic_difference(args[1], symbolic_sum(args[2:end]))
    elseif op == :*
        return symbolic_product(args)
    elseif op == :/
        return symbolic_quotient(args[1], args[2])
    elseif op == :^
        return symbolic_power(args[1], args[2])
    elseif op == :// && all(is_exact_integer, args)
        return symbolic_number(args[1].rational.num // args[2].rational.num)
    elseif op == :inv && length(args) == 1
        return symbolic_power(args[1], symbolic_minus_one())
    end
    return symbolic_call(op, args)
end

# ----------------------------------------------------------------------------
# Conversion to Julia expressions
# ----------------------------------------------------------------------------

function number_to_expr(n::SymbolicNode)
    if n.exact
        r = n.rational
        denominator(r) == 1 && return numerator(r)
        return Expr(:call, :/, numerator(r), denominator(r))
    end
    return n.value
end

"""Return `(is_negative, absolute_value_expression)` for a term of a sum."""
function signed_term_expr(term::SymbolicNode, cache::IdDict{SymbolicNode, Any})
    if term.kind == SYMBOLIC_NUMBER && term.value < 0
        return true, number_to_expr(symbolic_negate(term))
    end
    if term.kind == SYMBOLIC_PRODUCT && is_symbolic_number(term.args[1]) && term.args[1].value < 0
        return true, from_symbolic(symbolic_negate(term), cache)
    end
    return false, from_symbolic(term, cache)
end

function product_to_expr(node::SymbolicNode, cache::IdDict{SymbolicNode, Any})
    coefficient = symbolic_one()
    numerator_factors = Any[]
    denominator_factors = Any[]
    for f in node.args
        if f.kind == SYMBOLIC_NUMBER
            coefficient = f
        elseif f.kind == SYMBOLIC_POWER && is_symbolic_number(f.args[2]) && f.args[2].value < 0
            push!(denominator_factors, from_symbolic(symbolic_power(f.args[1], symbolic_negate(f.args[2])), cache))
        else
            push!(numerator_factors, from_symbolic(f, cache))
        end
    end

    negative = coefficient.value < 0
    if negative
        coefficient = symbolic_negate(coefficient)
    end

    if coefficient.exact
        p, q = numerator(coefficient.rational), denominator(coefficient.rational)
        p != 1 && pushfirst!(numerator_factors, p)
        q != 1 && pushfirst!(denominator_factors, q)
    elseif coefficient.value != 1
        pushfirst!(numerator_factors, coefficient.value)
    end

    numerator_expr = isempty(numerator_factors) ? 1 :
                     length(numerator_factors) == 1 ? numerator_factors[1] :
                     Expr(:call, :*, numerator_factors...)
    result = if isempty(denominator_factors)
        numerator_expr
    else
        denominator_expr = length(denominator_factors) == 1 ? denominator_factors[1] : Expr(:call, :*, denominator_factors...)
        Expr(:call, :/, numerator_expr, denominator_expr)
    end
    return negative ? Expr(:call, :-, result) : result
end

function power_to_expr(node::SymbolicNode, cache::IdDict{SymbolicNode, Any})
    base, exponent = node.args
    base_expr = from_symbolic(base, cache)
    if exponent.kind == SYMBOLIC_NUMBER
        if exponent.exact && exponent.rational == 1 // 2
            return Expr(:call, :sqrt, base_expr)
        end
        if exponent.value < 0
            positive = symbolic_negate(exponent)
            inner = is_symbolic_one(positive) && positive.exact ? base_expr :
                    positive.exact && positive.rational == 1 // 2 ? Expr(:call, :sqrt, base_expr) :
                    Expr(:call, :^, base_expr, number_to_expr(positive))
            return Expr(:call, :/, 1, inner)
        end
        return Expr(:call, :^, base_expr, number_to_expr(exponent))
    end
    return Expr(:call, :^, base_expr, from_symbolic(exponent, cache))
end

"""
    from_symbolic(node)

Convert a symbolic node to a Julia expression (`Expr`, `Symbol` or number).
"""
function from_symbolic(node::SymbolicNode, cache::IdDict{SymbolicNode, Any} = IdDict{SymbolicNode, Any}())
    haskey(cache, node) && return cache[node]
    result = if node.kind == SYMBOLIC_NUMBER
        number_to_expr(node)
    elseif node.kind == SYMBOLIC_SYMBOL
        node.name
    elseif node.kind == SYMBOLIC_SUM
        expr = nothing
        signed_terms = [signed_term_expr(term, cache) for term in node.args]
        # print positive terms first, as in `1 - δ` rather than `-δ + 1`
        for (negative, term_expr) in vcat(filter(t -> !t[1], signed_terms), filter(t -> t[1], signed_terms))
            if expr === nothing
                expr = negative ? Expr(:call, :-, term_expr) : term_expr
            elseif negative
                expr = Expr(:call, :-, expr, term_expr)
            elseif expr isa Expr && expr.head == :call && expr.args[1] == :+ && length(expr.args) > 2
                expr = Expr(:call, :+, expr.args[2:end]..., term_expr)
            else
                expr = Expr(:call, :+, expr, term_expr)
            end
        end
        expr
    elseif node.kind == SYMBOLIC_PRODUCT
        product_to_expr(node, cache)
    elseif node.kind == SYMBOLIC_POWER
        power_to_expr(node, cache)
    else
        if node.name ∈ (:&&, :||)
            Expr(node.name, [from_symbolic(a, cache) for a in node.args]...)
        else
            Expr(:call, node.name, [from_symbolic(a, cache) for a in node.args]...)
        end
    end
    cache[node] = result
    return result
end

"""
    simplify_symbolic(ex)

Canonicalize and locally simplify a Julia expression (combining like terms and
powers, folding numbers) and return it as a Julia expression.
"""
simplify_symbolic(ex::SymbolicExpression) = from_symbolic(to_symbolic(ex))

# ----------------------------------------------------------------------------
# Queries and substitution
# ----------------------------------------------------------------------------

function collect_free_symbols!(symbols::Set{Symbol}, node::SymbolicNode, seen::IdDict{SymbolicNode, Nothing})
    haskey(seen, node) && return symbols
    seen[node] = nothing
    if node.kind == SYMBOLIC_SYMBOL
        push!(symbols, node.name)
    else
        for a in node.args
            collect_free_symbols!(symbols, a, seen)
        end
    end
    return symbols
end

"""Return the set of symbols that occur in `node`."""
free_symbols(node::SymbolicNode) = collect_free_symbols!(Set{Symbol}(), node, IdDict{SymbolicNode, Nothing}())

function contains_symbol(node::SymbolicNode, name::Symbol, memo::IdDict{SymbolicNode, Bool} = IdDict{SymbolicNode, Bool}())::Bool
    node.kind == SYMBOLIC_SYMBOL && return node.name == name
    node.kind == SYMBOLIC_NUMBER && return false
    haskey(memo, node) && return memo[node]
    found = false
    for a in node.args
        if contains_symbol(a, name, memo)
            found = true
            break
        end
    end
    memo[node] = found
    return found
end

function substitute_symbolic(node::SymbolicNode, replacements::AbstractDict{Symbol, SymbolicNode},
                             memo::IdDict{SymbolicNode, SymbolicNode} = IdDict{SymbolicNode, SymbolicNode}())::SymbolicNode
    if node.kind == SYMBOLIC_SYMBOL
        return get(replacements, node.name, node)
    end
    node.kind == SYMBOLIC_NUMBER && return node
    haskey(memo, node) && return memo[node]
    changed = false
    new_args = Vector{SymbolicNode}(undef, length(node.args))
    for (i, a) in enumerate(node.args)
        new_args[i] = substitute_symbolic(a, replacements, memo)
        changed |= new_args[i] !== a
    end
    result = changed ? rebuild_symbolic_node(node, new_args) : node
    memo[node] = result
    return result
end

"""
    substitute_symbolic(ex, variable, replacement)

Substitute `replacement` for `variable` in the Julia expression `ex` and return
the simplified Julia expression.
"""
function substitute_symbolic(ex::SymbolicExpression, variable::Symbol, replacement::SymbolicExpression)
    node = to_symbolic(ex)
    return from_symbolic(substitute_symbolic(node, Dict{Symbol, SymbolicNode}(variable => to_symbolic(replacement))))
end

"""Count call operations, as `count_ops` does for Julia expressions."""
function count_symbolic_operations(node::SymbolicNode)::Int
    seen = IdDict{SymbolicNode, Nothing}()
    stack = SymbolicNode[node]
    count = 0
    while !isempty(stack)
        n = pop!(stack)
        haskey(seen, n) && continue
        seen[n] = nothing
        if n.kind ∉ (SYMBOLIC_NUMBER, SYMBOLIC_SYMBOL)
            count += 1
            append!(stack, n.args)
        end
    end
    return count
end

Base.show(io::IO, node::SymbolicNode) = print(io, from_symbolic(node))

Base.copy(node::SymbolicNode) = node
Base.transpose(node::SymbolicNode) = node
Base.adjoint(node::SymbolicNode) = node
