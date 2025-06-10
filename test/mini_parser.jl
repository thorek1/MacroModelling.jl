module MiniParser
using MacroTools: postwalk, @capture, flatten, striplines, unblock

function _condition_value(cond)
    cond isa Bool && return cond
    if cond isa Expr && cond.head == :call && cond.args[1] in (:(==), :(=))
        return cond.args[2] == cond.args[3]
    elseif cond isa Expr && cond.head == :call && cond.args[1] == :>=
        if cond.args[2] isa Int && cond.args[3] isa Int
            return cond.args[2] >= cond.args[3]
        end
    end
    return nothing
end

resolve_if_expr(x) = x

function resolve_if_expr(expr::Expr)
    if expr.head == :if
        val = _condition_value(expr.args[1])
        if val === nothing
            then_part = resolve_if_expr(expr.args[2])
            else_part = length(expr.args) == 3 ? resolve_if_expr(expr.args[3]) : Expr(:block)
            return Expr(:if, expr.args[1], then_part, else_part)
        elseif val
            return resolve_if_expr(expr.args[2])
        else
            return length(expr.args) == 3 ? resolve_if_expr(expr.args[3]) : Expr(:block)
        end
    elseif expr.head == :block
        return Expr(:block, map(resolve_if_expr, expr.args)...)
    else
        return expr
    end
end

function replace_var(expr, var, val)
    postwalk(x -> x == var ? val :
              (x isa Expr && x.head == :curly && x.args[2] == var ? Expr(:curly, x.args[1], val) : x), expr)
end

function expand_for(expr)
    if expr isa Expr && expr.head == :for
        var = expr.args[1].args[1]
        range = expr.args[1].args[2]
        indices = range.head == :vect ? range.args : eval(range)
        blocks = Any[]
        for idx in indices
            body = expand_for(replace_var(expr.args[2], var, idx))
            push!(blocks, body)
        end
        return Expr(:block, blocks...)
    elseif expr isa Expr
        return Expr(expr.head, map(expand_for, expr.args)...)
    else
        return expr
    end
end

parse_for_loops(expr) = flatten(expand_for(resolve_if_expr(expr)))

end # module
