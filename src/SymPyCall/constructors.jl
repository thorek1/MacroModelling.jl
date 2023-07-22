_symbols(a; kwargs...) = Sym(__sympy__.symbols(a; kwargs...))
# split on comma (add space?)
symbols(a; kwargs...) = contains(a, ",") ? _symbols.(split(a, ","); kwargs...) : _symbols(a; kwargs...)

macro syms(xs...)
    # If the user separates declaration with commas, the top-level expression is a tuple
    if length(xs) == 1 && isa(xs[1], Expr) && xs[1].head == :tuple
        _gensyms(xs[1].args...)
    elseif length(xs) > 0
        _gensyms(xs...)
    end
end
