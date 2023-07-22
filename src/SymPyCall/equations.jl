Base.:~(lhs::Number, rhs::Sym) = ~(promote(lhs, rhs)...)
Base.:~(lhs::Sym, rhs::Number) = ~(promote(lhs, rhs)...)
Base.:~(lhs::Sym, rhs::Sym) = asSymbolic(sympy.py.Eq(lhs.py, rhs.py))
