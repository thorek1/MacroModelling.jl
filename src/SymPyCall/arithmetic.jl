# arithmetic
Base.promote_rule(::Type{S}, ::Type{T})  where {T<:Sym, S<:Number}= T
Base.promote_rule(::Type{T}, ::Type{S})  where {T<:Sym, S<:Number}= T
Base.promote_rule(::Type{S}, ::Type{T})  where {T<:Sym, S<:Irrational}= T
Base.promote_rule(::Type{T}, ::Type{S})  where {T<:Sym, S<:Irrational}= T
Base.promote_rule(::Type{Sym}, ::Type{Bool}) = Sym
Base.promote_rule(::Type{Bool}, ::Type{Sym}) = Sym
Base.convert(::Type{T}, o::Py) where {T <: Sym} = T(o)
Base.convert(::Type{T}, o::Number) where {T <: Sym} = Sym(o)

Base.:(-)(x::Sym) = Sym(-unSym(x))
Base.:(+)(x::Sym, y::Sym) = Sym(+(unSym(x), unSym(y)))
Base.:(-)(x::Sym, y::Sym) = Sym(-(unSym(x), unSym(y)))
Base.:(*)(x::Sym, y::Sym) = Sym(*(unSym(x), unSym(y)))
Base.:(/)(x::Sym, y::Sym) = Sym(/(unSym(x), unSym(y)))
Base.:(^)(x::Sym, y::Int) = ↑(↓(x)^y)
Base.:(^)(x::Sym, y::Rational) = ^(promote(x, y)...)
Base.:(^)(x::Sym, y::Real) = ↑(↓(x)^y)
Base.:(^)(x::Sym, y::Sym) = Sym(^(unSym(x), unSym(y)))

# Bools
Base.:(+)(x::Sym, y::Bool) = x + Int(y)
Base.:(+)(x::Bool, y::Sym) = y + x
Base.:(-)(x::Sym, y::Bool) = x - Int(y)
Base.:(-)(x::Bool, y::Sym) = Int(x) - y
Base.:(*)(x::Sym, y::Bool) = x * Int(y)
Base.:(*)(x::Bool, y::Sym) = Int(x) * y
Base.:(/)(x::Sym, y::Bool) = x * Int(y)
Base.:(/)(x::Bool, y::Sym) = Int(x) * y
Base.:(^)(x::Sym, y::Bool) = x^Int(y)
Base.:(^)(x::Bool, y::Sym) = Int(x)^y
