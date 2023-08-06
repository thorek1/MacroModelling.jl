# generic functions
Base.hash(x::Sym, v::UInt64) = hash(Py(x), v)
Base.hash(x::Sym) = hash(Py(x))
Base.:(==)(x::Sym, y::Sym) = Bool(Py(x) == Py(y))

Base.isless(a::Sym, b::Sym) = (a != sympy.nan && b != sympy.nan) && sympy.Lt(a,b) == sympy.True
Base.isless(a::Sym, b::Number) = isless(promote(a,b)...)
Base.isless(a::Number, b::Sym) = isless(promote(a,b)...)




zero(::Sym) = zero(Sym)
zero(::Type{Sym}) = Sym(0)
one(::Sym) = one(Sym)
one(::Type{Sym}) = Sym(1)


# math functions which are not evaled into instanc
Base.abs2(x::SymbolicObject) = x * conj(x)
Base.cbrt(x::Sym) = x^(1//3)
Base.asech(z::Sym) = log(sqrt(1/z-1)*sqrt(1/z+1) + 1/z)
Base.acsch(z::Sym) = log(sqrt(1+1/z^2) + 1/z) ## http://mathworld.wolfram.com/InverseHyperbolicCosecant.html
Base.atan(y::Sym, x) = sympy.atan2(y,x)

Base.sinc(x::Sym) = iszero(x) ? one(x) : sin(PI*x)/(PI*x)
cosc(x::Sym) = diff(sinc(x))

Base.sincos(x::Sym) = (sin(x), cos(x))
Base.sinpi(x::Sym) = sin(PI * x)
Base.cospi(x::Sym) = cos(PI * x)
# XXX do cosd, ...?

Base.rad2deg(x::Sym) = (x * 180) / PI
Base.deg2rad(x::Sym) = (x * PI) / 180

Base.hypot(x::Sym, y::Number) = hypot(promote(x,y)...)
Base.hypot(xs::Sym...) = sqrt(sum(abs(xᵢ)^2 for xᵢ ∈ xs))

## exponential/Log
Base.log1p(x::Sym) = log(1 + x)
Base.log(b::Number, x::Sym) = sympy.log(x, b)
Base.log2(x::SymbolicObject) = log(2, x)
Base.log10(x::SymbolicObject) = log(10, x)


## --------------------------------------------------
## floating point stuff


# Floating point bits
Base.eps(::Type{Sym}) = zero(Sym)
Base.eps(::Sym) = zero(Sym)
Base.signbit(x::Sym) = x < 0
Base.copysign(x::Sym, y::Number) = abs(x) * sign(y)
Base.flipsign(x::Sym, y) = signbit(y) ? -x : x
Base.typemax(::Type{Sym}) = oo
Base.typemin(::Type{Sym}) = -oo

Base.fld(x::SymbolicObject, y) = floor(x/y)
Base.cld(x::SymbolicObject, y) = ceil(x/y)

#

Base.isfinite(x::Sym) = !is_(:infinite, x)
Base.isinf(x::Sym) = is_(:infinite, x)
Base.isnan(x::Sym) = x == sympy.nan
Base.isinteger(x::Sym) = is_(:integer, x)
Base.iseven(x::Sym) = is_(:even, x)
Base.isreal(x::Sym) = is_(:real, x)
Base.isodd(x::Sym) = is_(:odd, x)

# XXX right semantics?
function Base.float(x::Sym)
    u = N(x)
    u === x && return x
    float(u)
end

Base.Integer(x::Sym) = is_(:integer, x) ? N(x) : throw(DomainError("x can not be converted to an integer"))

Base.complex(::Type{Sym}) = Sym
Base.complex(r::Sym) = real(r) + imag(r) * im
function Base.complex(r::Sym, i)
    isreal(r) || throw(ArgumentError("r and i must not be complex"))
    isreal(i) || throw(ArgumentError("r and i must not be complex"))
    N(r) + N(i) * im
end
Base.complex(xs::AbstractArray{Sym}) = complex.(xs) # why is this in base?

Base.real(::Type{Sym}) = Sym
Base.real(x::Sym) = sympy.re(x)
Base.imag(x::Sym) =  sympy.im(x)

Base.angle(z::SymbolicObject) = atan(sympy.im(z), sympy.re(z))


Base.divrem(x::Sym, y) = sympy.div(x, y)
# needed for #390; but odd
Base.div(x::Sym, y::Union{Sym,Number}) = convert(Sym, sympy.floor(x/convert(Sym,y)))
Base.rem(x::Sym, y::Union{Sym,Number}) = x-Sym(y)*Sym(sympy.floor.(x/y))
