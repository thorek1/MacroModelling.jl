## TODO: add more types...
abstract type SymbolicObject <: Number end

struct Sym <: SymbolicObject
    py::Py
end

PythonCall.Py(x::Sym) = x.py
PythonCall.ispy(x::Sym) = true

function Base.show(io::IO, s::SymbolicObject)
    out = PythonCall.pyrepr(String, Py(s))
    out = replace(out, r"\*\*" => "^")
    print(io, out)
end

# allow dot access to underlying methods, properties
function Base.getproperty(x::Sym, a::Symbol)

    a == :py && return getfield(x,a)

    py = Py(x)
    val = pygetattr(py, string(a), nothing)
    val === nothing && return nothing

    # hacky, but seems necessary for sympy.Poly, sympy.Curve, and others
    contains(string(pytype(val)), "ManagedProperties") && return (args...;kwargs...) -> ↑(val(unSym.(args)...; unSymkwargs(kwargs)...))

    meth = pygetattr(val, "__call__", nothing)
    meth == nothing && return Sym(val)

    return SymbolicCallable(meth)

end

# Call
(ex::Sym)(xs...) = subs(ex, xs...)

Sym(x::AbstractString) = symbols(x)
Sym(x::Symbol) = Sym(string(x))
Sym(x::Sym) = x
Sym(x::Complex{Bool}) = 1*real(x) + 1*imag(x)*IM
Sym(x::Complex) = Sym(real(x)) + Sym(imag(x))*IM
Sym(x::Number) = sympy.sympify(x) #Sym(sympy.pyconvert(Py, x))
Sym(x::PyArray) = Sym.(x)
Sym(x::PyDict) = Dict(Sym(k) => Sym(v) for (k,v) ∈ x)
Sym(x::PyIterable) = (Sym(u) for u ∈ x)
Sym(x::PyList) = Sym.(x)
Sym(x::PySet) = Set(Sym.(x))
#Sym(x::PyTable) =

Sym(x::Irrational{:π}) = sympy.pi
Sym(x::Irrational{:ℯ}) = sympy.E
Sym(x::Irrational{:e}) = sympy.E



struct SymbolicCallable
    val
end
function (v::SymbolicCallable)(args...; kwargs...)
    ↑(v.val(unSym.(args)...; unSymkwargs(kwargs)...))
end

struct SymFunction <: SymbolicObject
    py::Py
end
PythonCall.Py(u::SymFunction) = u.py

function SymFunction(x::AbstractString; kwargs...)
    λ = __sympy__.Function(x; unSymkwargs(kwargs)...)
    SymFunction(λ)
end
(u::SymFunction)(xs...) = ↑(Py(u)(↓(xs)...))

# Steal this idea from ModelingToolkit
# better than the **hacky** f'(0) stuff
"""
    Differential(x)

Use to find (partial) derivatives.

## Example
```
@syms x y u()
Dx = Differential(x)
Dx(u(x,y))  # resolves to diff(u(x,y),x)
Dx(u)       # will evaluate diff(u(x), x)
```
"""
struct Differential
    x::Sym
end
(∂::Differential)(u::Sym) = diff(u, ∂.x)
(∂::Differential)(u::SymFunction) = diff(u(∂.x), ∂.x)
