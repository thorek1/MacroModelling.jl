
## Module for Introspection
module Introspection

import ..SymPyCall: Sym, asSymbolic
using PythonCall
import PythonCall: Py
export args, func, funcname, class, classname, getmembers


# utilities

"""
    funcname(x)

Return name or ""
"""
function funcname(x::Union{Sym, Py})
    y = Py(x)
    func = pygetattr(y, "func", nothing)
    func == nothing && return ""
    return string(func.__name__)
end

"""
   func(x)

Return function head from an expression

[Invariant:](http://docs.sympy.org/dev/tutorial/manipulation.html)

Every well-formed SymPy expression `ex` must either have `length(args(ex)) == 0` or
`func(ex)(args(ex)...) = ex`.
"""
func(ex::Sym) = return Py(ex).func

"""
    args(x)

Return arguments of `x`, as a tuple. (Empty if no `:args` property.)
"""
function args(x::Union{Sym, Py})
    asSymbolic(pygetattr(Py(x), "args", ()))
end

# return class or nothing
class(x::T) where {T <: Union{Sym, Py}} = getattr(x, "__class__", nothing)
classname(x::T) where {T <: Union{Sym, Py}} = (cls = class(x); isnothing(cls) ? "" : cls.__name__)

#function getmembers(x::T) where {T <: Union{Sym, PyObject}}
#    Dict(u=>v for (u,v) in inspect.getmembers(x))
#end

end
