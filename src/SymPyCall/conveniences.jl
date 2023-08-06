# extend to vectors
function CommonSolve.solve(eqns::Vector{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).solve)(↓(eqns), ↓(args)...; kwargs...))
end

function dsolve(eqns::Vector{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).dsolve)(↓(eqns), ↓(args)...; kwargs...))
end

function nsolve(eqns::Vector{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).nsolve)(↓(eqns), ↓(args)...; kwargs...))
end

function linsolve(eqns::AbstractArray{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).linsolve)(↓(eqns), ↓(args)...; kwargs...))
end
# function linsolve(eqns::NTuple{N, T}, args...; kwargs...) where {N <: Int, T <: SymbolicObject}
#     ↑((Py(sympy).linsolve)(↓(eqns), ↓(args)...; kwargs...))
# end

function nonlinsolve(eqns::Vector{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).nonlinsolve)(↓(eqns), ↓(args)...; kwargs...))
end

function solveset(eqns::Vector{T}, args...; kwargs...) where {T <: SymbolicObject}
    ↑((Py(sympy).solveset)(↓(eqns), ↓(args)...; kwargs...))
end



# disambiguate
SpecialFunctions.beta(x::Sym, y::Sym) = sympy.beta(x, y)
Base.log(b::Sym, x::Sym) = log(x)/log(b)

# special exports
export limit, subs,
    lhs, rhs, doit

export free_symbols
# export elements


function limit(ex::Sym, xc::Pair; kwargs...)
    ↑((Py(sympy).limit)(↓(ex), ↓(first(xc)), ↓(last(xc)); kwargs...))
end

function subs(ex::T, xs...; kwargs...) where {T <: SymbolicObject}
    fs = free_symbols(ex)
    length(fs) == length(xs) || throw(ArgumentError("free symbols mismatch"))
    subs(ex, (k=>v for (k,v) ∈ zip(fs, xs))...)
end

function subs(ex::T, d::Pair...; kwargs...)  where {T <: SymbolicObject}
    isnothing(pygetattr(Py(ex), "subs", nothing)) && return ex
    ↑(ex.py.subs(↓([(first(p), last(p)) for p in d])))
end

function subs(ex::T, d::Dict; kwargs...)  where {T <: SymbolicObject}
    for (k,v) ∈ d
        ex = ex.subs(k, v)
    end
    ex
end

# this one is in a test, but ...
subs(ex::T, y::Tuple{Any, Any}; kwargs...)          where {T <: SymbolicObject} = ex.subs(y[1], Sym(y[2]); kwargs...)
subs(ex::T, y::Tuple{Any, Any}, args...; kwargs...) where {T <: SymbolicObject} = subs(subs(ex, y), args...; kwargs...)

## curried versions to use with |>
#subs(x::SymbolicObject, y; kwargs...) = ex -> subs(ex, x, y; kwargs...)
subs(;kwargs...)                      = ex -> subs(ex; kwargs...)
subs(dict::Dict; kwargs...)           = ex -> subs(ex, dict...; kwargs...)
subs(d::Pair...; kwargs...)           = ex -> subs(ex, [(p.first, p.second) for p in d]...; kwargs...)


lhs(ex::SymbolicObject) = ↑(pygetattr(ex, "lhs", ↓(ex)))
rhs(ex::SymbolicObject) = ↑(pygetattr(ex, "rhs", ↓(ex)))
doit(ex::T; deep::Bool=false) where {T<:SymbolicObject} = ex.doit(deep=deep)
doit(; deep::Bool=false) = ex -> doit(ex, deep=deep)

rewrite(x::Sym, args...; kwargs...) = ↑(Py(x).rewrite(↓(args)...; kwargs...))

# extend to non-sym
simplify(x, args...; kwargs...) = x

# N is tricky
function N(x::Sym)
    length(free_symbols(x)) > 0 && return x # really

    u = Py(x)
    if is_(:real, x) || is_(:Float, x)
        is_(:infinite, x) && return is_(:negative, x) ? -Inf : Inf

        if is_(:integer, x)
            return pyconvert(Integer, u)
#            is_(:zero, x) && return 0
#            T = Bool(Py(abs(x)) < typemax(Int)) ? Int : BigInt
#            return pyconvert(T, Py(x))

        elseif is_(:rational, x)
            return N(numerator(x)) // N(denominator(x))
        else
            # Floating point
            is_(:zero, x) && return 0.0

            _prec = pygetattr(x, "_prec", nothing)
            if !isnothing(_prec)
                prec = pyconvert(Int, _prec)
                prec <= 64 && return pyconvert(Float64, x)
                return setprecision(BigFloat, prec) do
                    pyconvert(BigFloat, x)
                end
            else
                x == PI && return π
                x == sympy.E && return ℯ
                # _from_mpmath   ??
                return N(u.evalf(16))
                #error("not _prec?")
            end
        end
    elseif is_(:imaginary, x)
        return N(real(x)) * im
    elseif is_(:complex, x)

        is_(:infinite, x) && return complex(Inf)
        return complex(N(real(x)), N(imag(x)))


    else
        is_(:boolean, x) && return pyconvert(Bool, Py(x))
        # ...
        # check for Python object int, float, complex, mpfm mpc ,...
        u = Py(x)
        pyisinstance(u, pybuiltins.int) &&
            return pyconvert(Bool(u.bit_length() > 64) ? BigInt : Int, u)
        if pyisinstance(u, pybuiltins.float)
            prec = pyconvert(Int, pygetattr(u, "_prec", 64))
            prec <= 64 && return pyconvert(Float64, u)
            return setprecision(BigFloat, prec) do
                pyconvert(BigFloat, u)
            end
        end
        pyisinstance(u, pybuiltins.complex) &&
            return pyconvert(Complex, u)
        if pygetattr(u, "_numerator", nothing) !== nothing
            return N(↑(pygetattr(u, "_numerator"))) // N(↑(pygetattr(u, "_denominator")))
        end

    end

    _evalf = pygetattr(u, "evalf", nothing)
    !isnothing(_evalf) && return N(u.evalf(16))

    throw(ArgumentError("Could not convert $x"))
end
N(x) = N(Sym(x))
N(x::Sym, prec) = N(x.evalf(prec))

function Base.Bool(x::Sym)
    Bool(Py(x) == pybuiltins.True) && return true
    Bool(Py(x) == pybuiltins.False) && return false
    throw(ArgumentError("x is not boolean"))
end


##
function free_symbols(x::Sym)
    u = pygetattr(Py(x), "free_symbols", nothing)
    u == nothing && return Vector{Sym}[]

    fs = [Sym(uᵢ) for uᵢ ∈ PythonCall.pyconvert(PySet, u)]
    fs[sortperm(string.(fs))]   # some sorting order to rely on
end

free_symbols(x::Vector{T}) where {T <: Sym} = reduce(union, free_symbols.(x))

## setes
Base.contains(s::Sym, val) = Bool(↓(s.contains(val)))



# # Matrices
# # call SymMatrix method on Matrix{Sym}
# ## Eg. A.norm() where A = [x 1; 1 x], say
# function Base.getproperty(A::AbstractArray{T}, k::Symbol) where {T <: SymbolicObject}
#     if k in fieldnames(typeof(A))
#         return getfield(A,k)
#     else
#         u = ↓(A)
#         val = pygetattr(u, string(k), nothing)
#         if pycallable(val)
#             return SymbolicCallable(val)
#         end
#         val
#     end
# end

# Base.getindex(M::Sym, i::Int, j::Int) = M.py.__getitem__((i,j))
# Base.getindex(V::Sym, i::Int) = V.py.__getitem__((i,0))
# Base.getindex(M::Sym) = M
