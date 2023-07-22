"""
    SymPyCall

Module to call Python's SymPy package from Julia using PythonCall
"""
module SymPyCall

# usings/imports
using PythonCall
import PythonCall: Py
using SpecialFunctions
using LinearAlgebra
# using Markdown
import CommonSolve
import CommonSolve: solve
using CommonEq
# using RecipesBase
# using Latexify

# includes
include("types.jl")
include("utils.jl")
include("decl.jl")
include("constructors.jl")
include("generic.jl")
include("arithmetic.jl")
include("equations.jl")
include("gen_methods.jl")
include("conveniences.jl")
include("logical.jl")
include("matrix.jl")
include("assumptions.jl")
include("patternmatch.jl")
include("introspection.jl")
include("lambdify.jl")
# include("plot_recipes.jl")
# include("latexify_recipe.jl")
# export
export sympy,
    Sym,
    @syms, symbols,
    N,
    solve, Eq, Lt, Le, Ne, Ge, Gt,
    PI, E, oo, zoo, IM, TRUE, FALSE,
    𝑄, ask, refine,
    rewrite,
    Differential



# define sympy in int
const __sympy__ = PythonCall.pynew()
const sympy = Sym(__sympy__)

# core.sympy.numbers
const __PI__ = PythonCall.pynew()
const PI = Sym(__PI__)

const __E__ = PythonCall.pynew()
const E = Sym(__E__)


const __IM__ = PythonCall.pynew()
const IM = Sym(__IM__)

const __oo__ = PythonCall.pynew()
const oo = Sym(__oo__)

const __zoo__ = PythonCall.pynew()
const zoo = Sym(__zoo__)

const __TRUE__ = PythonCall.pynew()
const TRUE = Sym(__TRUE__)

const __FALSE__ = PythonCall.pynew()
const FALSE = Sym(__FALSE__)

const __𝑄__ = PythonCall.pynew()
const 𝑄 = Sym(__𝑄__)

function __init__()

    PythonCall.pycopy!(__sympy__, PythonCall.pyimport("sympy"))

    PythonCall.pycopy!(__PI__, __sympy__.pi)
    PythonCall.pycopy!(__E__, __sympy__.E)
    PythonCall.pycopy!(__IM__, __sympy__.I)
    PythonCall.pycopy!(__oo__, __sympy__.oo)
    PythonCall.pycopy!(__zoo__, __sympy__.zoo)
    PythonCall.pycopy!(__TRUE__, pyconvert(Py, true))
    PythonCall.pycopy!(__FALSE__, pyconvert(Py, false))
    PythonCall.pycopy!(__𝑄__, __sympy__.Q)


end




end
