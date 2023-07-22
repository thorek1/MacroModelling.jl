##################################################
##
## infix logical operators

## XXX Experimental! Not sure these are such a good idea ...
## but used with piecewise
Base.:&(x::Sym, y::Sym) = x.__and__(y)
Base.:|(x::Sym, y::Sym) = x.__or__(y)
Base.:!(x::Sym)         = x.__invert__()

## use ∨, ∧, ¬ for |,&,! (\vee<tab>, \wedge<tab>, \neg<tab>)
∨(x::Sym, y::Sym) = x | y
∧(x::Sym, y::Sym) = x & y
¬(x::Sym) = !x
export ∨, ∧, ¬


## In SymPy, symbolic equations are not represented by `=` or `==`
## rather ther function `Eq` is used. Here we use the unicode
## `\Equal<tab>` for an infix operator. There are also unicode values to represent analogs of `<`, `<=`, `>=`. `>`. These are

## * `<`  is `\ll<tab>`
## * `<=` is `\leqq<tab>
## * `==` is `\Equal<tab>`
## * `>=` is `\geqq<tab>`
## * `>`  is `\gg<tab>`


export ≪,≦,⩵,≧,≫, ≶, ≷ # from CommonEq
