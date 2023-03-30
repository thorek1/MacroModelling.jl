struct first_order_solution_function_tag end

function (ff::first_order_solution_function_tag)(x) return x end


function fff(x) return x end


typeof(ff)
first_order_solution_function_tag(first_order_solution_function_tag)
convert(first_order_solution_function_tag,func)


# Define a new function-like type with a tag "mytag"
struct MyFunction{T}
    tag::Symbol
end

# Implement the call method for MyFunction
function (f::MyFunction)(x::T) where T
    println("Called function with tag $(f.tag) and argument $x")
end

# Create a new instance of MyFunction with the tag "mytag"
myfunc = MyFunction{:mytag}()

# Call the MyFunction instance with an argument
myfunc(10)


# Define a new function-like type with a tag "mytag"
struct unction{T}
    nothing
end

# Implement the call method for MyFunction
function (f::unction)(x::T) where T
    x^2
end

function (f::unction)(x::T) where T
    x^5
end


# Create a new instance of MyFunction with the tag "mytag"
myfunca = unction{Int}(nothing)#(:mytag)

# Call the MyFunction instance with an argument
myfunca(2)

typeof(myfunca)

using ForwardDiff

@profview ForwardDiff.derivative(myfunca,10)



# Define a new function-like type with a tag "mytag"
struct MyFunction{T}
    tag::Symbol
end

# Implement the call method for MyFunction
function (f::MyFunction)(x::T) where T
    println("Called function with tag $(f.tag) and argument $x")
end

# Create a new instance of MyFunction with the tag "mytag"
myfunc = MyFunction{Int}(:mytag)

# Call the MyFunction instance with an argument
myfunc(10)

typeof(myfunc)





# Define a new function-like type with a tag "mytag"
struct myFunction end

# Implement the call method for MyFunction
function (f::myFunction)(x)
    println("Called function with tag $(f.tag) and argument $x")
end

# Create a new instance of MyFunction with the tag "mytag"
myfunc = MyFunction{Int}(:mytag)

# Call the MyFunction instance with an argument
myfunc(10)

func = x->x^2
typeof(func)



struct MyFunctionTag end

my_function(x) = x^2

my_function_typed(x::Int)::Function = my_function

my_function_typed(x::MyFunctionTag) = my_function

typeof(my_function)

typeof(my_function_typed)




struct MyFunctionTag end

my_function = x -> x^2

function my_function_typed(x::Int)::Function
    y -> my_function(y)
end

function my_function_typed(x::MyFunctionTag)
    my_function
end




# Define a custom function type
struct first_order_solution_function_tag end

function f(x, g::first_order_solution_function_tag)
    g(x)^2
end


# Define a function with the custom function type
function my_function(x::Float64)::Float64
    return x^2 + 1
end

# Call f with the custom function
f(2.0, my_function) # returns 25.0

# Redefine the custom function
function my_function(x::Float64)::Float64
    return x^2 + 2
end

# Call f with the redefined custom function
f(2.0, my_function) # returns 36.0, without recompiling f




# Define a custom function type
struct first_order_solution_function_tag end

# Define a function with the custom function type
# struct my_first_order_solution_function <: first_order_solution_function_tag end

function f(x, g::first_order_solution_function_tag)
    g(x)^2
end

# Define a method that matches the signature of the custom function type
function (g::first_order_solution_function_tag)(x::Float64)::Float64
    return x^2 + 2
end
function (h::first_order_solution_function_tag)(x::Int)::Int
    return x^2 + 1
end


# Call f with the custom function
f(2, first_order_solution_function_tag()) # returns 25.0

# Redefine the custom function
function (g::first_order_solution_function_tag)(x::Float64)::Float64
    return x^2 + 3
end

# Call f with the redefined custom function
f(2.0, first_order_solution_function_tag()) # returns 36.0, without recompiling f


using SnoopCompile

tinf = @snoopi_deep f(2.0, first_order_solution_function_tag())


lss = function (g::first_order_solution_function_tag)(x::Float64)::Float64
    return x^2 + 4+3
end

invalidations = @snoopr begin
    f(2.0, first_order_solution_function_tag())
end;

trees = SnoopCompile.invalidation_trees(invalidations);

@show length(SnoopCompile.uinvalidated(invalidations)) # show total invalidations

show(trees[end]) # show the most invalidated method

# Count number of children (number of invalidations per invalidated method)
n_invalidations = map(trees) do methinvs
    SnoopCompile.countchildren(methinvs)
end




function ff(x, g)
    g(x)^2
end

invalidations = @snoopr begin
    function g(x::Float64)::Float64
        return x^2 + 5
    end
    ff(2.0, g)
end;


trees = SnoopCompile.invalidation_trees(invalidations);

@show length(SnoopCompile.uinvalidated(invalidations)) # show total invalidations

show(trees[end]) # show the most invalidated method

# Count number of children (number of invalidations per invalidated method)
n_invalidations = map(trees) do methinvs
    SnoopCompile.countchildren(methinvs)
end



struct FWrap{F}
    f::F
end
(F::FWrap)(x...) = F.f(x...)

gggo = []
push!(gggo,FWrap(function ggogg(x)
    return x^2 + 4
end))

gg = FWrap(g)
# typeof(fff)




function ff(x, g::FWrap{typeof(g)})
    g(x)^2
end

ff(2.0,gg)

using ForwardDiff
ForwardDiff.derivative(s->ff(s,gg),2.0)
