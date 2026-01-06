module SymPyPythonCallExt

import MacroModelling
import MacroModelling: ℳ, SYMPYWORKSPACE_RESERVED_NAMES, convert_to_ss_equation, write_ss_check_function!
import RuntimeGeneratedFunctions: @RuntimeGeneratedFunction
import Subscripts: super, sub
import MacroTools: unblock, postwalk, prewalk, @capture, flatten

import SymPyPythonCall as SPyPyC
import PythonCall
import SpecialFunctions: erfcinv, erfc
import SparseArrays: spzeros
import Combinatorics: combinations
import BlockTriangularForm

using DispatchDoctor



# Module for SymPy symbol workspace to avoid polluting MacroModelling namespace
module SymPyWorkspace
    # Import SpecialFunctions
    using SpecialFunctions: erfcinv, erfc
    
    # Define density-related functions directly in the workspace
    # These need to be available for symbolic expressions
    function norminvcdf(p::T)::T where T
        -erfcinv(2*p) * 1.4142135623730951
    end
    norminv(p) = norminvcdf(p)
    qnorm(p) = norminvcdf(p)

    function normlogpdf(z::T)::T where T
        -(abs2(z) + 1.8378770664093453) / 2
    end

    function normpdf(z::T)::T where T
        exp(-abs2(z)/2) * 0.3989422804014327
    end

    function normcdf(z::T)::T where T
        erfc(-z * 0.7071067811865475) / 2
    end
    pnorm(p) = normcdf(p)
    dnorm(p) = normpdf(p)

    Max = max
    Min = min
end

@stable default_mode = "disable" begin

# Struct to hold symbolic representations of equations
struct symbolics
    ss_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    dyn_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    dyn_var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_exo_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    dyn_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    dyn_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_present_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_past_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_future_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    ss_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    var_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    par_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    calibration_equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}
    calibration_equations_parameters::Vector{SPyPyC.Sym{PythonCall.Core.Py}}

    vars_in_ss_equations::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    var::Set{SPyPyC.Sym{PythonCall.Core.Py}}
    ➕_vars::Set{SPyPyC.Sym{PythonCall.Core.Py}}

    ss_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
    par_calib_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}

    var_redundant_list::Vector{Set{SPyPyC.Sym{PythonCall.Core.Py}}}
end


function replace_with_one(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    tmp = SPyPyC.subs(equation, variable, 1)
    return replace_e(tmp)
end

function replace_e(equation::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    outraw = SPyPyC.subs(equation, SPyPyC.Sym(ℯ), exp(1))

    if outraw isa SPyPyC.Sym{PythonCall.Core.Py}
        out = outraw
    else
        out = collect(outraw)[1]
    end
    
    return out
end

function replace_symbolic(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py}, replacement::SPyPyC.Sym{PythonCall.Core.Py})::SPyPyC.Sym{PythonCall.Core.Py}
    return SPyPyC.subs(equation, variable, replacement)
end

function solve_symbolically(equation::SPyPyC.Sym{PythonCall.Core.Py}, variable::SPyPyC.Sym{PythonCall.Core.Py})::Union{Nothing,Vector{SPyPyC.Sym{PythonCall.Core.Py}}}
    soll = try SPyPyC.solve(equation, variable)
           catch
           end

    return soll
end

function solve_symbolically(equations::Vector{SPyPyC.Sym{PythonCall.Core.Py}}, variables::Vector{SPyPyC.Sym{PythonCall.Core.Py}})::Union{Nothing,Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}}
    soll = try SPyPyC.solve(equations, variables)
           catch
           end

    if soll == Any[]
        soll = Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}()
    elseif soll isa Vector
        soll = Dict{SPyPyC.Sym{PythonCall.Core.Py}, SPyPyC.Sym{PythonCall.Core.Py}}(variables .=> soll[1])
    end
    
    return soll
end

# compatibility with SymPy
Max = max
Min = min

function transform_expression(expr::Expr)
    # Dictionary to store the transformations for reversing
    reverse_transformations = Dict{Symbol, Expr}()

    # Counter for generating unique placeholders
    unique_counter = Ref(0)

    # Step 1: Replace min/max calls and record their original form
    function replace_min_max(expr)
        if expr isa Expr && expr.head == :call && (expr.args[1] == :min || expr.args[1] == :max)
            # Replace min/max functions with a placeholder
            # placeholder = Symbol("minimal__P", unique_counter[])
            placeholder = :minmax__P
            unique_counter[] += 1

            # Store the original min/max call for reversal
            reverse_transformations[placeholder] = expr

            return placeholder
        else
            return expr
        end
    end

    # Step 2: Transform :ref fields in the rest of the expression
    function transform_ref_fields(expr)
        if expr isa Expr && expr.head == :ref && isa(expr.args[1], Symbol)
            # Handle :ref expressions
            if isa(expr.args[2], Number) || isa(expr.args[2], Symbol)           
                if expr.args[2] < 0
                    new_symbol = Symbol(expr.args[1], "__", abs(expr.args[2]))
                else
                    new_symbol = Symbol(expr.args[1], "_", expr.args[2])
                end
            else
                # Generate a unique placeholder for complex :ref
                unique_counter[] += 1
                placeholder = Symbol("__placeholder", unique_counter[])
                new_symbol = placeholder
            end

            # Record the reverse transformation
            reverse_transformations[new_symbol] = expr

            return new_symbol
        else
            return expr
        end
    end


    # Replace equality sign with minus
    function replace_equality_with_minus(expr)
        if expr isa Expr && expr.head == :(=)
            return Expr(:call, :-, expr.args...)
        else
            return expr
        end
    end

    # Apply transformations
    expr = postwalk(replace_min_max, expr)
    expr = postwalk(transform_ref_fields, expr)
    transformed_expr = postwalk(replace_equality_with_minus, expr)
    
    return transformed_expr, reverse_transformations
end

function reverse_transformation(transformed_expr::Expr, reverse_dict::Dict{Symbol, Expr})
    # Function to replace the transformed symbols with their original form
    function revert_symbol(expr)
        if expr isa Symbol && haskey(reverse_dict, expr)
            return reverse_dict[expr]
        else
            return expr
        end
    end

    # Revert the expression using postwalk
    reverted_expr = postwalk(revert_symbol, transformed_expr)

    return reverted_expr
end

function count_ops(expr)::Int
    op_count = 0
    postwalk(x -> begin
        if x isa Expr && x.head == :call
            op_count += 1
        end
        x
    end, expr)
    return op_count
end

"""
    simplify(ex::Expr)

Simplify a Julia expression using SymPy's symbolic simplification.
"""
function simplify(ex::Expr)::Union{Expr,Symbol,Int}
    ex_ss = convert_to_ss_equation(ex)

    for x in get_symbols(ex_ss)
        sym_value = SPyPyC.symbols(string(x), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($x = $sym_value))
    end

    parsed = ex_ss |> x -> Core.eval(SymPyWorkspace, x) |> string |> Meta.parse

    postwalk(x -> x isa Expr ? 
                        x.args[1] == :conjugate ? 
                            x.args[2] : 
                        x : 
                    x, parsed)
end


"""
    transform_obc(ex::Expr; avoid_solve::Bool = false)

Transform an occasionally binding constraint expression using symbolic solving.
"""
function transform_obc(ex::Expr; avoid_solve::Bool = false)
    transformed_expr, reverse_dict = transform_expression(ex)

    for symbs in get_symbols(transformed_expr)
        sym_value = SPyPyC.symbols(string(symbs), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($symbs = $sym_value))
    end

    eq = Core.eval(SymPyWorkspace, transformed_expr)

    if avoid_solve || count_ops(Meta.parse(string(eq))) > 15
        soll = nothing
    else
        soll = solve_symbolically(eq, Core.eval(SymPyWorkspace, :minmax__P))
    end

    if !isempty(soll)
        sorted_minmax = Expr(:call, reverse_dict[:minmax__P].args[1], :($(reverse_dict[:minmax__P].args[2]) - $(Meta.parse(string(soll[1])))),  :($(reverse_dict[:minmax__P].args[3]) - $(Meta.parse(string(soll[1])))))
        return reverse_transformation(sorted_minmax, reverse_dict)
    else
        @error "Occasionally binding constraint not well-defined. See documentation for examples."
    end
end


"""
    create_symbols_eqs!(𝓂::ℳ)

Create symbolic representations of model equations using SymPy.
"""
function create_symbols_eqs!(𝓂::ℳ)::symbolics
    # create symbols in SymPyWorkspace to avoid polluting MacroModelling namespace
    symbols_in_dynamic_equations = reduce(union,get_symbols.(𝓂.dyn_equations))

    symbols_in_dynamic_equations_wo_subscripts = Symbol.(replace.(string.(symbols_in_dynamic_equations),r"₍₋?(₀|₁|ₛₛ|ₓ)₎$"=>""))

    symbols_in_ss_equations = reduce(union,get_symbols.(𝓂.ss_aux_equations))

    symbols_in_equation = union(𝓂.parameters_in_equations,𝓂.parameters,𝓂.parameters_as_function_of_parameters,symbols_in_dynamic_equations,symbols_in_dynamic_equations_wo_subscripts,symbols_in_ss_equations)

    symbols_pos = []
    symbols_neg = []
    symbols_none = []

    for symb in symbols_in_equation
        if haskey(𝓂.bounds, symb)
            if 𝓂.bounds[symb][1] >= 0
                push!(symbols_pos, symb)
            elseif 𝓂.bounds[symb][2] <= 0
                push!(symbols_neg, symb)
            else 
                push!(symbols_none, symb)
            end
        else
            push!(symbols_none, symb)
        end
    end

    # Create symbols in SymPyWorkspace instead of MacroModelling namespace
    for pos in symbols_pos
        sym_value = SPyPyC.symbols(string(pos), real = true, finite = true, positive = true)
        Core.eval(SymPyWorkspace, :($pos = $sym_value))
    end
    for neg in symbols_neg
        sym_value = SPyPyC.symbols(string(neg), real = true, finite = true, negative = true)
        Core.eval(SymPyWorkspace, :($neg = $sym_value))
    end
    for none in symbols_none
        sym_value = SPyPyC.symbols(string(none), real = true, finite = true)
        Core.eval(SymPyWorkspace, :($none = $sym_value))
    end

    symbolics(map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.ss_aux_equations),
                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.dyn_equations),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_var_present_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_var_past_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_var_future_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_exo_list),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_future_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_present_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.dyn_past_list),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.var_present_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.var_past_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.var_future_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.ss_list_aux_SS),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.var_list_aux_SS),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.par_list_aux_SS),

                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.calibration_equations),
                map(x->Core.eval(SymPyWorkspace, :($x)),𝓂.calibration_equations_parameters),

                Set(Core.eval(SymPyWorkspace, :([$(𝓂.vars_in_ss_equations...)]))),
                Set(Core.eval(SymPyWorkspace, :([$(𝓂.var...)]))),
                Set(Core.eval(SymPyWorkspace, :([$(𝓂.➕_vars...)]))),

                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.ss_calib_list),
                map(x->Set(Core.eval(SymPyWorkspace, :([$(x...)]))),𝓂.par_calib_list),

                [Set() for _ in 1:length(𝓂.ss_aux_equations)],
                )
    end


"""
    remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics; avoid_solve::Bool = false)

Remove redundant steady state variables using symbolic solving.
"""
function remove_redundant_SS_vars!(𝓂::ℳ, Symbolics::symbolics; avoid_solve::Bool = false)
    ss_equations = Symbolics.ss_equations

    # check variables which appear in two time periods. they might be redundant in steady state
    redundant_vars = intersect.(
        union.(
            intersect.(Symbolics.var_future_list,Symbolics.var_present_list),
            intersect.(Symbolics.var_future_list,Symbolics.var_past_list),
            intersect.(Symbolics.var_present_list,Symbolics.var_past_list),
            intersect.(Symbolics.ss_list,Symbolics.var_present_list),
            intersect.(Symbolics.ss_list,Symbolics.var_past_list),
            intersect.(Symbolics.ss_list,Symbolics.var_future_list)
        ),
    Symbolics.var_list)

    redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(Symbolics.var_list) .> 1))

    for i in redundant_idx
        for var_to_solve_for in redundant_vars[i]            
            if avoid_solve || count_ops(Meta.parse(string(ss_equations[i]))) > 15
                soll = nothing
            else
                soll = solve_symbolically(ss_equations[i],var_to_solve_for)
            end

            if isnothing(soll)
                continue
            end
            
            if isempty(soll) || soll == SPyPyC.Sym{PythonCall.Core.Py}[0] # take out variable if it is redundant from that equation only
                push!(Symbolics.var_redundant_list[i],var_to_solve_for)
                ss_equations[i] = replace_with_one(ss_equations[i], var_to_solve_for) # replace euler constant as it is not translated to julia properly
            end

        end
    end

end


function partial_solve(eqs_to_solve::Vector{E}, vars_to_solve::Vector{T}, incidence_matrix_subset; avoid_solve::Bool = false)::Tuple{Vector{T}, Vector{T}, Vector{E}, Vector{T}} where {E, T}
    for n in length(eqs_to_solve)-1:-1:2
        for eq_combo in combinations(1:length(eqs_to_solve), n)
            var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)

            var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0) 

            for var_combo in combinations(var_indices_to_select_from, n)
                remaining_vars_in_remaining_eqs = setdiff(var_indices_in_remaining_eqs, var_combo)
                # println("Solving for: ",vars_to_solve[var_combo]," in: ",eqs_to_solve[eq_combo])
                if length(remaining_vars_in_remaining_eqs) == length(eqs_to_solve) - n # not sure whether this condition needs to be there. could be because if the last remaining vars not solved for in the block is not present in the remaining block he will not be able to solve it for the same reasons he wasn't able to solve the unpartitioned block 
                    if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve[eq_combo]))) > 15
                        soll = nothing
                    else
                        soll = solve_symbolically(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                    end
                    
                    if !(isnothing(soll) || isempty(soll))
                        soll_collected = collect(values(soll))
                        
                        return (vars_to_solve[setdiff(1:length(eqs_to_solve),var_combo)],
                                vars_to_solve[var_combo],
                                eqs_to_solve[setdiff(1:length(eqs_to_solve),eq_combo)],
                                soll_collected)
                    end
                end
            end
        end
    end
    
    return (T[], T[], E[], T[])
end


function make_equation_robust_to_domain_errors(eqs,#::Vector{Union{Symbol,Expr}}, 
                                                vars_to_exclude::Vector{Vector{Symbol}}, 
                                                bounds::Dict{Symbol,Tuple{Float64,Float64}}, 
                                                ➕_vars::Vector{Symbol}, 
                                                unique_➕_eqs,#::Dict{Union{Expr,Symbol},Symbol}();
                                                precompile::Bool = false)
    ss_and_aux_equations = Expr[]
    ss_and_aux_equations_dep = Expr[]
    ss_and_aux_equations_error = Expr[]
    ss_and_aux_equations_error_dep = Expr[]
    rewritten_eqs = Union{Expr,Symbol}[]
    # write down ss equations including nonnegativity auxiliary variables
    # find nonegative variables, parameters, or terms
    for eq in eqs
        if eq isa Symbol
            push!(rewritten_eqs, eq)
        elseif eq isa Expr
            rewritten_eq = postwalk(x -> 
                x isa Expr ? 
                    # x.head == :(=) ? 
                    #     Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    #         x.head == :ref ?
                    #             occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 : # set shocks to zero and remove time scripts
                    #     x : 
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                Expr(:call, :*, x.args[3:end]..., x.args[2]) : # 2beta => beta * 2 
                            x :
                        x.args[1] ∈ [:^] ?
                            !(x.args[3] isa Int) ?
                                x.args[2] isa Symbol ? # nonnegative parameters 
                                    x.args[2] ∈ vars_to_exclude[1] ?
                                        begin
                                            bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                            x 
                                        end :
                                    begin
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if x.args[2] in vars_to_exclude[1]
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                        
                                        :($(replacement) ^ $(x.args[3]))
                                    end :
                                x.args[2] isa Float64 ?
                                    x :
                                x.args[2].head == :call ? # nonnegative expressions
                                    begin
                                        if precompile
                                            replacement = x.args[2]
                                        else
                                            replacement = simplify(x.args[2])
                                        end

                                        if !(replacement isa Int) # check if the nonnegative term is just a constant
                                            if haskey(unique_➕_eqs, x.args[2])
                                                replacement = unique_➕_eqs[x.args[2]]
                                            else
                                                if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                    push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                    push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                                else
                                                    push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                    push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                                end

                                                bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                                push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                                replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                                unique_➕_eqs[x.args[2]] = replacement
                                            end
                                        end

                                        :($(replacement) ^ $(x.args[3]))
                                    end :
                                x :
                            x :
                        x.args[2] isa Float64 ?
                            x :
                        x.args[1] ∈ [:log] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1e12)) : (eps(), 1e12)
                                        x 
                                    end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1e12,max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1e12)) : (eps(), 1e12)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:norminvcdf, :norminv, :qnorm] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 1-eps())) : (eps(), 1 - eps())
                                    x 
                                end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end

                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1 - eps())) : (eps(), 1 - eps())
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(1-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end

                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 1 - eps())) : (eps(), 1 - eps())
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:exp] ?
                            x.args[2] isa Symbol ? # have exp terms bound so they dont go to Inf
                                x.args[2] ∈ vars_to_exclude[1] ?
                                begin
                                    bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], -1e12), min(bounds[x.args[2]][2], 600)) : (-1e12, 600)
                                    x 
                                end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2]))))) 
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 600)) : (-1e12, 600)
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # have exp terms bound so they dont go to Inf
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(600,max(-1e12,$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end
                                            
                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], -1e12), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 600)) : (-1e12, 600)
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x.args[1] ∈ [:erfcinv] ?
                            x.args[2] isa Symbol ? # nonnegative parameters 
                                x.args[2] ∈ vars_to_exclude[1] ?
                                    begin
                                        bounds[x.args[2]] = haskey(bounds, x.args[2]) ? (max(bounds[x.args[2]][1], eps()), min(bounds[x.args[2]][2], 2 - eps())) : (eps(), 2 - eps())
                                        x 
                                    end :
                                begin
                                    if haskey(unique_➕_eqs, x.args[2])
                                        replacement = unique_➕_eqs[x.args[2]]
                                    else
                                        if x.args[2] in vars_to_exclude[1]
                                            push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        else
                                            push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                            push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                        end
                                        
                                        bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 2 - eps())) : (eps(), 2 - eps())
                                        push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                        replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                        unique_➕_eqs[x.args[2]] = replacement
                                    end
                                
                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x.args[2].head == :call ? # nonnegative expressions
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int) # check if the nonnegative term is just a constant
                                        if haskey(unique_➕_eqs, x.args[2])
                                            replacement = unique_➕_eqs[x.args[2]]
                                        else
                                            if isempty(intersect(get_symbols(x.args[2]), vars_to_exclude[1]))
                                                push!(ss_and_aux_equations, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            else
                                                push!(ss_and_aux_equations_dep, :($(Symbol("➕" * sub(string(length(➕_vars)+1)))) = min(2-eps(),max(eps(),$(x.args[2])))))
                                                push!(ss_and_aux_equations_error_dep, Expr(:call,:abs, Expr(:call,:-, :($(Symbol("➕" * sub(string(length(➕_vars)+1))))), x.args[2])))
                                            end
                                            
                                            bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))] = haskey(bounds, Symbol("➕" * sub(string(length(➕_vars)+1)))) ? (max(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][1], eps()), min(bounds[Symbol("➕" * sub(string(length(➕_vars)+1)))][2], 2 - eps())) : (eps(), 2 - eps())
                                            push!(➕_vars,Symbol("➕" * sub(string(length(➕_vars)+1))))
                                            replacement = Symbol("➕" * sub(string(length(➕_vars))))

                                            unique_➕_eqs[x.args[2]] = replacement
                                        end
                                    end

                                    :($(Expr(:call, x.args[1], replacement)))
                                end :
                            x :
                        x :
                    x :
                x,
            eq)
            push!(rewritten_eqs,rewritten_eq)
        else
            @assert typeof(eq) in [Symbol, Expr]
        end
    end

    vars_to_exclude_from_block = vcat(vars_to_exclude...)

    found_new_dependecy = true

    while found_new_dependecy
        found_new_dependecy = false

        for ssauxdep in ss_and_aux_equations_dep
            push!(vars_to_exclude_from_block, ssauxdep.args[1])
        end

        for (iii, ssaux) in enumerate(ss_and_aux_equations)
            if !isempty(intersect(get_symbols(ssaux), vars_to_exclude_from_block))
                found_new_dependecy = true
                push!(vars_to_exclude_from_block, ssaux.args[1])
                push!(ss_and_aux_equations_dep, ssaux)
                push!(ss_and_aux_equations_error_dep, ss_and_aux_equations_error[iii])
                deleteat!(ss_and_aux_equations, iii)
                deleteat!(ss_and_aux_equations_error, iii)
            end
        end
    end

    return rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep
end


"""
    solve_steady_state!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)

Solve steady state using symbolic methods when SymPy is available.
"""
function solve_steady_state!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)
    write_ss_check_function!(𝓂)

    unknowns = union(Symbolics.calibration_equations_parameters, Symbolics.vars_in_ss_equations)

    @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = spzeros(Int,length(unknowns),length(unknowns))

    eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list,
                                        Symbolics.ss_list),
                                    Symbolics.var_redundant_list),
                            Symbolics.par_list),
                    union.(Symbolics.ss_calib_list,
                            Symbolics.par_calib_list))

    for (i,u) in enumerate(unknowns)
        for (k,e) in enumerate(eq_list)
            incidence_matrix[i,k] = u ∈ e
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(incidence_matrix)
    R̂ = Int[]
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1] - 1]
    end
    push!(R̂,1)

    vars = hcat(P, R̂)'
    eqs = hcat(Q, R̂)'
    
    # @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations for: " * repr([collect(Symbol.(unknowns))[vars[1,eqs[1,:] .< 0]]...]) # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant equations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now." # repr([vcat(Symbolics.ss_equations,Symbolics.calibration_equations)[-eqs[1,eqs[1,:].<0]]...])
    
    n = n_blocks

    ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)# .|> SPyPyC.Sym
    # println(ss_equations)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    NSSS_solver_cache_init_tmp = []

    min_max_errors = []

    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    while n > 0 
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve_for = unknowns[vars[:,vars[2,:] .== n][1]]

            eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]

            # eliminate min/max from equations if solving for variables inside min/max. set to the variable we solve for automatically
            parsed_eq_to_solve_for = eq_to_solve |> string |> Meta.parse

            minmax_fixed_eqs = postwalk(x -> 
                x isa Expr ?
                    x.head == :call ? 
                        x.args[1] ∈ [:Max,:Min] ?
                            Symbol(var_to_solve_for) ∈ get_symbols(x.args[2]) ?
                                x.args[2] :
                            Symbol(var_to_solve_for) ∈ get_symbols(x.args[3]) ?
                                x.args[3] :
                            x :
                        x :
                    x :
                x,
            parsed_eq_to_solve_for)

            if parsed_eq_to_solve_for != minmax_fixed_eqs
                [push!(atoms_in_equations, a) for a in setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs))]
                push!(min_max_errors,:(solution_error += abs($parsed_eq_to_solve_for)))
                push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for min max terms in equations with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                eq_to_solve = eval(minmax_fixed_eqs)
            end
            
            if avoid_solve || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = solve_symbolically(eq_to_solve,var_to_solve_for)
            end

            if isnothing(soll) || isempty(soll)
                println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                
                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

                write_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                # write_domain_safe_block_solution!(𝓂, SS_solve_func, [var_to_solve_for], [eq_to_solve], relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
            elseif soll[1].is_number == true
                ss_equations = [replace_symbolic(eq, var_to_solve_for, soll[1]) for eq in ss_equations]
                
                push!(𝓂.solved_vars,Symbol(var_to_solve_for))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars) 
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = max(eps(),$(𝓂.solved_vals[end]))))
                else
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                end

                push!(atoms_in_equations_list,[])
            else
                push!(𝓂.solved_vars,Symbol(var_to_solve_for))
                push!(𝓂.solved_vals,Meta.parse(string(soll[1])))
                
                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)),Symbol.(soll[1].atoms()))))

                if (𝓂.solved_vars[end] ∈ 𝓂.➕_vars)
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vals[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2]))))
                    push!(SS_solve_func,:(solution_error += $(Expr(:call,:abs, Expr(:call, :-, 𝓂.solved_vars[end], 𝓂.solved_vals[end])))))
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical aux variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    
                    unique_➕_eqs[𝓂.solved_vals[end]] = 𝓂.solved_vars[end]
                else
                    vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.➕_vars), Symbol[]]
                    
                    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors([𝓂.solved_vals[end]], vars_to_exclude, 𝓂.bounds, 𝓂.➕_vars, unique_➕_eqs)
    
                    if length(vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)) > 0
                        push!(SS_solve_func,vcat(ss_and_aux_equations, ss_and_aux_equations_dep)...)
                        push!(SS_solve_func,:(solution_error += $(Expr(:call, :+, vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)...))))
                        push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for analytical variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                    end
                    
                    push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(rewritten_eqs[1])))
                end

                if haskey(𝓂.bounds, 𝓂.solved_vars[end]) && 𝓂.solved_vars[end] ∉ 𝓂.➕_vars
                    push!(SS_solve_func,:(solution_error += abs(min(max($(𝓂.bounds[𝓂.solved_vars[end]][1]), $(𝓂.solved_vars[end])), $(𝓂.bounds[𝓂.solved_vars[end]][2])) - $(𝓂.solved_vars[end]))))
                    push!(SS_solve_func, :(if solution_error > tol.NSSS_acceptance_tol if verbose println("Failed for bounded variables with error $solution_error") end; scale = scale * .3 + solved_scale * .7; continue end))
                end
            end
        else
            vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false
            
            if symbolic_SS
                if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = solve_symbolically(eqs_to_solve,vars_to_solve)
                end

                if isnothing(soll) || isempty(soll) || length(intersect((union(SPyPyC.free_symbols.(collect(values(soll)))...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end

                    numerical_sol = true
                else
                    if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end
                    
                    atoms = reduce(union,map(x->x.atoms(),collect(values(soll))))

                    for a in atoms push!(atoms_in_equations, Symbol(a)) end
                    
                    for vars in vars_to_solve
                        push!(𝓂.solved_vars,Symbol(vars))
                        push!(𝓂.solved_vals,Meta.parse(string(soll[vars]))) #using convert(Expr,x) leads to ugly expressions

                        push!(atoms_in_equations_list, Set(Symbol.(soll[vars].atoms())))
                        push!(SS_solve_func,:($(𝓂.solved_vars[end]) = $(𝓂.solved_vals[end])))
                    end
                end
            end

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

            incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eq_idx_in_block_to_solve]

            # try symbolically and use numerical if it does not work
            if numerical_sol || !symbolic_SS
                pv = sortperm(vars_to_solve, by = Symbol)
                pe = sortperm(eqs_to_solve, by = string)

                if length(pe) > 5
                    write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)
                    # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)
                else
                    solved_system = partial_solve(eqs_to_solve[pe], vars_to_solve[pv], incidence_matrix_subset[pv,pe], avoid_solve = avoid_solve)
                    
                    # if !isnothing(solved_system) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "LambertW")) && !any(contains.(string.(vcat(solved_system[3],solved_system[4])), "Heaviside")) 
                    #     write_reduced_block_solution!(𝓂, SS_solve_func, solved_system, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, 
                    #     𝓂.➕_vars, unique_➕_eqs)  
                    # else
                        write_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list)  
                        # write_domain_safe_block_solution!(𝓂, SS_solve_func, vars_to_solve, eqs_to_solve, relevant_pars_across, NSSS_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, unique_➕_eqs)  
                    # end
                end

                if !symbolic_SS && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
            end
        end
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp, fill(Inf, length(𝓂.parameters)))
    push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(unknowns)

    parameters_only_in_par_defs = Set()
    # add parameters from parameter definitions
    if length(𝓂.calibration_equations_no_var) > 0
		atoms = reduce(union, get_symbols.(𝓂.calibration_equations_no_var))
	    [push!(atoms_in_equations, a) for a in atoms]
	    [push!(parameters_only_in_par_defs, a) for a in atoms]
	end
    
    # 𝓂.par = union(𝓂.par,setdiff(parameters_only_in_par_defs,𝓂.parameters_as_function_of_parameters))
    
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.parameters) 
        if parss ∈ union(atoms_in_equations, relevant_pars_across)
            push!(parameters_in_equations, :($parss = parameters[$i]))
        end
    end
    
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies, 𝓂.solved_vars[i] => intersect(a, union(𝓂.var, 𝓂.parameters)))
    end

    push!(dependencies, :SS_relevant_calibration_parameters => intersect(reduce(union, atoms_in_equations_list), 𝓂.parameters))

    𝓂.SS_dependencies = dependencies
    

    
    dyn_exos = []
    for dex in union(𝓂.exo_past, 𝓂.exo_future)
        push!(dyn_exos,:($dex = 0))
    end

    push!(SS_solve_func,:($(dyn_exos...)))
    
    push!(SS_solve_func, min_max_errors...)
    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)] end))
    

    # push!(SS_solve_func,:(for pars in 𝓂.NSSS_solver_cache
    #                             latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
    #                             if latest <= current_best
    #                                 current_best = latest
    #                             end
    #                         end))
        push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol) && (scale == 1)
                                    reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
                            end))
    # push!(SS_solve_func,:(if length(𝓂.NSSS_solver_cache) > 100 popfirst!(𝓂.NSSS_solver_cache) end))
    
    # push!(SS_solve_func,:(SS_init_guess = ([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)])))

    # push!(SS_solve_func,:(𝓂.SS_init_guess = typeof(SS_init_guess) == Vector{Float64} ? SS_init_guess : ℱ.value.(SS_init_guess)))

    # push!(SS_solve_func,:(return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...]))))


    # fix parameter bounds
    par_bounds = []

    for varpar in intersect(𝓂.parameters,union(atoms_in_equations, relevant_pars_across))
        if haskey(𝓂.bounds, varpar)
            push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.bounds[varpar][1])),$(𝓂.bounds[varpar][2]))))
        end
    end

    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    # fail_fast_solvers_only::Bool, 
                                    tol::Tolerances,
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    initial_parameters_tmp = copy(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.NSSS_solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.NSSS_solver_cache[end]
                    
                    for pars in 𝓂.NSSS_solver_cache
                        copy!(initial_parameters_tmp, pars[end])

                        ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                        latest = sum(abs2,initial_parameters_tmp)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    # closest_solution = copy(closest_solution_init)
                    # solution_error = 1.0
                    # iters = 0
                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    # range_length = [ 1, 2, 4, 8,16,32,64,128,1024]
                    scale = 1.0

                    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
                    push!(NSSS_solver_cache_scale, closest_solution_init)
                    # fail_fast_solvers_only = true
                    while range_iters <= (cold_start ? 1 : 500) && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                        if abs(solved_scale - scale) < 1e-2
                            # println(NSSS_solver_cache_scale[end])
                            break 
                        end

                        # println("i: $range_iters - scale: $scale - solved_scale: $solved_scale")
                        # println(closest_solution[end])
                    # for range_ in range_length
                        # rangee = range(0,1,range_+1)
                        # for scale in rangee[2:end]
                            # scale = 6*scale^5 - 15*scale^4 + 10*scale^3 # smootherstep

                            # if scale <= solved_scale continue end

                            
                            current_best = sum(abs2,NSSS_solver_cache_scale[end][end] - initial_parameters)
                            closest_solution = NSSS_solver_cache_scale[end]

                            for pars in NSSS_solver_cache_scale
                                copy!(initial_parameters_tmp, pars[end])
                                
                                ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                                latest = sum(abs2,initial_parameters_tmp)

                                if latest <= current_best
                                    current_best = latest
                                    closest_solution = pars
                                end
                            end

                            # println(closest_solution)

                            if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                                parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                            else
                                parameters = copy(initial_parameters)
                            end
                            params_flt = parameters
                            
                            # println(parameters)

                            $(parameters_in_equations...)
                            $(par_bounds...)
                            $(𝓂.calibration_equations_no_var...)
                            NSSS_solver_cache_tmp = []
                            solution_error = 0.0
                            iters = 0
                            $(SS_solve_func...)

                            if solution_error < tol.NSSS_acceptance_tol
                                # println("solved for $scale; $range_iters")
                                solved_scale = scale
                                if scale == 1
                                    # return ComponentVector([$(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))...), $(𝓂.calibration_equations_parameters...)], Axis([sort(union(𝓂.exo_present,𝓂.var))...,𝓂.calibration_equations_parameters...])), solution_error
                                    # NSSS_solution = [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)]
                                    # NSSS_solution[abs.(NSSS_solution) .< 1e-12] .= 0 # doesn't work with Zygote
                                    return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                                else
                                    reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                                end

                                if scale > .95
                                    scale = 1
                                else
                                    # scale = (scale + 1) / 2
                                    scale = scale * .4 + .6
                                end
                            # else
                            #     println("no sol")
                            #     scale  = (scale + solved_scale) / 2
                            #     println("scale $scale")
                            # elseif scale == 1 && range_ == range_length[end]
                            #     return [$(Symbol.(replace.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))...), $(𝓂.calibration_equations_parameters...)], (solution_error, iters)
                            end
                    #     end
                    end
                    return zeros($(length(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)) + length(𝓂.calibration_equations_parameters))), (1, 0)
                end)


    𝓂.SS_solve_func = @RuntimeGeneratedFunction(solve_exp)
    # 𝓂.SS_solve_func = eval(solve_exp)

    return nothing
end

# Override the default numerical setup to provide symbolic solving capability
"""
    setup_steady_state_numerical!(𝓂::ℳ, verbose::Bool, silent::Bool)

SymPy extension override that handles both symbolic and numerical steady state setup.
- If precompile is false: Uses symbolic solving with SymPy
- If precompile is true: Falls back to numerical solving (calls the default implementation)
"""
function setup_steady_state_numerical!(𝓂::ℳ, verbose::Bool, silent::Bool)
    # Check if we should use symbolic solving
    if !𝓂.precompile
        # Use symbolic solving via SymPy
        start_time = time()

        if !silent print("Remove redundant variables in non-stochastic steady state problem:\t") end

        symbolics = create_symbols_eqs!(𝓂)

        remove_redundant_SS_vars!(𝓂, symbolics, avoid_solve = !𝓂.simplify)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end

        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        solve_steady_state!(𝓂, false, symbolics, verbose = verbose, avoid_solve = !𝓂.simplify)

        𝓂.obc_violation_equations = write_obc_violation_equations(𝓂)
        
        set_up_obc_violation_function!(𝓂)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    else
        # Fallback to numerical solving when precompile is true
        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        solve_steady_state!(𝓂, verbose = verbose)

        𝓂.obc_violation_equations = write_obc_violation_equations(𝓂)

        set_up_obc_violation_function!(𝓂)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    end
end

end # @stable


end # module
