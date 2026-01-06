module SymPyPythonCallExt

import MacroModelling
import MacroModelling: ℳ, SYMPYWORKSPACE_RESERVED_NAMES, _sympy_available, get_symbols, count_ops, convert_to_ss_equation, transform_expression, reverse_transformation, write_ss_check_function!, write_block_solution!, write_SS_solve_func!, postwalk

import SymPyPythonCall as SPyPyC
import PythonCall
import SpecialFunctions: erfcinv, erfc
import SparseArrays: spzeros
import Combinatorics: combinations
import BlockTriangularForm

using DispatchDoctor

# Set the flag to indicate SymPy is available
function __init__()
    _sympy_available[] = true
end

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
    
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant equations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now."
    
    n = n_blocks

    ss_equations = vcat(Symbolics.ss_equations,Symbolics.calibration_equations)

    SS_solve_func = []

    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    NSSS_solver_cache_init_tmp = []

    while n > 0
        vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        # try symbolically and use numerical if it does not work
        if length(vars_to_solve) == 1
            eq_to_solve = eqs_to_solve[1]
            var_to_solve_for = vars_to_solve[1]

            if avoid_solve || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = solve_symbolically(eq_to_solve,var_to_solve_for)
            end

            if isnothing(soll) || isempty(soll)
                if verbose println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve) end
                
                eq_idx_in_block_to_solve = Int[]
            else
                solved = Meta.parse(string(soll[1]))

                push!(𝓂.solved_vars, Symbol(var_to_solve_for))
                push!(𝓂.solved_vals, solved isa Expr ? postwalk(x -> x isa Float64 ? round(x, digits = 30) : x, solved) : solved)

                eq_idx_in_block_to_solve = Int[1]
            end
        else
            soll = nothing

            if !avoid_solve
                for combo in 1:length(vars_to_solve)
                    for var_combo in combinations(1:length(vars_to_solve),combo)
                        for eq_combo in combinations(1:length(vars_to_solve),combo)
                            if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve[eq_combo]))) > 15
                                soll = nothing
                            else
                                soll = solve_symbolically(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                            end
                            
                            if !(isnothing(soll) || isempty(soll))
                                soll_collected = collect(values(soll))
                                
                                if length(intersect((union(SPyPyC.free_symbols.(soll_collected)...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) == 0

                                    for (s,solved) in soll
                                        ssoll = Meta.parse(string(solved))
                                        push!(𝓂.solved_vars, Symbol(s))
                                        push!(𝓂.solved_vals, ssoll isa Expr ? postwalk(x -> x isa Float64 ? round(x, digits = 30) : x, ssoll) : ssoll)
                                    end
                                    
                                    break
                                else
                                    soll = nothing
                                end
                            end
                        end

                        if !(isnothing(soll) || isempty(soll)) break end
                    end
                    if !(isnothing(soll) || isempty(soll)) break end
                end
            end

            eq_idx_in_block_to_solve = Int[]

            if isnothing(soll) || isempty(soll)
                if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end
            else
                solved_symbolically = SPyPyC.:↓.(vars_to_solve) .∈ Ref(SPyPyC.:↓.(collect(keys(soll))))
                
                if !all(solved_symbolically)
                    eq_idx_in_block_to_solve = (1:length(eqs_to_solve))[.!solved_symbolically]
                end
            end
        end

        if length(eq_idx_in_block_to_solve) > 0 || (isnothing(soll) || isempty(soll))
            # Use the numerical solver path
            write_block_solution!(𝓂, 
                                    SS_solve_func, 
                                    length(eq_idx_in_block_to_solve) > 0 ? Symbol.(vars_to_solve[eq_idx_in_block_to_solve]) : Symbol.(vars_to_solve), 
                                    length(eq_idx_in_block_to_solve) > 0 ? Meta.parse.(string.(eqs_to_solve[eq_idx_in_block_to_solve])) : Meta.parse.(string.(eqs_to_solve)), 
                                    relevant_pars_across,
                                    NSSS_solver_cache_init_tmp, 
                                    eq_idx_in_block_to_solve, 
                                    atoms_in_equations_list)
        end

        n -= 1
    end

    # Write the final SS solve function
    write_SS_solve_func!(𝓂, SS_solve_func, relevant_pars_across, NSSS_solver_cache_init_tmp)

    return nothing
end

end # @stable

end # module
