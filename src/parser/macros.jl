# ── Macro helper functions (moved from MacroModelling.jl) ──

function evaluate_conditions(cond)
    if cond isa Bool
        return cond
    elseif cond isa Expr && cond.head == :call 
        a, b = cond.args[2], cond.args[3]

        if typeof(a) ∉ [Symbol, Number]
            a = eval(a)
        end

        if typeof(b) ∉ [Symbol, Number]
            b = eval(b)
        end
        
        if cond.args[1] == :(==)
            return a == b
        elseif cond.args[1] == :(!=)
            return a != b
        elseif cond.args[1] == :(<)
            return a < b
        elseif cond.args[1] == :(<=)
            return a <= b
        elseif cond.args[1] == :(>)
            return a > b
        elseif cond.args[1] == :(>=)
            return a >= b
        end
        # end
    end
    return nothing
end

function resolve_if_expr(ex::Expr)
    prewalk(ex) do node
        if node isa Expr && (node.head === :if || node.head === :elseif)
            cond = node.args[1]
            then_blk = node.args[2]
            if length(node.args) == 3
                else_blk = node.args[3]
            end
            val = evaluate_conditions(unblock(cond))

            if val === true
                # recurse into the selected branch
                return resolve_if_expr(unblock(then_blk))
            elseif val === false && length(node.args) == 3
                return resolve_if_expr(unblock(else_blk))
            elseif val === false && length(node.args) == 2
                return nothing
            elseif val === false && node.head === :elseif
                return resolve_if_expr(unblock(else_blk))
            end
        end
        return node
    end
end

function match_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) !== nothing, strings)
end

function contains_equation(expr)
    found = false
    postwalk(expr) do x
        if x isa Expr && x.head == :(=)
            found = true
        end
        return x
    end
    return found
end

# function remove_nothing(ex::Expr)
#     postwalk(ex) do node
#         # Only consider call-nodes with exactly two arguments
#         if node isa Expr && node.head === :call && length(node.args) == 3
#             fn, lhs, rhs = node.args
#             lhs2 = unblock(lhs)
#             rhs2 = unblock(rhs)

#             if rhs2 === :(nothing)
#                 # strip the call and recurse to clean deeper
#                 return remove_nothing(lhs2)
#             elseif lhs2 === :(nothing)
#                 return remove_nothing(rhs2)
#             # else
#             #     return remove_nothing(node.args)
#             end
#         end
#         return node
#     end
# end

function remove_nothing(ex::Expr)
    postwalk(ex) do node
        # Only consider call-expressions
        if node isa Expr && node.head === :call && any(node.args .=== nothing)
            fn = node.args[1]
            # Unblock and collect all the operands
            # raw_args = map(arg -> unblock(arg), node.args[2:end])
            # Drop any nothing
            kept = filter(arg -> !(unblock(arg) === nothing), node.args[2:end])
            if isempty(kept)
                return nothing
            elseif length(kept) == 1
                return kept[1]
            else
            # elseif length(kept) < length(raw_args)
                return Expr(:call, fn, kept...)
            # else
            #     return node
            end
        end
        return node
    end
end

function replace_indices_inside_for_loop(exxpr,index_variable,indices,concatenate, operator)
    @assert operator ∈ [:+,:*] "Only :+ and :* allowed as operators in for loops."
    calls = []
    indices = indices.args[1] == :(:) ? eval(indices) : [indices.args...]
    for idx in indices
        push!(calls, postwalk(x -> begin
            x isa Expr ?
                x.head == :ref ?
                    @capture(x, name_{index_}[time_]) ?
                        index == index_variable ?
                            :($(Expr(:ref, Symbol(string(name) * "{" * string(idx) * "}"),time))) :
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            x :
                        x :
                    @capture(x, name_[time_]) ?
                        time isa Expr || time isa Symbol ?
                            index_variable ∈ get_symbols(time) ?
                                :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx))))) :
                            # occursin("{" * string(index_variable) * "}", string(name)) ?
                            #     Expr(:ref, Symbol(replace(string(name), "{" * string(index_variable) * "}" => "◖" * string(idx) * "◗")), time) :
                            x :
                        # occursin("{" * string(index_variable) * "}", string(name)) ?
                        #     Expr(:ref, Symbol(replace(string(name), "{" * string(index_variable) * "}" => "◖" * string(idx) * "◗")), time) :
                        x :
                    x :
                x.head == :if ?
                    length(x.args) > 2 ?
                        Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                    replace_indices_inside_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock,
                                    replace_indices_inside_for_loop(x.args[3],index_variable,:([$idx]),false,:+) |> unblock) :
                    Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                replace_indices_inside_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock) :
                @capture(x, name_{index_}) ?
                    index == index_variable ?
                        :($(Symbol(string(name) * "{" * string(idx) * "}"))) :
                    x :
                x :
            @capture(x, name_) ?
                name == index_variable && idx isa Int ?
                    :($idx) :
                x isa Symbol ?
                    occursin("{" * string(index_variable) * "}", string(x)) ?
                Symbol(replace(string(x),  "{" * string(index_variable) * "}" => "{" * string(idx) * "}")) :
                    x :
                x :
            x
        end,
        exxpr))
    end
    
    if concatenate
        return :($(Expr(:call, operator, calls...)))
    else
        return :($(Expr(:block, calls...)))
        # return :($calls...)
        # return calls
    end
end

function write_out_for_loops(arg::Expr)::Expr
    postwalk(x -> begin
                    x = flatten(unblock(x))
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    x.args[1].head == :block ?
                                        # begin println("here"); 
                                        [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] : # end :
                                    # begin println("here2"); 
                                    [replace_indices_inside_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] : # end :
                                x :
                            x.args[2].head ∉ [:(=), :block] ?
                                x.args[1].head == :block ?
                                    # begin println("here3"); 
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    true,
                                                    x.args[1].args[1].args[2].value) : # end : # for loop part of equation
                                x.args[2].head == :if ?
                                    contains_equation(x.args[2]) ?
                                        # begin println("here5"); println(x)
                                        replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                            Symbol(x.args[1].args[1]), 
                                                            (x.args[1].args[2]),
                                                            false,
                                                            :+) : # end : # for loop part of equation
                                    # begin println("here6"); println(x)
                                    replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        (x.args[1].args[2]),
                                                        true,
                                                        :+) : # end : # for loop part of equation
                                # begin println("here4"); println(x)
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    (x.args[1].args[2]),
                                                    true,
                                                    :+) : # end : # for loop part of equation
                            x.args[1].head == :block ?
                                # begin println("here5"); 
                                replace_indices_inside_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    false,
                                                    x.args[1].args[1].args[2].value) : # end :
                                                # end 
                                                # : # for loop part of equation
                            # begin println(x); 
                            # begin println("here7"); println(x)
                            replace_indices_inside_for_loop(unblock(x.args[2]), 
                                            Symbol(x.args[1].args[1]), 
                                            (x.args[1].args[2]),
                                            false,
                                            :+) : # end :
                                            # println(out); 
                                            # return out end 
                                            # :
                        x :
                    x
                end,
    arg) #|> unblock |> flatten
end

# function parse_for_loops(equations_block)
#     eqs = Expr[]  # Initialize an empty array to collect expressions

#     # Define a helper recursive function
#     function recurse(arg)
#             if arg isa Expr
#                 if arg.head == :block
#                     for b in arg.args
#                         if b isa Expr
#                             # If the result is an Expr, process and add to eqs
#                             push!(eqs, unblock(replace_indices(b)))
#                         elseif b isa Array
#                             recurse(b)
#                         end
#                     end
#                 end
#             elseif arg isa Array
#                 # If the result is an Array, iterate and recurse
#                 for B in arg
#                     println((B))
#                     recurse(B)
#                 end
#             end
#     end

#     for arg in equations_block.args
#         if isa(arg,Expr)
#             parsed_eqs = write_out_for_loops(arg)
#             recurse(parsed_eqs)
#         end
#     end

#     # Return the collected expressions as a block
#     return Expr(:block, eqs...)
# end


function parse_for_loops(equations_block)::Expr
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = write_out_for_loops(arg)
            # println(parsed_eqs)
            if parsed_eqs isa Expr
                push!(eqs,unblock(replace_indices(parsed_eqs)))
            elseif parsed_eqs isa Array
                for B in parsed_eqs
                    if B isa Array
                        for b in B
                            push!(eqs,unblock(replace_indices(b)))
                        end
                    elseif B isa Expr
                        if B.head == :block
                            for b in B.args
                                if b isa Expr
                                    push!(eqs,replace_indices(b))
                                end
                            end
                        else
                            push!(eqs,unblock(replace_indices(B)))
                        end
                    else
                        push!(eqs,unblock(replace_indices(B)))
                    end
                end
            end

        end
    end
    return Expr(:block,eqs...) |> flatten
end

function decompose_name(name::Symbol)
    name = string(name)
    matches = eachmatch(r"◖([\p{L}\p{N}]+)◗|([\p{L}\p{N}]+[^◖◗]*)", name)

    result = []
    nested = []

    for m in matches
        if m.captures[1] !== nothing
            push!(nested, m.captures[1])
        else
            if !isempty(nested)
                push!(result, Symbol.(nested))
                nested = []
            end
            push!(result, Symbol(m.captures[2]))
        end
    end

    if !isempty(nested)
        push!(result, (nested))
    end

    return result
end

function get_possible_indices_for_name(name::Symbol, all_names::Vector{Symbol})
    indices = filter(x -> length(x) < 3 && x[1] == name, decompose_name.(all_names))

    indexset = []

    for i in indices
        if length(i) > 1
            push!(indexset, Symbol.(i[2])...)
        end
    end

    return indexset
end

function expand_calibration_equations(calibration_equation_parameters::Vector{Symbol}, calibration_equations::Vector{Expr}, ss_calib_list::Vector, par_calib_list::Vector, all_names::Vector{Symbol})
    expanded_parameters = Symbol[]
    expanded_equations = Expr[]
    expanded_ss_var_list = []
    expanded_par_var_list = []

    for (u,par) in enumerate(calibration_equation_parameters)
        indices_in_calibration_equation = Set()
        indexed_names = []
        for i in get_symbols(calibration_equations[u])
            indices = get_possible_indices_for_name(i, all_names)
            if indices != Any[]
                push!(indices_in_calibration_equation, indices)
                push!(indexed_names,i)
            end
        end

        par_indices = get_possible_indices_for_name(par, all_names)
        
        if length(par_indices) > 0
            push!(indices_in_calibration_equation, par_indices)
        end
        
        @assert length(indices_in_calibration_equation) <= 1 "Calibration equations cannot have more than one index in the equations or for the parameter."
        
        if length(indices_in_calibration_equation) == 0
            push!(expanded_parameters,par)
            push!(expanded_equations,calibration_equations[u])
            push!(expanded_ss_var_list,ss_calib_list[u])
            push!(expanded_par_var_list,par_calib_list[u])
        else
            for i in collect(indices_in_calibration_equation)[1]
                expanded_ss_var = Set()
                expanded_par_var = Set()
                push!(expanded_parameters, Symbol(string(par) * "◖" * string(i) * "◗"))
                push!(expanded_equations, postwalk(x -> x ∈ indexed_names ? Symbol(string(x) * "◖" * string(i) * "◗") : x, calibration_equations[u]))
                for ss in ss_calib_list[u]
                    if ss ∈ indexed_names
                        push!(expanded_ss_var,Symbol(string(ss) * "◖" * string(i) * "◗"))
                    else
                        push!(expanded_ss_var,ss)
                    end
                end
                # Handle parameters from par_calib_list - expand indexed ones, keep non-indexed
                for p in par_calib_list[u]
                    if p ∈ indexed_names
                        push!(expanded_par_var, Symbol(string(p) * "◖" * string(i) * "◗"))
                    else
                        push!(expanded_par_var, p)
                    end
                end
                push!(expanded_ss_var_list, expanded_ss_var)
                push!(expanded_par_var_list, expanded_par_var)
            end
        end
    end

    return expanded_parameters, expanded_equations, expanded_ss_var_list, expanded_par_var_list
end

function expand_indices(compressed_inputs::Vector{Symbol}, compressed_values::Vector{T}, expanded_list::Vector{Symbol}) where T
    expanded_inputs = Symbol[]
    expanded_values = T[]

    for (i,par) in enumerate(compressed_inputs)
        par_idx = findall(x -> string(par) == x, first.(split.(string.(expanded_list ), "◖")))

        if length(par_idx) > 1
            for idx in par_idx
                push!(expanded_inputs, expanded_list[idx])
                push!(expanded_values, compressed_values[i])
            end
        else#if par ∈ expanded_list ## breaks parameters defined in parameter block
            push!(expanded_inputs, par)
            push!(expanded_values, compressed_values[i])
        end
    end
    return expanded_inputs, expanded_values
end


const all_available_algorithms = [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order]


"""
$(SIGNATURES)
Parses the model equations and assigns them to an object.

# Arguments
- `𝓂`: name of the object to be created containing the model information.
- `ex`: equations

# Optional arguments to be placed between `𝓂` and `ex`
- `max_obc_horizon` [Default: `40`, Type: `Int`]: maximum length of anticipated shocks and corresponding unconditional forecast horizon over which the occasionally binding constraint is to be enforced. Increase this number if no solution is found to enforce the constraint.

Variables must be defined with their time subscript in square brackets.
Endogenous variables can have the following:
- present: `c[0]`
- non-stochastic steady state: `c[ss]` instead of `ss` any of the following is also a valid flag for the non-stochastic steady state: `ss`, `stst`, `steady`, `steadystate`, `steady_state`, and the parser is case-insensitive (`SS` or `sTst` will work as well).
- past: `c[-1]` or any negative Integer: e.g. `c[-12]`
- future: `c[1]` or any positive Integer: e.g. `c[16]` or `c[+16]`
Signed integers are recognised and parsed as such.

Exogenous variables (shocks) can have the following:
- present: `eps_z[x]` instead of `x` any of the following is also a valid flag for exogenous variables: `ex`, `exo`, `exogenous`, and the parser is case-insensitive (`Ex` or `exoGenous` will work as well).
- past: `eps_z[x-1]`
- future: `eps_z[x+1]`

Parameters enter the equations without square brackets.

If an equation contains a `max` or `min` operator, the default dynamic (first order) solution of the model will enforce the occasionally binding constraint. This enforcement can be disabled by setting `ignore_obc = true` in the relevant function calls.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end
```

# Programmatic model writing

Parameters and variables can be indexed using curly braces: e.g. `c{H}[0]`, `eps_z{F}[x]`, or `α{H}`.

`for` loops can be used to write models programmatically. They can either be used to generate expressions where the time index or the index in curly braces is iterated over:
- generate equation with different indices in curly braces: `for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end`
- generate multiple equations with different indices in curly braces: `for co in [H, F] K{co}[0] = (1-delta{co}) * K{co}[-1] + S{co}[0] end`
- generate equation with different time indices: `Y_annual[0] = for lag in -3:0 Y[lag] end` or `R_annual[0] = for operator = :*, lag in -3:0 R[lag] end`

# Returns
- `Nothing`. The macro creates the model `𝓂` in the calling scope.
"""
macro model(𝓂, ex...)
    # parse options
    verbose = false
    precompile = false
    max_obc_horizon = 40

    for exp in ex[1:end-1]
        postwalk(x ->
            x isa Expr ?
                x.head == :(=) ?
                    x.args[1] == :verbose && x.args[2] isa Bool ?
                        verbose = x.args[2] :
                    x.args[1] == :precompile && x.args[2] isa Bool ?
                        precompile = x.args[2] :
                    x.args[1] == :max_obc_horizon && x.args[2] isa Int ?
                        max_obc_horizon = x.args[2] :
                    begin
                        @warn "Invalid option `$(x.args[1])` ignored. See docs: `?@model` for valid options."
                        x
                    end :
                x :
            x,
        exp)
    end

    model_name = string(𝓂)
    model_block = ex[end]

    # Heavy lifting is delegated to `process_model_equations` in
    # src/parser/equation_processing.jl, which is also used by the
    # equation-modification reprocess pipeline. Keeping a single source of
    # truth avoids drift between the two callers.
    return quote
        local _T, _eqs, _ℂ, _𝓦 = MacroModelling.process_model_equations(
            $(QuoteNode(model_block)),
            $max_obc_horizon,
            $precompile,
        )

        global $𝓂 = ℳ(
            $model_name,
            Float64[],            # parameter_values, populated by @parameters
            _eqs,
            caches(
                valid_for_caches(),
                zeros(0,0), # jacobian
                zeros(0,0), # jacobian_parameters
                zeros(0,0), # jacobian_SS_and_pars
                zeros(0,0), # hessian
                zeros(0,0), # hessian_parameters
                zeros(0,0), # hessian_SS_and_pars
                zeros(0,0), # third_order_derivatives
                zeros(0,0), # third_order_derivatives_parameters
                zeros(0,0), # third_order_derivatives_SS_and_pars
                zeros(0,0), # first_order_solution_matrix
                zeros(0,0), # first_order_obc_solution_matrix
                zeros(0,0), # qme_solution
                Float64[],  # second_order_stochastic_steady_state
                SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # second_order_solution
                Float64[],  # pruned_second_order_stochastic_steady_state
                Float64[],  # third_order_stochastic_steady_state
                SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # third_order_solution
                Float64[],  # pruned_third_order_stochastic_steady_state
                Float64[],  # non_stochastic_steady_state
                CircularBuffer{Vector{Vector{Float64}}}(500),  # solver
                zeros(0,0), # NSSS_∂equations_∂parameters
                zeros(0,0), # NSSS_∂equations_∂SS_and_pars
                zeros(0,0), # covariance_first_order
                zeros(0,0), # covariance_second_order
                zeros(0,0), # covariance_third_order
                zeros(0,0), # covariance_third_order_autocorr
            ),
            _ℂ,
            _𝓦,
            model_functions(
                x->x, # NSSS_check_func
                nothing, # NSSS_custom_function
                x->x, # NSSS_∂equations_∂parameters_func
                x->x, # NSSS_∂equations_∂SS_and_pars_func
                NSSSSolverFunctions(),
                nothing, # nsss_param_prep!
                jacobian_functions(x->x, x->x, x->x),
                hessian_functions(x->x, x->x, x->x),
                third_order_derivatives_functions(x->x, x->x, x->x),
                x->x, # obc_violation
                Tuple{Int,Int,Float64}[], # obc_constraint_info
                false, # functions_written
            ),
            SolveCounters(),
            RevisionEntry[],
        );
    end
end






"""
$(SIGNATURES)
Adds parameter values and calibration equations to the previously defined model. Allows to provide an initial guess for the non-stochastic steady state (NSSS).

# Arguments
- `𝓂`: name of the object previously created containing the model information.
- `ex`: parameter, parameters values, and calibration equations

Parameters can be defined in either of the following ways:
- plain number: `δ = 0.02`
- expression containing numbers: `δ = 1/50`
- expression containing other parameters: `δ = 2 * std_z` in this case it is irrelevant if `std_z` is defined before or after. The definitions including other parameters are treated as a system of equations and solved accordingly.
- expressions containing a target parameter and an equations with endogenous variables in the non-stochastic steady state, and other parameters, or numbers: `k[ss] / (4 * q[ss]) = 1.5 | δ` or `α | 4 * q[ss] = δ * k[ss]` in this case the target parameter will be solved simultaneously with the non-stochastic steady state using the equation defined with it.

# Optional arguments to be placed between `𝓂` and `ex`
- `guess` [Type: `Dict{Symbol, <:Real}` or `Dict{String, <:Real}`]: Guess for the non-stochastic steady state. The keys must be variable (and calibrated parameter) names and the values the guesses. Missing values are filled with standard starting values.
- $STEADY_STATE_FUNCTION®
- `verbose` [Default: `false`, Type: `Bool`]: print more information about how the non-stochastic steady state is solved
- `silent` [Default: `false`, Type: `Bool`]: do not print any information
- `ss_symbolic_mode` [Default: `:single_equation`, Type: `Symbol`]: controls symbolic steps in non-stochastic steady state (NSSS) setup. Use `:none` for numerical-only setup, `:single_equation` to allow symbolic solves only for single-equation blocks, or `:full` to allow symbolic solves for both single- and multi-equation blocks.
- `perturbation_order` [Default: `1`, Type: `Int`]: take derivatives only up to the specified order at this stage. When working with higher order perturbation later on, respective derivatives will be taken at that stage.
- `ss_solver_parameters_algorithm` [Default: `:ESCH`, Type: `Symbol`]: global optimization routine used when searching for steady-state solver parameters after an initial failure; choose `:ESCH` (evolutionary) or `:SAMIN` (simulated annealing). `:SAMIN` is available only when Optim.jl is loaded.
- `ss_solver_parameters_maxtime` [Default: `120.0`, Type: `Real`]: time budget in seconds for the steady-state solver parameter search when `ss_solver_parameters_algorithm` is invoked

# Delayed parameter definition
Not all parameters need to be defined in the `@parameters` macro. Calibration equations using the `|` syntax and parameters defined as functions of other parameters must be declared here, but simple parameter value assignments (e.g., `α = 0.5`) can be deferred and provided later by passing them to any function that accepts the `parameters` argument (e.g., [`get_irf`](@ref), [`get_steady_state`](@ref), [`simulate`](@ref)). 

**Parameter ordering:** When some parameters are not defined in `@parameters`, the final parameter vector follows a specific order: first come the parameters defined in `@parameters` (in their declaration order), followed by any missing parameters (in alphabetical order). This ordering is important when passing parameter values by position rather than by name in subsequent function calls.

# Examples
```julia
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC verbose = true begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

@model RBC_calibrated begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC_calibrated verbose = true guess = Dict(:k => 3) begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end
```

# Programmatic model writing
Variables and parameters indexed with curly braces can be either referenced specifically (e.g. `c{H}[ss]`) or generally (e.g. `alpha`). If they are referenced generally the parse assumes all instances (indices) are meant. For example, in a model where `alpha` has two indices `H` and `F`, the expression `alpha = 0.3` is interpreted as two expressions: `alpha{H} = 0.3` and `alpha{F} = 0.3`. The same goes for calibration equations.

# Returns
- `Nothing`. The macro assigns parameter values and calibration equations to `𝓂` in the calling scope.
"""
macro parameters(𝓂, ex...)
    # parse options
    verbose = false
    silent = false
    ss_symbolic_mode = :single_equation
    precompile = false
    report_missing_parameters = true
    perturbation_order = 1
    guess = Dict{Symbol,Float64}()
    steady_state_function = nothing
    ss_solver_parameters_algorithm = :ESCH
    ss_solver_parameters_maxtime = 120.0

    for exp in ex[1:end-1]
        postwalk(x ->
            x isa Expr ?
                x.head == :(=) ?
                    (x.args[1] == :ss_symbolic_mode && (x.args[2] isa Symbol || (x.args[2] isa QuoteNode && x.args[2].value isa Symbol))) ?
                        ss_symbolic_mode = x.args[2] isa QuoteNode ? x.args[2].value : x.args[2] :
                    (x.args[1] == :verbose && x.args[2] isa Bool) ?
                        verbose = x.args[2] :
                    (x.args[1] == :silent && x.args[2] isa Bool) ?
                        silent = x.args[2] :
                    (x.args[1] == :report_missing_parameters && x.args[2] isa Bool) ?
                        report_missing_parameters = x.args[2] :
                    (x.args[1] == :precompile && x.args[2] isa Bool) ?
                        precompile = x.args[2] :
                    (x.args[1] == :perturbation_order && x.args[2] isa Int) ?
                        perturbation_order = x.args[2] :
                    (x.args[1] == :guess && (isa(eval(x.args[2]), Dict{Symbol, <:Real}) || isa(eval(x.args[2]), Dict{String, <:Real}))) ?
                        guess = x.args[2] :
                    (x.args[1] == :ss_solver_parameters_algorithm && (x.args[2] isa Symbol || (x.args[2] isa QuoteNode && x.args[2].value isa Symbol))) ?
                        ss_solver_parameters_algorithm = x.args[2] isa QuoteNode ? x.args[2].value : x.args[2] :
                    (x.args[1] == :steady_state_function && x.args[2] isa Symbol) ?
                        steady_state_function = esc(x.args[2]) :
                    (x.args[1] == :ss_solver_parameters_maxtime && x.args[2] isa Real) ?
                        ss_solver_parameters_maxtime = x.args[2] :
                    begin
                        @warn "Invalid option `$(x.args[1])` ignored. See docs: `?@parameters` for valid options."
                        x
                    end :
                x :
            x,
        exp)
    end

    @assert ss_symbolic_mode ∈ [:none, :single_equation, :full] "ss_symbolic_mode must be :none, :single_equation, or :full. Got $ss_symbolic_mode."

    @assert ss_solver_parameters_algorithm ∈ [:ESCH, :SAMIN] "ss_solver_parameters_algorithm must be :ESCH or :SAMIN. Got $ss_solver_parameters_algorithm. Using default :ESCH."

    parameter_block = ex[end]

    # Parsing of the calibration block is delegated to
    # `process_parameter_definitions` in src/parser/equation_processing.jl,
    # which is also used by the equation-modification reprocess pipeline.
    return quote
        mod = @__MODULE__

        local _parsed = MacroModelling.process_parameter_definitions(
            $(QuoteNode(parameter_block)),
            mod.$𝓂.constants.post_model_macro,
        )

        # Merge guess option with any guess already on the model.
        local _guess_dict = mod.$𝓂.constants.post_parameters_macro.guess
        if isa($guess, Dict{String, <:Real})
            _guess_dict = Dict{Symbol, Float64}()
            for (key, value) in $guess
                if key isa String
                    key = replace_indices(key)
                end
                _guess_dict[replace_indices(key)] = value
            end
        elseif isa($guess, Dict{Symbol, <:Real})
            _guess_dict = $guess
        end

        # Merge bounds returned by the parser with bounds already on the model.
        local _bounds_dict = copy(mod.$𝓂.constants.post_parameters_macro.bounds)
        for (k, v) in _parsed.bounds
            _bounds_dict[k] = haskey(_bounds_dict, k) ?
                (max(_bounds_dict[k][1], v[1]), min(_bounds_dict[k][2], v[2])) :
                (v[1], v[2])
        end

        local _invalid_bounds = Symbol[]
        for (k, v) in _bounds_dict
            if v[1] >= v[2]
                push!(_invalid_bounds, k)
            end
        end
        @assert isempty(_invalid_bounds) "Invalid bounds: " * repr(_invalid_bounds)

        mod.$𝓂.constants.post_parameters_macro = post_parameters_macro(
            _parsed.calib_parameters_no_var,
            $precompile,
            $(QuoteNode(ss_symbolic_mode)),
            $(QuoteNode(ss_solver_parameters_algorithm)),
            $ss_solver_parameters_maxtime,
            _guess_dict,
            _parsed.ss_calib_list,
            _parsed.par_calib_list,
            _bounds_dict,
        )

        mod.$𝓂.equations.calibration            = _parsed.equations.calibration
        mod.$𝓂.equations.calibration_no_var     = _parsed.equations.calibration_no_var
        mod.$𝓂.equations.calibration_parameters = _parsed.equations.calibration_parameters
        mod.$𝓂.equations.calibration_original   = _parsed.equations.calibration_original

        mod.$𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            mod.$𝓂.constants.post_complete_parameters;
            parameters         = _parsed.parameters,
            missing_parameters = _parsed.missing_parameters,
        )
        mod.$𝓂.parameter_values = _parsed.parameter_values

        local _missing_params = _parsed.missing_parameters
        local _has_missing_parameters = !isempty(_missing_params)

        set_custom_steady_state_function!(mod.$𝓂, $steady_state_function)

        mod.$𝓂.functions.functions_written = false

        if !isnothing($steady_state_function)
            write_ss_check_function!(mod.$𝓂)
        else
            if !_has_missing_parameters
                set_up_steady_state_solver!(mod.$𝓂, verbose = $verbose, silent = $silent, ss_symbolic_mode = $(QuoteNode(ss_symbolic_mode)))
            end
        end

        if !_has_missing_parameters
            opts = merge_calculation_options(verbose = $verbose)

            SS_and_pars, solution_error, found_solution = solve_steady_state!(mod.$𝓂, opts, $(QuoteNode(ss_solver_parameters_algorithm)), $ss_solver_parameters_maxtime, silent = $silent)

            write_symbolic_derivatives!(mod.$𝓂; perturbation_order = $perturbation_order, silent = $silent)

            mod.$𝓂.functions.functions_written = true
        end

        if _has_missing_parameters && $report_missing_parameters
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(_missing_params). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined. Provide missing parameter values via the `parameters` keyword argument in functions like `get_irf`, `get_steady_state`, `simulate`, etc."
        end

        if !$silent && $report_missing_parameters
            Base.show(mod.$𝓂)
        end

        nothing
    end
end
