# =============================================================================
# Constants
# =============================================================================
const BOUND_INFINITY = 1e12
const EXP_UPPER_BOUND = 600
const SMALL_EPS = eps(Float32)

# Bounded functions with their (lower_bound, upper_bound) for arguments
const BOUNDED_FUNCTIONS = Dict{Symbol, Tuple{Float64, Float64}}(
    :log => (eps(), BOUND_INFINITY),
    :^ => (eps(), BOUND_INFINITY),  # when exponent is non-integer
    :exp => (-BOUND_INFINITY, EXP_UPPER_BOUND),
    :norminvcdf => (eps(), 1 - eps()),
    :norminv => (eps(), 1 - eps()),
    :qnorm => (eps(), 1 - eps()),
    :erfcinv => (eps(), 2 - eps()),
)

# Regex pattern for auxiliary variables with lead/lag notation (e.g., xᴸ⁽²⁾)
const AUX_VAR_REGEX = r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾"

# =============================================================================
# Timing marker predicates
# =============================================================================

"""Check if timing marker indicates an exogenous shock (x, ex, exo, exogenous)."""
is_exo_marker(t) = occursin(r"^(x|ex|exo|exogenous){1}$"i, string(t))

"""Check if timing marker indicates steady state (ss, stst, steady, steadystate, steady_state)."""
is_steady_state_marker(t) = occursin(r"^(ss|stst|steady|steadystate|steady_state){1}$"i, string(t))

"""Check if timing marker is exogenous with a lead offset (e.g., 'x + 2')."""
is_exo_lead_offset(t) = occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\+{1}\s{1}\d+$))"i, string(t))

"""Check if timing marker is exogenous with a lag offset (e.g., 'x - 2')."""
is_exo_lag_offset(t) = occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}\-{1}\s{1}\d+$))"i, string(t))

"""Check if timing marker starts with exogenous prefix (for any offset)."""
is_exo_with_offset(t) = occursin(r"^(x|ex|exo|exogenous){1}(?=(\s{1}(\-|\+){1}\s{1}\d+$))"i, string(t))

"""Check if a symbol is an operator or timing keyword (not a parameter)."""
is_operator_or_timing_keyword(s) = occursin(r"^(\+|\-|\*|\/|\^|ss|stst|steady|steadystate|steady_state){1}$"i, string(s))

function process_model_equations(
    model_block::Expr,
    max_obc_horizon::Int,
    precompile::Bool,
)
    model_ex = preprocess_model_block(model_block, max_obc_horizon)

    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    dyn_equations = []
    ➕_vars = []
    ss_and_aux_equations = []
    ss_equations = []
    ss_aux_equations = Expr[]
    aux_vars_created = Set()
    unique_➕_eqs = Dict{Union{Expr,Symbol},Expr}()
    ss_equations_with_aux_variables = Int[]
    dyn_eq_aux_ind = Int[]

    build_dynamic_and_ss_equations!(model_ex, dyn_equations, ss_equations, aux_vars_created, dyn_eq_aux_ind)
    create_nonnegativity_auxiliaries!(model_ex, ss_and_aux_equations, ➕_vars, bounds, ss_equations_with_aux_variables, unique_➕_eqs, precompile)

    # go through changed SS equations including nonnegativity auxiliary variables
    var_list_aux_SS = []
    ss_list_aux_SS = []
    par_list_aux_SS = []

    var_future_list_aux_SS = []
    var_present_list_aux_SS = []
    var_past_list_aux_SS = []

    process_auxiliary_ss_equations!(
        ss_and_aux_equations,
        ss_aux_equations,
        var_list_aux_SS,
        ss_list_aux_SS,
        par_list_aux_SS,
        var_future_list_aux_SS,
        var_present_list_aux_SS,
        var_past_list_aux_SS,
        ss_equations_with_aux_variables,
        precompile,
    )

    # go through dynamic equations and label
    # create timings
    dyn_var_future_list, dyn_var_present_list, dyn_var_past_list, dyn_exo_list, dyn_ss_list = extract_dyn_symbol_lists(dyn_equations)

    all_symbols = reduce(union,collect.(get_symbols.(dyn_equations)))
    parameters_in_equations = sort(collect(setdiff(all_symbols, filter_by_pattern(all_symbols, r"₎$"))))
    
    dyn_var_future  =  sort(collect(reduce(union,dyn_var_future_list)))
    dyn_var_present =  sort(collect(reduce(union,dyn_var_present_list)))
    dyn_var_past    =  sort(collect(reduce(union,dyn_var_past_list)))
    dyn_var_ss      =  sort(collect(reduce(union,dyn_ss_list)))

    all_dyn_vars        = union(dyn_var_future, dyn_var_present, dyn_var_past)

    @assert length(setdiff(dyn_var_ss, all_dyn_vars)) == 0 "The following variables are (and cannot be) defined only in steady state (`[ss]`): $(setdiff(dyn_var_ss, all_dyn_vars))"

    present_only              = sort(setdiff(dyn_var_present,union(dyn_var_past,dyn_var_future)))
    future_not_past           = sort(setdiff(dyn_var_future, dyn_var_past))
    past_not_future           = sort(setdiff(dyn_var_past, dyn_var_future))
    mixed                     = sort(setdiff(dyn_var_present, union(present_only, future_not_past, past_not_future)))
    future_not_past_and_mixed = sort(union(future_not_past,mixed))
    past_not_future_and_mixed = sort(union(past_not_future,mixed))
    present_but_not_only      = sort(setdiff(dyn_var_present,present_only))
    mixed_in_past             = sort(intersect(dyn_var_past, mixed))
    not_mixed_in_past         = sort(setdiff(dyn_var_past,mixed_in_past))
    mixed_in_future           = sort(intersect(dyn_var_future, mixed))
    exo                       = sort(collect(reduce(union,dyn_exo_list)))
    var                       = sort(dyn_var_present)
    aux_tmp                   = sort(filter(x->occursin(AUX_VAR_REGEX,string(x)), dyn_var_present))
    aux                       = sort(aux_tmp[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∉ exo, aux_tmp)])
    exo_future                = dyn_var_future[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∈ exo, dyn_var_future)]
    exo_present               = dyn_var_present[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∈ exo, dyn_var_present)]
    exo_past                  = dyn_var_past[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∈ exo, dyn_var_past)]

    nPresent_only              = length(present_only)
    nMixed                     = length(mixed)
    nFuture_not_past_and_mixed = length(future_not_past_and_mixed)
    nPast_not_future_and_mixed = length(past_not_future_and_mixed)
    nVars                      = length(union(all_dyn_vars, dyn_var_ss))
    nExo                       = length(collect(exo))

    present_only_idx              = indexin(present_only,var)
    present_but_not_only_idx      = indexin(present_but_not_only,var)
    future_not_past_and_mixed_idx = indexin(future_not_past_and_mixed,var)
    past_not_future_and_mixed_idx = indexin(past_not_future_and_mixed,var)
    mixed_in_future_idx           = indexin(mixed_in_future,dyn_var_future)
    mixed_in_past_idx             = indexin(mixed_in_past,dyn_var_past)
    not_mixed_in_past_idx         = indexin(not_mixed_in_past,dyn_var_past)
    past_not_future_idx           = indexin(past_not_future,var)

    reorder       = indexin(var, [present_only; past_not_future; future_not_past_and_mixed])
    dynamic_order = indexin(present_but_not_only, [past_not_future; future_not_past_and_mixed])

    @assert length(intersect(union(var,exo),parameters_in_equations)) == 0 "Parameters and variables cannot have the same name. This is the case for: " * repr(sort([intersect(union(var,exo),parameters_in_equations)...]))

    # Check that no variable, shock, or parameter names conflict with SymPyWorkspace reserved names
    reserved_conflicts_vars = intersect(var, SYMPYWORKSPACE_RESERVED_NAMES)
    reserved_conflicts_exo = intersect(exo, SYMPYWORKSPACE_RESERVED_NAMES)
    reserved_conflicts_params = intersect(parameters_in_equations, SYMPYWORKSPACE_RESERVED_NAMES)
    
    @assert length(reserved_conflicts_vars) == 0 "The following variable names are reserved and cannot be used: " * repr(sort([reserved_conflicts_vars...]))
    @assert length(reserved_conflicts_exo) == 0 "The following shock names are reserved and cannot be used: " * repr(sort([reserved_conflicts_exo...]))
    @assert length(reserved_conflicts_params) == 0 "The following parameter names are reserved and cannot be used: " * repr(sort([reserved_conflicts_params...]))

    @assert !any(isnothing, future_not_past_and_mixed_idx) "The following variables appear in the future only (and should at least appear in the present as well): $(setdiff(future_not_past_and_mixed, var)))"

    @assert !any(isnothing, past_not_future_and_mixed_idx) "The following variables appear in the past only (and should at least appear in the present as well): $(setdiff(future_not_past_and_mixed, var)))"

    aux_future_tmp  = sort(filter(x->occursin(AUX_VAR_REGEX,string(x)), dyn_var_future))
    aux_future      = aux_future_tmp[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∉ exo, aux_future_tmp)]

    aux_past_tmp    = sort(filter(x->occursin(AUX_VAR_REGEX,string(x)), dyn_var_past))
    aux_past        = aux_past_tmp[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∉ exo, aux_past_tmp)]

    aux_present_tmp = sort(filter(x->occursin(AUX_VAR_REGEX,string(x)), dyn_var_present))
    aux_present     = aux_present_tmp[map(x->Symbol(replace(string(x),AUX_VAR_REGEX => "")) ∉ exo, aux_present_tmp)]

    vars_in_ss_equations = sort(collect(setdiff(reduce(union, get_symbols.(ss_aux_equations)), parameters_in_equations)))
    vars_in_ss_equations_no_aux = setdiff(vars_in_ss_equations, ➕_vars)

    dyn_future_list =   filter_by_pattern.(get_symbols.(dyn_equations), r"₍₁₎")
    dyn_present_list =  filter_by_pattern.(get_symbols.(dyn_equations), r"₍₀₎")
    dyn_past_list =     filter_by_pattern.(get_symbols.(dyn_equations), r"₍₋₁₎")
    dyn_exo_list =      filter_by_pattern.(get_symbols.(dyn_equations), r"₍ₓ₎")

    T = post_model_macro(
                max_obc_horizon,
                # present_only,
                # future_not_past,
                # past_not_future,
                # mixed,
                future_not_past_and_mixed,
                past_not_future_and_mixed,
                # present_but_not_only,
                # mixed_in_past,
                # not_mixed_in_past,
                # mixed_in_future,

                var,

                parameters_in_equations,

                exo,
                exo_past,
                exo_present,
                exo_future,

                aux,
                aux_past,
                aux_present,
                aux_future,

                ➕_vars,

                nPresent_only,
                nMixed,
                nFuture_not_past_and_mixed,
                nPast_not_future_and_mixed,
                # nPresent_but_not_only,
                nVars,
                nExo,

                present_only_idx,
                present_but_not_only_idx,
                future_not_past_and_mixed_idx,
                not_mixed_in_past_idx,
                past_not_future_and_mixed_idx,
                mixed_in_past_idx,
                mixed_in_future_idx,
                past_not_future_idx,

                reorder,
                dynamic_order,
                vars_in_ss_equations,
                vars_in_ss_equations_no_aux,

                dyn_var_future_list,
                dyn_var_present_list,
                dyn_var_past_list,
                dyn_ss_list,
                dyn_exo_list,

                dyn_future_list,
                dyn_present_list,
                dyn_past_list,

                var_list_aux_SS,
                ss_list_aux_SS,
                par_list_aux_SS,
                var_future_list_aux_SS,
                var_present_list_aux_SS,
                var_past_list_aux_SS,
                ss_equations_with_aux_variables)

    original_equations = []
    for (i,arg) in enumerate(model_ex.args)
        if isa(arg,Expr)
            prs_exx = postwalk(x -> x isa Expr ? unblock(x) : x, model_ex.args[i])
            push!(original_equations, unblock(prs_exx))
        end
    end

    single_dyn_vars_equations = findall(length.(vcat.(collect.(dyn_var_future_list),
                                                      collect.(dyn_var_present_list),
                                                      collect.(dyn_var_past_list),
                                                      collect.(dyn_exo_list))) .== 1)
                                                    
    @assert length(single_dyn_vars_equations) == 0 "Equations must contain more than 1 dynamic variable. This is not the case for: " * repr([original_equations[indexin(single_dyn_vars_equations,setdiff(1:length(dyn_equations),dyn_eq_aux_ind .- 1))]...])
    
    duplicate_equations = []
    for item in unique(dyn_equations)
        indices = findall(x -> x == item, dyn_equations)
        if length(indices) > 1
            push!(duplicate_equations, indices)
        end
    end
    
    @assert length(duplicate_equations) == 0 "The following equations appear more than once (and should only appear once): \n" * join(["$(original_equations[eq_idxs[1]])" for eq_idxs in duplicate_equations], "\n")

    equations_struct = equations(
        original_equations,
        dyn_equations,
        ss_equations,
        ss_aux_equations,
        Expr[],
        Expr[],
        Expr[],
        Symbol[],
        Expr[],
    )

    return T, equations_struct
end

function preprocess_model_block(model_block::Expr, max_obc_horizon::Int)
    model_ex = expand_for_loops_block(model_block)
    model_ex = resolve_if_blocks(model_ex::Expr)::Expr
    model_ex = drop_nothing_calls(model_ex::Expr)::Expr
    return expand_obc_constraints(model_ex::Expr, max_obc_horizon = max_obc_horizon)::Expr
end

function expand_for_loops_block(equations_block)::Expr
    eqs = Expr[]
    for arg in equations_block.args
        if isa(arg,Expr)
            parsed_eqs = expand_for_loop_expr(arg)
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

function expand_for_loop_expr(arg::Expr)::Expr
    postwalk(x -> begin
                    x = flatten(unblock(x))
                    x isa Expr ?
                        x.head == :for ?
                            x.args[2] isa Array ?
                                length(x.args[2]) >= 1 ?
                                    x.args[1].head == :block ?
                                        [replace_indices_in_for_loop(X, Symbol(x.args[1].args[2].args[1]), (x.args[1].args[2].args[2]), false, x.args[1].args[1].args[2].value) for X in x.args[2]] :
                                    [replace_indices_in_for_loop(X, Symbol(x.args[1].args[1]), (x.args[1].args[2]), false, :+) for X in x.args[2]] :
                                x :
                            x.args[2].head ∉ [:(=), :block] ?
                                x.args[1].head == :block ?
                                    replace_indices_in_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    true,
                                                    x.args[1].args[1].args[2].value) :
                                x.args[2].head == :if ?
                                    has_equation(x.args[2]) ?
                                        replace_indices_in_for_loop(unblock(x.args[2]), 
                                                            Symbol(x.args[1].args[1]), 
                                                            (x.args[1].args[2]),
                                                            false,
                                                            :+) :
                                    replace_indices_in_for_loop(unblock(x.args[2]), 
                                                        Symbol(x.args[1].args[1]), 
                                                        (x.args[1].args[2]),
                                                        true,
                                                        :+) :
                                replace_indices_in_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[1]), 
                                                    (x.args[1].args[2]),
                                                    true,
                                                    :+) :
                            x.args[1].head == :block ?
                                replace_indices_in_for_loop(unblock(x.args[2]), 
                                                    Symbol(x.args[1].args[2].args[1]), 
                                                    (x.args[1].args[2].args[2]),
                                                    false,
                                                    x.args[1].args[1].args[2].value) :
                            replace_indices_in_for_loop(unblock(x.args[2]), 
                                            Symbol(x.args[1].args[1]), 
                                            (x.args[1].args[2]),
                                            false,
                                            :+) :
                        x :
                    x
                end,
    arg)
end

function replace_indices_in_for_loop(exxpr,index_variable,indices,concatenate, operator)
    @assert operator ∈ [:+,:*] "Only :+ and :* allowed as operators in for loops."
    calls = []
    indices = indices.args[1] == :(:) ? eval(indices) : [indices.args...]
    for idx in indices
        push!(calls, postwalk(x -> begin
            if x isa Expr
                if x.head == :ref
                    if @capture(x, name_{index_}[time_])
                        if index == index_variable
                            return :($(Expr(:ref, Symbol(string(name) * "{" * string(idx) * "}"),time)))
                        elseif time isa Expr || time isa Symbol
                            if index_variable ∈ get_symbols(time)
                                return :($(Expr(:ref, Expr(:curly,name,index), Meta.parse(replace(string(time), string(index_variable) => idx)))))
                            end
                        end
                        return x
                    elseif @capture(x, name_[time_])
                        if time isa Expr || time isa Symbol
                            if index_variable ∈ get_symbols(time)
                                return :($(Expr(:ref, name, Meta.parse(replace(string(time), string(index_variable) => idx)))))
                            end
                        end
                        return x
                    end
                elseif x.head == :if
                    if length(x.args) > 2
                        return Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                    replace_indices_in_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock,
                                    replace_indices_in_for_loop(x.args[3],index_variable,:([$idx]),false,:+) |> unblock)
                    end
                    return Expr(:if,   postwalk(x -> x == index_variable ? idx : x, x.args[1]),
                                replace_indices_in_for_loop(x.args[2],index_variable,:([$idx]),false,:+) |> unblock)
                elseif @capture(x, name_{index_})
                    if index == index_variable
                        return :($(Symbol(string(name) * "{" * string(idx) * "}")))
                    end
                end
            elseif @capture(x, name_)
                if name == index_variable && idx isa Int
                    return :($idx)
                elseif x isa Symbol && occursin("{" * string(index_variable) * "}", string(x))
                    return Symbol(replace(string(x),  "{" * string(index_variable) * "}" => "{" * string(idx) * "}"))
                end
            end
            return x
        end,
        exxpr))
    end
    
    if concatenate
        return :($(Expr(:call, operator, calls...)))
    else
        return :($(Expr(:block, calls...)))
    end
end

function has_equation(expr)
    found = false
    postwalk(expr) do x
        if @capture(x, _ = _)
            found = true
        end
        return x
    end
    return found
end

function resolve_if_blocks(ex::Expr)
    prewalk(ex) do node
        if node isa Expr && (node.head === :if || node.head === :elseif)
            cond = node.args[1]
            then_blk = node.args[2]
            if length(node.args) == 3
                else_blk = node.args[3]
            end
            val = eval_condition(unblock(cond))

            if val === true
                # recurse into the selected branch
                return resolve_if_blocks(unblock(then_blk))
            elseif val === false && length(node.args) == 3
                return resolve_if_blocks(unblock(else_blk))
            elseif val === false && length(node.args) == 2
                return nothing
            elseif val === false && node.head === :elseif
                return resolve_if_blocks(unblock(else_blk))
            end
        end
        return node
    end
end

function eval_condition(cond)
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
    end
    return nothing
end

function drop_nothing_calls(ex::Expr)
    postwalk(ex) do node
        # Only consider call-expressions
        if node isa Expr && node.head === :call && any(node.args .=== nothing)
            fn = node.args[1]
            kept = filter(arg -> !(unblock(arg) === nothing), node.args[2:end])
            if isempty(kept)
                return nothing
            elseif length(kept) == 1
                return kept[1]
            else
                return Expr(:call, fn, kept...)
            end
        end
        return node
    end
end

function expand_obc_constraints(equations_block; max_obc_horizon::Int = 40)
    eqs = []
    obc_shocks = Expr[]

    for arg in equations_block.args
        if isa(arg,Expr)
            if check_for_minmax(arg)
                arg_trans = transform_obc(arg)
            else
                arg_trans = arg
            end

            eq = postwalk(x -> begin
                    x isa Expr || return x
                    if @capture(x, max_(lhs_, rhs_)) && max == :max
                        obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                        obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                        if !(lhs isa Symbol) && check_for_dynamic_variables(lhs)
                            push!(eqs, :($obc_vars_left = $(lhs)))
                        else
                            obc_vars_left = lhs
                        end

                        if !(rhs isa Symbol) && check_for_dynamic_variables(rhs)
                            push!(eqs, :($obc_vars_right = $(rhs)))
                        else
                            obc_vars_right = rhs
                        end

                        obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                        push!(eqs, :($obc_inequality = $(Expr(:call, :max, obc_vars_left, obc_vars_right))))

                        obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁺ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                        push!(obc_shocks, obc_shock)

                        return :($obc_inequality - $obc_shock)
                    elseif @capture(x, min_(lhs_, rhs_)) && min == :min
                        obc_vars_left = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝˡ" ), 0)
                        obc_vars_right = Expr(:ref, Meta.parse("χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝʳ" ), 0)

                        if !(lhs isa Symbol) && check_for_dynamic_variables(lhs)
                            push!(eqs, :($obc_vars_left = $(lhs)))
                        else
                            obc_vars_left = lhs
                        end

                        if !(rhs isa Symbol) && check_for_dynamic_variables(rhs)
                            push!(eqs, :($obc_vars_right = $(rhs)))
                        else
                            obc_vars_right = rhs
                        end

                        obc_inequality = Expr(:ref, Meta.parse("Χᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ" ), 0)

                        push!(eqs, :($obc_inequality = $(Expr(:call, :min, obc_vars_left, obc_vars_right))))

                        obc_shock = Expr(:ref, Meta.parse("ϵᵒᵇᶜ⁻ꜝ" * super(string(length(obc_shocks) + 1)) * "ꜝ"), 0)

                        push!(obc_shocks, obc_shock)

                        return :($obc_inequality - $obc_shock)
                    end

                    return x
            end, arg_trans)

            push!(eqs, eq)
        end
    end

    for obc in obc_shocks
        push!(eqs, :($(obc) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(max_obc_horizon)) * "⁾"), 0))))

        push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻⁰⁾"), 0)) = activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon)) * "⁾"), :x))))

        for i in 1:max_obc_horizon
            push!(eqs, :($(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i)) * "⁾"), 0)) = $(Expr(:ref, Meta.parse(string(obc.args[1]) * "ᴸ⁽⁻" * super(string(i-1)) * "⁾"), -1)) + activeᵒᵇᶜshocks * $(Expr(:ref, Meta.parse(string(obc.args[1]) * "⁽" * super(string(max_obc_horizon-i)) * "⁾"), :x))))
        end
    end

    return Expr(:block, eqs...)
end

function set_up_obc_violation_function!(𝓂)
    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    present_varss = collect(reduce(union,filter_by_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎$")))

    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))

    # write indices in auxiliary objects
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(filter_by_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎")))

    dyn_var_present = Symbol.(replace.(string.(sort(collect(reduce(union,dyn_var_present_list)))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

    SS_and_pars_names = ms.SS_and_pars_names

    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)

    alll = []
    for (i,var) in enumerate(present_varss)
        if !(match(r"^χᵒᵇᶜ", string(var)) === nothing)
            push!(alll,:($var = Y[$(dyn_var_present_idx[i]),1:max(periods, 1)]))
        end
    end

    calc_obc_violation = :(function calculate_obc_violation(x, p)
        state, state_update, reference_steady_state, 𝓂, algorithm, periods, shock_values = p

        T = 𝓂.constants.post_model_macro

        Y = zeros(typeof(x[1]), T.nVars, periods+1)

        shock_values = convert(typeof(x), shock_values)

        shock_values[contains.(string.(T.exo),"ᵒᵇᶜ")] .= x

        zero_shock = zero(shock_values)

        if algorithm ∈ [:pruned_second_order, :pruned_third_order]
            states = state_update(state, shock_values)
            Y[:,1] = sum(states)
        else
            Y[:,1] = state_update(state, shock_values)
        end

        for t in 1:periods
            if algorithm ∈ [:pruned_second_order, :pruned_third_order]
                states = state_update(states, zero_shock)
                Y[:,t+1] = sum(states)
            else
                Y[:,t+1] = state_update(Y[:,t], zero_shock)
            end
        end

        Y .+= reference_steady_state[1:T.nVars]

        $(alll...)

        constraint_values = Vector[]

        $(𝓂.equations.obc_violation...)

        return vcat(constraint_values...)
    end)

    𝓂.functions.obc_violation = @RuntimeGeneratedFunction(calc_obc_violation)

    return nothing
end

function write_obc_violation_equations(𝓂)
    eqs = Expr[]
    for eq in 𝓂.equations.dynamic
        if check_for_minmax(eq)
            minmax_fixed_eqs = postwalk(x -> begin
                x isa Expr || return x
                if x.head == :call && length(x.args) == 3 && x.args[3] isa Expr
                    if @capture(x.args[3], minmax_(arg1_, arg2_)) && minmax ∈ [:Min, :min, :Max, :max]
                        placeholder = Symbol(replace(string(x.args[2]), "₍₀₎" => ""))
                        maximisation = contains(string(placeholder), "⁺")

                        cond = Expr[]
                        if maximisation
                            push!(cond, :(push!(constraint_values, [sum($arg1 .* $arg2)])))
                            push!(cond, :(push!(constraint_values, $arg1)))
                            push!(cond, :(push!(constraint_values, $arg2)))
                        else
                            push!(cond, :(push!(constraint_values, [sum($arg1 .* $arg2)])))
                            push!(cond, :(push!(constraint_values, -$arg1)))
                            push!(cond, :(push!(constraint_values, -$arg2)))
                        end

                        return :($(Expr(:block, cond...)))
                    end
                end
                return x
            end, eq)

            push!(eqs, minmax_fixed_eqs)
        end
    end

    return eqs
end

function build_dynamic_and_ss_equations!(
    model_ex::Expr,
    dyn_equations,
    ss_equations,
    aux_vars_created,
    dyn_eq_aux_ind,
)
    # write down dynamic equations and add auxiliary variables for leads and lags > 1
    for arg in model_ex.args
        if isa(arg, Expr)
            # write down dynamic equations
            t_ex = postwalk(x -> begin
                x isa Expr || return x
                
                # Convert = to subtraction
                if x.head == :(=)
                    return Expr(:call, :-, x.args[1], x.args[2])
                end
                
                # Handle variable references with timing
                if x.head == :ref
                    return transform_var_ref(x, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
                end
                
                return unblock(x)
            end, arg)

            push!(dyn_equations, unblock(t_ex))
            
            # write down ss equations using the helper function
            eqs = postwalk(x -> transform_to_ss(x), arg)
            push!(ss_equations, flatten(unblock(eqs)))
        end
    end
end

function create_nonnegativity_auxiliaries!(
    model_ex::Expr,
    ss_and_aux_equations,
    ➕_vars,
    bounds,
    ss_equations_with_aux_variables,
    unique_➕_eqs,
    precompile,
)
    # write down ss equations including nonnegativity auxiliary variables
    # find nonegative variables, parameters, or terms
    for arg in model_ex.args
        if isa(arg, Expr)
            eqs = postwalk(x -> begin
                x isa Expr || return x
                
                # Convert = to subtraction
                if x.head == :(=)
                    return Expr(:call, :-, x.args[1], x.args[2])
                end
                
                # Handle variable references - exogenous become 0, others keep their name
                if x.head == :ref
                    if length(x.args) >= 2 && occursin(r"^(x|ex|exo|exogenous){1}"i, string(x.args[2]))
                        return 0
                    end
                    return x
                end
                
                # Handle function calls
                if x.head == :call
                    # Reorder integer multiplication: 2*beta => beta*2
                    if x.args[1] == :* && length(x.args) >= 3
                        if x.args[2] isa Int && !(x.args[3] isa Int)
                            return Expr(:call, :*, x.args[3:end]..., x.args[2])
                        end
                    end
                    
                    # Handle bounded functions (log, exp, ^, norminvcdf, erfcinv)
                    return handle_bounded_call(x, bounds, ss_and_aux_equations, ss_equations_with_aux_variables, ➕_vars, unique_➕_eqs, precompile)
                end
                
                return x
            end, arg)
            
            push!(ss_and_aux_equations, unblock(eqs))
        end
    end
end

function process_auxiliary_ss_equations!(
    ss_and_aux_equations,
    ss_aux_equations,
    var_list_aux_SS,
    ss_list_aux_SS,
    par_list_aux_SS,
    var_future_list_aux_SS,
    var_present_list_aux_SS,
    var_past_list_aux_SS,
    ss_equations_with_aux_variables,
    precompile)
    # go through changed SS equations including nonnegativity auxiliary variables
    for (idx,eq) in enumerate(ss_and_aux_equations)
        var_tmp = Set()
        ss_tmp = Set()
        par_tmp = Set()
        var_future_tmp = Set()
        var_present_tmp = Set()
        var_past_tmp = Set()

        # remove terms multiplied with 0
        eq = postwalk(remove_zero_terms, eq)

        # label all variables parameters and exogenous variables and timings for individual equations
        classify_vars_by_timing!(eq, var_future_tmp, var_present_tmp, var_past_tmp, ss_tmp, par_tmp)

        var_tmp = union(var_future_tmp,var_present_tmp,var_past_tmp)
        
        push!(var_list_aux_SS,var_tmp)
        push!(ss_list_aux_SS,ss_tmp)
        push!(par_list_aux_SS,par_tmp)
        push!(var_future_list_aux_SS,var_future_tmp)
        push!(var_present_list_aux_SS,var_present_tmp)
        push!(var_past_list_aux_SS,var_past_tmp)

        # write down SS equations including nonnegativity auxiliary variables
        prs_ex = convert_to_ss_equation(eq)
        
        if idx ∈ ss_equations_with_aux_variables
            if precompile
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],unblock(prs_ex).args[3]) 
            else
                ss_aux_equation = Expr(:call,:-,unblock(prs_ex).args[2],simplify(unblock(prs_ex).args[3])) # simplify RHS if nonnegative auxiliary variable
            end
        else
            if precompile
                ss_aux_equation = unblock(prs_ex)
            else
                ss_aux_equation = simplify(unblock(prs_ex))
            end
        end
        
        if ss_aux_equation isa Symbol 
            push!(ss_aux_equations, Expr(:call,:-,ss_aux_equation,0))
        else#if !(ss_aux_equation isa Int)
            push!(ss_aux_equations, ss_aux_equation)
        end
    end
end

function extract_dyn_symbol_lists(dyn_equations)
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(filter_by_pattern.(get_symbols.(dyn_equations), r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(filter_by_pattern.(get_symbols.(dyn_equations), r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎"=> "")),x)),collect.(filter_by_pattern.(get_symbols.(dyn_equations), r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(filter_by_pattern.(get_symbols.(dyn_equations), r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(filter_by_pattern.(get_symbols.(dyn_equations), r"₍ₛₛ₎")))
    return dyn_var_future_list, dyn_var_present_list, dyn_var_past_list, dyn_exo_list, dyn_ss_list
end

function filter_by_pattern(strings::Union{Set,Vector}, pattern::Regex)
    return filter(r -> match(pattern, string(r)) !== nothing, strings)
end

# =============================================================================
# Expression transformation helpers
# =============================================================================

"""
Extract all steady-state variable references from an expression.
Returns a Set of variable names that appear as `var[ss]`.
"""
function extract_ss_vars(eq)
    ss_vars = Set{Symbol}()
    postwalk(eq) do x
        if @capture(x, name_[timing_]) && name isa Symbol && is_steady_state_marker(timing)
            push!(ss_vars, name)
        end
        x
    end
    return ss_vars
end

"""
Extract parameters from an expression (symbols that are not SS variables or operators).
"""
function extract_parameters(eq, ss_vars::Set)
    params = Set{Symbol}()
    all_syms = get_symbols(eq)
    postwalk(eq) do x
        if x isa Symbol && !is_operator_or_timing_keyword(x) && x ∉ ss_vars && x ∈ all_syms
            push!(params, x)
        end
        x
    end
    return params
end

"""
Classify variable references by timing (future, present, past, ss).
Updates the provided sets in place.
"""
function classify_vars_by_timing!(eq, var_future::Set, var_present::Set, var_past::Set, ss_vars::Set, params::Set)
    all_syms = get_symbols(eq)
    postwalk(eq) do x
        # Handle variable references with timing
        if @capture(x, name_[timing_])
            if name isa Symbol
                if timing isa Int
                    if timing == 0
                        push!(var_present, name)
                    elseif timing > 0
                        push!(var_future, name)
                    else  # timing < 0
                        push!(var_past, name)
                    end
                elseif is_steady_state_marker(timing)
                    push!(ss_vars, name)
                elseif is_exo_lag_offset(timing)
                    push!(var_past, name)
                elseif is_exo_lead_offset(timing)
                    push!(var_future, name)
                end
            end
        # Handle symbols in function calls (potential parameters)
        elseif x isa Symbol && !is_operator_or_timing_keyword(x) && x ∈ all_syms
            # Only add if not already classified as a variable
            if x ∉ var_future && x ∉ var_present && x ∉ var_past && x ∉ ss_vars
                push!(params, x)
            end
        end
        x
    end
end

"""
Remove terms multiplied by zero: `a * 0 * b => 0`.
"""
function remove_zero_terms(x)
    if @capture(x, *(args__))
        if any(a -> a == 0, args)
            return 0
        end
    end
    return x
end

"""
Reorder integer multiplication: `2 * beta => beta * 2` (needed for SymPy).
"""
function reorder_int_mult(x)
    if @capture(x, n_ * rest__)
        if n isa Int && length(rest) >= 1 && !(rest[1] isa Int)
            return Expr(:call, :*, rest..., n)
        end
    end
    return x
end

"""
Transform a calibration equation for solving.
- Converts `a = b` to `a - b` (when `to_subtraction=true`)
- Strips steady state markers: `K[ss] => K`
- Reorders integer multiplication
"""
function transform_calibration_eq(eq; to_subtraction::Bool=true)
    postwalk(eq) do x
        # Convert equality to subtraction (if requested)
        if to_subtraction && @capture(x, lhs_ = rhs_)
            return Expr(:call, :-, lhs, rhs)
        end
        # Strip steady state markers
        if @capture(x, name_[timing_]) && is_steady_state_marker(timing)
            return name
        end
        # Reorder integer multiplication
        x = reorder_int_mult(x)
        # Unblock
        if x isa Expr
            return unblock(x)
        end
        return x
    end
end

# =============================================================================
# Parameter definition parsing helpers
# =============================================================================

"""
Check if expression is a pipe call `a | b` and return (value, param) or nothing.
Note: @capture doesn't work well with | operator due to precedence, so we check manually.
"""
function match_pipe_expr(ex::Expr)
    if ex.head == :call && length(ex.args) >= 3 && ex.args[1] == :|
        return (ex.args[2], ex.args[3])  # (value, param)
    end
    return nothing
end
match_pipe_expr(::Any) = nothing

"""
Check if expression is a leading pipe calibration: `| param = target = value`.
Returns (calib_param, target_expr, rhs) or nothing.
"""
function match_leading_pipe_calibration(x::Expr)
    # Pattern: (| param = target) = rhs
    if x.head == :(=) && x.args[1] isa Expr && x.args[1].head == :call && 
       length(x.args[1].args) >= 3 && x.args[1].args[1] == :|
        calib_param = x.args[1].args[2]
        target_expr = x.args[1].args[3]
        rhs = x.args[2]
        if calib_param isa Symbol
            return (calib_param, target_expr, rhs)
        end
    end
    return nothing
end

"""
Check if expression is a bounds constraint: `0 < x < 1`, `x > 0`, etc.
"""
is_bounds_expr(x::Expr) = x.head == :comparison || (x.head == :call && x.args[1] ∈ [:<, :>, :<=, :>=])

# =============================================================================
# Bounds parsing
# =============================================================================

"""
Parse comparison bounds like `0 < x < 1`, `0 <= x < 1`, etc.
Updates the bounds dict in place.
"""
function parse_comparison_bound!(bounds::Dict{Symbol,Tuple{Float64,Float64}}, x::Expr)
    x.head == :comparison || return
    length(x.args) >= 5 || return
    
    lo_val = x.args[1]
    lo_op = x.args[2]
    param = x.args[3]
    hi_op = x.args[4]
    hi_val = x.args[5]
    
    param isa Symbol || return
    
    # Determine if bounds are strict (<) or inclusive (<=)
    lo_strict = lo_op == :(<) || lo_op == :(>)
    hi_strict = hi_op == :(<) || hi_op == :(>)
    
    # Handle reversed comparisons (> and >=)
    if lo_op == :(>) || lo_op == :(>=)
        lo_val, hi_val = hi_val, lo_val
        lo_strict, hi_strict = hi_strict, lo_strict
    end
    
    # Apply epsilon for strict bounds
    lb = lo_strict ? lo_val + SMALL_EPS : lo_val
    ub = hi_strict ? hi_val - SMALL_EPS : hi_val
    
    # Merge with existing bounds
    if haskey(bounds, param)
        bounds[param] = (max(bounds[param][1], lb), min(bounds[param][2], ub))
    else
        bounds[param] = (lb, ub)
    end
end

"""
Parse single-sided bounds like `x < 1`, `x > 0`, `x <= 1`, `x >= 0`.
Updates the bounds dict in place.
"""
function parse_call_bound!(bounds::Dict{Symbol,Tuple{Float64,Float64}}, x::Expr)
    x.head == :call || return
    length(x.args) >= 3 || return
    
    op = x.args[1]
    op ∈ [:(<), :(>), :(<=), :(>=)] || return
    
    lhs = x.args[2]
    rhs = x.args[3]
    
    # Determine which side has the symbol
    if lhs isa Symbol
        param = lhs
        val = rhs
        # param op val: e.g., x < 1 means upper bound
        if op == :(<)
            lb, ub = -BOUND_INFINITY + rand(), val - SMALL_EPS
        elseif op == :(<=)
            lb, ub = -BOUND_INFINITY + rand(), val
        elseif op == :(>)
            lb, ub = val + SMALL_EPS, BOUND_INFINITY + rand()
        elseif op == :(>=)
            lb, ub = val, BOUND_INFINITY + rand()
        else
            return
        end
    elseif rhs isa Symbol
        param = rhs
        val = lhs
        # val op param: e.g., 0 < x means lower bound
        if op == :(<)
            lb, ub = val + SMALL_EPS, BOUND_INFINITY + rand()
        elseif op == :(<=)
            lb, ub = val, BOUND_INFINITY + rand()
        elseif op == :(>)
            lb, ub = -BOUND_INFINITY + rand(), val - SMALL_EPS
        elseif op == :(>=)
            lb, ub = -BOUND_INFINITY + rand(), val
        else
            return
        end
    else
        return
    end
    
    # Merge with existing bounds
    if haskey(bounds, param)
        bounds[param] = (max(bounds[param][1], lb), min(bounds[param][2], ub))
    else
        bounds[param] = (lb, ub)
    end
end

"""
Parse a list of bound expressions and return a bounds dictionary.
"""
function parse_bounds(bounded_vars::Vector)
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    
    for bound in bounded_vars
        postwalk(x -> begin
            if x isa Expr
                if x.head == :comparison
                    parse_comparison_bound!(bounds, x)
                elseif x.head == :call
                    parse_call_bound!(bounds, x)
                end
            end
            x
        end, bound)
    end
    
    return bounds
end

# =============================================================================
# Helper functions for auxiliary variable creation (leads/lags > 1)
# =============================================================================

"""
Create auxiliary equations for variables with lead > 1.
Returns the symbol to use in the transformed expression.
"""
function create_lead_aux!(name::Symbol, k::Int, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
    name_str = string(name)
    
    # Create auxiliary dynamic equations for k > 2
    while k > 2
        aux_sym = Symbol(name_str * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎")
        if aux_sym ∈ aux_vars_created
            break
        end
        push!(aux_vars_created, aux_sym)
        aux_next = Symbol(name_str * "ᴸ⁽" * super(string(abs(k - 2))) * "⁾₍₁₎")
        push!(dyn_equations, Expr(:call, :-, aux_sym, aux_next))
        push!(dyn_eq_aux_ind, length(dyn_equations))
        k -= 1
    end
    
    # Create the final auxiliary equation connecting to x₍₁₎
    aux_sym = Symbol(name_str * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₀₎")
    if aux_sym ∉ aux_vars_created && k > 1
        push!(aux_vars_created, aux_sym)
        base_future = Symbol(name_str * "₍₁₎")
        push!(dyn_equations, Expr(:call, :-, aux_sym, base_future))
        push!(dyn_eq_aux_ind, length(dyn_equations))
    end
end

"""
Create auxiliary equations for variables with lag < -1.
Returns the symbol to use in the transformed expression.
"""
function create_lag_aux!(name::Symbol, k::Int, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
    name_str = string(name)
    
    # Create auxiliary dynamic equations for k < -2
    while k < -2
        aux_sym = Symbol(name_str * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎")
        if aux_sym ∈ aux_vars_created
            break
        end
        push!(aux_vars_created, aux_sym)
        aux_prev = Symbol(name_str * "ᴸ⁽⁻" * super(string(abs(k + 2))) * "⁾₍₋₁₎")
        push!(dyn_equations, Expr(:call, :-, aux_sym, aux_prev))
        push!(dyn_eq_aux_ind, length(dyn_equations))
        k += 1
    end
    
    # Create the final auxiliary equation connecting to x₍₋₁₎
    aux_sym = Symbol(name_str * "ᴸ⁽⁻" * super(string(abs(k + 1))) * "⁾₍₀₎")
    if aux_sym ∉ aux_vars_created && k < -1
        push!(aux_vars_created, aux_sym)
        base_past = Symbol(name_str * "₍₋₁₎")
        push!(dyn_equations, Expr(:call, :-, aux_sym, base_past))
        push!(dyn_eq_aux_ind, length(dyn_equations))
    end
end

"""
Create auxiliary equations for exogenous variables with lead/lag, including the x₍₀₎ = x₍ₓ₎ equation.
"""
function create_exo_aux!(name::Symbol, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
    name_str = string(name)
    present_sym = Symbol(name_str * "₍₀₎")
    if present_sym ∉ aux_vars_created
        push!(aux_vars_created, present_sym)
        exo_sym = Symbol(name_str * "₍ₓ₎")
        push!(dyn_equations, Expr(:call, :-, present_sym, exo_sym))
        push!(dyn_eq_aux_ind, length(dyn_equations))
    end
end

"""
Transform a variable reference expression to its dynamic form.
Handles time indices: [0], [1], [-1], [ss], [x], [x+k], [x-k], and leads/lags > 1.
"""
function transform_var_ref(x::Expr, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
    # Use @capture to match var[timing] pattern
    @capture(x, name_[timing_]) || return x
    name isa Symbol || return x
    name_str = string(name)
    
    # Case 1: Simple exogenous marker (x, ex, exo, exogenous)
    if is_exo_marker(timing)
        return Symbol(name_str * "₍ₓ₎")
    end
    
    # Case 2: Steady state marker
    if is_steady_state_marker(timing)
        return Symbol(name_str * "₍ₛₛ₎")
    end
    
    # Case 3: Exogenous with offset (x + k or x - k)
    if is_exo_with_offset(timing)
        if timing isa Expr && timing.head == :call
            op = timing.args[1]
            offset = timing.args[3]
            
            if op == :(+)  # Lead (x + k)
                k = offset
                create_lead_aux!(name, k, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
                create_exo_aux!(name, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
                
                if k > 1
                    return Symbol(name_str * "ᴸ⁽" * super(string(abs(k - 1))) * "⁾₍₁₎")
                else
                    return Symbol(name_str * "₍₁₎")
                end
                
            elseif op == :(-)  # Lag (x - k)
                k = -offset
                create_lag_aux!(name, k, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
                create_exo_aux!(name, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
                
                if k < -1
                    return Symbol(name_str * "ᴸ⁽⁻" * super(string(abs(offset - 1))) * "⁾₍₋₁₎")
                else
                    return Symbol(name_str * "₍₋₁₎")
                end
            end
        end
        return name  # Fallback
    end
    
    # Case 4: Integer time index
    if timing isa Int
        if timing > 1  # Lead > 1
            create_lead_aux!(name, timing, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
            return Symbol(name_str * "ᴸ⁽" * super(string(abs(timing - 1))) * "⁾₍₁₎")
            
        elseif timing >= 0 && timing <= 1  # Present or future (0 or 1)
            return Symbol(name_str * "₍" * sub(string(timing)) * "₎")
            
        elseif timing >= -1 && timing < 0  # Past (-1)
            return Symbol(name_str * "₍₋" * sub(string(timing)) * "₎")
            
        elseif timing < -1  # Lag < -1
            create_lag_aux!(name, timing, aux_vars_created, dyn_equations, dyn_eq_aux_ind)
            return Symbol(name_str * "ᴸ⁽⁻" * super(string(abs(timing + 1))) * "⁾₍₋₁₎")
        end
    end
    
    # Fallback: return the variable name
    return name
end

"""
Transform an expression for steady-state equations.
Sets shocks to zero, removes time indices, and reorders integer multiplication.
"""
function transform_to_ss(x)
    x isa Expr || return x
    
    # Convert = to subtraction
    if x.head == :(=)
        return Expr(:call, :-, x.args[1], x.args[2])
    end
    
    # Handle variable references
    if x.head == :ref
        # Exogenous variables become 0 in steady state
        if length(x.args) >= 2 && occursin(r"^(x|ex|exo|exogenous){1}"i, string(x.args[2]))
            return 0
        end
        # Other variables: strip time index
        return x.args[1]
    end
    
    # Reorder integer multiplication: 2*beta => beta*2 (needed for sympy)
    if x.head == :call && x.args[1] == :*
        if length(x.args) >= 3 && x.args[2] isa Int && !(x.args[3] isa Int)
            return Expr(:call, :*, x.args[3:end]..., x.args[2])
        end
    end
    
    return x
end

# =============================================================================
# Helper functions for nonnegativity/bounded constraint handling
# =============================================================================

"""
Update bounds for a symbol, merging with existing bounds if present.
"""
function update_bounds!(bounds::Dict{Symbol,Tuple{Float64,Float64}}, sym::Symbol, lb::Float64, ub::Float64)
    if haskey(bounds, sym)
        bounds[sym] = (max(bounds[sym][1], lb), min(bounds[sym][2], ub))
    else
        bounds[sym] = (lb, ub)
    end
end

"""
Create an auxiliary variable for a bounded expression and return the replacement expression.
"""
function create_bounded_aux!(
    arg_expr::Expr,
    lb::Float64,
    ub::Float64,
    bounds,
    ss_and_aux_equations,
    ss_equations_with_aux_variables,
    ➕_vars,
    unique_➕_eqs,
    precompile::Bool,
)
    replacement = precompile ? arg_expr : simplify(arg_expr)
    
    # Check if it's just a constant
    if replacement isa Int
        return replacement
    end
    
    # Check if we already created an auxiliary for this expression
    if haskey(unique_➕_eqs, arg_expr)
        return unique_➕_eqs[arg_expr]
    end
    
    # Create new auxiliary variable
    aux_name = Symbol("➕" * sub(string(length(➕_vars) + 1)))
    
    # Add the auxiliary equation: aux[0] - expression = 0
    push!(ss_and_aux_equations, Expr(:call, :-, Expr(:ref, aux_name, 0), arg_expr))
    
    # Set bounds for the auxiliary variable
    update_bounds!(bounds, aux_name, lb, ub)
    
    push!(ss_equations_with_aux_variables, length(ss_and_aux_equations))
    push!(➕_vars, aux_name)
    
    # Create the replacement reference
    replacement = Expr(:ref, aux_name, 0)
    unique_➕_eqs[arg_expr] = replacement
    
    return replacement
end

"""
Handle bounded function calls (log, exp, ^, norminvcdf, erfcinv, etc.).
Sets bounds on arguments and creates auxiliary variables for complex expressions.
Returns the (potentially modified) expression.
"""
function handle_bounded_call(
    x::Expr,
    bounds,
    ss_and_aux_equations,
    ss_equations_with_aux_variables,
    ➕_vars,
    unique_➕_eqs,
    precompile::Bool,
)
    # Must be a function call
    x.head == :call || return x
    length(x.args) >= 2 || return x
    
    func = x.args[1]
    arg = x.args[2]
    
    # Skip if argument is a Float64 literal
    arg isa Float64 && return x
    
    # Special handling for power operator: only applies when exponent is non-integer
    if func == :^
        length(x.args) >= 3 || return x
        exponent = x.args[3]
        exponent isa Int && return x  # Integer exponent doesn't require bounds
        
        lb, ub = BOUNDED_FUNCTIONS[:^]
        
        # Handle symbol argument (parameter)
        if arg isa Symbol
            update_bounds!(bounds, arg, lb, ub)
            return x
        end
        
        # Handle variable reference
        if arg isa Expr && arg.head == :ref && arg.args[1] isa Symbol
            update_bounds!(bounds, arg.args[1], lb, ub)
            return x
        end
        
        # Handle complex expression - create auxiliary
        if arg isa Expr && arg.head == :call
            replacement = create_bounded_aux!(arg, lb, ub, bounds, ss_and_aux_equations, ss_equations_with_aux_variables, ➕_vars, unique_➕_eqs, precompile)
            return :($(replacement) ^ $(exponent))
        end
        
        return x
    end
    
    # Handle other bounded functions (log, exp, norminvcdf, erfcinv, etc.)
    func_key = func ∈ [:norminv, :qnorm] ? :norminvcdf : func
    haskey(BOUNDED_FUNCTIONS, func_key) || return x
    
    lb, ub = BOUNDED_FUNCTIONS[func_key]
    
    # Handle symbol argument (parameter)
    if arg isa Symbol
        update_bounds!(bounds, arg, lb, ub)
        return x
    end
    
    # Handle variable reference
    if arg isa Expr && arg.head == :ref && arg.args[1] isa Symbol
        update_bounds!(bounds, arg.args[1], lb, ub)
        return x
    end
    
    # Handle complex expression - create auxiliary
    if arg isa Expr && arg.head == :call
        replacement = create_bounded_aux!(arg, lb, ub, bounds, ss_and_aux_equations, ss_equations_with_aux_variables, ➕_vars, unique_➕_eqs, precompile)
        return Expr(:call, func, replacement)
    end
    
    return x
end

function parse_parameter_definitions_raw(parameter_block::Expr)
    calib_equations = Expr[]
    calib_equations_original = Expr[]
    calib_equations_no_var = Expr[]
    calib_values_no_var = Any[]
    calib_parameters_no_var = Symbol[]
    calib_eq_parameters = Symbol[]
    calib_equations_list = Expr[]

    ss_calib_list = []
    par_calib_list = []

    calib_equations_no_var_list = Expr[]

    ss_no_var_calib_list = []
    par_no_var_calib_list = []
    par_no_var_calib_rhs_list = []

    calib_parameters = Symbol[]
    calib_values = Float64[]

    par_defined_more_than_once = Set{Symbol}()

    bounded_vars = []

    parameter_definitions = replace_indices(parameter_block)

    # Helper to check for duplicate parameter definitions
    function check_duplicate!(param::Symbol)
        all_defined = union(calib_parameters, calib_parameters_no_var, calib_eq_parameters)
        if param ∈ all_defined
            push!(par_defined_more_than_once, param)
        end
    end

    # Helper to add a calibration equation
    function add_calibration_eq!(calib_param::Symbol, lhs_expr, rhs_expr)
        check_duplicate!(calib_param)
        push!(calib_eq_parameters, calib_param)
        push!(calib_equations, Expr(:(=), lhs_expr, unblock(rhs_expr)))
        push!(calib_equations_original, Expr(:(=), lhs_expr, Expr(:call, :|, unblock(rhs_expr), calib_param)))
    end

    # First pass: parse simple assignments, leading pipe calibrations, and bounds
    postwalk(parameter_definitions) do x
        if !(x isa Expr)
            return x
        end
        
        # Bounds: 0 < x < 1, x > 0, etc.
        if is_bounds_expr(x)
            push!(bounded_vars, x)
            return x
        end
        
        # Leading pipe calibration: | param = target = value
        leading_pipe = match_leading_pipe_calibration(x)
        if leading_pipe !== nothing
            calib_param, target_expr, rhs = leading_pipe
            add_calibration_eq!(calib_param, target_expr, rhs)
            return x
        end
        
        # Simple assignment: param = value
        if x.head == :(=) && x.args[1] isa Symbol
            param = x.args[1]
            val = x.args[2]
            
            if val isa Int || val isa Float64
                check_duplicate!(param)
                push!(calib_values, val)
                push!(calib_parameters, param)
            elseif val isa Symbol
                check_duplicate!(param)
                push!(calib_values_no_var, val)
                push!(calib_parameters_no_var, param)
            elseif val isa Expr && match_pipe_expr(val) === nothing
                # Expression without pipe (trailing pipe handled in second pass)
                check_duplicate!(param)
                push!(calib_values_no_var, unblock(val))
                push!(calib_parameters_no_var, param)
            end
        end
        x
    end

    # Second pass: handle trailing pipe calibrations
    postwalk(parameter_definitions) do x
        if !(x isa Expr) || x.head != :(=)
            return x
        end
        
        # Skip numeric values (already processed)
        if x.args[2] isa Int || x.args[2] isa Float64
            return x
        end
        
        lhs = x.args[1]
        rhs = x.args[2]
        
        # Trailing pipe: lhs = expr | calib_param
        if rhs isa Expr
            pipe_match = match_pipe_expr(rhs)
            if pipe_match !== nothing
                target_expr, calib_param = pipe_match
                if calib_param isa Symbol
                    add_calibration_eq!(calib_param, lhs, target_expr)
                    return x
                end
            end
            
            # Block expression with embedded pipe (may have LineNumberNodes)
            if rhs.head == :block
                inner_idx = findfirst(a -> a isa Expr, rhs.args)
                if inner_idx !== nothing
                    inner = rhs.args[inner_idx]
                    inner_pipe = match_pipe_expr(inner)
                    if inner_pipe !== nothing
                        inner_expr, calib_param = inner_pipe
                        if calib_param isa Symbol
                            # Skip if already handled as leading pipe
                            is_leading_pipe = lhs isa Expr && lhs.head == :call && lhs.args[1] == :|
                            if !is_leading_pipe
                                add_calibration_eq!(calib_param, lhs, inner_expr)
                            end
                            return x
                        end
                    elseif !(lhs isa Expr && lhs.head == :call && lhs.args[1] == :|)
                        @warn "Invalid parameter input ignored: " * repr(x)
                    end
                end
            end
        end
        x
    end

    @assert length(par_defined_more_than_once) == 0 "Parameters can only be defined once. This is not the case for: " * repr([par_defined_more_than_once...])

    # Check that no parameter names conflict with SymPyWorkspace reserved names
    all_params = union(calib_parameters, calib_parameters_no_var, calib_eq_parameters)
    reserved_conflicts_params = intersect(all_params, SYMPYWORKSPACE_RESERVED_NAMES)
    @assert length(reserved_conflicts_params) == 0 "The following parameter names are reserved and cannot be used: " * repr(sort([reserved_conflicts_params...]))

    # Evaluate expressions that can be computed to Float64: log(1/3), exp(2), etc.
    for (i, v) in enumerate(calib_values_no_var)
        out = try eval(v) catch e end
        if out isa Float64
            push!(calib_parameters, calib_parameters_no_var[i])
            push!(calib_values, out)
        else
            push!(calib_equations_no_var, Expr(:(=),calib_parameters_no_var[i], calib_values_no_var[i]))
        end
    end

    calib_parameters_no_var = setdiff(calib_parameters_no_var, calib_parameters)

    for (i, cal_eq) in enumerate(calib_equations)
        # Extract SS variables and parameters using @capture-based helpers
        ss_tmp = extract_ss_vars(cal_eq)
        par_tmp = extract_parameters(cal_eq, ss_tmp)

        push!(ss_calib_list, ss_tmp)
        push!(par_calib_list, par_tmp)
        
        prs_ex = transform_calibration_eq(cal_eq)
        push!(calib_equations_list, unblock(prs_ex))
    end

    # parse calibration equations without a variable present: eta = Pi_bar /2 (Pi_bar is also a parameter)
    for (i, cal_eq) in enumerate(calib_equations_no_var)
        # Extract SS variables and parameters using @capture-based helpers
        ss_tmp = extract_ss_vars(cal_eq)
        par_tmp = extract_parameters(cal_eq, ss_tmp)

        push!(ss_no_var_calib_list, ss_tmp)
        push!(par_no_var_calib_list, setdiff(par_tmp, calib_parameters))
        push!(par_no_var_calib_rhs_list, intersect(par_tmp, calib_parameters))
        
        prs_ex = transform_calibration_eq(cal_eq; to_subtraction=false)
        push!(calib_equations_no_var_list, unblock(prs_ex))
    end

    # arrange calibration equations where they use parameters defined in parameters block so that they appear in right order (Pi_bar is defined before it is used later on: eta = Pi_bar / 2)
    if length(calib_equations_no_var_list) > 0
        incidence_matrix = fill(0,length(calib_parameters_no_var),length(calib_parameters_no_var))
        
        for i in 1:length(calib_parameters_no_var)
            for k in 1:length(calib_parameters_no_var)
                incidence_matrix[i,k] = collect(calib_parameters_no_var)[i] ∈ collect(par_no_var_calib_list)[k]
            end
        end
        
        Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(incidence_matrix))
        
        @assert length(Q) == n_blocks "Check the parameter definitions. They are either incomplete or have more than only the defined parameter on the LHS."
        
        calib_equations_no_var_list = calib_equations_no_var_list[Q]
    end

    # parse bounds
    bounds = parse_bounds(bounded_vars)

    return (
        calib_parameters = calib_parameters,
        calib_values = calib_values,
        calib_eq_parameters = calib_eq_parameters,
        calib_equations_list = calib_equations_list,
        calib_equations_original = calib_equations_original,
        ss_calib_list = ss_calib_list,
        par_calib_list = par_calib_list,
        calib_parameters_no_var = calib_parameters_no_var,
        calib_equations_no_var_list = calib_equations_no_var_list,
        par_no_var_calib_list = par_no_var_calib_list,
        par_no_var_calib_rhs_list = par_no_var_calib_rhs_list,
        bounds = bounds,
    )
end

function build_guess_dict(guess)
    guess_dict = Dict{Symbol, Float64}()
    if isa(guess, Dict{String, <:Real})
        for (key, value) in guess
            if key isa String
                key = replace_indices(key)
            end
            guess_dict[replace_indices(key)] = value
        end
    elseif isa(guess, Dict{Symbol, <:Real})
        guess_dict = guess
    end
    return guess_dict
end

"""
Reconstruct a parameter block expression from model state.
Optionally override calibration equations (original form) for updates.

When calibration equations change, this function handles the case where:
- A parameter switches from calibrated to having a value assignment
- A parameter switches from having a value assignment to being calibrated

The function extracts which parameters are calibrated from the calibration equations,
and adjusts the parameter value assignments accordingly.
"""
function reconstruct_parameter_block(
    𝓂::ℳ;
    calibration_original_override::Union{Nothing, Vector{Expr}} = nothing,
)
    lines = Expr[]

    # Calibration equations (original form)
    calibration_original = calibration_original_override === nothing ? 𝓂.equations.calibration_original : calibration_original_override

    # Extract calibrated parameters from calibration equations
    # These are the parameters after the | symbol
    new_calib_params = Set{Symbol}()
    for eq in calibration_original
        # Look for pattern: lhs = rhs | param or | param = rhs
        calib_param = extract_calibrated_parameter(eq)
        if calib_param !== nothing
            push!(new_calib_params, calib_param)
        end
    end

    # Original calibrated parameters
    old_calib_params = Set{Symbol}(𝓂.equations.calibration_parameters)

    # Parameters that are now calibrated but weren't before - exclude them from value assignments
    params_becoming_calibrated = setdiff(new_calib_params, old_calib_params)

    # Parameters that were calibrated but no longer are - add them with their SS value
    params_no_longer_calibrated = setdiff(old_calib_params, new_calib_params)

    # Parameters with values - exclude those that are now calibrated
    params = 𝓂.constants.post_complete_parameters.parameters
    values = 𝓂.parameter_values
    for (p, v) in zip(params, values)
        if !isnan(v) && p ∉ params_becoming_calibrated
            push!(lines, Expr(:(=), p, v))
        end
    end

    # Add parameters that were calibrated but now need value assignments
    # Use their value from the non-stochastic steady state cache
    if !isempty(params_no_longer_calibrated)
        # The NSSS cache contains [variables..., calibrated_parameters...]
        n_vars = 𝓂.constants.post_model_macro.nVars
        old_calib_param_list = 𝓂.equations.calibration_parameters
        for param in params_no_longer_calibrated
            idx = findfirst(==(param), old_calib_param_list)
            if idx !== nothing && !isempty(𝓂.caches.non_stochastic_steady_state)
                val = 𝓂.caches.non_stochastic_steady_state[n_vars + idx]
                push!(lines, Expr(:(=), param, val))
            end
        end
    end

    # Parameters defined as functions of other parameters
    for eq in 𝓂.equations.calibration_no_var
        push!(lines, eq)
    end

    # Calibration equations (original form)
    for eq in calibration_original
        push!(lines, eq)
    end

    # Bounds
    for (param, (lo, hi)) in 𝓂.constants.post_parameters_macro.bounds
        push!(lines, Expr(:comparison, lo, :(<), param, :(<), hi))
    end

    return Expr(:block, lines...)
end

"""
Extract the calibrated parameter from a calibration equation.
Handles formats like:
- `:(k[ss] / (4 * q[ss]) = 1.5 | α)` returns `:α`
- `:(| α = k[ss] / (4 * q[ss]) - 1.5)` returns `:α`
Returns `nothing` if no calibrated parameter found.
"""
function extract_calibrated_parameter(eq::Expr)::Union{Symbol, Nothing}
    result = Ref{Union{Symbol, Nothing}}(nothing)
    
    postwalk(eq) do x
        if x isa Expr && x.head == :call && x.args[1] == :|
            # Pattern: Expr(:call, :|, rhs, param)
            if length(x.args) >= 3 && x.args[end] isa Symbol
                result[] = x.args[end]
            end
        elseif x isa Expr && x.head == :| 
            # Pattern: Expr(:|, param, ...)
            if length(x.args) >= 1 && x.args[1] isa Symbol
                result[] = x.args[1]
            end
        end
        x
    end
    
    return result[]
end

function process_parameter_definitions(
    parameter_block::Expr,
    post_model_macro::post_model_macro
)
    parsed = parse_parameter_definitions_raw(parameter_block)

    calib_parameters = parsed.calib_parameters
    calib_values = parsed.calib_values
    calib_eq_parameters = parsed.calib_eq_parameters
    calib_equations_list = parsed.calib_equations_list
    calib_equations_original = parsed.calib_equations_original
    ss_calib_list = parsed.ss_calib_list
    par_calib_list = parsed.par_calib_list
    calib_parameters_no_var = parsed.calib_parameters_no_var
    calib_equations_no_var_list = parsed.calib_equations_no_var_list
    par_no_var_calib_list = parsed.par_no_var_calib_list
    par_no_var_calib_rhs_list = parsed.par_no_var_calib_rhs_list

    calib_eq_parameters_raw = calib_eq_parameters
    ss_calib_list_raw = ss_calib_list
    par_calib_list_raw = par_calib_list
    all_names = [post_model_macro.parameters_in_equations; post_model_macro.var]

    if any(contains.(string.(post_model_macro.var), "ᵒᵇᶜ"))
        push!(calib_parameters, :activeᵒᵇᶜshocks)
        push!(calib_values, 0)
    end

    calib_parameters, calib_values = expand_indices(calib_parameters, calib_values, all_names)
    calib_eq_parameters, calib_equations_list, ss_calib_list, par_calib_list = expand_calibration_equations(calib_eq_parameters_raw, calib_equations_list, ss_calib_list_raw, par_calib_list_raw, all_names)
    _, calib_equations_original, _, _ = expand_calibration_equations(calib_eq_parameters_raw, calib_equations_original, ss_calib_list_raw, par_calib_list_raw, all_names)
    calib_parameters_no_var, calib_equations_no_var_list = expand_indices(calib_parameters_no_var, calib_equations_no_var_list, all_names)

    # Calculate missing parameters instead of asserting
    # Include parameters from:
    # 1. par_calib_list - parameters used in calibration equations (e.g., K_ss in "K[ss] = K_ss | beta")
    # 2. parameters_in_equations - parameters used in model equations
    # 3. par_no_var_calib_list - parameters used in parameter definitions (e.g., rho{H}{H} in "rho{F}{F} = rho{H}{H}")
    # Subtract:
    # 1. calib_parameters - parameters with explicit values (e.g., "α = 0.5")
    # 2. calib_parameters_no_var - parameters defined as functions of other parameters (e.g., "α = alpha_param")
    # 3. calib_eq_parameters - parameters determined by calibration equations (e.g., "beta" in "K[ss] = K_ss | beta")
    # Start with directly required parameters
    all_required_params = union(
        reduce(union, par_calib_list, init = Set{Symbol}()),
        reduce(union, par_no_var_calib_rhs_list, init = Set{Symbol}()),
        Set{Symbol}(post_model_macro.parameters_in_equations)
    )

    # Add parameters from parameter definitions, but only if the target parameter is needed
    # This handles the case where parameter X = f(Y, Z) but X is not used in the model.
    # In that case, Y and Z should not be required either.
    # We need to check if target is in all_required_params OR in calib_eq_parameters (parameters used in calibration equations)
    par_no_var_calib_filtered = mapreduce(i -> par_no_var_calib_list[i], union, findall(target_param -> target_param ∈ all_required_params, calib_parameters_no_var), init = Set{Symbol}())

    all_required_params = union(all_required_params, par_no_var_calib_filtered)

    defined_params = union(
        Set{Symbol}(calib_parameters),
        Set{Symbol}(calib_parameters_no_var),
        Set{Symbol}(calib_eq_parameters)
    )

    ignored_params = collect(setdiff(defined_params, all_required_params))

    if !isempty(ignored_params) @warn "Parameters not part of the model are ignored: $ignored_params" end

    missing_params_unsorted = collect(setdiff(all_required_params, defined_params))
    missing_params = sort(missing_params_unsorted)

    bounds_dict = Dict{Symbol,Tuple{Float64,Float64}}()
    for (k,v) in parsed.bounds
        bounds_dict[k] = haskey(bounds_dict, k) ? (max(bounds_dict[k][1], v[1]), min(bounds_dict[k][2], v[2])) : (v[1], v[2])
    end

    invalid_bounds = Symbol[]

    for (k,v) in bounds_dict
        if v[1] >= v[2]
            push!(invalid_bounds, k)
        end
    end

    @assert isempty(invalid_bounds) "Invalid bounds: " * repr(invalid_bounds)

    # Keep calib_parameters in declaration order, append missing_params at end
    # This preserves declaration order for estimation and method of moments
    all_params = vcat(calib_parameters, missing_params)
    all_values = vcat(calib_values, fill(NaN, length(missing_params)))

    defined_params_idx = indexin(setdiff(intersect(all_params, defined_params), ignored_params), collect(all_params))

    return (
        equations = (
            calibration = calib_equations_list,
            calibration_no_var = calib_equations_no_var_list,
            calibration_parameters = calib_eq_parameters,
            calibration_original = calib_equations_original,
        ),
        calib_parameters_no_var = calib_parameters_no_var,
        ss_calib_list = ss_calib_list,
        par_calib_list = par_calib_list,
        bounds = bounds_dict,
        parameters = all_params[defined_params_idx],
        missing_parameters = missing_params,
        parameter_values = all_values[defined_params_idx],
    )
end
