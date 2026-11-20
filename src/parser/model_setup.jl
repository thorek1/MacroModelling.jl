"""
    solve_symbolically(equation, variable; positive = false)

Return the closed-form solutions of `equation = 0` for `variable` (empty if no
closed form is found).
"""
function solve_symbolically(equation::SymbolicExpression, variable::Symbol; positive::Bool = false)::Vector{SymbolicExpression}
    solution = solve_for_variable(to_symbolic(equation), variable, positive = positive)
    solution === nothing && return SymbolicExpression[]
    return SymbolicExpression[from_symbolic(solution)]
end

"""
    solve_symbolically(equations, variables; positive_symbols = Set{Symbol}())

Return the closed-form solution of the system `equations = 0` for `variables`
as a dictionary, or `nothing` if no closed form is found.
"""
function solve_symbolically(equations::AbstractVector, variables::AbstractVector; positive_symbols::Set{Symbol} = Set{Symbol}())::Union{Nothing,Dict{Symbol, SymbolicExpression}}
    solutions = solve_system_for_variables(SymbolicNode[to_symbolic(e) for e in equations], Symbol[Symbol(v) for v in variables], positive = positive_symbols)
    solutions === nothing && return nothing
    return Dict{Symbol, SymbolicExpression}(v => from_symbolic(sol) for (v, sol) in solutions)
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


function get_relevant_steady_states(𝓂::ℳ, 
                                    algorithm::Symbol;
                                    opts::CalculationOptions = merge_calculation_options())::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    full_NSSS = ms.full_NSSS_display

    relevant_SS = get_steady_state(𝓂, algorithm = algorithm, 
                                    stochastic = algorithm != :first_order,
                                    return_variables_only = true, 
                                    derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    reference_steady_state = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_SS(s) for s in full_NSSS]

    relevant_NSSS = get_steady_state(𝓂, algorithm = :first_order, 
                                    stochastic = false, 
                                    return_variables_only = true, 
                                    derivatives = false, 
                                    verbose = opts.verbose,
                                    tol = opts.tol,
                                    quadratic_matrix_equation_algorithm = opts.quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm = [opts.sylvester_algorithm², opts.sylvester_algorithm³])

    NSSS = [s ∈ 𝓂.constants.post_model_macro.exo_present ? 0.0 : relevant_NSSS(s) for s in full_NSSS]

    SSS_delta = NSSS - reference_steady_state

    return reference_steady_state, NSSS, SSS_delta
end

Max = max
Min = min

function simplify(ex::Expr)::SymbolicExpression
    return simplify_symbolic(convert_to_ss_equation(ex))
end

function convert_to_ss_equation(eq::Expr)::Expr
    postwalk(x -> 
        x isa Expr ? 
            x.head == :(=) ? 
                Expr(:call,:(-),x.args[1],x.args[2]) : #convert = to -
                    x.head == :ref ?
                        occursin(r"^(x|ex|exo|exogenous){1}"i,string(x.args[2])) ? 0 :
                x.args[1] : 
            x.head == :call ?
                x.args[1] == :* ?
                    x.args[2] isa Int ?
                        x.args[3] isa Int ?
                            x :
                        :($(x.args[3]) * $(x.args[2])) : # avoid 2X syntax
                    x :
                x :
            unblock(x) : 
        x,
    eq)
end


replace_indices(x::Symbol) = x

replace_indices_special(x::Symbol) = x

replace_indices(x::String) = Symbol(replace(x, "{" => "◖", "}" => "◗"))

replace_indices_in_symbol(x::Symbol) = replace(string(x), "◖" => "{", "◗" => "}")

function replace_indices(exxpr::Expr)::Union{Expr,Symbol}
    postwalk(x -> begin
        x isa Symbol ?
            replace_indices(string(x)) :
        x isa Expr ?
            x.head == :curly ?
                Symbol(string(x.args[1]) * "◖" * string(x.args[2]) * "◗") :
            x :
        x
    end, exxpr)
end

function replace_indices_special(exxpr::Expr)::Union{Expr,Symbol}
    postwalk(x -> begin
        x isa Symbol ?
            replace_indices(string(x)) :
        x isa Expr ?
            x.head == :curly ?
                Symbol(string(x.args[1]) * "◖" * string(x.args[2]) * "◗") :
            x.head == :call ?
                x.args[1] == :(*) ?
                    Symbol(string(x.args[2]), string(x.args[3])) :
                x :
            x :
        x
    end, exxpr)
end


function expand_steady_state(SS_and_pars::Vector{M}, ms::post_complete_parameters) where M
    X = ms.steady_state_expand_matrix
    return X * SS_and_pars
end



function create_symbols_eqs!(𝓂::ℳ)::symbolics
    symbols_in_dynamic_equations = reduce(union, get_symbols.(𝓂.equations.dynamic))

    symbols_in_dynamic_equations_wo_subscripts = Symbol.(replace.(string.(symbols_in_dynamic_equations), r"₍₋?(₀|₁|ₛₛ|ₓ)₎$"=>""))

    symbols_in_ss_equations = reduce(union,get_symbols.(𝓂.equations.steady_state_aux))

    symbols_in_equation = union(𝓂.constants.post_model_macro.parameters_in_equations, 
                                𝓂.constants.post_complete_parameters.parameters, 
                                𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters,
                                symbols_in_dynamic_equations,
                                symbols_in_dynamic_equations_wo_subscripts,
                                symbols_in_ss_equations)

    # symbols with a nonnegative lower bound are treated as positive when solving symbolically
    positive_symbols = Set{Symbol}()

    for symb in symbols_in_equation
        if symb isa Symbol && haskey(𝓂.constants.post_parameters_macro.bounds, symb) && 𝓂.constants.post_parameters_macro.bounds[symb][1] >= 0
            push!(positive_symbols, symb)
        end
    end

    to_symbol_sets(lists) = [Set{Symbol}(Symbol.(collect(x))) for x in lists]

    symbolics(
                SymbolicExpression[x for x in 𝓂.equations.steady_state_aux],
                to_symbol_sets(𝓂.constants.post_model_macro.var_present_list_aux_SS),
                to_symbol_sets(𝓂.constants.post_model_macro.var_past_list_aux_SS),
                to_symbol_sets(𝓂.constants.post_model_macro.var_future_list_aux_SS),
                to_symbol_sets(𝓂.constants.post_model_macro.ss_list_aux_SS),
                to_symbol_sets(𝓂.constants.post_model_macro.var_list_aux_SS),
                to_symbol_sets(𝓂.constants.post_model_macro.par_list_aux_SS),
                SymbolicExpression[x for x in 𝓂.equations.calibration],
                Symbol.(𝓂.equations.calibration_parameters),
                Set{Symbol}(Symbol.(collect(𝓂.constants.post_model_macro.vars_in_ss_equations))),
                to_symbol_sets(𝓂.constants.post_parameters_macro.ss_calib_list),
                to_symbol_sets(𝓂.constants.post_parameters_macro.par_calib_list),
                [Set{Symbol}() for _ in 1:length(𝓂.equations.steady_state_aux)],
                positive_symbols,
                )
end



function remove_redundant_SS_vars!(𝓂::ℳ, symbolics_data::symbolics; avoid_solve::Bool = false)
    ss_equations = symbolics_data.ss_equations

    # check variables which appear in two time periods. they might be redundant in steady state
    redundant_vars = intersect.(
        union.(
            intersect.(symbolics_data.var_future_list_aux_SS, symbolics_data.var_present_list_aux_SS),
            intersect.(symbolics_data.var_future_list_aux_SS, symbolics_data.var_past_list_aux_SS),
            intersect.(symbolics_data.var_present_list_aux_SS, symbolics_data.var_past_list_aux_SS),
            intersect.(symbolics_data.ss_list_aux_SS, symbolics_data.var_present_list_aux_SS),
            intersect.(symbolics_data.ss_list_aux_SS, symbolics_data.var_past_list_aux_SS),
            intersect.(symbolics_data.ss_list_aux_SS, symbolics_data.var_future_list_aux_SS)
        ),
    symbolics_data.var_list_aux_SS)

    redundant_idx = getindex(1:length(redundant_vars), (length.(redundant_vars) .> 0) .& (length.(symbolics_data.var_list_aux_SS) .> 1))
    for i in redundant_idx
        for var_to_solve_for in redundant_vars[i]
            if var_to_solve_for ∉ get_symbols(ss_equations[i])
                # variable already absent (e.g. previous redundancy rewrites removed it)
                push!(symbolics_data.var_redundant_list[i], var_to_solve_for)
                continue
            end

            if avoid_solve || count_ops(ss_equations[i]) > 15
                continue
            end

            # the variable is redundant if the equation does not determine it, e.g. because it only enters as a common factor
            if variable_is_redundant(to_symbolic(ss_equations[i]), var_to_solve_for)
                push!(symbolics_data.var_redundant_list[i], var_to_solve_for)
                ss_equations[i] = substitute_symbolic(ss_equations[i], var_to_solve_for, 1)
            end
        end
    end

end



function write_ss_check_function!(𝓂::ℳ;
                                    cse = true,
                                    skipzeros = true, 
                                    density_threshold::Float64 = .1,
                                    nnz_parallel_threshold::Int = 1000000,
                                    min_length::Int = 10000)
    unknowns = union(setdiff(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.constants.post_model_macro.➕_vars), 𝓂.equations.calibration_parameters)

    ss_equations = vcat(𝓂.equations.steady_state, 𝓂.equations.calibration)



    np = length(𝓂.constants.post_complete_parameters.parameters)
    nu = length(unknowns)
    # nc = length(𝓂.calibration_equations_no_var)

    𝔓 = symbolic_placeholders("𝔓", np)
    𝔘 = symbolic_placeholders("𝔘", nu)

    parameter_dict = Dict{Symbol, Symbol}()
    calib_vars = Symbol[]
    calib_expr = []


    for (i,v) in enumerate(𝓂.constants.post_complete_parameters.parameters)
        push!(parameter_dict, v => 𝔓[i])
    end

    for (i,v) in enumerate(unknowns)
        push!(parameter_dict, v => 𝔘[i])
    end

    for (i,v) in enumerate(𝓂.equations.calibration_no_var)
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end

    calib_replacements = Dict{Symbol, Union{Expr, Symbol, Number}}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict{Symbol, Union{Expr, Symbol, Number}}(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    ss_equations_sub = ss_equations |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> SymbolicNode[to_symbolic(e) for e in x]


    lennz = length(ss_equations_sub)


    func_exprs = build_symbolic_function(ss_equations_sub, [𝔓, 𝔘])


    𝓂.functions.NSSS_check = func_exprs

    # Ensure check_residual buffer is sized for the NSSS_check function
    nres = length(ss_equations)
    cr = 𝓂.workspaces.nsss_solver.check_residual
    if length(cr) != nres
        resize!(cr, nres)
        fill!(cr, 0.0)
    end


    # SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars))))), 𝓂.calibration_equations_parameters))

    # eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    # nx = length(𝓂.parameter_values)

    # np = length(SS_and_pars)

    nϵˢ = length(ss_equations)

    # nc = length(𝓂.calibration_equations_no_var)

    # Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    # ϵˢ = zeros(Symbolics.Num, nϵˢ)

    # calib_vals = zeros(Symbolics.Num, nc)

    # 𝓂.SS_calib_func(calib_vals, 𝔓)

    # 𝓂.functions.NSSS_check(ϵˢ, 𝔓, 𝔘, calib_vals)

    ∂SS_equations_∂parameters = symbolic_sparse_jacobian(ss_equations_sub, 𝔓) # nϵ x nx

    lennz = nnz(∂SS_equations_∂parameters)

    if (lennz / length(∂SS_equations_∂parameters) > density_threshold) || (length(∂SS_equations_∂parameters) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂parameters)
        buffer = zeros(Float64, size(∂SS_equations_∂parameters))
    else
        derivatives_mat = ∂SS_equations_∂parameters
        buffer = similar(∂SS_equations_∂parameters, Float64)
        buffer.nzval .= 0
    end

    
    func_exprs = build_symbolic_function(derivatives_mat, [𝔓, 𝔘])

    𝓂.caches.NSSS_∂equations_∂parameters = buffer
    𝓂.functions.NSSS_∂equations_∂parameters = func_exprs



    ∂SS_equations_∂SS_and_pars = symbolic_sparse_jacobian(ss_equations_sub, 𝔘) # nϵ x nx

    lennz = nnz(∂SS_equations_∂SS_and_pars)

    if (lennz / length(∂SS_equations_∂SS_and_pars) > density_threshold) || (length(∂SS_equations_∂SS_and_pars) < min_length)
        derivatives_mat = convert(Matrix, ∂SS_equations_∂SS_and_pars)
        buffer = zeros(Float64, size(∂SS_equations_∂SS_and_pars))
    else
        derivatives_mat = ∂SS_equations_∂SS_and_pars
        buffer = similar(∂SS_equations_∂SS_and_pars, Float64)
        buffer.nzval .= 0
    end


    func_exprs = build_symbolic_function(derivatives_mat, [𝔓, 𝔘])

    𝓂.caches.NSSS_∂equations_∂SS_and_pars = buffer
    𝓂.functions.NSSS_∂equations_∂SS_and_pars = func_exprs

    return nothing
end


function write_symbolic_derivatives!(𝓂::ℳ; perturbation_order::Int = 1, silent::Bool = false)
    start_time = time()

    if !silent
        if perturbation_order == 1
            print("Take symbolic derivatives up to first order:\t\t\t\t")
        elseif perturbation_order == 2
            print("Take symbolic derivatives up to second order:\t\t\t\t")
        elseif perturbation_order == 3
            print("Take symbolic derivatives up to third order:\t\t\t\t")
        end
    end

    write_auxiliary_indices!(𝓂)
    
    write_functions_mapping!(𝓂, perturbation_order)

    if !silent
        println(round(time() - start_time, digits = 3), " seconds")
    end

    return nothing
end


function steady_state_symbolic_mode_flags(ss_symbolic_mode::Symbol, precompile::Bool = false)
    precompile && (ss_symbolic_mode = :none)
    ss_symbolic_mode == :none && return true, false
    ss_symbolic_mode == :single_equation && return false, false
    ss_symbolic_mode == :full && return false, true
    error("Invalid ss_symbolic_mode $(ss_symbolic_mode). Expected :none, :single_equation, or :full.")
end

function set_up_steady_state_solver!(𝓂::ℳ; verbose::Bool, silent::Bool, ss_symbolic_mode::Symbol = :single_equation)
    avoid_solve, symbolic_enabled = steady_state_symbolic_mode_flags(ss_symbolic_mode, 𝓂.constants.post_parameters_macro.precompile)
    use_symbolics = !𝓂.constants.post_parameters_macro.precompile

    if use_symbolics
        start_time = time()

        if !silent print("Remove redundant variables in non-stochastic steady state problem:\t") end

        symbolics = create_symbols_eqs!(𝓂)

        remove_redundant_SS_vars!(𝓂, symbolics, avoid_solve = avoid_solve)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end

        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        write_ss_check_function!(𝓂)

        write_steady_state_solver_function!(𝓂, symbolic_enabled, symbolics, verbose = verbose, avoid_solve = avoid_solve)

        𝓂.equations.obc_violation = write_obc_violation_equations(𝓂)

        set_up_obc_violation_function!(𝓂)

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    else
        start_time = time()

        if !silent print("Set up non-stochastic steady state problem:\t\t\t\t") end

        write_ss_check_function!(𝓂)

        write_steady_state_solver_function!(𝓂, false, nothing, verbose = verbose, avoid_solve = avoid_solve)

        if !𝓂.constants.post_parameters_macro.precompile
            𝓂.equations.obc_violation = write_obc_violation_equations(𝓂)
            set_up_obc_violation_function!(𝓂)
        end

        if !silent println(round(time() - start_time, digits = 3), " seconds") end
    end

    return nothing
end


function take_nth_order_derivatives(
    dyn_equations::Vector{T},
    𝔙::Vector{Symbol},
    𝔓::Vector{Symbol},
    SS_mapping::Dict{Symbol, T},
    nps::Int,
    nxs::Int;
    max_perturbation_order::Int = 1,
    output_compressed::Bool = true # Controls compression for X derivatives (order >= 2)
)::Vector{Tuple{SparseMatrixCSC{T, Int}, SparseMatrixCSC{T, Int}}} where T <: SymbolicNode
    
    nx = BigInt(length(𝔙)::Int)
    # np = length(𝔓)::BigInt
    nϵ = length(dyn_equations)::Int

    if max_perturbation_order < 1
        throw(ArgumentError("max_perturbation_order must be at least 1"))
    end

    results = [] # To store pairs of sparse matrices (X_matrix, P_matrix) for each order

    # --- Order 1 ---
    # Compute the 1st order derivative with respect to X (Jacobian)
    spX_order_1 = symbolic_sparse_jacobian(dyn_equations, 𝔙) # nϵ x nx


    spX_order_1_sub = copy(spX_order_1)

    steady_state_memo = IdDict{SymbolicNode, SymbolicNode}()
    spX_order_1_sub.nzval .= [substitute_symbolic(x, SS_mapping, steady_state_memo) for x in spX_order_1_sub.nzval]

    # Compute the derivative of the non-zeros of the 1st X-derivative w.r.t. P
    # This is an intermediate step. The final P matrix will be built from this.
    spP_of_flatX_nzval_order_1 = symbolic_sparse_jacobian(spX_order_1_sub.nzval, vcat(𝔓[1:nps], 𝔙[1:nxs])) # nnz(spX_order_1) x np

    # Determine dimensions for the Order 1 P matrix
    X_nrows_1 = nϵ
    X_ncols_1 = nx
    P_nrows_1 = X_nrows_1 * X_ncols_1
    P_ncols_1 = nps + nxs

    # Build the Order 1 P matrix (dimensions nϵ*nx x np)
    sparse_rows_1_P = Int[] # Row index in the flattened space of spX_order_1
    sparse_cols_1_P = Int[] # Column index for parameters (1 to np)
    sparse_vals_1_P = SymbolicNode[]

    # Map linear index in spX_order_1.nzval to its (row, col) in spX_order_1
    nz_lin_to_rc_1 = Dict{Int, Tuple{Int, Int}}()
    k_lin = 1
    for j = 1:size(spX_order_1, 2) # col
        for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
                r = spX_order_1.rowval[ptr] # row
                nz_lin_to_rc_1[k_lin] = (r, j)
                k_lin += 1
        end
    end


    # Iterate through the non-zero entries of spP_of_flatX_nzval_order_1
    k_temp_P = 1 # linear index counter for nzval
    for p_col = 1:size(spP_of_flatX_nzval_order_1, 2) # Parameter index
        for i_ptr_temp_P = spP_of_flatX_nzval_order_1.colptr[p_col]:(spP_of_flatX_nzval_order_1.colptr[p_col+1]-1)
            temp_row = spP_of_flatX_nzval_order_1.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval (corresponds to temp_row-th nzval of spX_order_1)
            p_val = spP_of_flatX_nzval_order_1.nzval[i_ptr_temp_P] # Derivative value w.r.t. parameter

            # Get the (row, col) in spX_order_1 corresponding to this derivative
            r_X1, c_X1 = nz_lin_to_rc_1[temp_row]

            # Calculate the row index in spP_order_1 (flattened index of spX_order_1)
            # P_row_idx = (r_X1 - 1) * X_ncols_1 + c_X1
            P_row_idx = (c_X1 - 1) * X_nrows_1 + r_X1
            P_col_idx = p_col # Parameter column index

            push!(sparse_rows_1_P, P_row_idx)
            push!(sparse_cols_1_P, P_col_idx)
            push!(sparse_vals_1_P, p_val)

            k_temp_P += 1
        end
    end

    spP_order_1 = sparse!(sparse_rows_1_P, sparse_cols_1_P, sparse_vals_1_P, P_nrows_1, P_ncols_1)


    # Store the pair for order 1
    push!(results, (spX_order_1_sub, spP_order_1))

    if max_perturbation_order > 1
        # --- Prepare for higher orders (Order 2 to max_perturbation_order) ---
        # Initialize map for Order 1: linear index in spX_order_1.nzval -> (row, (v1,))
        # This map is needed to trace indices for Order 2
        # We already built nz_lin_to_rc_1 above, reuse it and wrap the variable index in a Tuple
        nz_to_indices_prev = Dict{Int, Tuple{Int, Tuple{Int}}}()
        k_lin = 1
        for j = 1:size(spX_order_1, 2)
            for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
                r = spX_order_1.rowval[ptr]
                nz_to_indices_prev[k_lin] = (r, (j,)) # Store (equation row, (v1,))
                k_lin += 1
            end
        end

        nzvals_prev = spX_order_1.nzval # nzvals from Order 1 X-matrix

        # --- Iterate for orders n = 2, 3, ..., max_perturbation_order ---
        for n = 2:max_perturbation_order

            # Compute the Jacobian of the previous level's nzval w.r.t. 𝔛
            # This gives a flat matrix where rows correspond to non-zeros from order n-1 X-matrix
            # and columns correspond to the n-th variable we differentiate by (x_vn).
            sp_flat_curr_X_rn = symbolic_sparse_jacobian(nzvals_prev, 𝔙) # nnz(spX_order_(n-1)) x nx

            sp_flat_curr_X = copy(sp_flat_curr_X_rn)

            sp_flat_curr_X.nzval .= [substitute_symbolic(x, SS_mapping, steady_state_memo) for x in sp_flat_curr_X.nzval]

            # Build the nz_to_indices map for the *current* level (order n)
            # Map: linear index in sp_flat_curr_X.nzval -> (original_row_f, (v_1, ..., v_n))
            nz_to_indices_curr = Dict{Int, Tuple{Int, Tuple{Vararg{Int}}}}()
            k_lin_curr = 1 # linear index counter for nzval of sp_flat_curr_X
            # Iterate through the non-zeros of the current flat Jacobian
            for col_curr = 1:size(sp_flat_curr_X, 2) # Column index in sp_flat_curr_X (corresponds to v_n)
                for ptr_curr = sp_flat_curr_X.colptr[col_curr]:(sp_flat_curr_X.colptr[col_curr+1]-1)
                    row_curr = sp_flat_curr_X.rowval[ptr_curr] # Row index in sp_flat_curr_X (corresponds to the row_curr-th nzval of previous level)

                    # Get previous indices info from the map of order n-1
                    prev_info = nz_to_indices_prev[row_curr]
                    orig_row_f = prev_info[1] # Original equation row
                    vars_prev = prev_info[2] # Tuple of variables from previous order (v_1, ..., v_{n-1})

                    # Append the current variable index (v_n)
                    vars_curr = (vars_prev..., col_curr) # Full tuple (v_1, ..., v_n)

                    # Store info for the current level's non-zero
                    nz_to_indices_curr[k_lin_curr] = (orig_row_f, vars_curr)
                    k_lin_curr += 1
                end
            end

            # --- Construct the X-derivative sparse matrix for order n (compressed or uncompressed) ---
            local spX_order_n # Declare variable to hold the resulting X matrix
            local X_ncols_n # Number of columns in the resulting spX_order_n matrix

            if output_compressed
                # COMPRESSED output: nϵ x binomial(nx + n - 1, n)
                sparse_rows_n = Int[]
                sparse_cols_n = Int[] # This will store the compressed column index
                sparse_vals_n = SymbolicNode[]

                # Calculate the total number of compressed columns for order n
                X_ncols_n = Int(binomial(nx + n - 1, n))

                # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
                k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
                for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                    for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                        # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                        val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                        # Get the full info for this non-zero from the map
                        # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                        orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                        # Check the compression rule: v_n <= v_{n-1} <= ... <= v_1
                        is_compressed = true
                        for k_rule = 1:(n-1)
                            # Check v_{n-k_rule+1} <= v_{n-k_rule}
                            if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                                is_compressed = false
                                break
                            end
                        end

                        if is_compressed
                            # Calculate the compressed column index c_n for the tuple (v_1, ..., v_n)
                            # using the derived formula: c_n = sum_{k=1}^{n-1} binomial(v_k + n - k - 1, n - k + 1) + v_n
                            compressed_col_idx = 0
                            for k_formula = 1:(n-1)
                                term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                                compressed_col_idx += term
                            end
                            # Add the last term: v_n (var_indices_full[n])
                            compressed_col_idx += var_indices_full[n]

                            push!(sparse_rows_n, orig_row_f)
                            push!(sparse_cols_n, compressed_col_idx)
                            push!(sparse_vals_n, val)
                        end

                        k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                    end
                end
                # Construct the compressed sparse matrix for order n
                spX_order_n = sparse!(sparse_rows_n, sparse_cols_n, sparse_vals_n, X_nrows_1, X_ncols_n)

            else # output_compressed == false
                # UNCOMPRESSED output: nϵ x nx^n
                sparse_rows_n_uncomp = Int[]
                sparse_cols_n_uncomp = Int[] # Uncompressed column index (1 to nx^n)
                sparse_vals_n_uncomp = SymbolicNode[]

                # Total number of uncompressed columns
                X_ncols_n = nx^n # Use BigInt for the power calculation, cast to Int

                # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
                k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
                for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                    for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                        # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                        val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                        # Get the full info for this non-zero from the map
                        # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                        orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                        # Calculate the UNCOMPRESSED column index for the tuple (v_1, ..., v_n)
                        # This maps the tuple (v1, ..., vn) to a unique index from 1 to nx^n
                        # Formula: 1 + (v1-1)*nx^(n-1) + (v2-1)*nx^(n-2) + ... + (vn-1)*nx^0
                        uncompressed_col_idx = 1 # 1-based
                        power_of_nx = nx^(n-1) # Start with nx^(n-1) for v1 term
                        for i = 1:n
                            uncompressed_col_col_idx_term = (var_indices_full[i] - 1) * power_of_nx
                            # Check for overflow before adding
                            # if (uncompressed_col_idx > 0 && uncompressed_col_col_idx_term > 0 && uncompressed_col_idx + uncompressed_col_col_idx_term <= uncompressed_col_idx) ||
                            #    (uncompressed_col_idx < 0 && uncompressed_col_col_idx_term < 0 && uncompressed_col_idx + uncompressed_col_col_idx_term >= uncompressed_col_idx)
                            #    error("Integer overflow calculating uncompressed column index")
                            # end
                            uncompressed_col_idx += uncompressed_col_col_idx_term

                            if i < n # Avoid nx^-1
                                power_of_nx = div(power_of_nx, nx) # Integer division
                            end
                        end

                        push!(sparse_rows_n_uncomp, orig_row_f)
                        push!(sparse_cols_n_uncomp, Int(uncompressed_col_idx)) # Cast to Int
                        push!(sparse_vals_n_uncomp, val)

                        k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                    end
                end
                # Construct the uncompressed sparse matrix for order n
                spX_order_n = sparse!(sparse_rows_n_uncomp, sparse_cols_n_uncomp, sparse_vals_n_uncomp, X_nrows_1, X_ncols_n)

            end # End of if output_compressed / else


            # --- Compute the P-derivative sparse matrix for order n ---
            # This is the Jacobian of the nzval of the intermediate flat X-Jacobian (sp_flat_curr_X) w.r.t. 𝔓.
            # sp_flat_curr_X.nzval contains expressions for d^n f_i / (dx_v1 ... dx_vn) for all
            # non-zero such values that were propagated from the previous step.
            spP_of_flatX_nzval_curr = symbolic_sparse_jacobian(sp_flat_curr_X.nzval, vcat(𝔓[1:nps], 𝔙[1:nxs])) # nnz(sp_flat_curr_X) x np
            
            # Determine the desired dimensions of spP_order_n
            # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
            P_nrows_n = nϵ * X_ncols_n
            P_ncols_n = nps + nxs

            sparse_rows_n_P = Int[] # Row index in the flattened space of spX_order_n (1 to P_nrows_n)
            sparse_cols_n_P = Int[] # Column index for parameters (1 to np)
            sparse_vals_n_P = SymbolicNode[]

            # Iterate through the non-zero entries of spP_of_flatX_nzval_curr
            # Its rows correspond to the non-zeros in sp_flat_curr_X
            k_temp_P = 1 # linear index counter for nzval of spP_of_flatX_nzval_curr
            for p_col = 1:size(spP_of_flatX_nzval_curr, 2) # Column index in spP_of_flatX_nzval_curr (corresponds to parameter index)
                for i_ptr_temp_P = spP_of_flatX_nzval_curr.colptr[p_col]:(spP_of_flatX_nzval_curr.colptr[p_col+1]-1)
                    temp_row = spP_of_flatX_nzval_curr.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval_curr (corresponds to the temp_row-th nzval of sp_flat_curr_X)
                    p_val = spP_of_flatX_nzval_curr.nzval[i_ptr_temp_P] # The derivative w.r.t. parameter value

                    # Get the full info for the X-derivative term that this P-derivative is from
                    # temp_row is the linear index in sp_flat_curr_X.nzval
                    # This corresponds to the derivative d^n f_orig_row_f / (dx_v1 ... dx_vn)
                    orig_row_f, var_indices_full = nz_to_indices_curr[temp_row] # (v_1, ..., v_n)

                    # We need to find the column index (X_col_idx) this term corresponds to
                    # in the final spX_order_n matrix (which might be compressed or uncompressed)
                    local X_col_idx # Column index in the final spX_order_n matrix (1 to X_ncols_n)

                    if output_compressed
                        # For compressed output, only include entries where variable indices
                        # are in non-increasing order (v_n <= v_{n-1} <= ... <= v_1).
                        # This matches the compression rule used for the X-matrix.
                        # Unsorted tuples represent the same derivative (by symmetry of
                        # mixed partials) but the compressed column formula maps them to
                        # WRONG positions, corrupting the Jacobian.
                        is_compressed_P = true
                        for k_rule = 1:(n-1)
                            if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                                is_compressed_P = false
                                break
                            end
                        end

                        if !is_compressed_P
                            k_temp_P += 1
                            continue
                        end

                        # Calculate the compressed column index
                        compressed_col_idx = 0
                        for k_formula = 1:(n-1)
                            term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                            compressed_col_idx += term
                        end
                        compressed_col_idx += var_indices_full[n]
                        X_col_idx = compressed_col_idx # The column in spX_order_n is the compressed one

                    else # output_compressed == false
                        # Calculate the uncompressed column index
                        uncompressed_col_idx = 1
                        power_of_nx = nx^(n-1)
                        for i = 1:n
                            uncompressed_col_idx += (var_indices_full[i] - 1) * power_of_nx
                            if i < n
                                power_of_nx = div(power_of_nx, nx)
                            end
                        end
                        X_col_idx = Int(uncompressed_col_idx) # The column in spX_order_n is the uncompressed one
                    end

                    # Calculate the row index in spP_order_n
                    # This maps the (orig_row_f, X_col_idx) pair in spX_order_n's grid to a linear index
                    # Formula: (row_in_X - 1) * num_cols_in_X + col_in_X
                    # P_row_idx = (orig_row_f - 1) * X_ncols_n + X_col_idx
                    P_row_idx = (X_col_idx - 1) * nϵ + orig_row_f

                    # The column index in spP_order_n is the parameter index
                    P_col_idx = p_col

                    push!(sparse_rows_n_P, P_row_idx)
                    push!(sparse_cols_n_P, P_col_idx)
                    push!(sparse_vals_n_P, p_val)

                    k_temp_P += 1 # Increment linear index counter for spP_of_flatX_nzval_curr.nzval
                end
            end

            # Construct the P-derivative sparse matrix for order n
            # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
            spP_order_n = sparse!(sparse_rows_n_P, sparse_cols_n_P, sparse_vals_n_P, P_nrows_n, P_ncols_n)

            # Store the pair (X-matrix, P-matrix) for order n
            push!(results, (spX_order_n, spP_order_n))


            # Prepare for the next iteration (order n+1)
            # The nzvals for the next X-Jacobian step are the nzvals of the current flat X-Jacobian
            nzvals_prev = sp_flat_curr_X_rn.nzval
            # The map for the next step should provide info for order n derivatives
            nz_to_indices_prev = nz_to_indices_curr

        end # End of loop for orders n = 2 to max_perturbation_order
    end

    return results #, (𝔛, 𝔓) # Return results as a tuple of (X_matrix, P_matrix) pairs
end


function write_functions_mapping!(𝓂::ℳ, max_perturbation_order::Int; 
                                    density_threshold::Float64 = .1, 
                                    min_length::Int = 1000,
                                    nnz_parallel_threshold::Int = 1000000,
                                    cse = true,
                                    skipzeros = true)

    future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₁₎$")))
    present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎$")))
    past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₋₁₎$")))
    shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₓ₎$")))
    ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₛₛ₎$")))

    sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
    sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
    sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
    sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
    sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

    dyn_future_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_future_list))
    dyn_present_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_present_list))
    dyn_past_list = collect(reduce(union, 𝓂.constants.post_model_macro.dyn_past_list))
    dyn_exo_list = collect(reduce(union,𝓂.constants.post_model_macro.dyn_exo_list))
    dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.constants.post_model_macro.dyn_ss_list))) .* "₍ₛₛ₎")

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
    stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))

    vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                    dyn_present_list[indexin(sort(present),present)],
                    dyn_past_list[indexin(sort(past),past)],
                    dyn_exo_list[indexin(sort(exo),exo)])

    dyn_var_future_idx = 𝓂.constants.post_complete_parameters.dyn_var_future_idx
    dyn_var_present_idx = 𝓂.constants.post_complete_parameters.dyn_var_present_idx
    dyn_var_past_idx = 𝓂.constants.post_complete_parameters.dyn_var_past_idx
    dyn_ss_idx = 𝓂.constants.post_complete_parameters.dyn_ss_idx

    dyn_var_idxs = vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)

    pars_ext = vcat(𝓂.constants.post_complete_parameters.parameters, 𝓂.equations.calibration_parameters)
    parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst),stst)])

    np = length(parameters_and_SS)
    nv = length(vars_raw)
    nc = length(𝓂.equations.calibration)
    nps = length(𝓂.constants.post_complete_parameters.parameters)
    nxs = maximum(dyn_var_idxs) + nc

    𝔓 = symbolic_placeholders("𝔓", np)
    𝔙 = symbolic_placeholders("𝔙", nv)

    parameter_dict = Dict{Symbol, Symbol}()
    calib_vars = Symbol[]
    calib_expr = []
    SS_mapping = Dict{Symbol, SymbolicNode}()


    for (i,v) in enumerate(parameters_and_SS)
        push!(parameter_dict, v => 𝔓[i])
        if i > nps
            if i > length(pars_ext)
                push!(SS_mapping, 𝔓[i] => symbolic_symbol(𝔙[dyn_ss_idx[i-length(pars_ext)]]))
            else
                push!(SS_mapping, 𝔓[i] => symbolic_symbol(𝔙[nxs + i - nps - nc]))
            end
        end
    end

    for (i,v) in enumerate(vars_raw)
        push!(parameter_dict, v => 𝔙[i])
        if i <= length(dyn_var_idxs)
            push!(SS_mapping, 𝔙[i] => symbolic_symbol(𝔙[dyn_var_idxs[i]]))
        else
            push!(SS_mapping, 𝔙[i] => symbolic_zero())
        end
    end


    for v in 𝓂.equations.calibration_no_var
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end


    calib_replacements = Dict{Symbol, Union{Expr, Symbol, Number}}()
    for (i,x) in enumerate(calib_vars)
        replacement = Dict{Symbol, Union{Expr, Symbol, Number}}(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end


    dyn_equations = 𝓂.equations.dynamic |> 
        x -> replace_symbols.(x, Ref(calib_replacements)) |> 
        x -> replace_symbols.(x, Ref(parameter_dict)) |> 
        x -> SymbolicNode[to_symbolic(e) for e in x]

    derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs)

    function prepare_sensitivity_buffer(derivative_sensitivities)
        transposed = derivative_sensitivities isa SparseMatrixCSC ? sparse(transpose(derivative_sensitivities)) : permutedims(derivative_sensitivities)
        local nz_count = nnz(transposed)

        if (nz_count / length(transposed) > density_threshold) || (length(transposed) < min_length)
            return convert(Matrix, transposed), zeros(Float64, size(transposed)), nz_count
        end

        local buf = similar(transposed, Float64)
        buf.nzval .= 0
        return transposed, buf, nz_count
    end


    ∇₁_dyn = derivatives[1][1]

    lennz = nnz(∇₁_dyn)

    jacobian_dense_by_heuristic = (lennz / length(∇₁_dyn) > density_threshold) || (length(∇₁_dyn) < min_length)
    # Re-enable `jacobian_dense_by_heuristic` directly to restore sparse Jacobian path switching.
    if jacobian_dense_by_heuristic
        derivatives_mat = convert(Matrix, ∇₁_dyn)
        buffer = zeros(Float64, size(∇₁_dyn))
    else
        derivatives_mat = ∇₁_dyn
        buffer = similar(∇₁_dyn, Float64)
        buffer.nzval .= 0
    end
    
    
    func_exprs = build_symbolic_function(derivatives_mat, [𝔓, 𝔙])

    𝓂.caches.jacobian = buffer


    ∇₁_parameters_mat, buffer_parameters, lennz = prepare_sensitivity_buffer(derivatives[1][2][:,1:nps])


    func_∇₁_parameters = build_symbolic_function(∇₁_parameters_mat, [𝔓, 𝔙])

    𝓂.caches.jacobian_parameters = buffer_parameters
 

    ∇₁_SS_and_pars_mat, buffer_SS_and_pars, lennz = prepare_sensitivity_buffer(derivatives[1][2][:,nps+1:end])


    func_∇₁_SS_and_pars = build_symbolic_function(∇₁_SS_and_pars_mat, [𝔓, 𝔙])

    𝓂.caches.jacobian_SS_and_pars = buffer_SS_and_pars
    
    # Create jacobian_functions struct with all three functions
    𝓂.functions.jacobian = jacobian_functions(func_exprs, func_∇₁_parameters, func_∇₁_SS_and_pars)




    # if max_perturbation_order >= 1
    #     SS_and_pars = Symbol.(vcat(string.(sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.constants.post_model_macro.parameters_in_equations,𝓂.constants.post_model_macro.➕_vars))))), 𝓂.calibration_equations_parameters))

    #     eqs = vcat(𝓂.ss_equations, 𝓂.calibration_equations)

    #     nx = length(𝓂.parameter_values)

    #     np = length(SS_and_pars)

    #     nϵˢ = length(eqs)

    #     nc = length(𝓂.calibration_equations_no_var)

    #     Symbolics.@variables 𝔛¹[1:nx] 𝔓¹[1:np]

    #     ϵˢ = zeros(Symbolics.Num, nϵˢ)
    
    #     calib_vals = zeros(Symbolics.Num, nc)

    #     𝓂.SS_calib_func(calib_vals, 𝔛¹)
    
    #     𝓂.functions.NSSS_check(ϵˢ, 𝔛¹, 𝔓¹, calib_vals)
    # println(ϵˢ)
    #     ∂SS_equations_∂parameters = symbolic_sparse_jacobian(ϵˢ, 𝔛¹) # nϵ x nx
    
    #     lennz = nnz(∂SS_equations_∂parameters)

    #     if (lennz / length(∂SS_equations_∂parameters) > density_threshold) || (length(∂SS_equations_∂parameters) < min_length)
    #         derivatives_mat = convert(Matrix, ∂SS_equations_∂parameters)
    #         buffer = zeros(Float64, size(∂SS_equations_∂parameters))
    #     else
    #         derivatives_mat = ∂SS_equations_∂parameters
    #         buffer = similar(∂SS_equations_∂parameters, Float64)
    #         buffer.nzval .= 0
    #     end

    #     if lennz > nnz_parallel_threshold
    #         parallel = Symbolics.ShardedForm(1500,4)
    #     else
    #         parallel = Symbolics.SerialForm()
    #     end
        
    #     func_exprs = build_symbolic_function(derivatives_mat, [𝔛¹, 𝔓¹])

    #     𝓂.functions.NSSS_∂equations_∂parameters = func_exprs



    #     ∂SS_equations_∂SS_and_pars = symbolic_sparse_jacobian(ϵˢ, 𝔓¹) # nϵ x nx
    
    #     lennz = nnz(∂SS_equations_∂SS_and_pars)

    #     if (lennz / length(∂SS_equations_∂SS_and_pars) > density_threshold) || (length(∂SS_equations_∂SS_and_pars) < min_length)
    #         derivatives_mat = convert(Matrix, ∂SS_equations_∂SS_and_pars)
    #         buffer = zeros(Float64, size(∂SS_equations_∂SS_and_pars))
    #     else
    #         derivatives_mat = ∂SS_equations_∂SS_and_pars
    #         buffer = similar(∂SS_equations_∂SS_and_pars, Float64)
    #         buffer.nzval .= 0
    #     end

    #     if lennz > nnz_parallel_threshold
    #         parallel = Symbolics.ShardedForm(1500,4)
    #     else
    #         parallel = Symbolics.SerialForm()
    #     end

    #     func_exprs = build_symbolic_function(derivatives_mat, [𝔛¹, 𝔓¹])

    #     𝓂.functions.NSSS_∂equations_∂SS_and_pars = func_exprs
    # end
        
    if max_perturbation_order >= 2
    # second order
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = 2, output_compressed = true)

        if 𝓂.constants.second_order.𝛔 == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            ∇₂_dyn = derivatives[2][1]

            𝓂.constants.second_order = create_second_order_auxiliary_matrices(𝓂.constants)
            𝓂.constants.second_order.∇₂_nonempty_col_as_kron_rowmask = findall(@view(∇₂_dyn.colptr[1:end-1]) .< @view(∇₂_dyn.colptr[2:end]))

            lennz = nnz(∇₂_dyn)

            if (lennz / length(∇₂_dyn) > density_threshold) || (length(∇₂_dyn) < min_length)
                derivatives_mat = convert(Matrix, ∇₂_dyn)
                buffer = zeros(Float64, size(∇₂_dyn))
            else
                derivatives_mat = ∇₂_dyn
                buffer = similar(∇₂_dyn, Float64)
                buffer.nzval .= 0
            end


            func_exprs = build_symbolic_function(derivatives_mat, [𝔓, 𝔙])

            𝓂.caches.hessian = buffer


            ∇₂_parameters_mat, buffer_parameters, lennz = prepare_sensitivity_buffer(derivatives[2][2][:,1:nps])


            func_∇₂_parameters = build_symbolic_function(∇₂_parameters_mat, [𝔓, 𝔙])

            𝓂.caches.hessian_parameters = buffer_parameters
        

            ∇₂_SS_and_pars_mat, buffer_SS_and_pars, lennz = prepare_sensitivity_buffer(derivatives[2][2][:,nps+1:end])


            func_∇₂_SS_and_pars = build_symbolic_function(∇₂_SS_and_pars_mat, [𝔓, 𝔙])

            𝓂.caches.hessian_SS_and_pars = buffer_SS_and_pars
            
            # Create hessian_functions struct with all three functions
            𝓂.functions.hessian = hessian_functions(func_exprs, func_∇₂_parameters, func_∇₂_SS_and_pars)
        end
    end

    if max_perturbation_order == 3
        derivatives = take_nth_order_derivatives(dyn_equations, 𝔙, 𝔓, SS_mapping, nps, nxs; max_perturbation_order = max_perturbation_order, output_compressed = true)
    # third order
        if 𝓂.constants.third_order.𝐂₃ == SparseMatrixCSC{Int, Int64}(ℒ.I,0,0)
            I,J,V = findnz(derivatives[3][1])
            𝓂.constants.third_order = create_third_order_auxiliary_matrices(𝓂.constants, unique(J))
        
            ∇₃_dyn = derivatives[3][1]

            lennz = nnz(∇₃_dyn)

            if (lennz / length(∇₃_dyn) > density_threshold) || (length(∇₃_dyn) < min_length)
                derivatives_mat = convert(Matrix, ∇₃_dyn)
                buffer = zeros(Float64, size(∇₃_dyn))
            else
                derivatives_mat = ∇₃_dyn
                buffer = similar(∇₃_dyn, Float64)
                buffer.nzval .= 0
            end


            func_exprs = build_symbolic_function(derivatives_mat, [𝔓, 𝔙])

            𝓂.caches.third_order_derivatives = buffer


            ∇₃_parameters_mat, buffer_parameters, lennz = prepare_sensitivity_buffer(derivatives[3][2][:,1:nps])


            func_∇₃_parameters = build_symbolic_function(∇₃_parameters_mat, [𝔓, 𝔙])

            𝓂.caches.third_order_derivatives_parameters = buffer_parameters
        

            ∇₃_SS_and_pars_mat, buffer_SS_and_pars, lennz = prepare_sensitivity_buffer(derivatives[3][2][:,nps+1:end])


            func_∇₃_SS_and_pars = build_symbolic_function(∇₃_SS_and_pars_mat, [𝔓, 𝔙])

            𝓂.caches.third_order_derivatives_SS_and_pars = buffer_SS_and_pars
            
            # Create third_order_derivatives_functions struct with all three functions
            𝓂.functions.third_order_derivatives = third_order_derivatives_functions(func_exprs, func_∇₃_parameters, func_∇₃_SS_and_pars)
        end
    end

    # Invalidate derivative stamps since buffers were replaced with fresh (zeroed) content.
    # Without this, calculate_jacobian/hessian/third_order_derivatives would return stale
    # zero-filled buffers on a cache hit, causing downstream DimensionMismatch errors.
    𝓂.caches.valid_for.jacobian = Float64[]

    return nothing
end


function write_auxiliary_indices!(𝓂::ℳ)
    # write indices in auxiliary objects
    dyn_var_future_list  = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₁₎")))
    dyn_var_present_list = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₀₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₀₎")))
    dyn_var_past_list    = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍₋₁₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍₋₁₎")))
    dyn_exo_list         = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₓ₎")))
    dyn_ss_list          = map(x->Set{Symbol}(map(x->Symbol(replace(string(x),"₍ₛₛ₎" => "")),x)),collect.(match_pattern.(get_symbols.(𝓂.equations.dynamic),r"₍ₛₛ₎")))

    dyn_var_future  = Symbol.(string.(sort(collect(reduce(union,dyn_var_future_list)))))
    dyn_var_present = Symbol.(string.(sort(collect(reduce(union,dyn_var_present_list)))))
    dyn_var_past    = Symbol.(string.(sort(collect(reduce(union,dyn_var_past_list)))))
    dyn_exo         = Symbol.(string.(sort(collect(reduce(union,dyn_exo_list)))))
    dyn_ss          = Symbol.(string.(sort(collect(reduce(union,dyn_ss_list)))))

    SS_and_pars_names = vcat(Symbol.(string.(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_past,𝓂.constants.post_model_macro.exo_future)))), 𝓂.equations.calibration_parameters)

    dyn_var_future_idx  = indexin(dyn_var_future    , SS_and_pars_names)
    dyn_var_present_idx = indexin(dyn_var_present   , SS_and_pars_names)
    dyn_var_past_idx    = indexin(dyn_var_past      , SS_and_pars_names)
    dyn_ss_idx          = indexin(dyn_ss            , SS_and_pars_names)

    shocks_ss = zeros(length(dyn_exo))

    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        dyn_var_future_idx = dyn_var_future_idx,
        dyn_var_present_idx = dyn_var_present_idx,
        dyn_var_past_idx = dyn_var_past_idx,
        dyn_ss_idx = dyn_ss_idx,
        shocks_ss = shocks_ss,
    )

    return nothing
end
