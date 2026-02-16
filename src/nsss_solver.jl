# Non-stochastic steady state (NSSS) solver
# 
# This file contains:
# 1. Builder for accumulating step data into model sub-structs
# 2. Step execution function dispatching on step type
# 3. The solve_nsss_steps orchestrator that iterates over steps
# 4. The solve_nsss_wrapper that handles cache management and continuation method

# ============================================================================
# Pipeline builder
# ============================================================================

const EMPTY_NSSS_STEP_CACHE = Vector{Vector{Float64}}()

"""
Mutable accumulator used during `write_steady_state_solver_function!` to collect step data.
After all steps are appended, call `build_nsss_solver!(𝓂, builder, param_prep!)` to assign
data into the model's functions, constants, and workspaces sub-structs.
"""
mutable struct NSSSSolverBuilder
    # Per-step parallel vectors (functions)
    aux_funcs::Vector{Union{Nothing, Function}}
    error_funcs::Vector{Union{Nothing, Function}}
    eval_funcs::Vector{Union{Nothing, Function}}
    solve_blocks::Vector{Union{Nothing, ss_solve_block}}
    # Per-step metadata
    step_types::Vector{UInt8}
    descriptions::Vector{String}
    block_indices::Vector{Int}
    # Flat index accumulators
    write_indices::Vector{Int}
    write_ranges::Vector{UnitRange{Int}}
    aux_write_indices::Vector{Int}
    aux_write_ranges::Vector{UnitRange{Int}}
    param_gather_indices::Vector{Int}
    param_gather_ranges::Vector{UnitRange{Int}}
    var_gather_indices::Vector{Int}
    var_gather_ranges::Vector{UnitRange{Int}}
    # Flat bounds accumulators (analytical)
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    has_bounds::BitVector
    bounds_ranges::Vector{UnitRange{Int}}
    # Flat bounds accumulators (numerical)
    numerical_lbs::Vector{Float64}
    numerical_ubs::Vector{Float64}
    numerical_bounds_ranges::Vector{UnitRange{Int}}
    # Error sizes
    error_sizes::Vector{Int}
    aux_error_sizes::Vector{Int}
    # Workspace size tracking
    max_main_buffer::Int
    max_aux_buffer::Int
    max_error_buffer::Int
    max_guess_buffer::Int
end

function NSSSSolverBuilder()
    NSSSSolverBuilder(
        Union{Nothing,Function}[], Union{Nothing,Function}[],
        Union{Nothing,Function}[], Union{Nothing,ss_solve_block}[],
        UInt8[], String[], Int[],
        Int[], UnitRange{Int}[],
        Int[], UnitRange{Int}[],
        Int[], UnitRange{Int}[],
        Int[], UnitRange{Int}[],
        Float64[], Float64[], BitVector(), UnitRange{Int}[],
        Float64[], Float64[], UnitRange{Int}[],
        Int[], Int[],
        0, 0, 0, 0,
    )
end

"""Append an analytical step to the builder."""
function push_analytical_step!(b::NSSSSolverBuilder;
                               aux_func!::Union{Nothing,Function} = nothing,
                               aux_write_indices::Vector{Int} = Int[],
                               error_func!::Union{Nothing,Function} = nothing,
                               error_size::Int = 0,
                               eval_func!::Function,
                               write_indices::Vector{Int},
                               lower_bounds::Vector{Float64} = Float64[],
                               upper_bounds::Vector{Float64} = Float64[],
                               has_bounds::BitVector = falses(length(write_indices)),
                               description::String = "")
    push!(b.step_types, ANALYTICAL_STEP)
    push!(b.descriptions, description)
    push!(b.block_indices, 0)

    # Functions
    push!(b.aux_funcs, aux_func!)
    push!(b.error_funcs, error_func!)
    push!(b.eval_funcs, eval_func!)
    push!(b.solve_blocks, nothing)

    # Write indices
    off = length(b.write_indices)
    append!(b.write_indices, write_indices)
    push!(b.write_ranges, (off+1):(off+length(write_indices)))

    # Aux write indices
    off = length(b.aux_write_indices)
    append!(b.aux_write_indices, aux_write_indices)
    push!(b.aux_write_ranges, (off+1):(off+length(aux_write_indices)))

    # No param/var gather for analytical
    push!(b.param_gather_ranges, 1:0)
    push!(b.var_gather_ranges, 1:0)

    # Bounds (analytical)
    off = length(b.lower_bounds)
    append!(b.lower_bounds, lower_bounds)
    append!(b.upper_bounds, upper_bounds)
    append!(b.has_bounds, has_bounds)
    push!(b.bounds_ranges, (off+1):(off+length(lower_bounds)))

    # No numerical bounds
    push!(b.numerical_bounds_ranges, 1:0)

    # Error sizes
    push!(b.error_sizes, error_size)
    push!(b.aux_error_sizes, 0)

    # Update workspace max sizes
    b.max_main_buffer = max(b.max_main_buffer, length(write_indices))
    b.max_aux_buffer = max(b.max_aux_buffer, length(aux_write_indices))
    b.max_error_buffer = max(b.max_error_buffer, error_size)
end

"""Append a numerical step to the builder."""
function push_numerical_step!(b::NSSSSolverBuilder;
                              solve_block::ss_solve_block,
                              block_index::Int,
                              write_indices::Vector{Int},
                              param_gather_indices::Vector{Int},
                              var_gather_indices::Vector{Int},
                              lbs::Vector{Float64},
                              ubs::Vector{Float64},
                              aux_func!::Union{Nothing,Function} = nothing,
                              aux_write_indices::Vector{Int} = Int[],
                              aux_error_func!::Union{Nothing,Function} = nothing,
                              aux_error_size::Int = 0,
                              description::String = "")
    push!(b.step_types, NUMERICAL_STEP)
    push!(b.descriptions, description)
    push!(b.block_indices, block_index)

    # Functions
    push!(b.aux_funcs, aux_func!)
    push!(b.error_funcs, aux_error_func!)   # numerical steps use error_funcs slot for aux_error
    push!(b.eval_funcs, nothing)
    push!(b.solve_blocks, solve_block)

    # Write indices
    off = length(b.write_indices)
    append!(b.write_indices, write_indices)
    push!(b.write_ranges, (off+1):(off+length(write_indices)))

    # Aux write indices
    off = length(b.aux_write_indices)
    append!(b.aux_write_indices, aux_write_indices)
    push!(b.aux_write_ranges, (off+1):(off+length(aux_write_indices)))

    # Param/var gather indices
    off = length(b.param_gather_indices)
    append!(b.param_gather_indices, param_gather_indices)
    push!(b.param_gather_ranges, (off+1):(off+length(param_gather_indices)))

    off = length(b.var_gather_indices)
    append!(b.var_gather_indices, var_gather_indices)
    push!(b.var_gather_ranges, (off+1):(off+length(var_gather_indices)))

    # No analytical bounds
    push!(b.bounds_ranges, 1:0)

    # Numerical bounds
    off = length(b.numerical_lbs)
    append!(b.numerical_lbs, lbs)
    append!(b.numerical_ubs, ubs)
    push!(b.numerical_bounds_ranges, (off+1):(off+length(lbs)))

    # Error sizes
    push!(b.error_sizes, 0)
    push!(b.aux_error_sizes, aux_error_size)

    # Update workspace max sizes
    gather_size = length(param_gather_indices) + length(var_gather_indices)
    b.max_main_buffer = max(b.max_main_buffer, gather_size)
    b.max_aux_buffer = max(b.max_aux_buffer, length(aux_write_indices))
    b.max_error_buffer = max(b.max_error_buffer, aux_error_size)
    b.max_guess_buffer = max(b.max_guess_buffer, length(write_indices))
end

"""Assign the solver functions, constants, and workspace from builder data into `𝓂`."""
function build_nsss_solver!(𝓂::ℳ, b::NSSSSolverBuilder, param_prep!::Union{Nothing,Function})
    n = length(b.step_types)
    n_ext_params = length(𝓂.constants.post_complete_parameters.parameters) + length(𝓂.equations.calibration_no_var)
    𝓂.functions.nsss_solver = NSSSSolverFunctions(
        b.aux_funcs, b.error_funcs, b.eval_funcs, b.solve_blocks,
    )
    𝓂.functions.nsss_param_prep! = param_prep!
    𝓂.constants.nsss_solver = NSSSSolverConstants(
        n,
        n_ext_params,
        b.step_types, b.descriptions, b.block_indices,
        b.write_indices, b.write_ranges,
        b.aux_write_indices, b.aux_write_ranges,
        b.param_gather_indices, b.param_gather_ranges,
        b.var_gather_indices, b.var_gather_ranges,
        b.lower_bounds, b.upper_bounds, b.has_bounds, b.bounds_ranges,
        b.numerical_lbs, b.numerical_ubs, b.numerical_bounds_ranges,
        b.error_sizes, b.aux_error_sizes,
    )
    𝓂.workspaces.nsss_solver = NSSSSolverWorkspace(
        zeros(Float64, max(b.max_main_buffer, 1)),
        zeros(Float64, max(b.max_aux_buffer, 1)),
        zeros(Float64, max(b.max_error_buffer, 1)),
        zeros(Float64, max(𝓂.constants.nsss_solver.n_ext_params, 1)),
        Float64[],
        zeros(Float64, max(b.max_guess_buffer, 1)),
        [zeros(Float64, max(b.max_guess_buffer, 1)), Float64[Inf]],
        zeros(Float64, max(b.max_main_buffer, 1)),
        zeros(Float64, max(b.max_guess_buffer, 1)),
        zeros(Float64, max(b.max_guess_buffer, 1)),
    )
    return nothing
end

@unstable begin
    function replace_symbols(exprs, remap::AbstractDict{Symbol, <:Any})
        postwalk(node ->
            (node isa Symbol && haskey(remap, node)) ? remap[node] : node,
            exprs,
        )
    end
end

function write_block_solution!(𝓂,
                                vars_to_solve,
                                eqs_to_solve,
                                relevant_pars_across,
                                nsss_solver_cache_init_tmp,
                                eq_idx_in_block_to_solve,
                                atoms_in_equations_list,
                                solved_vars,
                                solved_vals;
                                block_index::Int,
                                cse = true,
                                skipzeros = true,
                                density_threshold::Float64 = .1,
                                nnz_parallel_threshold::Int = 1000000,
                                min_length::Int = 10000)

    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    vars_to_exclude = [vcat(Symbol.(vars_to_solve), 𝓂.constants.post_model_macro.➕_vars),Symbol[]]

    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors(Meta.parse.(string.(eqs_to_solve)), vars_to_exclude, 𝓂.constants.post_parameters_macro.bounds, 𝓂.constants.post_model_macro.➕_vars, unique_➕_eqs)

    push!(solved_vars, Symbol.(vars_to_solve))
    push!(solved_vals, rewritten_eqs)

    syms_in_eqs = Set{Symbol}()
    for i in vcat(ss_and_aux_equations_dep, ss_and_aux_equations, rewritten_eqs)
        push!(syms_in_eqs, get_symbols(i)...)
    end

    setdiff!(syms_in_eqs,𝓂.constants.post_model_macro.➕_vars)

    syms_in_eqs2 = Set{Symbol}()
    for i in ss_and_aux_equations
        push!(syms_in_eqs2, get_symbols(i)...)
    end

    ➕_vars_alread_in_eqs = intersect(𝓂.constants.post_model_macro.➕_vars,reduce(union,get_symbols.(Meta.parse.(string.(eqs_to_solve)))))

    union!(syms_in_eqs, intersect(union(➕_vars_alread_in_eqs, syms_in_eqs2), 𝓂.constants.post_model_macro.➕_vars))

    push!(atoms_in_equations_list,setdiff(syms_in_eqs, solved_vars[end]))

    calib_pars_input = Symbol[]

    relevant_pars = union(intersect(reduce(union, vcat(𝓂.constants.post_model_macro.par_list_aux_SS, 𝓂.constants.post_parameters_macro.par_calib_list)[eq_idx_in_block_to_solve]), syms_in_eqs),intersect(syms_in_eqs, 𝓂.constants.post_model_macro.➕_vars))
    union!(relevant_pars_across, relevant_pars)

    sorted_vars = sort(Symbol.(vars_to_solve))

    iii = 1
    for parss in union(𝓂.constants.post_complete_parameters.parameters, 𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters)
        if :($parss) ∈ relevant_pars
            push!(calib_pars_input, :($parss))
            iii += 1
        end
    end

    other_vrs_eliminated_by_sympy = Set{Symbol}()
    for (i,val) in enumerate(solved_vals[end])
        if eq_idx_in_block_to_solve[i] ∈ 𝓂.constants.post_model_macro.ss_equations_with_aux_variables
            val = vcat(𝓂.equations.steady_state_aux, 𝓂.equations.calibration)[eq_idx_in_block_to_solve[i]]
            push!(other_vrs_eliminated_by_sympy, val.args[2])
        end
    end

    solved_vals_local = Union{Expr, Symbol}[]
    for (i,val) in enumerate(rewritten_eqs)
        push!(solved_vals_local, postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
    end

    other_vars_input = Symbol[]
    other_vrs = intersect( setdiff( union(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters, 𝓂.constants.post_model_macro.➕_vars),
                                        sort(solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy ) )

    for var in other_vrs
        push!(other_vars_input,:($(var)))
        iii += 1
    end

    parameters_and_solved_vars = vcat(calib_pars_input, other_vrs)

    ng = length(sorted_vars)
    np = length(parameters_and_solved_vars)
    nd = length(ss_and_aux_equations_dep)
    nx = iii - 1

    Symbolics.@variables 𝔊[1:ng] 𝔓[1:np]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
    aux_vars = Symbol[]
    aux_expr = []

    for (i,v) in enumerate(sorted_vars)
        push!(parameter_dict, v => :($(Symbol("𝔊_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔊_$i"))), @__MODULE__) => 𝔊[i])
    end

    for (i,v) in enumerate(parameters_and_solved_vars)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end

    for (i,v) in enumerate(ss_and_aux_equations_dep)
        push!(aux_vars, v.args[1])
        push!(aux_expr, v.args[2])
    end

    aux_replacements = Dict{Symbol, Union{Expr, Symbol, Number}}()
    for (i,x) in enumerate(aux_vars)
        replacement = Dict{Symbol, Union{Expr, Symbol, Number}}(x => aux_expr[i])
        for ii in i+1:length(aux_vars)
            aux_expr[ii] = replace_symbols(aux_expr[ii], replacement)
        end
        push!(aux_replacements, x => aux_expr[i])
    end

    replaced_solved_vals = solved_vals_local |>
        x -> replace_symbols.(x, Ref(aux_replacements)) |>
        x -> replace_symbols.(x, Ref(parameter_dict)) |>
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))

    lennz = length(replaced_solved_vals)
    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, calc_block! = Symbolics.build_function(replaced_solved_vals, 𝔊, 𝔓,
                                                cse = cse,
                                                skipzeros = skipzeros,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    ϵˢ = zeros(Symbolics.Num, ng)
    ϵ = zeros(ng)

    ∂block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals, 𝔊)

    lennz = nnz(∂block_∂parameters_and_solved_vars)
    if (lennz / length(∂block_∂parameters_and_solved_vars) > density_threshold) || (length(∂block_∂parameters_and_solved_vars) < min_length)
        derivatives_mat = convert(Matrix, ∂block_∂parameters_and_solved_vars)
        buffer = zeros(Float64, size(∂block_∂parameters_and_solved_vars))
    else
        derivatives_mat = ∂block_∂parameters_and_solved_vars
        buffer = similar(∂block_∂parameters_and_solved_vars, Float64)
        buffer.nzval .= 1
    end

    chol_buff = buffer * buffer'
    chol_buff += ℒ.I

    prob = 𝒮.LinearProblem(chol_buff, ϵ)
    chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

    lu_factorization = issparse(buffer) ? 𝒮.LUFactorization() : 𝒮.FastLUFactorization()
    prob = 𝒮.LinearProblem(buffer, ϵ)
    lu_buffer = 𝒮.init(prob, lu_factorization, verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔊, 𝔓,
                                                cse = cse,
                                                skipzeros = skipzeros,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    Symbolics.@variables 𝔊[1:ng+nx]

    ext_diff = Symbolics.Num[]
    for i in 1:nx
        push!(ext_diff, 𝔓[i] - 𝔊[ng + i])
    end
    replaced_solved_vals_ext = vcat(replaced_solved_vals, ext_diff)

    _, calc_ext_block! = Symbolics.build_function(replaced_solved_vals_ext, 𝔊, 𝔓,
                                                cse = cse,
                                                skipzeros = skipzeros,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    ϵᵉ = zeros(ng + nx)
    ∂ext_block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals_ext, 𝔊)

    lennz = nnz(∂ext_block_∂parameters_and_solved_vars)
    if (lennz / length(∂ext_block_∂parameters_and_solved_vars) > density_threshold) || (length(∂ext_block_∂parameters_and_solved_vars) < min_length)
        derivatives_mat_ext = convert(Matrix, ∂ext_block_∂parameters_and_solved_vars)
        ext_buffer = zeros(Float64, size(∂ext_block_∂parameters_and_solved_vars))
    else
        derivatives_mat_ext = ∂ext_block_∂parameters_and_solved_vars
        ext_buffer = similar(∂ext_block_∂parameters_and_solved_vars, Float64)
        ext_buffer.nzval .= 1
    end

    ext_chol_buff = ext_buffer * ext_buffer'
    ext_chol_buff += ℒ.I

    prob = 𝒮.LinearProblem(ext_chol_buff, ϵᵉ)
    ext_chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

    ext_lu_factorization = issparse(ext_buffer) ? 𝒮.LUFactorization() : 𝒮.FastLUFactorization()
    prob = 𝒮.LinearProblem(ext_buffer, ϵᵉ)
    ext_lu_buffer = 𝒮.init(prob, ext_lu_factorization, verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

    if lennz > nnz_parallel_threshold
        parallel = Symbolics.ShardedForm(1500,4)
    else
        parallel = Symbolics.SerialForm()
    end

    _, ext_func_exprs = Symbolics.build_function(derivatives_mat_ext, 𝔊, 𝔓,
                                                cse = cse,
                                                skipzeros = skipzeros,
                                                parallel = parallel,
                                                expression_module = @__MODULE__,
                                                expression = Val(false))::Tuple{<:Function, <:Function}

    push!(nsss_solver_cache_init_tmp, [haskey(𝓂.constants.post_parameters_macro.guess, v) ? 𝓂.constants.post_parameters_macro.guess[v] : Inf for v in sorted_vars])
    push!(nsss_solver_cache_init_tmp, [Inf])

    lbs = Float64[]
    ubs = Float64[]
    limit_boundaries = 1e12

    for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
        if haskey(𝓂.constants.post_parameters_macro.bounds,i)
            push!(lbs,𝓂.constants.post_parameters_macro.bounds[i][1])
            push!(ubs,𝓂.constants.post_parameters_macro.bounds[i][2])
        else
            push!(lbs,-limit_boundaries)
            push!(ubs, limit_boundaries)
        end
    end

    n_block = block_index

    workspace = Nonlinear_solver_workspace(ϵ, buffer, chol_buffer, lu_buffer)
    ext_workspace = Nonlinear_solver_workspace(ϵᵉ, ext_buffer, ext_chol_buffer, ext_lu_buffer)

    solve_block = ss_solve_block(
            function_and_jacobian(calc_block!::Function, func_exprs::Function, workspace),
            function_and_jacobian(calc_ext_block!::Function, ext_func_exprs::Function, ext_workspace)
        )

    return (sorted_vars = sorted_vars,
            calib_pars_input = Symbol.(calib_pars_input),
            other_vars_input = Symbol.(other_vars_input),
            lbs = lbs,
            ubs = ubs,
            n_block = n_block,
            solve_block = solve_block,
            ss_and_aux_equations = ss_and_aux_equations,
            ss_and_aux_equations_error = ss_and_aux_equations_error)
end

struct PartialSolveResult{T,E}
    remaining_vars::Vector{T}
    solved_vars::Vector{T}
    remaining_eqs::Vector{E}
    solved_exprs::Vector{E}
    remaining_var_indices::Vector{Int}
    solved_var_indices::Vector{Int}
    remaining_eq_indices::Vector{Int}
    solved_eq_indices::Vector{Int}
end

function partial_solve(eqs_to_solve::Vector{E}, vars_to_solve::Vector{T}, incidence_matrix_subset; avoid_solve::Bool = false)::PartialSolveResult{T,E} where {E, T}
    for n in length(eqs_to_solve)-1:-1:2
        for eq_combo in combinations(1:length(eqs_to_solve), n)
            var_indices_to_select_from = findall([sum(incidence_matrix_subset[:,eq_combo],dims = 2)...] .> 0)
            var_indices_in_remaining_eqs = findall([sum(incidence_matrix_subset[:,setdiff(1:length(eqs_to_solve),eq_combo)],dims = 2)...] .> 0)

            for var_combo in combinations(var_indices_to_select_from, n)
                remaining_vars_in_remaining_eqs = setdiff(var_indices_in_remaining_eqs, var_combo)
                if length(remaining_vars_in_remaining_eqs) == length(eqs_to_solve) - n
                    if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve[eq_combo]))) > 15
                        soll = nothing
                    else
                        soll = solve_symbolically(eqs_to_solve[eq_combo], vars_to_solve[var_combo])
                    end

                    if !(isnothing(soll) || isempty(soll))
                        soll_collected = E.(collect(values(soll)))
                        solved_var_indices = Int[var_combo...]
                        remaining_var_indices = [i for i in 1:length(eqs_to_solve) if i ∉ solved_var_indices]
                        solved_eq_indices = Int[eq_combo...]
                        remaining_eq_indices = [i for i in 1:length(eqs_to_solve) if i ∉ solved_eq_indices]

                        return PartialSolveResult(
                            vars_to_solve[remaining_var_indices],
                            vars_to_solve[solved_var_indices],
                            eqs_to_solve[remaining_eq_indices],
                            soll_collected,
                            remaining_var_indices,
                            solved_var_indices,
                            remaining_eq_indices,
                            solved_eq_indices,
                        )
                    end
                end
            end
        end
    end

    return PartialSolveResult(T[], T[], E[], E[], Int[], Int[], Int[], Int[])
end

function make_equation_robust_to_domain_errors(eqs,
                                                vars_to_exclude::Vector{Vector{Symbol}},
                                                bounds::Dict{Symbol,Tuple{Float64,Float64}},
                                                ➕_vars::Vector{Symbol},
                                                unique_➕_eqs;
                                                precompile::Bool = false)
    ss_and_aux_equations = Expr[]
    ss_and_aux_equations_dep = Expr[]
    ss_and_aux_equations_error = Expr[]
    ss_and_aux_equations_error_dep = Expr[]
    rewritten_eqs = Union{Expr,Symbol}[]
    for eq in eqs
        if eq isa Symbol
            push!(rewritten_eqs, eq)
        elseif eq isa Expr
            rewritten_eq = postwalk(x ->
                x isa Expr ?
                    x.head == :call ?
                        x.args[1] == :* ?
                            x.args[2] isa Int ?
                                x.args[3] isa Int ?
                                    x :
                                Expr(:call, :*, x.args[3:end]..., x.args[2]) :
                            x :
                        x.args[1] ∈ [:^] ?
                            !(x.args[3] isa Int) ?
                                x.args[2] isa Symbol ?
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
                                x.args[2].head == :call ?
                                    begin
                                        if precompile
                                            replacement = x.args[2]
                                        else
                                            replacement = simplify(x.args[2])
                                        end

                                        if !(replacement isa Int)
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
                            x.args[2] isa Symbol ?
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
                            x.args[2].head == :call ?
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int)
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
                            x.args[2] isa Symbol ?
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
                            x.args[2].head == :call ?
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int)
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
                            x.args[2] isa Symbol ?
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
                            x.args[2].head == :call ?
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int)
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
                            x.args[2] isa Symbol ?
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
                            x.args[2].head == :call ?
                                begin
                                    if precompile
                                        replacement = x.args[2]
                                    else
                                        replacement = simplify(x.args[2])
                                    end

                                    if !(replacement isa Int)
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

function compile_exprs_to_func(exprs::Vector, 𝔖, 𝔓_ext, placeholder_dict, back_to_array_dict;
                                cse = true, skipzeros = true, nnz_parallel_threshold::Int = 1000000)
    sym_exprs = Symbolics.Num[]
    for expr in exprs
        if expr isa Number
            push!(sym_exprs, Symbolics.Num(expr))
        else
            clean_expr = postwalk(x -> x isa Expr && length(x.args) >= 2 && x.args[1] == :conjugate ? x.args[2] : x, expr)
            replaced = replace_symbols(clean_expr, placeholder_dict)
            sym = Symbolics.parse_expr_to_symbolic(replaced, @__MODULE__)
            sym = Symbolics.substitute(sym, back_to_array_dict)
            push!(sym_exprs, sym)
        end
    end

    lennz = length(sym_exprs)
    parallel = lennz > nnz_parallel_threshold ?
        Symbolics.ShardedForm(1500, 4) : Symbolics.SerialForm()

    _, func! = Symbolics.build_function(sym_exprs, 𝔖, 𝔓_ext,
        cse = cse, skipzeros = skipzeros,
        parallel = parallel,
        expression_module = @__MODULE__,
        expression = Val(false))::Tuple{<:Function, <:Function}

    return func!
end

function append_numerical_step!(builder::NSSSSolverBuilder, block_meta, sol_name_to_index, ext_param_to_index,
                               𝔖, 𝔓_ext, placeholder_dict, back_to_array_dict,
                               global_solvetime_aux_sub::Dict{Symbol, Union{Symbol, Expr}} = Dict{Symbol, Union{Symbol, Expr}}())
    write_indices = [sol_name_to_index[v] for v in block_meta.sorted_vars]
    param_gather_indices = [ext_param_to_index[p] for p in block_meta.calib_pars_input]
    var_gather_indices = [sol_name_to_index[v] for v in block_meta.other_vars_input]

    aux_func! = nothing
    aux_write_indices = Int[]
    aux_error_func! = nothing
    aux_error_size = 0

    if !isempty(block_meta.ss_and_aux_equations)
        model_aux_names = Symbol[]
        model_aux_rhs = Any[]
        model_aux_sub = Dict{Symbol, Any}()
        for eq in block_meta.ss_and_aux_equations
            if eq isa Expr && eq.head == :(=)
                lhs = eq.args[1]
                rhs = eq.args[2]
                expanded_rhs = isempty(global_solvetime_aux_sub) ? rhs : replace_symbols(rhs, global_solvetime_aux_sub)
                expanded_rhs = isempty(model_aux_sub) ? expanded_rhs : replace_symbols(expanded_rhs, model_aux_sub)
                if haskey(sol_name_to_index, lhs)
                    push!(model_aux_names, lhs)
                    push!(model_aux_rhs, expanded_rhs)
                    model_aux_sub[lhs] = expanded_rhs
                else
                    global_solvetime_aux_sub[lhs] = expanded_rhs
                end
            end
        end
        if !isempty(model_aux_rhs)
            aux_write_indices = [sol_name_to_index[v] for v in model_aux_names]
            aux_func! = compile_exprs_to_func(model_aux_rhs, 𝔖, 𝔓_ext, placeholder_dict, back_to_array_dict)
        end
    end

    if !isempty(block_meta.ss_and_aux_equations_error)
        inlined_errors = isempty(global_solvetime_aux_sub) ? block_meta.ss_and_aux_equations_error : [replace_symbols(e, global_solvetime_aux_sub) for e in block_meta.ss_and_aux_equations_error]
        aux_error_size = length(inlined_errors)
        aux_error_func! = compile_exprs_to_func(inlined_errors,
                                                 𝔖, 𝔓_ext, placeholder_dict, back_to_array_dict)
    end

    desc = "Numerical block $(block_meta.n_block): $(join(string.(block_meta.sorted_vars), ", "))"

    push_numerical_step!(builder;
        solve_block = block_meta.solve_block,
        block_index = block_meta.n_block,
        write_indices = write_indices,
        param_gather_indices = param_gather_indices,
        var_gather_indices = var_gather_indices,
        lbs = block_meta.lbs,
        ubs = block_meta.ubs,
        aux_func! = aux_func!,
        aux_write_indices = aux_write_indices,
        aux_error_func! = aux_error_func!,
        aux_error_size = aux_error_size,
        description = desc,
    )
end

function write_steady_state_solver_function!(𝓂::ℳ, symbolic_enabled::Bool = false, symbolics_data::Union{Nothing, symbolics} = nothing;
                                            verbose::Bool = false,
                                            avoid_solve::Bool = false)
    symbolic_enabled = symbolic_enabled && (symbolics_data !== nothing)

    unknowns = if symbolics_data === nothing
        union(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.equations.calibration_parameters)
    else
        union(symbolics_data.calibration_equations_parameters, symbolics_data.vars_in_ss_equations)
    end

    n_equations_total = if symbolics_data === nothing
        length(𝓂.equations.steady_state_aux) + length(𝓂.equations.calibration)
    else
        length(symbolics_data.ss_equations) + length(symbolics_data.calibration_equations)
    end
    @assert length(unknowns) <= n_equations_total "Unable to solve steady state. More unknowns than equations."

    incidence_matrix = spzeros(Int, length(unknowns), length(unknowns))

    eq_list = if symbolics_data === nothing
        empty_var_redundant_list = [Symbol[] for _ in eachindex(𝓂.constants.post_model_macro.var_list_aux_SS)]
        vcat(
            union.(
                setdiff.(
                    union.(
                        𝓂.constants.post_model_macro.var_list_aux_SS,
                        𝓂.constants.post_model_macro.ss_list_aux_SS,
                    ),
                    empty_var_redundant_list,
                ),
                𝓂.constants.post_model_macro.par_list_aux_SS,
            ),
            union.(
                𝓂.constants.post_parameters_macro.ss_calib_list,
                𝓂.constants.post_parameters_macro.par_calib_list,
            ),
        )
    else
        vcat(
            union.(
                setdiff.(
                    union.(
                        symbolics_data.var_list_aux_SS,
                        symbolics_data.ss_list_aux_SS,
                    ),
                    symbolics_data.var_redundant_list,
                ),
                symbolics_data.par_list_aux_SS,
            ),
            union.(
                symbolics_data.ss_calib_list,
                symbolics_data.par_calib_list,
            ),
        )
    end

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

    ss_equations = if symbolics_data === nothing
        vcat(𝓂.equations.steady_state_aux, 𝓂.equations.calibration)
    else
        vcat(symbolics_data.ss_equations, symbolics_data.calibration_equations)
    end

    output_var_names = unique(Symbol.(replace.(string.(sort(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")))
    calib_param_names = 𝓂.equations.calibration_parameters
    plus_var_names = Symbol.(𝓂.constants.post_model_macro.➕_vars)
    all_sol_names = vcat(output_var_names, calib_param_names, plus_var_names)
    n_sol = length(all_sol_names)
    sol_name_to_index = Dict(name => i for (i, name) in enumerate(all_sol_names))
    plus_var_count_at_start = length(plus_var_names)

    for d in union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future)
        raw_name = Symbol(d)
        stripped_name = Symbol(replace(string(d), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
        if raw_name != stripped_name && haskey(sol_name_to_index, stripped_name)
            sol_name_to_index[raw_name] = sol_name_to_index[stripped_name]
        end
    end

    output_names_full = vcat(
        Symbol.(replace.(string.(sort(union(
            𝓂.constants.post_model_macro.var,
            𝓂.constants.post_model_macro.exo_past,
            𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),
        calib_param_names
    )
    output_indices = [sol_name_to_index[name] for name in output_names_full]

    raw_param_names = collect(𝓂.constants.post_complete_parameters.parameters)
    n_raw_params = length(raw_param_names)
    calib_no_var_names = Symbol[expr.args[1] for expr in 𝓂.equations.calibration_no_var]
    ext_param_names = vcat(raw_param_names, calib_no_var_names)
    n_ext_params = length(ext_param_names)
    ext_param_to_index = Dict(name => i for (i, name) in enumerate(ext_param_names))

    exo_zero_indices = Int[]
    for d in union(𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future)
        dns = Symbol(replace(string(d), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
        if haskey(sol_name_to_index, dns)
            push!(exo_zero_indices, sol_name_to_index[dns])
        end
    end

    n_sol_max = n_sol + 2 * length(ss_equations)
    MacroModelling.Symbolics.@variables 𝔖[1:n_sol_max] 𝔓_ext[1:n_ext_params]

    global_placeholder = Dict{Symbol, Symbol}()
    global_back_to_array = Dict{MacroModelling.Symbolics.Num, MacroModelling.Symbolics.Num}()

    for (name, idx) in sol_name_to_index
        sym = Symbol("𝔖_$idx")
        global_placeholder[name] = sym
        global_back_to_array[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = 𝔖[idx]
    end
    for (name, idx) in ext_param_to_index
        sym = Symbol("𝔓e_$idx")
        global_placeholder[name] = sym
        global_back_to_array[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = 𝔓_ext[idx]
    end

    MacroModelling.Symbolics.@variables P_raw[1:n_raw_params]

    pp_back = Dict{MacroModelling.Symbolics.Num, MacroModelling.Symbolics.Num}()
    for i in 1:n_raw_params
        sym = Symbol("Praw_$i")
        pp_back[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = P_raw[i]
    end

    bounded_param_exprs_for_sub = Dict{Symbol, Union{Symbol, Expr}}()
    for (i, par) in enumerate(raw_param_names)
        if haskey(𝓂.constants.post_parameters_macro.bounds, par)
            lb, ub = 𝓂.constants.post_parameters_macro.bounds[par]
            bounded_param_exprs_for_sub[par] = :(min(max($(Symbol("Praw_$i")), $lb), $ub))
        else
            bounded_param_exprs_for_sub[par] = Symbol("Praw_$i")
        end
    end

    ext_param_sym_exprs = MacroModelling.Symbolics.Num[]
    for (i, par) in enumerate(raw_param_names)
        if haskey(𝓂.constants.post_parameters_macro.bounds, par)
            lb, ub = 𝓂.constants.post_parameters_macro.bounds[par]
            push!(ext_param_sym_exprs, min(max(P_raw[i], lb), ub))
        else
            push!(ext_param_sym_exprs, P_raw[i])
        end
    end

    calib_expr_replacements = Dict{Symbol, Union{Symbol, Expr}}()
    for expr in 𝓂.equations.calibration_no_var
        lhs = expr.args[1]
        rhs = expr.args[2]
        rhs_expanded = replace_symbols(rhs, calib_expr_replacements)
        rhs_final = replace_symbols(rhs_expanded, bounded_param_exprs_for_sub)
        calib_expr_replacements[lhs] = rhs_final

        sym_expr = MacroModelling.Symbolics.parse_expr_to_symbolic(rhs_final, @__MODULE__)
        sym_expr = MacroModelling.Symbolics.substitute(sym_expr, pp_back)
        push!(ext_param_sym_exprs, sym_expr)
    end

    _, param_prep_func! = MacroModelling.Symbolics.build_function(ext_param_sym_exprs, P_raw,
        cse = true, skipzeros = true,
        parallel = MacroModelling.Symbolics.SerialForm(),
        expression_module = @__MODULE__,
        expression = Val(false))::Tuple{<:Function, <:Function}

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    nsss_solver_cache_init_tmp = []

    solved_vars = []
    solved_vals = []

    min_max_error_exprs = []
    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()
    global_solvetime_aux_sub = Dict{Symbol, Union{Symbol, Expr}}()
    builder = NSSSSolverBuilder()
    numerical_block_count = 0

    while n > 0
        if length(eqs[:,eqs[2,:] .== n]) == 2
            var_to_solve_for = unknowns[vars[:,vars[2,:] .== n][1]]

            eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]
            minmax_rewritten = false

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
                push!(min_max_error_exprs, parsed_eq_to_solve_for)
                eq_to_solve = minmax_fixed_eqs
                minmax_rewritten = true
            end

            if symbolics_data === nothing || avoid_solve || minmax_rewritten || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                if eq_to_solve isa SPyPyC.Sym{PythonCall.Core.Py} && var_to_solve_for isa SPyPyC.Sym{PythonCall.Core.Py}
                    soll = solve_symbolically(eq_to_solve, var_to_solve_for)
                else
                    soll = nothing
                end
            end

            if isnothing(soll) || isempty(soll)
                if verbose && symbolic_enabled
                    println("Failed finding solution symbolically for: ",var_to_solve_for," in: ",eq_to_solve)
                end

                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]

                numerical_block_count += 1
                block_meta = write_block_solution!(𝓂, [var_to_solve_for], [eq_to_solve], relevant_pars_across, nsss_solver_cache_init_tmp, eq_idx_in_block_to_solve, atoms_in_equations_list, solved_vars, solved_vals, block_index = numerical_block_count)

                current_plus_count = length(𝓂.constants.post_model_macro.➕_vars)
                if current_plus_count > plus_var_count_at_start
                    for pvi in (plus_var_count_at_start + 1):current_plus_count
                        pv = Symbol(𝓂.constants.post_model_macro.➕_vars[pvi])
                        if !haskey(sol_name_to_index, pv)
                            push!(all_sol_names, pv)
                            idx = length(all_sol_names)
                            sol_name_to_index[pv] = idx
                            sym = Symbol("𝔖_$idx")
                            global_placeholder[pv] = sym
                            global_back_to_array[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = 𝔖[idx]
                        end
                    end
                    plus_var_count_at_start = current_plus_count
                end

                append_numerical_step!(builder, block_meta, sol_name_to_index, ext_param_to_index,
                                       𝔖, 𝔓_ext, global_placeholder, global_back_to_array, global_solvetime_aux_sub)

            elseif soll[1].is_number == true
                if var_to_solve_for isa SPyPyC.Sym{PythonCall.Core.Py} && soll[1] isa SPyPyC.Sym{PythonCall.Core.Py}
                    ss_equations = [eq isa SPyPyC.Sym{PythonCall.Core.Py} ? replace_symbolic(eq, var_to_solve_for, soll[1]) : eq for eq in ss_equations]
                end

                push!(solved_vars, Symbol(var_to_solve_for))
                push!(solved_vals, Meta.parse(string(soll[1])))
                push!(atoms_in_equations_list, [])

                var_name = solved_vars[end]
                val = solved_vals[end]
                widx = sol_name_to_index[var_name]

                if var_name ∈ 𝓂.constants.post_model_macro.➕_vars
                    step_expr = :(max(eps(), $val))
                    eval_func! = compile_exprs_to_func([step_expr], 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)
                else
                    constant_value = Float64(soll[1])
                    eval_func! = let constant_value = constant_value
                        (out, _sol_vec, _params_vec) -> begin
                            out[1] = constant_value
                            return nothing
                        end
                    end
                end

                push_analytical_step!(builder;
                    eval_func! = eval_func!,
                    write_indices = [widx],
                    description = "Constant: $var_name = $val",
                )

            else
                push!(solved_vars, Symbol(var_to_solve_for))
                push!(solved_vals, Meta.parse(string(soll[1])))

                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                push!(atoms_in_equations_list, Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)), Symbol.(soll[1].atoms()))))

                var_name = solved_vars[end]
                val_expr = solved_vals[end]
                widx = sol_name_to_index[var_name]

                if var_name ∈ 𝓂.constants.post_model_macro.➕_vars
                    bounds_tuple = get(𝓂.constants.post_parameters_macro.bounds, var_name, (eps(), 1e12))
                    lb, ub = Float64(bounds_tuple[1]), Float64(bounds_tuple[2])

                    eval_func! = compile_exprs_to_func([val_expr], 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)

                    push_analytical_step!(builder;
                        eval_func! = eval_func!,
                        write_indices = [widx],
                        lower_bounds = [lb],
                        upper_bounds = [ub],
                        has_bounds = trues(1),
                        description = "Analytical ➕: $var_name",
                    )

                    unique_➕_eqs[val_expr] = var_name
                else
                    vars_to_exclude = [vcat(Symbol.(var_to_solve_for), 𝓂.constants.post_model_macro.➕_vars), Symbol[]]

                    rewritten_eqs, ss_and_aux_equations, ss_and_aux_equations_dep, ss_and_aux_equations_error, ss_and_aux_equations_error_dep = make_equation_robust_to_domain_errors([val_expr], vars_to_exclude, 𝓂.constants.post_parameters_macro.bounds, 𝓂.constants.post_model_macro.➕_vars, unique_➕_eqs)

                    current_plus_count = length(𝓂.constants.post_model_macro.➕_vars)
                    if current_plus_count > plus_var_count_at_start
                        for pvi in (plus_var_count_at_start + 1):current_plus_count
                            pv = Symbol(𝓂.constants.post_model_macro.➕_vars[pvi])
                            if !haskey(sol_name_to_index, pv)
                                push!(all_sol_names, pv)
                                idx = length(all_sol_names)
                                sol_name_to_index[pv] = idx
                                sym = Symbol("𝔖_$idx")
                                global_placeholder[pv] = sym
                                global_back_to_array[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = 𝔖[idx]
                            end
                        end
                        plus_var_count_at_start = current_plus_count
                    end

                    all_aux_eqs = vcat(ss_and_aux_equations, ss_and_aux_equations_dep)
                    all_aux_errors = vcat(ss_and_aux_equations_error, ss_and_aux_equations_error_dep)

                    aux_func! = nothing
                    aux_write_indices = Int[]
                    error_func! = nothing
                    error_size = 0

                    model_aux_names = Symbol[]
                    model_aux_rhs = Any[]
                    model_aux_sub = Dict{Symbol, Any}()

                    for eq in all_aux_eqs
                        if eq isa Expr && eq.head == :(=)
                            lhs = eq.args[1]
                            rhs = eq.args[2]
                            expanded_rhs = isempty(global_solvetime_aux_sub) ? rhs : replace_symbols(rhs, global_solvetime_aux_sub)
                            expanded_rhs = isempty(model_aux_sub) ? expanded_rhs : replace_symbols(expanded_rhs, model_aux_sub)
                            if haskey(sol_name_to_index, lhs)
                                push!(model_aux_names, lhs)
                                push!(model_aux_rhs, expanded_rhs)
                                model_aux_sub[lhs] = expanded_rhs
                            else
                                global_solvetime_aux_sub[lhs] = expanded_rhs
                            end
                        end
                    end

                    if !isempty(model_aux_rhs)
                        aux_write_indices = [sol_name_to_index[v] for v in model_aux_names]
                        aux_func! = compile_exprs_to_func(model_aux_rhs, 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)
                    end

                    main_expr = isempty(global_solvetime_aux_sub) ? rewritten_eqs[1] : replace_symbols(rewritten_eqs[1], global_solvetime_aux_sub)
                    eval_func! = compile_exprs_to_func([main_expr], 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)

                    if !isempty(all_aux_errors)
                        inlined_errors = isempty(global_solvetime_aux_sub) ? all_aux_errors : [replace_symbols(e, global_solvetime_aux_sub) for e in all_aux_errors]
                        error_size = length(inlined_errors)
                        error_func! = compile_exprs_to_func(inlined_errors, 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)
                    end

                    has_user_bounds = haskey(𝓂.constants.post_parameters_macro.bounds, var_name) && var_name ∉ 𝓂.constants.post_model_macro.➕_vars
                    if has_user_bounds
                        lb = Float64(𝓂.constants.post_parameters_macro.bounds[var_name][1])
                        ub = Float64(𝓂.constants.post_parameters_macro.bounds[var_name][2])
                        push_analytical_step!(builder;
                            aux_func! = aux_func!,
                            aux_write_indices = aux_write_indices,
                            error_func! = error_func!,
                            error_size = error_size,
                            eval_func! = eval_func!,
                            write_indices = [widx],
                            lower_bounds = [lb],
                            upper_bounds = [ub],
                            has_bounds = trues(1),
                            description = "Analytical bounded: $var_name",
                        )
                    else
                        push_analytical_step!(builder;
                            aux_func! = aux_func!,
                            aux_write_indices = aux_write_indices,
                            error_func! = error_func!,
                            error_size = error_size,
                            eval_func! = eval_func!,
                            write_indices = [widx],
                            description = "Analytical: $var_name",
                        )
                    end
                end
            end
        else
            vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]
            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false

            if symbolic_enabled
                if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = solve_symbolically(eqs_to_solve::Vector{SPyPyC.Sym{PythonCall.Core.Py}}, vars_to_solve::Vector{SPyPyC.Sym{PythonCall.Core.Py}})
                end

                if isnothing(soll) || isempty(soll) || length(intersect((union(SPyPyC.free_symbols.(collect(values(soll)))...) .|> SPyPyC.:↓),(vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose println("Failed finding solution symbolically for: ",vars_to_solve," in: ",eqs_to_solve,". Solving numerically.") end
                    numerical_sol = true
                else
                    if verbose println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " symbolically.") end

                    atoms = reduce(union,map(x->x.atoms(),collect(values(soll))))
                    for a in atoms push!(atoms_in_equations, Symbol(a)) end

                    step_exprs = []
                    step_write_indices = Int[]

                    for v in vars_to_solve
                        push!(solved_vars, Symbol(v))
                        push!(solved_vals, Meta.parse(string(soll[v])))
                        push!(atoms_in_equations_list, Set(Symbol.(soll[v].atoms())))
                        push!(step_exprs, solved_vals[end])
                        push!(step_write_indices, sol_name_to_index[Symbol(v)])
                    end

                    eval_func! = compile_exprs_to_func(step_exprs, 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)

                    push_analytical_step!(builder;
                        eval_func! = eval_func!,
                        write_indices = step_write_indices,
                        description = "Analytical multi: $(join(string.(Symbol.(vars_to_solve)), ", "))",
                    )
                end
            end

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]
            incidence_matrix_subset = incidence_matrix[vars[:,vars[2,:] .== n][1,:], eq_idx_in_block_to_solve]

            if numerical_sol || !symbolic_enabled
                vars_to_solve_reduced = vars_to_solve
                eqs_to_solve_reduced = eqs_to_solve
                eq_idx_in_block_to_solve_reduced = eq_idx_in_block_to_solve

                numerical_block_count += 1
                block_meta = write_block_solution!(𝓂, vars_to_solve_reduced, eqs_to_solve_reduced, relevant_pars_across, nsss_solver_cache_init_tmp, eq_idx_in_block_to_solve_reduced, atoms_in_equations_list, solved_vars, solved_vals, block_index = numerical_block_count)

                if !isnothing(block_meta)
                    current_plus_count = length(𝓂.constants.post_model_macro.➕_vars)
                    if current_plus_count > plus_var_count_at_start
                        for pvi in (plus_var_count_at_start + 1):current_plus_count
                            pv = Symbol(𝓂.constants.post_model_macro.➕_vars[pvi])
                            if !haskey(sol_name_to_index, pv)
                                push!(all_sol_names, pv)
                                idx = length(all_sol_names)
                                sol_name_to_index[pv] = idx
                                sym = Symbol("𝔖_$idx")
                                global_placeholder[pv] = sym
                                global_back_to_array[MacroModelling.Symbolics.parse_expr_to_symbolic(sym, @__MODULE__)] = 𝔖[idx]
                            end
                        end
                        plus_var_count_at_start = current_plus_count
                    end

                    append_numerical_step!(builder, block_meta, sol_name_to_index, ext_param_to_index,
                                           𝔖, 𝔓_ext, global_placeholder, global_back_to_array, global_solvetime_aux_sub)
                end

                if !symbolic_enabled && verbose
                    println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
                end
            end
        end
        n -= 1
    end

    push!(nsss_solver_cache_init_tmp, fill(Inf, length(𝓂.constants.post_complete_parameters.parameters)))
    push!(𝓂.caches.solver_cache, nsss_solver_cache_init_tmp)

    parameters_only_in_par_defs = Set()
    if length(𝓂.equations.calibration_no_var) > 0
        atoms = reduce(union, get_symbols.(𝓂.equations.calibration_no_var))
        [push!(atoms_in_equations, a) for a in atoms]
        [push!(parameters_only_in_par_defs, a) for a in atoms]
    end

    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies, solved_vars[i] => intersect(a, union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_complete_parameters.parameters)))
    end

    push!(dependencies, :SS_relevant_calibration_parameters => intersect(reduce(union, atoms_in_equations_list), 𝓂.constants.post_complete_parameters.parameters))
    if !isempty(min_max_error_exprs)
        minmax_error_func! = compile_exprs_to_func(min_max_error_exprs, 𝔖, 𝔓_ext, global_placeholder, global_back_to_array)
        n_errors = length(min_max_error_exprs)
        push_analytical_step!(builder;
            error_func! = minmax_error_func!,
            error_size = n_errors,
            eval_func! = compile_exprs_to_func([0.0], 𝔖, 𝔓_ext, global_placeholder, global_back_to_array),
            write_indices = Int[],
            description = "Min/Max validation",
        )
    end

    # Patch bounds on ➕ steps in the builder's flat arrays
    if !isempty(𝓂.constants.post_parameters_macro.bounds)
        for i in 1:length(builder.step_types)
            if builder.step_types[i] == ANALYTICAL_STEP && startswith(builder.descriptions[i], "Analytical ➕:")
                wr = builder.write_ranges[i]
                br = builder.bounds_ranges[i]
                for (j_local, j_wr) in enumerate(wr)
                    widx = builder.write_indices[j_wr]
                    name = all_sol_names[widx]
                    if haskey(𝓂.constants.post_parameters_macro.bounds, name)
                        bt = 𝓂.constants.post_parameters_macro.bounds[name]
                        j_br = br[j_local]
                        builder.lower_bounds[j_br] = Float64(bt[1])
                        builder.upper_bounds[j_br] = Float64(bt[2])
                        builder.has_bounds[j_br] = true
                    end
                end
            end
        end
    end

    build_nsss_solver!(𝓂, builder, param_prep_func!)
    n_sol = length(all_sol_names)
    𝓂.constants.post_complete_parameters = update_post_complete_parameters(
        𝓂.constants.post_complete_parameters;
        nsss_dependencies = dependencies,
        nsss_n_sol = n_sol,
        nsss_output_indices = output_indices,
        nsss_n_ext_params = n_ext_params,
        nsss_sol_names = all_sol_names,
        nsss_exo_zero_indices = exo_zero_indices,
        nsss_param_names_ext = ext_param_names,
    )

    return nothing
end

function find_closest_solution(cache, initial_parameters::Vector{Float64}, expected_length::Int)
    current_best = Inf
    closest_solution = cache[end]

    for pars in cache
        if length(pars) < expected_length || !(pars[end] isa Vector{Float64}) || length(pars[end]) != length(initial_parameters)
            continue
        end
        squared_distance = 0.0
        @inbounds for i in eachindex(initial_parameters)
            d = pars[end][i] - initial_parameters[i]
            squared_distance += d * d
        end
        if squared_distance <= current_best
            current_best = squared_distance
            closest_solution = pars
        end
    end

    if !isfinite(current_best)
        if (closest_solution[end] isa Vector{Float64}) && (length(closest_solution[end]) == length(initial_parameters))
            current_best = 0.0
            @inbounds for i in eachindex(initial_parameters)
                d = closest_solution[end][i] - initial_parameters[i]
                current_best += d * d
            end
        else
            current_best = Inf
        end
    end

    return current_best, closest_solution
end

"""
    execute_step!(step_idx, sol_vec, params_vec, closest_solution, 𝓂, ...)

Execute a single NSSS solve step.
Dispatches on `𝓂.constants.nsss_solver.step_types[step_idx]` (ANALYTICAL_STEP or NUMERICAL_STEP).

Uses shared workspace buffers for scratch computations, avoiding per-step allocation.

Returns: (error, iterations, cache_entries::Vector{Vector{Float64}})
"""
function execute_step!(step_idx::Int,
                       sol_vec::Vector{Float64}, params_vec::Vector{Float64},
                       closest_solution, 𝓂, tol, fail_fast_solvers_only,
                       cold_start, solver_parameters, preferred_solver_parameter_idx::Int, verbose)
    
    c = 𝓂.constants.nsss_solver
    f = 𝓂.functions.nsss_solver
    w = 𝓂.workspaces.nsss_solver
    step_type = c.step_types[step_idx]

    error = 0.0

    # Phase 1: Compute auxiliary variables (shared across both step types)
    if f.aux_funcs[step_idx] !== nothing
        aux_wr = c.aux_write_ranges[step_idx]
        n_aux = length(aux_wr)
        aux_buf = @view w.aux_buffer[1:n_aux]
        f.aux_funcs[step_idx](aux_buf, sol_vec, params_vec)
        @inbounds for j in 1:n_aux
            sol_vec[c.aux_write_indices[aux_wr[j]]] = aux_buf[j]
        end
    end

    if step_type == ANALYTICAL_STEP
        # Error check (analytical domain-safety)
        if f.error_funcs[step_idx] !== nothing
            err_n = c.error_sizes[step_idx]
            err_buf = @view w.error_buffer[1:err_n]
            f.error_funcs[step_idx](err_buf, sol_vec, params_vec)
            error += sum(abs, err_buf)
        end

        # Main evaluation
        wr = c.write_ranges[step_idx]
        n_write = length(wr)
        if n_write > 0
            main_buf = @view w.main_buffer[1:n_write]
            f.eval_funcs[step_idx](main_buf, sol_vec, params_vec)
            br = c.bounds_ranges[step_idx]
            @inbounds for j in 1:n_write
                raw = main_buf[j]
                widx = c.write_indices[wr[j]]
                if !isempty(br) && c.has_bounds[br[j]]
                    clamped = clamp(raw, c.lower_bounds[br[j]], c.upper_bounds[br[j]])
                    error += abs(clamped - raw)
                    sol_vec[widx] = clamped
                else
                    sol_vec[widx] = raw
                end
            end
        elseif f.eval_funcs[step_idx] !== nothing
            # Min/Max validation step: no writes but eval_func exists
            f.eval_funcs[step_idx](@view(w.main_buffer[1:1]), sol_vec, params_vec)
        end

        return error, 0, EMPTY_NSSS_STEP_CACHE

    else # NUMERICAL_STEP
        # Gather params_and_solved_vars into shared main_buffer
        pgr = c.param_gather_ranges[step_idx]
        vgr = c.var_gather_ranges[step_idx]
        n_params = length(pgr)
        n_vars = length(vgr)
        gather_size = n_params + n_vars

        params_and_solved_vars = w.params_and_solved_vars_buffer
        resize!(params_and_solved_vars, gather_size)
        @inbounds for j in 1:n_params
            params_and_solved_vars[j] = params_vec[c.param_gather_indices[pgr[j]]]
        end
        @inbounds for j in 1:n_vars
            params_and_solved_vars[n_params + j] = sol_vec[c.var_gather_indices[vgr[j]]]
        end

        # Build initial guesses
        block_idx = c.block_indices[step_idx]
        cache_sol_idx = 2*(block_idx-1)+1
        cache_par_idx = 2*block_idx
        cache_sol = cache_sol_idx <= length(closest_solution) ? closest_solution[cache_sol_idx] : Float64[]
        cache_par = cache_par_idx <= length(closest_solution) ? closest_solution[cache_par_idx] : Float64[Inf]

        wr = c.write_ranges[step_idx]
        n_write = length(wr)
        nbr = c.numerical_bounds_ranges[step_idx]
        guess_len = min(n_write, length(nbr))

        guess_buf = @view w.guess_buffer[1:guess_len]
        copy_len = min(length(cache_sol), guess_len)
        @inbounds for i in 1:copy_len
            guess_buf[i] = clamp(cache_sol[i], c.numerical_lbs[nbr[i]], c.numerical_ubs[nbr[i]])
        end
        @inbounds for i in (copy_len + 1):guess_len
            guess_buf[i] = clamp(0.5 * (c.numerical_lbs[nbr[i]] + c.numerical_ubs[nbr[i]]),
                                 c.numerical_lbs[nbr[i]], c.numerical_ubs[nbr[i]])
        end

        # Use workspace inits container
        resize!(w.inits[1], guess_len)
        if guess_len > 0
            copyto!(w.inits[1], 1, guess_buf, 1, guess_len)
        end
        w.inits[2] = cache_par

        lbs = w.lbs_buffer
        ubs = w.ubs_buffer
        n_bounds = length(nbr)
        resize!(lbs, n_bounds)
        resize!(ubs, n_bounds)
        @inbounds for i in 1:n_bounds
            lbs[i] = c.numerical_lbs[nbr[i]]
            ubs[i] = c.numerical_ubs[nbr[i]]
        end

        # Call block solver
        solve_block = f.solve_blocks[step_idx]
        if solve_block === nothing
            if verbose
                println("Missing numerical solve block for step $(step_idx)")
            end
            return Inf, 0, EMPTY_NSSS_STEP_CACHE
        end

        solution = block_solver(
            params_and_solved_vars,
            block_idx,
            solve_block,
            w.inits,
            lbs,
            ubs,
            solver_parameters,
            preferred_solver_parameter_idx,
            fail_fast_solvers_only,
            cold_start,
            verbose
        )

        error += solution[2][1]
        iters = solution[2][2]
        if error > tol.NSSS_acceptance_tol
            if verbose
                println("Failed after solving block with error $error")
            end
            return error, iters, EMPTY_NSSS_STEP_CACHE
        end

        # Domain safety error check after block solve
        if f.error_funcs[step_idx] !== nothing
            err_n = c.aux_error_sizes[step_idx]
            err_buf = @view w.error_buffer[1:err_n]
            f.error_funcs[step_idx](err_buf, sol_vec, params_vec)
            error += sum(abs, err_buf)
            if error > tol.NSSS_acceptance_tol
                if verbose
                    println("Failed for aux variables with error $error")
                end
                return error, iters, EMPTY_NSSS_STEP_CACHE
            end
        end

        # Write results to solution vector
        sol = solution[1]
        @inbounds for j in 1:n_write
            sol_vec[c.write_indices[wr[j]]] = sol[j]
        end

        # Build cache entries for this block
        cache_entries = [
            typeof(sol) == Vector{Float64} ? copy(sol) : ℱ.value.(sol),
            typeof(params_and_solved_vars) == Vector{Float64} ? copy(params_and_solved_vars) : ℱ.value.(params_and_solved_vars)
        ]

        return error, iters, cache_entries
    end
end


# ============================================================================
# Orchestrator: solve_nsss_steps
# ============================================================================

"""
    solve_nsss_steps(parameters, 𝓂, tol, verbose, fail_fast_solvers_only,
                     closest_solution, cold_start, solver_params)

Solve the NSSS by executing pipeline steps in a single pass.

Steps are dispatched via `execute_step!` which uses the pipeline's shared
workspace buffers. Steps are executed in order, filling the solution vector
progressively.

Returns: (SS_and_pars, (solution_error, iters), nsss_solver_cache_tmp)
"""
function solve_nsss_steps(
    parameters::Vector{Float64},
    𝓂::ℳ,
    tol::Tolerances,
    verbose::Bool,
    fail_fast_solvers_only::Bool,
    closest_solution,
    cold_start::Bool,
    solver_params::Vector{solver_parameters},
    preferred_solver_parameter_idx::Int
)
    nsss_n_ext_params = 𝓂.constants.post_complete_parameters.nsss_n_ext_params
    nsss_n_sol = 𝓂.constants.post_complete_parameters.nsss_n_sol
    nsss_output_indices = 𝓂.constants.post_complete_parameters.nsss_output_indices
    nsss_consts = 𝓂.constants.nsss_solver
    nsss_ws = 𝓂.workspaces.nsss_solver
    
    # Prepare extended parameter vector (raw params → bounded + calibration_no_var)
    params_vec = nsss_ws.params_vec_buffer
    if length(params_vec) != nsss_n_ext_params
        resize!(params_vec, nsss_n_ext_params)
    end
    𝓂.functions.nsss_param_prep!(params_vec, parameters)
    
    # Initialize solution vector from workspace buffer
    sol_vec = nsss_ws.sol_vec_buffer
    if length(sol_vec) != nsss_n_sol
        resize!(sol_vec, nsss_n_sol)
    end
    fill!(sol_vec, 0.0)
    
    # Single pass through all steps
    nsss_solver_cache_tmp = Vector{Float64}[]
    solution_error = 0.0
    iters = 0
    
    n_steps = nsss_consts.n_steps
    for step_idx in 1:n_steps
        step_error, step_iters, step_cache = execute_step!(
            step_idx, sol_vec, params_vec, closest_solution, 𝓂, tol,
            fail_fast_solvers_only, cold_start, solver_params, preferred_solver_parameter_idx, verbose
        )
        
        solution_error += step_error
        iters += step_iters
        if !isempty(step_cache)
            append!(nsss_solver_cache_tmp, step_cache)
        end
        
        if solution_error > tol.NSSS_acceptance_tol
            if verbose
                println("Step '$(nsss_consts.descriptions[step_idx])' failed with accumulated error $solution_error")
            end
            break
        end
    end
    
    # Build SS_and_pars from solution vector using output indices
    SS_and_pars = sol_vec[nsss_output_indices]
    
    # If failed to converge, return zeros
    if solution_error >= tol.NSSS_acceptance_tol
        fill!(SS_and_pars, 0.0)
    end
    
    # Append parameters to cache 
    if isempty(nsss_solver_cache_tmp)
        nsss_solver_cache_tmp = [parameters]
    else
        push!(nsss_solver_cache_tmp, parameters)
    end
    
    return SS_and_pars, (solution_error, iters), nsss_solver_cache_tmp
end


# ============================================================================
# Wrapper: solve_nsss_wrapper (handles cache + continuation method)
# ============================================================================

"""
    solve_nsss_wrapper(
        parameter_values::Vector{<:Real},
        𝓂::ℳ,
        tol::Tolerances,
        verbose::Bool,
        cold_start::Bool,
        solver_params::Vector{solver_parameters}
    )::Tuple{Vector, Tuple{Real, Int}}

Normal Julia function wrapper for NSSS solving.

This function handles cache management and continuation scaling for solving
the non-stochastic steady state using the step-based NSSS solver.

The continuation method gradually transitions from a cached solution to the
target parameters using a scaling approach, which improves convergence.

# Arguments
- `parameter_values`: Parameter values to solve at
- `𝓂`: Model structure
- `tol`: Tolerance settings
- `verbose`: Whether to print verbose output
- `cold_start`: Whether this is a cold start (limits iterations to 1)
- `solver_params`: Solver configuration

# Keyword arguments
- `continuation_cache_capacity`: Size of local continuation cache buffer
- `continuation_max_iters`: Maximum continuation iterations for warm starts
- `stall_tolerance`: Threshold to stop when continuation scale no longer moves
- `cache_push_distance_tol`: Distance threshold before pushing solved cache to model cache
- `scale_snap_threshold`: Scale above which continuation snaps directly to `1.0`
- `scale_success_weight`: Weight on current scale after successful continuation step
- `scale_failure_weight`: Weight on current scale after failed continuation step

# Returns
- Tuple of (solution_vector, (solution_error, iterations))
"""
function solve_nsss_wrapper(
    parameter_values::Vector{<:Real},
    𝓂::ℳ,
    tol::Tolerances,
    verbose::Bool,
    cold_start::Bool,
    solver_params::Vector{solver_parameters}
    ;
    continuation_cache_capacity::Int = 500,
    continuation_max_iters::Int = 500,
    stall_tolerance::Float64 = 1e-2,
    cache_push_distance_tol::Float64 = 1e-8,
    scale_snap_threshold::Float64 = 0.95,
    scale_success_weight::Float64 = 0.4,
    scale_failure_weight::Float64 = 0.3,
    preferred_solver_parameter_idx::Int = 1,
)::Tuple{Vector, Tuple{Real, Int}}

    n_numerical_steps = count(==(NUMERICAL_STEP), 𝓂.constants.nsss_solver.step_types)
    
    # Type conversion for AD compatibility
    initial_parameters = parameter_values isa Vector{Float64} ? 
                        parameter_values : 
                        ℱ.value.(parameter_values)

    n_solver_parameters = length(solver_params)
    @assert n_solver_parameters > 0 "At least one steady-state solver parameter set is required."
    preferred_idx = clamp(preferred_solver_parameter_idx, 1, n_solver_parameters)
    
    # Find closest cached solution as starting point
    expected_cache_length = 2 * n_numerical_steps + 1
    current_best_init, closest_solution_init = find_closest_solution(𝓂.caches.solver_cache, initial_parameters, expected_cache_length)

    if !cold_start && !isfinite(current_best_init)
        SS_and_pars, (solution_error, iters), nsss_solver_cache_tmp = solve_nsss_steps(
            initial_parameters,
            𝓂,
            tol,
            verbose,
            false,
            closest_solution_init,
            true,
            solver_params,
            preferred_idx,
        )

        if solution_error < tol.NSSS_acceptance_tol
            reverse_diff_friendly_push!(𝓂.caches.solver_cache, nsss_solver_cache_tmp)
            return SS_and_pars, (solution_error, iters)
        end
    end
    
    # Initialize continuation method variables
    range_iters = 0
    solution_error = 1.0
    solved_scale = 0.0
    scale = 1.0
    SS_and_pars = Float64[]
    
    # Local intermediate cache for warm starts at intermediate scales
    continuation_cache = CircularBuffer{Vector{Vector{Float64}}}(continuation_cache_capacity)
    push!(continuation_cache, closest_solution_init)
    scaled_parameters = similar(initial_parameters)
    
    # Continuation method: iterate with scaling to gradually approach target
    max_iters = cold_start ? 1 : continuation_max_iters

    while range_iters <= max_iters && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
        range_iters += 1
        fail_fast_solvers_only = range_iters > 1

        # Stall detection: stop if scale hasn't moved
        if abs(solved_scale - scale) < stall_tolerance
            break
        end

        # Find closest solution from local intermediate cache
        current_best, closest_solution = find_closest_solution(continuation_cache, initial_parameters, expected_cache_length)
        
        # Interpolate parameters between target and cached solution
        if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
            @inbounds for i in eachindex(initial_parameters)
                scaled_parameters[i] = scale * initial_parameters[i] + (1 - scale) * closest_solution_init[end][i]
            end
            parameters = scaled_parameters
        else
            parameters = initial_parameters
        end
        
        # Call step-based solver
        SS_and_pars, (solution_error, iters), nsss_solver_cache_tmp = solve_nsss_steps(
            parameters,
            𝓂,
            tol,
            verbose,
            fail_fast_solvers_only,
            closest_solution,
            cold_start,
            solver_params,
            preferred_idx
        )
        
        # Check convergence and update scaling
        if solution_error < tol.NSSS_acceptance_tol
            solved_scale = scale
            
            if scale == 1
                if current_best > cache_push_distance_tol
                    reverse_diff_friendly_push!(𝓂.caches.solver_cache, nsss_solver_cache_tmp)
                end
                return SS_and_pars, (solution_error, iters)
            end
            
            # Cache intermediate result for warm starts
            push!(continuation_cache, nsss_solver_cache_tmp)
            
            # Advance scale toward 1.0
            if scale > scale_snap_threshold
                scale = 1.0
            else
                scale = scale * scale_success_weight + (1 - scale_success_weight)
            end
        else
            # Failed: pull scale back toward last successful scale
            scale = scale * scale_failure_weight + solved_scale * (1 - scale_failure_weight)
        end
    end

    # Warm-start continuation failed: retry once with a direct cold-start pass.
    SS_and_pars, (solution_error, iters), nsss_solver_cache_tmp = solve_nsss_steps(
        initial_parameters,
        𝓂,
        tol,
        verbose,
        false,
        closest_solution_init,
        true,
        solver_params,
        preferred_idx,
    )

    if solution_error < tol.NSSS_acceptance_tol
        reverse_diff_friendly_push!(𝓂.caches.solver_cache, nsss_solver_cache_tmp)
        return SS_and_pars, (solution_error, iters)
    end
    
    # Failed to converge - return zeros with matching output length
    n_output = length(𝓂.constants.post_complete_parameters.nsss_output_indices)
    
    return zeros(n_output), (1.0, 0)
end
