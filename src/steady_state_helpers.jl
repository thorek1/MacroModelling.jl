# Shared helper functions for steady-state solver generation
# These functions eliminate code duplication between the two versions of
# write_steady_state_solver_function! (symbolic and precompiled/numerical)

"""
    compute_block_triangularization(unknowns, eq_list)

Compute block triangular form ordering for the steady-state equation system.
Uses BlockTriangularForm.order to decompose the system into ordered blocks.

# Arguments
- `unknowns`: Vector of unknown variables/parameters to solve for
- `eq_list`: Vector of sets, each containing symbols that appear in that equation

# Returns
- `vars`: Variable ordering with block assignments (2×n matrix)
- `eqs`: Equation ordering with block assignments (2×n matrix)
- `n_blocks`: Number of blocks
- `incidence_matrix`: Sparse incidence matrix
"""
function compute_block_triangularization(unknowns, eq_list)
    incidence_matrix = spzeros(Int, length(unknowns), length(unknowns))

    for (i, u) in enumerate(unknowns)
        for (k, e) in enumerate(eq_list)
            incidence_matrix[i, k] = u ∈ e
        end
    end

    Q, P, R, nmatch, n_blocks = BlockTriangularForm.order(incidence_matrix)
    R̂ = Int[]
    for i in 1:n_blocks
        [push!(R̂, n_blocks - i + 1) for ii in R[i]:R[i+1]-1]
    end
    push!(R̂, 1)

    vars = hcat(P, R̂)'
    eqs = hcat(Q, R̂)'
    
    @assert all(eqs[1,:] .> 0) "Could not solve system of steady state and calibration equations. Number of redundant equations: " * repr(sum(eqs[1,:] .< 0)) * ". Try defining some steady state values as parameters (e.g. r[ss] -> r̄). Nonstationary variables are not supported as of now."
    
    return vars, eqs, n_blocks, incidence_matrix
end


"""
    build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)

Build the parameter assignment expressions for use in the generated solve_SS function.

# Returns
Vector of expressions like `:(param = parameters[i])` for parameters that appear
in the steady-state equations.
"""
function build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    parameters_in_equations = []

    for (i, parss) in enumerate(𝓂.constants.post_complete_parameters.parameters) 
        if parss ∈ union(atoms_in_equations, relevant_pars_across)
            push!(parameters_in_equations, :($parss = parameters[$i]))
        end
    end
    
    return parameters_in_equations
end


"""
    build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)

Build and store the dependency tracking information in 𝓂.NSSS.dependencies.
Tracks which parameters and variables each solved variable depends on.

# Returns
The dependencies vector.
"""
function build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)
    dependencies = []
    for (i, a) in enumerate(atoms_in_equations_list)
        push!(dependencies, solved_vars[i] => intersect(a, union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_complete_parameters.parameters)))
    end

    push!(dependencies, :SS_relevant_calibration_parameters => intersect(reduce(union, atoms_in_equations_list), 𝓂.constants.post_complete_parameters.parameters))

    𝓂.NSSS.dependencies = dependencies
    
    return dependencies
end


"""
    build_dyn_exos_expressions(𝓂)

Build expressions that set dynamic exogenous variables (past/future) to zero for steady state.

# Returns
Vector of expressions like `:(exo_var = 0)`.
"""
function build_dyn_exos_expressions(𝓂)
    dyn_exos = []
    for dex in union(𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future)
        push!(dyn_exos, :($dex = 0))
    end
    return dyn_exos
end


"""
    build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

Build expressions that clamp parameters to their defined bounds.

# Returns
Vector of bound-clamping expressions like `:(param = min(max(param, lb), ub))`.
"""
function build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)
    par_bounds = []
    
    for varpar in intersect(𝓂.constants.post_complete_parameters.parameters, union(atoms_in_equations, relevant_pars_across))
        if haskey(𝓂.constants.post_parameters_macro.bounds, varpar)
            push!(par_bounds, :($varpar = min(max($varpar,$(𝓂.constants.post_parameters_macro.bounds[varpar][1])),$(𝓂.constants.post_parameters_macro.bounds[varpar][2]))))
        end
    end
    
    return par_bounds
end


"""
    collect_calibration_no_var_parameters!(atoms_in_equations::Set{Symbol}, 𝓂)

Add parameters from calibration_no_var equations to the atoms_in_equations set.
Also returns a set of parameters_only_in_par_defs for internal use.
"""
function collect_calibration_no_var_parameters!(atoms_in_equations::Set{Symbol}, 𝓂)
    parameters_only_in_par_defs = Set()
    if length(𝓂.equations.calibration_no_var) > 0
        atoms = reduce(union, get_symbols.(𝓂.equations.calibration_no_var))
        [push!(atoms_in_equations, a) for a in atoms]
        [push!(parameters_only_in_par_defs, a) for a in atoms]
    end
    return parameters_only_in_par_defs
end


"""
    build_return_variables(𝓂)

Build the return variable list for solve_SS function.

# Returns
- Symbols for variables in return statement
- Length of return vector
"""
function build_return_variables(𝓂)
    vars_expr = Symbol.(replace.(string.(sort(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future
    ))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))
    
    return_length = length(union(
        𝓂.constants.post_model_macro.var,
        𝓂.constants.post_model_macro.exo_past,
        𝓂.constants.post_model_macro.exo_future
    )) + length(𝓂.equations.calibration_parameters)
    
    return vars_expr, return_length
end


"""
    build_solve_SS_expression(𝓂, parameters_in_equations, par_bounds, SS_solve_func)

Build the solve_SS function expression that will be compiled with @RuntimeGeneratedFunction.
This is shared between both versions of write_steady_state_solver_function!.

# Arguments
- `𝓂`: The model struct
- `parameters_in_equations`: Parameter assignment expressions
- `par_bounds`: Parameter bounds expressions  
- `SS_solve_func`: Block solving expressions
- `precompiled::Bool`: Whether this is the precompiled version (V2) with multi-element cache structure

# Returns
An Expr representing the solve_SS function.
"""
function build_solve_SS_expression(𝓂, parameters_in_equations, par_bounds, SS_solve_func; precompiled::Bool = false)
    vars_expr, return_length = build_return_variables(𝓂)
    
    # Zero initial value check only for precompiled version (V2) which has multi-element cache entries
    zero_init_check = precompiled ? quote
        # Zero initial value if starting without guess
        if !isfinite(sum(abs,closest_solution[2]))
            closest_solution = copy(closest_solution)
            for i in 1:2:length(closest_solution)
                closest_solution[i] = zeros(length(closest_solution[i]))
            end
        end
    end : :()
    
    solve_exp = :(function solve_SS(initial_parameters::Vector{Real}, 
                                    𝓂::ℳ,
                                    tol::Tolerances,
                                    verbose::Bool, 
                                    cold_start::Bool,
                                    solver_parameters::Vector{solver_parameters})
                    initial_parameters = typeof(initial_parameters) == Vector{Float64} ? initial_parameters : ℱ.value.(initial_parameters)

                    initial_parameters_tmp = copy(initial_parameters)

                    parameters = copy(initial_parameters)
                    params_flt = copy(initial_parameters)
                    
                    current_best = sum(abs2,𝓂.caches.solver_cache[end][end] - initial_parameters)
                    closest_solution_init = 𝓂.caches.solver_cache[end]
                    
                    for pars in 𝓂.caches.solver_cache
                        copy!(initial_parameters_tmp, pars[end])

                        ℒ.axpy!(-1,initial_parameters,initial_parameters_tmp)

                        latest = sum(abs2,initial_parameters_tmp)
                        if latest <= current_best
                            current_best = latest
                            closest_solution_init = pars
                        end
                    end

                    range_iters = 0
                    solution_error = 1.0
                    solved_scale = 0
                    scale = 1.0

                    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
                    push!(NSSS_solver_cache_scale, closest_solution_init)

                    while range_iters <= (cold_start ? 1 : 500) && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)
                        range_iters += 1
                        fail_fast_solvers_only = range_iters > 1 ? true : false

                        if abs(solved_scale - scale) < 1e-2
                            break 
                        end

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

                        $zero_init_check

                        if all(isfinite,closest_solution[end]) && initial_parameters != closest_solution_init[end]
                            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
                        else
                            parameters = copy(initial_parameters)
                        end
                        params_flt = parameters

                        $(parameters_in_equations...)
                        $(par_bounds...)
                        $(𝓂.equations.calibration_no_var...)
                        NSSS_solver_cache_tmp = []
                        solution_error = 0.0
                        iters = 0
                        $(SS_solve_func...)

                        if solution_error < tol.NSSS_acceptance_tol
                            solved_scale = scale
                            if scale == 1
                                return [$(vars_expr...), $(𝓂.equations.calibration_parameters...)], (solution_error, iters)
                            else
                                reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
                            end

                            if scale > .95
                                scale = 1
                            else
                                scale = scale * .4 + .6
                            end
                        end
                    end
                    return zeros($return_length), (1, 0)
                end)

    return solve_exp
end


"""
    build_model_specific_rtgfs!(𝓂, parameters_in_equations, par_bounds, dyn_exos,
                                 block_calib_pars_input, block_other_vars_input,
                                 block_result_exprs, block_lbs, block_ubs)

Build and store model-specific RuntimeGeneratedFunctions for the refactored steady state solver.
This creates separate RTGFs for:
1. setup_parameters_and_bounds: parameter vector → named parameters with bounds
2. get_block_inputs (per block): named vars → block input vector
3. update_solution (per block): named vars + solution → updated named vars
4. set_dynamic_exogenous: named vars → named vars (with exogenous = 0)
5. extract_solution_vector: named vars → solution vector

# Arguments
- `𝓂`: The model struct
- `parameters_in_equations`: Parameter assignment expressions
- `par_bounds`: Parameter bounds expressions
- `dyn_exos`: Dynamic exogenous variable assignments
- `block_calib_pars_input`: Vector of vectors of calibration parameter symbols per block
- `block_other_vars_input`: Vector of vectors of other variable symbols per block
- `block_result_exprs`: Vector of vectors of result assignment expressions per block
- `block_lbs`: Vector of lower bound vectors per block
- `block_ubs`: Vector of upper bound vectors per block
"""
function build_model_specific_rtgfs!(𝓂, parameters_in_equations, par_bounds, dyn_exos,
                                      block_calib_pars_input, block_other_vars_input,
                                      block_result_exprs, block_lbs, block_ubs)
    # 1. Build setup_parameters_and_bounds RTGF
    # This takes the parameter vector and returns a NamedTuple with all named parameters
    vars_expr, _ = build_return_variables(𝓂)

    # Create NamedTuple fields for all variables and calibration parameters
    all_var_names = sort(union(𝓂.constants.post_model_macro.var,
                                𝓂.constants.post_model_macro.exo_past,
                                𝓂.constants.post_model_macro.exo_future))
    all_names = vcat(all_var_names, 𝓂.equations.calibration_parameters)

    # Initialize all variables to NaN (will be set by block solutions)
    init_exprs = [:($(nm) = NaN) for nm in all_names]

    setup_expr = :(function setup_params(parameters::Vector{T}) where T <: Real
        $(parameters_in_equations...)
        $(par_bounds...)
        $(𝓂.equations.calibration_no_var...)
        $(init_exprs...)
        return (; $(all_names...))
    end)

    𝓂.NSSS.setup_parameters_and_bounds = @RuntimeGeneratedFunction(setup_expr)

    # 2. Build per-block get_block_inputs RTGFs
    # Each takes a NamedTuple and returns a Vector of the required inputs for that block
    n_blocks = length(block_calib_pars_input)
    𝓂.NSSS.block_metadata = ss_block_metadata[]

    for i in 1:n_blocks
        inputs = vcat(block_calib_pars_input[i], block_other_vars_input[i])

        if length(inputs) == 0
            get_inputs_expr = :(function get_inputs(named_vars::NamedTuple)
                return Float64[]
            end)
        else
            get_inputs_expr = :(function get_inputs(named_vars::NamedTuple)
                $([:($nm = named_vars.$nm) for nm in inputs]...)
                return [$(inputs...)]
            end)
        end

        get_inputs_func = @RuntimeGeneratedFunction(get_inputs_expr)

        # 3. Build per-block update_solution RTGFs
        # Each takes a NamedTuple and solution vector, returns updated NamedTuple
        result_assignments = block_result_exprs[i]
        update_expr = :(function update_sol(named_vars::NamedTuple, sol::Vector{T}) where T <: Real
            # Unpack current named variables
            $([:($nm = named_vars.$nm) for nm in all_names]...)
            # Apply solution
            $(result_assignments...)
            # Return updated NamedTuple
            return (; $(all_names...))
        end)

        update_func = @RuntimeGeneratedFunction(update_expr)

        # Store in block_metadata
        push!(𝓂.NSSS.block_metadata, ss_block_metadata(
            block_lbs[i],
            block_ubs[i],
            get_inputs_func,
            update_func
        ))
    end

    # 4. Build set_dynamic_exogenous RTGF
    set_exo_expr = :(function set_exo(named_vars::NamedTuple)
        # Unpack current named variables
        $([:($nm = named_vars.$nm) for nm in all_names]...)
        # Set dynamic exogenous to zero
        $(dyn_exos...)
        # Return updated NamedTuple
        return (; $(all_names...))
    end)

    𝓂.NSSS.set_dynamic_exogenous = @RuntimeGeneratedFunction(set_exo_expr)

    # 5. Build extract_solution_vector RTGF
    extract_expr = :(function extract_sol(named_vars::NamedTuple)
        $([:($nm = named_vars.$nm) for nm in all_names]...)
        return [$(vars_expr...), $(𝓂.equations.calibration_parameters...)]
    end)

    𝓂.NSSS.extract_solution_vector = @RuntimeGeneratedFunction(extract_expr)

    # Store solution vector length
    𝓂.NSSS.solution_vector_length = length(all_names)

    return nothing
end
