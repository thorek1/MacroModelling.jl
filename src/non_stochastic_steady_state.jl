# Refactored symbolic steady state solver - separates orchestration from model-specific code
# This version generates RTGFs for symbolic evaluations and uses the same infrastructure
# as the precompiled version for numerical blocks
function write_steady_state_solver_function!(𝓂::ℳ, symbolic_SS, Symbolics::symbolics; verbose::Bool = false, avoid_solve::Bool = false)
    unknowns = union(Symbolics.calibration_equations_parameters, Symbolics.vars_in_ss_equations)

    @assert length(unknowns) <= length(Symbolics.ss_equations) + length(Symbolics.calibration_equations) "Unable to solve steady state. More unknowns than equations."

    eq_list = vcat(union.(setdiff.(union.(Symbolics.var_list_aux_SS,
                                        Symbolics.ss_list_aux_SS),
                                    Symbolics.var_redundant_list),
                            Symbolics.par_list_aux_SS),
                    union.(Symbolics.ss_calib_list,
                            Symbolics.par_calib_list))

    vars, eqs, n_blocks, incidence_matrix = compute_block_triangularization(unknowns, eq_list)

    n = n_blocks

    ss_equations = vcat(Symbolics.ss_equations, Symbolics.calibration_equations)

    # Data collection for refactored solver
    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = Symbol[]
    NSSS_solver_cache_init_tmp = []

    # Symbolic solution data - only for truly symbolic solutions (not numerical)
    solved_vars_symbolic = []  # Renamed to make clear these are ONLY symbolic
    solved_vals_symbolic = []
    min_max_errors = []

    # Numerical block data (same structure as precompiled version)
    block_calib_pars_input = []
    block_other_vars_input = []
    block_result_exprs = []
    block_lbs = []
    block_ubs = []

    unique_➕_eqs = Dict{Union{Expr,Symbol},Symbol}()

    # Track all solved variables for dependency analysis
    all_solved_vars = []
    all_atoms_list = []

    # Main block loop - collect data for RTGFs
    while n > 0
        if length(eqs[:,eqs[2,:] .== n]) == 2
            # Single equation, single variable
            var_to_solve_for = unknowns[vars[:,vars[2,:] .== n][1]]
            eq_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1]]

            # Handle min/max constraints
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
                push!(min_max_errors, :(minmax_error += abs($parsed_eq_to_solve_for)))
                eq_to_solve = eval(minmax_fixed_eqs)
            end

            # Try symbolic solving
            if avoid_solve || count_ops(Meta.parse(string(eq_to_solve))) > 15
                soll = nothing
            else
                soll = solve_symbolically(eq_to_solve, var_to_solve_for)
            end

            if isnothing(soll) || isempty(soll)
                # Symbolic solving failed - use numerical block solver
                if verbose
                    println("Failed finding solution symbolically for: ", var_to_solve_for, " in: ", eq_to_solve)
                end
                # Fall back to numerical - treat as 1-variable numerical block
                # For simplicity, collect as if it's a multi-variable block
                vars_to_solve = [var_to_solve_for]
                eqs_to_solve = [eq_to_solve]
                # Jump to numerical block handling
                numerical_sol = true
            elseif soll[1].is_number == true
                # Symbolic constant solution
                ss_equations = [replace_symbolic(eq, var_to_solve_for, soll[1]) for eq in ss_equations]

                push!(solved_vars_symbolic, Symbol(var_to_solve_for))
                push!(solved_vals_symbolic, Meta.parse(string(soll[1])))
                push!(all_solved_vars, Symbol(var_to_solve_for))
                push!(all_atoms_list, [])
                numerical_sol = false
            else
                # Symbolic expression solution
                push!(solved_vars_symbolic, Symbol(var_to_solve_for))
                push!(solved_vals_symbolic, Meta.parse(string(soll[1])))
                push!(all_solved_vars, Symbol(var_to_solve_for))

                [push!(atoms_in_equations, Symbol(a)) for a in soll[1].atoms()]
                atoms_set = Set(union(setdiff(get_symbols(parsed_eq_to_solve_for), get_symbols(minmax_fixed_eqs)), Symbol.(soll[1].atoms())))
                push!(all_atoms_list, atoms_set)
                numerical_sol = false
            end

            # If we fell back to numerical for single variable, handle it
            if numerical_sol
                eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]
                # Process as numerical block below
            else
                n -= 1
                continue
            end
        else
            vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]
            eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

            numerical_sol = false

            if symbolic_SS
                # Try symbolic solving for multiple variables
                if avoid_solve || count_ops(Meta.parse(string(eqs_to_solve))) > 15
                    soll = nothing
                else
                    soll = solve_symbolically(eqs_to_solve, vars_to_solve)
                end

                if isnothing(soll) || isempty(soll) || length(intersect((union(SPyPyC.free_symbols.(collect(values(soll)))...) .|> SPyPyC.:↓), (vars_to_solve .|> SPyPyC.:↓))) > 0
                    if verbose
                        println("Failed finding solution symbolically for: ", vars_to_solve, " in: ", eqs_to_solve, ". Solving numerically.")
                    end
                    numerical_sol = true
                else
                    if verbose
                        println("Solved: ", string.(eqs_to_solve), " for: ", Symbol.(vars_to_solve), " symbolically.")
                    end

                    atoms = reduce(union, map(x->x.atoms(), collect(values(soll))))
                    for a in atoms
                        push!(atoms_in_equations, Symbol(a))
                    end

                    for var in vars_to_solve
                        push!(solved_vars_symbolic, Symbol(var))
                        push!(solved_vals_symbolic, Meta.parse(string(soll[var])))
                        push!(all_solved_vars, Symbol(var))
                        push!(all_atoms_list, Set(Symbol.(soll[var].atoms())))
                    end
                end
            else
                numerical_sol = true
            end

            eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]
        end

        # Handle numerical blocks (same approach as precompiled version)
        if numerical_sol || !symbolic_SS
            if verbose && !symbolic_SS
                println("Solved: ", string.(eqs_to_solve), " for: ", Symbol.(vars_to_solve), " numerically.")
            end

            # Collect symbols in equations
            syms_in_eqs = Set()
            for i in eqs_to_solve
                push!(syms_in_eqs, get_symbols(i)...)
            end

            push!(all_atoms_list, setdiff(syms_in_eqs, Symbol.(vars_to_solve)))
            [push!(all_solved_vars, Symbol(v)) for v in vars_to_solve]

            # Calibration parameters needed for this block
            calib_pars_input = []
            relevant_pars = reduce(union, vcat(𝓂.constants.post_model_macro.par_list_aux_SS, 𝓂.constants.post_parameters_macro.par_calib_list)[eq_idx_in_block_to_solve])
            relevant_pars_across = union(relevant_pars_across, relevant_pars)

            for parss in union(𝓂.constants.post_complete_parameters.parameters, 𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters)
                if :($parss) ∈ relevant_pars
                    push!(calib_pars_input, :($parss))
                end
            end

            # Result assignments
            result = Expr[]
            sorted_vars = sort(Symbol.(vars_to_solve))
            for (i, parss) in enumerate(sorted_vars)
                push!(result, :($parss = sol[$i]))
            end

            # Other variables needed
            other_vars_input = []
            other_vrs = intersect(
                setdiff(union(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters), sorted_vars),
                syms_in_eqs
            )

            for var in other_vrs
                push!(other_vars_input, :($(var)))
            end

            # Store block data
            push!(block_calib_pars_input, calib_pars_input)
            push!(block_other_vars_input, other_vars_input)
            push!(block_result_exprs, result)

            # Bounds (simplified - use defaults)
            push!(block_lbs, fill(-1e12, length(sorted_vars)))
            push!(block_ubs, fill(1e12, length(sorted_vars)))

            # Cache initialization
            push!(NSSS_solver_cache_init_tmp, fill(NaN, length(sorted_vars)))
            push!(NSSS_solver_cache_init_tmp, length(vcat(calib_pars_input, other_vars_input)) == 0 ? [Inf] : fill(Inf, length(vcat(calib_pars_input, other_vars_input))))
        end

        n -= 1
    end

    # Initialize cache
    if length(NSSS_solver_cache_init_tmp) == 0
        push!(NSSS_solver_cache_init_tmp, fill(Inf, length(𝓂.constants.post_complete_parameters.parameters)))
    else
        push!(NSSS_solver_cache_init_tmp, fill(Inf, length(𝓂.constants.post_complete_parameters.parameters)))
    end
    push!(𝓂.caches.solver_cache, NSSS_solver_cache_init_tmp)

    # Build common data structures
    collect_calibration_no_var_parameters!(atoms_in_equations, 𝓂)
    parameters_in_equations = build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    build_dependencies!(𝓂, all_atoms_list, all_solved_vars)
    dyn_exos = build_dyn_exos_expressions(𝓂)
    par_bounds = build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

    # Build model-specific RTGFs
    # 1. Build RTGFs for numerical blocks
    if length(block_calib_pars_input) > 0
        build_model_specific_rtgfs!(𝓂, parameters_in_equations, par_bounds, dyn_exos,
                                      block_calib_pars_input, block_other_vars_input,
                                      block_result_exprs, block_lbs, block_ubs)
    else
        # No numerical blocks - initialize with empty structures
        build_model_specific_rtgfs!(𝓂, parameters_in_equations, par_bounds, dyn_exos,
                                      [], [], [], [], [])
    end

    # 2. Build RTGFs for symbolic-specific operations
    build_symbolic_rtgfs!(𝓂, solved_vars_symbolic, solved_vals_symbolic, min_max_errors,
                          parameters_in_equations, par_bounds, dyn_exos)

    # 3. Set NSSS_solve to call the general orchestration function
    𝓂.functions.NSSS_solve = function(parameters, model, tol, verbose, cold_start, solver_params)
        solve_steady_state_symbolic(parameters, model, tol, verbose, cold_start, solver_params)
    end

    return nothing
end



function write_steady_state_solver_function!(𝓂::ℳ;
                            cse = true,
                            skipzeros = true,
                            density_threshold::Float64 = .1,
                            nnz_parallel_threshold::Int = 1000000,
                            min_length::Int = 1000,
                            verbose::Bool = false)
    unknowns = union(𝓂.constants.post_model_macro.vars_in_ss_equations, 𝓂.equations.calibration_parameters)

    @assert length(unknowns) <= length(𝓂.equations.steady_state_aux) + length(𝓂.equations.calibration) "Unable to solve steady state. More unknowns than equations."

    eq_list = vcat(union.(union.(𝓂.constants.post_model_macro.var_list_aux_SS,
                    𝓂.constants.post_model_macro.ss_list_aux_SS),
                𝓂.constants.post_model_macro.par_list_aux_SS),
            union.(𝓂.constants.post_parameters_macro.ss_calib_list,
                𝓂.constants.post_parameters_macro.par_calib_list))

    vars, eqs, n_blocks, incidence_matrix = compute_block_triangularization(unknowns, eq_list)
    
    n = n_blocks

    ss_equations = vcat(𝓂.equations.steady_state_aux,𝓂.equations.calibration)

    SS_solve_func = []

    atoms_in_equations = Set{Symbol}()
    atoms_in_equations_list = []
    relevant_pars_across = []
    NSSS_solver_cache_init_tmp = []

    solved_vars = []
    solved_vals = []

    # Collect block-specific data for refactored solver
    block_calib_pars_input = []
    block_other_vars_input = []
    block_result_exprs = []
    block_lbs = []
    block_ubs = []

    n_block = 1

    while n > 0
        vars_to_solve = unknowns[vars[:,vars[2,:] .== n][1,:]]

        eqs_to_solve = ss_equations[eqs[:,eqs[2,:] .== n][1,:]]

        # try symbolically and use numerical if it does not work
        if verbose
            println("Solved: ",string.(eqs_to_solve)," for: ",Symbol.(vars_to_solve), " numerically.")
        end
        
        push!(solved_vars,Symbol.(vars_to_solve))
        push!(solved_vals,Meta.parse.(string.(eqs_to_solve)))

        syms_in_eqs = Set()

        for i in eqs_to_solve
            push!(syms_in_eqs, get_symbols(i)...)
        end

        # println(syms_in_eqs)
        push!(atoms_in_equations_list,setdiff(syms_in_eqs, solved_vars[end]))

        # calib_pars = []
        calib_pars_input = []
        relevant_pars = reduce(union,vcat(𝓂.constants.post_model_macro.par_list_aux_SS,𝓂.constants.post_parameters_macro.par_calib_list)[eqs[:,eqs[2,:] .== n][1,:]])
        relevant_pars_across = union(relevant_pars_across,relevant_pars)
        
        iii = 1
        for parss in union(𝓂.constants.post_complete_parameters.parameters,𝓂.constants.post_parameters_macro.parameters_as_function_of_parameters)
            # valss   = 𝓂.parameter_values[i]
            if :($parss) ∈ relevant_pars
                # push!(calib_pars,:($parss = parameters_and_solved_vars[$iii]))
                push!(calib_pars_input,:($parss))
                iii += 1
            end
        end


        # guess = Expr[]
        # untransformed_guess = Expr[]
        result = Expr[]
        sorted_vars = sort(solved_vars[end])
        # sorted_vars = sort(setdiff(solved_vars[end],𝓂.constants.post_model_macro.➕_vars))
        for (i, parss) in enumerate(sorted_vars) 
            # push!(guess,:($parss = guess[$i]))
            # push!(untransformed_guess,:($parss = undo_transform(guess[$i],transformation_level)))
            push!(result,:($parss = sol[$i]))
        end

        
        # separate out auxiliary variables (nonnegativity)
        nnaux = []
        # nnaux_linear = []
        # nnaux_error = []
        # push!(nnaux_error, :(aux_error = 0))
        solved_vals_local = Expr[]
        # solved_vals_in_place = Expr[]
        
        eq_idx_in_block_to_solve = eqs[:,eqs[2,:] .== n][1,:]


        other_vrs_eliminated_by_sympy = Set()

        for (i,val) in enumerate(solved_vals[end])
            if typeof(val) ∈ [Symbol,Float64,Int]
                push!(solved_vals_local,val)
                # push!(solved_vals_in_place, :(ℰ[$i] = $val))
            else
                if eq_idx_in_block_to_solve[i] ∈ 𝓂.constants.post_model_macro.ss_equations_with_aux_variables
                    val = vcat(𝓂.equations.steady_state_aux,𝓂.equations.calibration)[eq_idx_in_block_to_solve[i]]
                    push!(nnaux,:($(val.args[2]) = max(eps(),$(val.args[3]))))
                    push!(other_vrs_eliminated_by_sympy, val.args[2])
                    # push!(nnaux_linear,:($val))
                    push!(solved_vals_local,:($val))
                    # push!(solved_vals_in_place,:(ℰ[$i] = $val))
                    # push!(nnaux_error, :(aux_error += min(eps(),$(val.args[3]))))
                else
                    push!(solved_vals_local,postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))
                    # push!(solved_vals_in_place, :(ℰ[$i] = $(postwalk(x -> x isa Expr ? x.args[1] == :conjugate ? x.args[2] : x : x, val))))
                end
            end
        end

        # println(other_vrs_eliminated_by_sympy)
        # sort nnaux vars so that they enter in right order. avoid using a variable before it is declared
        # println(nnaux)
        if length(nnaux) > 1
            all_symbols = map(x->x.args[1],nnaux) #relevant symbols come first in respective equations

            nn_symbols = map(x->intersect(all_symbols,x), get_symbols.(nnaux))
            
            inc_matrix = fill(0,length(all_symbols),length(all_symbols))

            for i in 1:length(all_symbols)
                for k in 1:length(nn_symbols)
                    inc_matrix[i,k] = collect(all_symbols)[i] ∈ collect(nn_symbols)[k]
                end
            end

            QQ, P, R, nmatch, n_blocks = BlockTriangularForm.order(sparse(inc_matrix))

            nnaux = nnaux[QQ]
            # nnaux_linear = nnaux_linear[QQ]
        end


        # other_vars = []
        other_vars_input = []
        # other_vars_inverse = []
        other_vrs = intersect( setdiff( union(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters, 𝓂.constants.post_model_macro.➕_vars),
                                            sort(solved_vars[end]) ),
                                union(syms_in_eqs, other_vrs_eliminated_by_sympy, setdiff(reduce(union, get_symbols.(nnaux), init = []), map(x->x.args[1],nnaux)) ) )

        for var in other_vrs
            # var_idx = findfirst(x -> x == var, union(𝓂.constants.post_model_macro.var,𝓂.calibration_equations_parameters))
            # push!(other_vars,:($(var) = parameters_and_solved_vars[$iii]))
            push!(other_vars_input,:($(var)))
            iii += 1
            # push!(other_vars_inverse,:(𝓂.SS_init_guess[$var_idx] = $(var)))
        end

        parameters_and_solved_vars = vcat(calib_pars_input, other_vrs)

        ng = length(sorted_vars)
        np = length(parameters_and_solved_vars)
        nd = 0
        nx = iii - 1
    

        Symbolics.@variables 𝔊[1:ng] 𝔓[1:np]


        parameter_dict = Dict{Symbol, Symbol}()
        back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()
        # aux_vars = Symbol[]
        # aux_expr = []
    
    
        for (i,v) in enumerate(sorted_vars)
            push!(parameter_dict, v => :($(Symbol("𝔊_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔊_$i"))), @__MODULE__) => 𝔊[i])
        end
    
        for (i,v) in enumerate(parameters_and_solved_vars)
            push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
            push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
        end
    
        # for (i,v) in enumerate(ss_and_aux_equations_dep)
        #     push!(aux_vars, v.args[1])
        #     push!(aux_expr, v.args[2])
        # end
    
        # aux_replacements = Dict(aux_vars .=> aux_expr)
    
        replaced_solved_vals = solved_vals_local |> 
            # x -> replace_symbols.(x, Ref(aux_replacements)) |> 
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
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        # 𝐷 = zeros(Symbolics.Num, nd)
    
        # ϵᵃ = zeros(nd)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        ϵˢ = zeros(Symbolics.Num, ng)
    
        ϵ = zeros(ng)
    
        # calc_block!(ϵˢ, 𝔊, 𝔓, 𝐷)
    
        ∂block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals, 𝔊) # nϵ x nx
    
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

        prob = 𝒮.LinearProblem(chol_buff, ϵ, 𝒮.CholeskyFactorization())

        chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        prob = 𝒮.LinearProblem(buffer, ϵ, 𝒮.LUFactorization())

        lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        if lennz > nnz_parallel_threshold
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, func_exprs = Symbolics.build_function(derivatives_mat, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
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
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    
        ϵᵉ = zeros(ng + nx)
        
        # ϵˢᵉ = zeros(Symbolics.Num, ng + nx)
    
        # calc_block_aux!(𝐷, 𝔊, 𝔓)
    
        # Evaluate the function symbolically
        # calc_ext_block!(ϵˢᵉ, 𝔊, 𝔓, 𝐷)
    
        ∂ext_block_∂parameters_and_solved_vars = Symbolics.sparsejacobian(replaced_solved_vals_ext, 𝔊) # nϵ x nx
    
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

        prob = 𝒮.LinearProblem(ext_chol_buff, ϵᵉ, 𝒮.CholeskyFactorization())

        ext_chol_buffer = 𝒮.init(prob, 𝒮.CholeskyFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        prob = 𝒮.LinearProblem(ext_buffer, ϵᵉ, 𝒮.LUFactorization())

        ext_lu_buffer = 𝒮.init(prob, 𝒮.LUFactorization(), verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)

        if lennz > nnz_parallel_threshold
            parallel = Symbolics.ShardedForm(1500,4)
        else
            parallel = Symbolics.SerialForm()
        end
        
        _, ext_func_exprs = Symbolics.build_function(derivatives_mat_ext, 𝔊, 𝔓,
                                                    cse = cse, 
                                                    skipzeros = skipzeros, 
                                                    # nanmath = false,
                                                    parallel = parallel,
                                                    expression_module = @__MODULE__,
                                                    expression = Val(false))::Tuple{<:Function, <:Function}
    


        push!(NSSS_solver_cache_init_tmp,fill(1.205996189998029, length(sorted_vars)))
        push!(NSSS_solver_cache_init_tmp,[Inf])

        # WARNING: infinite bounds are transformed to 1e12
        lbs = []
        ubs = []
        
        limit_boundaries = 1e12

        for i in vcat(sorted_vars, calib_pars_input, other_vars_input)
            if haskey(𝓂.constants.post_parameters_macro.bounds, i)
                push!(lbs,𝓂.constants.post_parameters_macro.bounds[i][1] == -Inf ? -limit_boundaries+rand() : 𝓂.constants.post_parameters_macro.bounds[i][1])
                push!(ubs,𝓂.constants.post_parameters_macro.bounds[i][2] ==  Inf ?  limit_boundaries-rand() : 𝓂.constants.post_parameters_macro.bounds[i][2])
            else
                push!(lbs,-limit_boundaries+rand())
                push!(ubs,limit_boundaries+rand())
            end
        end

        push!(SS_solve_func,:(params_and_solved_vars = [$(calib_pars_input...),$(other_vars_input...)]))

        push!(SS_solve_func,:(lbs = [$(lbs...)]))
        push!(SS_solve_func,:(ubs = [$(ubs...)]))
        
        push!(SS_solve_func,:(inits = [max.(lbs[1:length(closest_solution[$(2*(n_block-1)+1)])], min.(ubs[1:length(closest_solution[$(2*(n_block-1)+1)])], closest_solution[$(2*(n_block-1)+1)])), closest_solution[$(2*n_block)]]))

        push!(SS_solve_func,:(solution = block_solver(length(params_and_solved_vars) == 0 ? [0.0] : params_and_solved_vars,
                                                                $(n_block), 
                                                                𝓂.NSSS.solve_blocks_in_place[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks[$(n_block)], 
                                                                # 𝓂.ss_solve_blocks_no_transform[$(n_block)], 
                                                                # f, 
                                                                inits,
                                                                lbs, 
                                                                ubs,
                                                                solver_parameters,
                                                                fail_fast_solvers_only,
                                                                cold_start,
                                                                verbose)))
        
        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        push!(SS_solve_func,:(iters += solution[2][2])) 
        push!(SS_solve_func,:(solution_error += solution[2][1])) 
        push!(SS_solve_func,:(sol = solution[1]))

        # push!(SS_solve_func,:(solution = block_solver_RD(length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)])))#, 
        
        # push!(SS_solve_func,:(solution_error += sum(abs2,𝓂.ss_solve_blocks[$(n_block)](length([$(calib_pars_input...),$(other_vars_input...)]) == 0 ? [0.0] : [$(calib_pars_input...),$(other_vars_input...)],solution))))

        push!(SS_solve_func,:($(result...)))   
        
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(sol) == Vector{Float64} ? sol : ℱ.value.(sol)]))
        push!(SS_solve_func,:(NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., typeof(params_and_solved_vars) == Vector{Float64} ? params_and_solved_vars : ℱ.value.(params_and_solved_vars)]))

        # Create nonlinear solver workspaces for regular and extended problems
        workspace = Nonlinear_solver_workspace(ϵ, buffer, chol_buffer, lu_buffer)
        ext_workspace = Nonlinear_solver_workspace(ϵᵉ, ext_buffer, ext_chol_buffer, ext_lu_buffer)

        push!(𝓂.NSSS.solve_blocks_in_place,
            ss_solve_block(
                function_and_jacobian(calc_block!::Function, func_exprs::Function, workspace),
                function_and_jacobian(calc_ext_block!::Function, ext_func_exprs::Function, ext_workspace)
            )
        )

        # Collect block-specific data for refactored solver
        push!(block_calib_pars_input, calib_pars_input)
        push!(block_other_vars_input, other_vars_input)
        push!(block_result_exprs, result)
        push!(block_lbs, lbs)
        push!(block_ubs, ubs)

        n_block += 1
        
        n -= 1
    end

    push!(NSSS_solver_cache_init_tmp,[Inf])
    push!(NSSS_solver_cache_init_tmp,fill(Inf,length(𝓂.constants.post_complete_parameters.parameters)))
    push!(𝓂.caches.solver_cache,NSSS_solver_cache_init_tmp)

    unknwns = Symbol.(unknowns)

    collect_calibration_no_var_parameters!(atoms_in_equations, 𝓂)
    
    parameters_in_equations = build_parameters_in_equations(𝓂, atoms_in_equations, relevant_pars_across)
    
    build_dependencies!(𝓂, atoms_in_equations_list, solved_vars)

    dyn_exos = build_dyn_exos_expressions(𝓂)

    push!(SS_solve_func,:($(dyn_exos...)))

    # push!(SS_solve_func,:(push!(NSSS_solver_cache_tmp, params_scaled_flt)))
    push!(SS_solve_func,:(if length(NSSS_solver_cache_tmp) == 0 NSSS_solver_cache_tmp = [copy(params_flt)] else NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)] end))
    
    push!(SS_solve_func,:(current_best = sqrt(sum(abs2,𝓂.caches.solver_cache[end][end] - params_flt))))# / max(sum(abs2,𝓂.caches.solver_cache[end][end]), sum(abs2,params_flt))))

    push!(SS_solve_func,:(for pars in 𝓂.caches.solver_cache
                                latest = sqrt(sum(abs2,pars[end] - params_flt))# / max(sum(abs2,pars[end]), sum(abs,params_flt))
                                if latest <= current_best
                                    current_best = latest
                                end
                            end))

    push!(SS_solve_func,:(if (current_best > 1e-8) && (solution_error < tol.NSSS_acceptance_tol)
                                    reverse_diff_friendly_push!(𝓂.caches.solver_cache, NSSS_solver_cache_tmp)
                                # solved_scale = scale
                            end))

    # fix parameter bounds
    par_bounds = build_parameter_bounds_expressions(𝓂, atoms_in_equations, relevant_pars_across)

    dyn_exos = build_dyn_exos_expressions(𝓂)

    # Build model-specific RTGFs for the refactored solver
    build_model_specific_rtgfs!(𝓂, parameters_in_equations, par_bounds, dyn_exos,
                                 block_calib_pars_input, block_other_vars_input,
                                 block_result_exprs, block_lbs, block_ubs)

    # Set the NSSS_solve function to the general orchestration function
    𝓂.functions.NSSS_solve = (parameters, model, tol, verbose, cold_start, solver_params) ->
        solve_steady_state_precompiled(parameters, model, tol, verbose, cold_start, solver_params)

    return nothing
end
