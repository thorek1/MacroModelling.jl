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
macro model(𝓂,ex...)
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

    # create data containers
    parameter_values = Vector{Float64}(undef,0)
    
    T, equations_struct = process_model_equations(
        ex[end],
        max_obc_horizon,
        precompile,
    )

    ℂ = Constants(T)

    𝓦 = Workspaces()

    # default_optimizer = nlboxsolve
    # default_optimizer = Optimisers.Adam
    # default_optimizer = NLopt.LN_BOBYQA
    
    #assemble data container
    model_name = string(𝓂)
    quote
        global $𝓂 =  ℳ(
                        $model_name,
                        
                        $parameter_values,

                        non_stochastic_steady_state(
                            $(ss_solve_block[]), # NSSS_solve_blocks_in_place
                            $nothing # NSSS_dependencies
                        ),

                        $equations_struct, 

                        caches(
                            outdated_caches(
                                true, # non_stochastic_steady_state
                                true, # jacobian
                                true, # hessian
                                true, # third_order_derivatives
                                true, # first_order_solution
                                true, # second_order_solution
                                true, # pruned_second_order_solution
                                true, # third_order_solution
                                true, # pruned_third_order_solution
                            ),
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
                            zeros(0,0), # qme_solution
                            Float64[],  # second_order_stochastic_steady_state
                            SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # second_order_solution
                            Float64[],  # pruned_second_order_stochastic_steady_state
                            Float64[],  # third_order_stochastic_steady_state
                            SparseMatrixCSC{Float64, Int64}(ℒ.I,0,0), # third_order_solution
                            Float64[],  # pruned_third_order_stochastic_steady_state
                            Float64[],  # non_stochastic_steady_state
                            $(CircularBuffer{Vector{Vector{Float64}}}(500)),  # NSSS_solver_cache
                            $(zeros(0,0)),  # NSSS_∂equations_∂parameters
                            $(zeros(0,0)),  # NSSS_∂equations_∂SS_and_pars
                        ),
                        
                        $ℂ,
                        $𝓦,

                        model_functions(
                            $(x->x), # NSSS_solve_func
                            $(x->x), # NSSS_check_func
                            $nothing, # NSSS_custom_function
                            $(x->x), # NSSS_∂equations_∂parameters_func
                            $(x->x), # NSSS_∂equations_∂SS_and_pars_func
                            jacobian_functions(x->x, x->x, x->x), # jacobian, jacobian_parameters, jacobian_SS_and_pars
                            hessian_functions(x->x, x->x, x->x), # hessian, hessian_parameters, hessian_SS_and_pars
                            third_order_derivatives_functions(x->x, x->x, x->x), # third_order_derivatives, third_order_derivatives_parameters, third_order_derivatives_SS_and_pars
                            (x,y)->nothing, # first_order_state_update
                            (x,y)->nothing, # first_order_state_update_obc
                            (x,y)->nothing, # second_order_state_update
                            (x,y)->nothing, # second_order_state_update_obc
                            (x,y)->nothing, # pruned_second_order_state_update
                            (x,y)->nothing, # pruned_second_order_state_update_obc
                            (x,y)->nothing, # third_order_state_update
                            (x,y)->nothing, # third_order_state_update_obc
                            (x,y)->nothing, # pruned_third_order_state_update
                            (x,y)->nothing, # pruned_third_order_state_update_obc
                            x->x, # obc_violation
                            false # functions_written
                        ),

                        SolveCounters(),

                        RevisionEntry[] # revision_history
                    );
    end
end





"""
$(SIGNATURES)
Adds parameter values and calibration equations to the previously defined model. Allows to provide an initial guess for the non-stochastic steady state (NSSS).
- $STEADY_STATE_FUNCTION®
- `verbose` [Default: `false`, Type: `Bool`]: print more information about how the non-stochastic steady state is solved
- `silent` [Default: `false`, Type: `Bool`]: do not print any information
- `symbolic` [Default: `false`, Type: `Bool`]: try to solve the non-stochastic steady state symbolically and fall back to a numerical solution if not possible
- `perturbation_order` [Default: `1`, Type: `Int`]: take derivatives only up to the specified order at this stage. When working with higher order perturbation later on, respective derivatives will be taken at that stage.
- `simplify` [Default: `true`, Type: `Bool`]: whether to eliminate redundant variables and simplify the non-stochastic steady state (NSSS) problem. Setting this to `false` can speed up the process, but might make it harder to find the NSSS. If the model does not parse at all (at step 1 or 2), setting this option to `false` might solve it.
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
macro parameters(𝓂,ex...)
    # parse options
    verbose = false
    silent = false
    symbolic = false
    precompile = false
    report_missing_parameters = true
    perturbation_order = 1
    guess = Dict{Symbol,Float64}()
    simplify = true
    steady_state_function = nothing
    ss_solver_parameters_algorithm = :ESCH
    ss_solver_parameters_maxtime = 120.0

    for exp in ex[1:end-1]
        postwalk(x -> 
            x isa Expr ?
                x.head == :(=) ?  
                    (x.args[1] == :symbolic && x.args[2] isa Bool) ?
                        symbolic = x.args[2] :
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
                    (x.args[1] == :simplify && x.args[2] isa Bool) ?
                        simplify = x.args[2] :
                    (x.args[1] == :steady_state_function && x.args[2] isa Symbol) ? # allow Symbol, anonymous fn, or any callable expr
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
    
    @assert ss_solver_parameters_algorithm ∈ [:ESCH, :SAMIN] "ss_solver_parameters_algorithm must be :ESCH or :SAMIN. Got $ss_solver_parameters_algorithm. Using default :ESCH."
    
    return quote
        mod = @__MODULE__

        parsed_parameters = process_parameter_definitions(
            $(QuoteNode(ex[end])),
            mod.$𝓂.constants.post_model_macro
        )

        guess_dict = build_guess_dict($guess)

        mod.$𝓂.constants.post_parameters_macro = post_parameters_macro(
            parsed_parameters.calib_parameters_no_var,
            $precompile,
            $simplify,
            guess_dict,
            parsed_parameters.ss_calib_list,
            parsed_parameters.par_calib_list,
            # parsed_parameters.ss_no_var_calib_list,
            # parsed_parameters.par_no_var_calib_list,
            parsed_parameters.bounds,
            $(QuoteNode(ss_solver_parameters_algorithm)),
            $ss_solver_parameters_maxtime,
        )

        # Update equations struct with calibration fields
        mod.$𝓂.equations.calibration = parsed_parameters.equations.calibration
        mod.$𝓂.equations.calibration_no_var = parsed_parameters.equations.calibration_no_var
        mod.$𝓂.equations.calibration_parameters = parsed_parameters.equations.calibration_parameters

        mod.$𝓂.constants.post_complete_parameters = update_post_complete_parameters(
            mod.$𝓂.constants.post_complete_parameters;
            parameters = parsed_parameters.parameters,
            missing_parameters = parsed_parameters.missing_parameters,
        )
        mod.$𝓂.parameter_values = parsed_parameters.parameter_values

        has_missing_parameters = !isempty(mod.$𝓂.constants.post_complete_parameters.missing_parameters)
        missing_params = mod.$𝓂.constants.post_complete_parameters.missing_parameters
        # mod.$𝓂.caches.outdated_NSSS = true
        
        # Store precompile and simplify flag in model container
        
        # Set custom steady state function if provided
        # if !isnothing($steady_state_function)
        set_custom_steady_state_function!(mod.$𝓂, $steady_state_function)
        # end

        mod.$𝓂.functions.functions_written = false

        # time_symbolics = @elapsed 
        # time_rm_red_SS_vars = @elapsed
        if !isnothing($steady_state_function)
            write_ss_check_function!(mod.$𝓂)
        else
            if !has_missing_parameters
                set_up_steady_state_solver!(mod.$𝓂, verbose = $verbose, silent = $silent, avoid_solve = !$simplify, symbolic = $symbolic)
            end
        end

        if !has_missing_parameters
            opts = merge_calculation_options(verbose = $verbose)
            
            SS_and_pars, solution_error, found_solution = solve_steady_state!(mod.$𝓂, opts, $(QuoteNode(ss_solver_parameters_algorithm)), $ss_solver_parameters_maxtime, silent = $silent)
            
            write_symbolic_derivatives!(mod.$𝓂; perturbation_order = $perturbation_order, silent = $silent)

            mod.$𝓂.functions.functions_written = true
        end

        if has_missing_parameters && $report_missing_parameters
            @warn "Model has been set up with incomplete parameter definitions. Missing parameters: $(missing_params). The non-stochastic steady state and perturbation solution cannot be computed until all parameters are defined. Provide missing parameter values via the `parameters` keyword argument in functions like `get_irf`, `get_SS`, `simulate`, etc."
        end

        if !$silent && $report_missing_parameters Base.show(mod.$𝓂) end
        nothing
    end
end
