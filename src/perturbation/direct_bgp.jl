"""
Cached raw-equation representation used to solve a BGP NSSS.

The affine route evaluates the original equations at three consecutive time
origins. Every endogenous level `x` receives an intercept `xᴬ` and a
multiplier `xᴳ`, with the recurrence `x[t+1] = xᴬ + xᴳ*x[t]`. The NSSS root
therefore identifies additive and multiplicative growth from the same
3N-equation system. Once those numerical coefficients are solved, the raw
equations are detrended by their affine recurrences and passed through the
ordinary symbolic derivative generator. No equation-form or `exp` pattern is
needed in that perturbation path.
"""
mutable struct affine_bgp_cache
    steady_state_model::Any
    raw_equations::Vector{Expr}
    steady_state_variables::Vector{Symbol}
    steady_state_solution::Vector{Float64}
    steady_state_parameters::Vector{Float64}
    steady_state_error::Float64
    perturbation_model::Any
    perturbation_order::Int
end

const direct_bgp_cache = affine_bgp_cache

function direct_bgp_equations_supported(𝓂::ℳ)
    all(equation -> stationarization_equation_sides(equation) !== nothing,
        𝓂.equations.original)
end

function direct_bgp_cache_matches(cache, 𝓂::ℳ)
    cache isa affine_bgp_cache && cache.raw_equations == 𝓂.equations.original
end

function ensure_direct_bgp_cache!(𝓂::ℳ)
    if direct_bgp_cache_matches(𝓂.direct_bgp_cache, 𝓂) &&
       𝓂.direct_bgp_cache.steady_state_model isa ℳ
        return 𝓂.direct_bgp_cache
    end

    variables = Symbol[name for name in 𝓂.constants.post_model_macro.var
                       if !is_bgp_affine_name(name)]
    𝓂.direct_bgp_cache = affine_bgp_cache(
        nothing,
        copy(𝓂.equations.original),
        variables,
        Float64[],
        Float64[],
        Inf,
        nothing,
        0,
    )
    𝓂.direct_bgp_cache
end

function affine_bgp_growth_kind(cache::affine_bgp_cache,
                                name::Symbol;
                                tolerance::Float64 = 1e-8)
    names = cache.steady_state_model.constants.post_complete_parameters.nsss_sol_names
    indices = Dict(variable => index for (index, variable) in enumerate(names))
    intercept_name = Symbol(string(name) * "ᴬ")
    growth_name = Symbol(string(name) * "ᴳ")
    haskey(indices, intercept_name) && haskey(indices, growth_name) ||
        throw(ArgumentError("No affine BGP coefficients were solved for $(name)."))
    solution = cache.steady_state_solution
    intercept = solution[indices[intercept_name]]
    growth = solution[indices[growth_name]]
    additive = abs(growth - 1.0) <= tolerance && abs(intercept) > tolerance
    multiplicative = abs(intercept) <= tolerance && abs(growth - 1.0) > tolerance
    additive && return :additive
    multiplicative && return :multiplicative
    abs(intercept) <= tolerance && abs(growth - 1.0) <= tolerance && return :stationary
    :affine
end

function affine_bgp_growth_expression(name::Symbol)
    Expr(:ref, Symbol(string(name) * "ᴳ"), 0)
end

function affine_bgp_intercept_expression(name::Symbol)
    Expr(:ref, Symbol(string(name) * "ᴬ"), 0)
end

function affine_bgp_reference(name::Symbol, shift::Int)
    current = Expr(:ref, name, 0)
    intercept = affine_bgp_intercept_expression(name)
    growth = affine_bgp_growth_expression(name)
    if shift > 0
        for _ in 1:shift
            current = Expr(:call, :+, intercept, Expr(:call, :*, growth, current))
        end
    elseif shift < 0
        for _ in 1:(-shift)
            current = Expr(:call, :/, Expr(:call, :-, current, intercept), growth)
        end
    end
    current
end

function affine_bgp_reference(node, exogenous::Set{Symbol}, shift::Int)
    node isa Expr && node.head == :ref && length(node.args) == 2 || return node
    name, timing = node.args
    timing == :x && return 0
    timing == :ss && return name ∈ exogenous ? 0 : Expr(:ref, name, 0)
    timing isa Int || return node
    name isa Symbol || return node
    name ∈ exogenous && return 0
    affine_bgp_reference(name, timing + shift)
end

function affine_bgp_steady_state_equation(equation,
                                          exogenous::Set{Symbol},
                                          shift::Int)
    sides = stationarization_equation_sides(equation)
    sides === nothing && throw(ArgumentError("Expected an equation, got $(equation)."))
    lhs, rhs = sides
    transformed = Expr(:(=),
                       postwalk(node -> affine_bgp_reference(node, exogenous, shift), lhs),
                       postwalk(node -> affine_bgp_reference(node, exogenous, shift), rhs))
    transformed
end

function ensure_direct_bgp_steady_state_model!(𝓂::ℳ,
                                               cache::direct_bgp_cache)
    metadata = 𝓂.equations.stationarization
    raw_model = deepcopy(𝓂)
    restore_raw_model!(raw_model;
                       rebuild_solver = false,
                       allow_duplicate_equations = true)
    raw_variables = copy(raw_model.constants.post_model_macro.var)

    if cache.steady_state_model isa ℳ &&
       cache.steady_state_variables == raw_variables &&
       direct_bgp_cache_matches(cache, 𝓂)
        return cache.steady_state_model
    end

    raw_equations = copy(raw_model.equations.original)
    exogenous = Set(raw_model.constants.post_model_macro.exo)
    transformed_equations = Expr[]

    # Three points provide one level, one intercept, and one multiplier per
    # endogenous variable. No driver law is identified or skipped specially:
    # additive and multiplicative growth are selected by the solved affine
    # coefficients themselves.
    for shift in (0, 1, 2)
        for equation in raw_equations
            push!(transformed_equations,
                  affine_bgp_steady_state_equation(equation, exogenous, shift))
        end
    end

    T, equations_struct, constants, workspaces = process_model_equations(
        Expr(:block, transformed_equations...),
        raw_model.constants.post_model_macro.max_obc_horizon,
        raw_model.constants.post_parameters_macro.precompile;
        allow_single_variable_equations = true,
        allow_duplicate_equations = true,
    )

    equations_struct.original = raw_equations
    equations_struct.ss_anchors = raw_model.equations.ss_anchors
    equations_struct.bgp_detection = raw_model.equations.bgp_detection
    equations_struct.stationarization = nothing
    equations_struct.calibration = raw_model.equations.calibration
    equations_struct.calibration_no_var = raw_model.equations.calibration_no_var
    equations_struct.calibration_parameters = raw_model.equations.calibration_parameters
    equations_struct.calibration_original = raw_model.equations.calibration_original

    raw_model.constants = constants
    raw_model.constants.post_parameters_macro = 𝓂.constants.post_parameters_macro
    raw_model.constants.post_complete_parameters = update_post_complete_parameters(
        raw_model.constants.post_complete_parameters;
        parameters = 𝓂.constants.post_complete_parameters.parameters,
        missing_parameters = 𝓂.constants.post_complete_parameters.missing_parameters,
    )
    raw_model.equations = equations_struct
    raw_model.workspaces = workspaces
    raw_model.equations.ss_anchors = copy(raw_model.equations.ss_anchors)
    profile = raw_model.equations.bgp_detection
    if profile !== nothing
        for driver in profile.active_drivers
            raw_model.equations.ss_anchors[driver] =
                driver ∈ profile.additive_log_candidates ? 0 : 1
        end
    end
    if metadata !== nothing
        for variable in raw_variables
            haskey(metadata.growth_exponents, variable) || continue
            exponents = metadata.growth_exponents[variable]
            additive_log = variable ∈ metadata.additive_log_drivers
            if any(abs.(exponents) .> 1e-10)
                # An additive-log trend stores its additive increment in the
                # affine intercept. Multiplicative trends require a zero
                # intercept, while the base level is normalized separately
                # through the BGP-driver anchor above.
                additive_log || (raw_model.equations.ss_anchors[Symbol(string(variable) * "ᴬ")] = 0)
            else
                raw_model.equations.ss_anchors[Symbol(string(variable) * "ᴬ")] = 0
                raw_model.equations.ss_anchors[Symbol(string(variable) * "ᴳ")] = 1
            end
        end
    end
    # Lagged affine references are represented by division by the solved
    # multiplier. Keep every multiplier away from the singular value zero;
    # positive gross growth factors are also the domain required by the
    # perturbation and level-IRF mappings.
    affine_bounds = copy(raw_model.constants.post_parameters_macro.bounds)
    for variable in raw_variables
        growth_name = Symbol(string(variable) * "ᴳ")
        lower, upper = get(affine_bounds, growth_name, (1e-8, 1e12))
        affine_bounds[growth_name] = (max(lower, 1e-8), min(upper, 1e12))
    end
    affine_guesses = copy(raw_model.constants.post_parameters_macro.guess)
    for variable in raw_variables
        get!(affine_guesses, Symbol(string(variable) * "ᴬ"), 0.0)
        get!(affine_guesses, Symbol(string(variable) * "ᴳ"), 1.0)
    end
    raw_model.constants.post_parameters_macro = update_post_parameters_macro(
        raw_model.constants.post_parameters_macro;
        bounds = affine_bounds,
        guess = affine_guesses,
    )
    reset_solver_state!(raw_model)
    raw_model.functions.NSSS_custom = nothing
    raw_model.functions.functions_written = false
    # The affine system is coupled through lagged references. Keep it as a
    # numerical system instead of eliminating one variable at a time; the
    # latter can evaluate a singular intermediate affine multiplier before the
    # valid root is reached.
    set_up_steady_state_solver!(raw_model; verbose = false, silent = true, ss_symbolic_mode = :none)
    reset_nsss_solver_cache!(raw_model)

    cache.steady_state_model = raw_model
    cache.perturbation_model = nothing
    cache.perturbation_order = 0
    cache.steady_state_variables = raw_variables
    empty!(cache.steady_state_solution)
    empty!(cache.steady_state_parameters)
    cache.steady_state_error = Inf
    raw_model
end

function direct_bgp_affine_initial_solution!(𝓂::ℳ,
                                             raw_model::ℳ,
                                             metadata,
                                             parameters::Vector{Float64},
                                             opts::CalculationOptions)
    metadata === nothing && return nothing
    if 𝓂.constants.nsss_solver.n_steps == 0
        set_up_steady_state_solver!(
            𝓂;
            verbose = false,
            silent = true,
            ss_symbolic_mode = 𝓂.constants.post_parameters_macro.ss_symbolic_mode,
        )
    end
    active_names = 𝓂.constants.post_complete_parameters.nsss_sol_names
    active_index = Dict(name => index for (index, name) in enumerate(active_names))
    growth_names = stationary_growth_symbol.(metadata.trend_drivers)
    all(haskey(active_index, name) for name in growth_names) || return nothing

    # The stationary representation is used only to seed the nonlinear affine
    # solve. The accepted NSSS root remains the raw three-date solution.
    reset_nsss_solver_cache!(𝓂)
    _, (active_error, _) = solve_nsss_wrapper(
        parameters,
        𝓂,
        opts.tol,
        false,
        false,
        DEFAULT_SOLVER_PARAMETERS,
    )
    active_error <= opts.tol.nsss.acceptance_tol || return nothing
    active_solution = 𝓂.workspaces.nsss_solver.sol_vec_buffer
    driver_growth = Dict(
        driver => active_solution[active_index[stationary_growth_symbol(driver)]]
        for driver in metadata.trend_drivers
    )

    raw_names = raw_model.constants.post_complete_parameters.nsss_sol_names
    raw_index = Dict(name => index for (index, name) in enumerate(raw_names))
    initial = zeros(Float64, length(raw_names))
    for name in raw_names
        haskey(active_index, name) && (initial[raw_index[name]] = active_solution[active_index[name]])
    end
    for (name, exponents) in metadata.growth_exponents
        growth = prod(driver_growth[driver]^exponent
                      for (exponent, driver) in zip(exponents, metadata.trend_drivers))
        affine_intercept = name ∈ metadata.additive_log_drivers ? log(growth) : 0.0
        intercept_name = Symbol(string(name) * "ᴬ")
        growth_name = Symbol(string(name) * "ᴳ")
        haskey(raw_index, intercept_name) && (initial[raw_index[intercept_name]] = affine_intercept)
        haskey(raw_index, growth_name) && (initial[raw_index[growth_name]] =
                                           name ∈ metadata.additive_log_drivers ? 1.0 : growth)
    end
    initial
end

function direct_bgp_generic_initial_solution(raw_model::ℳ,
                                             parameters::Vector{Float64})
    names = raw_model.constants.post_complete_parameters.nsss_sol_names
    parameter_names = raw_model.constants.post_complete_parameters.parameters
    parameter_indices = Dict(name => index for (index, name) in enumerate(parameter_names))
    initial = zeros(Float64, length(names))
    for (index, name) in enumerate(names)
        if is_bgp_affine_intercept_name(name)
            initial[index] = 0.0
        elseif is_bgp_growth_name(name)
            initial[index] = 1.0
        elseif haskey(parameter_indices, name)
            initial[index] = parameters[parameter_indices[name]]
        elseif haskey(raw_model.constants.post_parameters_macro.guess, name)
            initial[index] = raw_model.constants.post_parameters_macro.guess[name]
        else
            # A neutral positive level avoids the singular zero-level branch
            # in nonlinear equations while leaving the affine coefficients
            # free to identify additive versus multiplicative growth.
            initial[index] = 1.0
        end
    end
    initial
end

function seed_direct_bgp_solver!(raw_model::ℳ, initial::Vector{Float64}, parameters::Vector{Float64})
    reset_nsss_solver_cache!(raw_model)
    seed = raw_model.caches.solver[1]
    block = 0
    constants = raw_model.constants.nsss_solver
    for step_index in eachindex(constants.step_types)
        constants.step_types[step_index] == NUMERICAL_STEP || continue
        block += 1
        guesses = seed[2 * block - 1]
        write_range = constants.write_ranges[step_index]
        for index in eachindex(guesses)
            write_index = constants.write_indices[write_range[index]]
            guesses[index] = initial[write_index]
        end
    end
    seed[end] = copy(parameters)
    nothing
end

function direct_bgp_affine_residual(raw_model::ℳ,
                                    solution::Vector{Float64},
                                    parameters::Vector{Float64})
    check_unknowns = union(
        setdiff(raw_model.constants.post_model_macro.vars_in_ss_equations,
                raw_model.constants.post_model_macro.➕_vars),
        raw_model.equations.calibration_parameters,
    )
    indices = indexin(check_unknowns, raw_model.constants.post_complete_parameters.nsss_sol_names)
    any(isnothing, indices) && return Inf
    residual = raw_model.workspaces.nsss_solver.check_residual
    fill!(residual, 0.0)
    raw_model.functions.NSSS_check(residual, parameters, solution[Int.(indices)])
    ℒ.norm(residual)
end

function direct_bgp_nsss_solution!(𝓂::ℳ,
                                   parameters::Vector{Float64};
                                   opts::CalculationOptions,
                                   cold_start::Bool,
                                   caching::Bool)
    cache = ensure_direct_bgp_cache!(𝓂)
    raw_model = ensure_direct_bgp_steady_state_model!(𝓂, cache)
    parameter_values = Float64.(parameters)
    cache_hit = cache.steady_state_parameters == parameter_values &&
                !isempty(cache.steady_state_solution) &&
                isfinite(cache.steady_state_error) &&
                cache.steady_state_error <= opts.tol.nsss.acceptance_tol

    if !cache_hit
        initial_solution = direct_bgp_affine_initial_solution!(
            𝓂,
            raw_model,
            𝓂.equations.stationarization,
            parameter_values,
            opts,
        )
        initial_solution === nothing &&
            (initial_solution = direct_bgp_generic_initial_solution(raw_model, parameter_values))
        initial_error = initial_solution === nothing ? Inf :
                        direct_bgp_affine_residual(raw_model, initial_solution, parameter_values)
        if initial_error <= opts.tol.nsss.acceptance_tol
            raw_solution = initial_solution
            solution_error = initial_error
            iters = 0
        else
            # A neutral generic seed is already installed by the numerical
            # solver setup. Seeding its block buffers with the full 3N vector
            # can misalign eliminated blocks; only use the mapped stationary
            # seed when the processed representation supplied one.
            𝓂.equations.stationarization === nothing ||
                seed_direct_bgp_solver!(raw_model, initial_solution, parameter_values)
            _, (solution_error, iters) = get_NSSS_and_parameters(
                raw_model,
                parameter_values;
                opts = opts,
                # The affine 3N system can need continuation even when the outer
                # solve is a cold start; restricting it to one iteration can leave
                # a valid BGP root unsolved.
                cold_start = cold_start,
                caching = false,
                allow_bgp_fallback = false,
            )
            raw_solution = copy(raw_model.workspaces.nsss_solver.sol_vec_buffer)
            # The step solver may validate a domain-safe rewrite rather than
            # the full three-date affine equations (for example when a
            # non-integer power introduces a `➕` variable). Re-evaluate the
            # actual transformed system before accepting a direct root.
            affine_error = direct_bgp_affine_residual(
                raw_model, raw_solution, parameter_values)
            if !isfinite(affine_error)
                solution_error = Inf
            elseif affine_error > solution_error
                solution_error = affine_error
            end
        end
        cache.steady_state_error = solution_error
        if isfinite(solution_error) && solution_error <= opts.tol.nsss.acceptance_tol
            cache.steady_state_solution = raw_solution
            cache.steady_state_parameters = copy(parameter_values)
        else
            empty!(cache.steady_state_solution)
            empty!(cache.steady_state_parameters)
        end
        iterations = iters
    else
        solution_error = cache.steady_state_error
        iterations = 0
        raw_solution = cache.steady_state_solution
    end

    raw_names = raw_model.constants.post_complete_parameters.nsss_sol_names
    raw_index = Dict(name => index for (index, name) in enumerate(raw_names))
    active_names = 𝓂.constants.post_complete_parameters.nsss_sol_names
    active_solution = 𝓂.workspaces.nsss_solver.sol_vec_buffer
    length(active_solution) == length(active_names) || resize!(active_solution, length(active_names))
    metadata = 𝓂.equations.stationarization
    additive_log_drivers = metadata === nothing ? Symbol[] : metadata.additive_log_drivers
    for (index, name) in enumerate(active_names)
        base_name = is_bgp_growth_name(name) ?
                    Symbol(chop(string(name), tail = 1)) : name
        if base_name ∈ additive_log_drivers && is_bgp_growth_name(name)
            source_name = Symbol(string(base_name) * "ᴬ")
            source_index = get(raw_index, source_name, 0)
            source_index == 0 &&
                throw(ArgumentError("Direct BGP steady state did not solve additive log intercept $(source_name)."))
            active_solution[index] = exp(raw_solution[source_index])
            continue
        end
        source_index = get(raw_index, name, 0)
        source_index == 0 &&
            throw(ArgumentError("Direct BGP steady state did not solve internal variable $(name)."))
        active_solution[index] = raw_solution[source_index]
    end

    output_indices = 𝓂.constants.post_complete_parameters.nsss_output_indices
    output = 𝓂.workspaces.nsss_solver.output_buffer
    length(output) == length(output_indices) || resize!(output, length(output_indices))
    for (index, source_index) in enumerate(output_indices)
        output[index] = active_solution[source_index]
    end

    solved = isfinite(solution_error) && solution_error <= opts.tol.nsss.acceptance_tol
    if caching && solved
        cache_output = 𝓂.caches.non_stochastic_steady_state
        length(cache_output) == length(output) || resize!(cache_output, length(output))
        copyto!(cache_output, output)
        𝓂.caches.valid_for.non_stochastic_steady_state = copy(parameter_values)
    elseif caching
        empty!(𝓂.caches.non_stochastic_steady_state)
        𝓂.caches.valid_for.non_stochastic_steady_state = Float64[]
    end

    output, (solution_error, iterations), cache_hit
end

function direct_bgp_nsss_and_parameters(𝓂::ℳ,
                                        parameter_values::Vector{Float64};
                                        opts::CalculationOptions,
                                        cold_start::Bool,
                                        estimation::Bool,
                                        caching::Bool)
    output, result, cache_hit = direct_bgp_nsss_solution!(𝓂, parameter_values;
                                                          opts = opts,
                                                          cold_start = cold_start,
                                                          caching = caching)
    profile = 𝓂.equations.bgp_detection
    if result[1] > opts.tol.nsss.acceptance_tol &&
       𝓂.equations.stationarization === nothing &&
       profile !== nothing && profile.mode == BGP_UNSUPPORTED_MODE
        # A failed block decomposition can leave the affine solver at a
        # non-root even though the full 3N system has a valid root. Rebuild
        # the isolated raw affine model once and retry with the same generic
        # initialization; this keeps the fallback deterministic without
        # entering the expensive global solver-parameter search.
        cache = ensure_direct_bgp_cache!(𝓂)
        cache.steady_state_model = nothing
        cache.perturbation_model = nothing
        cache.perturbation_order = 0
        output, result, cache_hit = direct_bgp_nsss_solution!(𝓂, parameter_values;
                                                              opts = opts,
                                                              cold_start = length(cache.steady_state_variables) <= 50,
                                                              caching = caching)
    end
    if !cache_hit
        update_ss_counter!(𝓂.counters,
                           result[1] <= opts.tol.nsss.acceptance_tol,
                           estimation = estimation)
    end
    output, result
end

function direct_bgp_solution_valid(𝓂::ℳ)
    cache = 𝓂.direct_bgp_cache
    cache isa affine_bgp_cache || return false
    raw_model = cache.steady_state_model
    raw_model isa ℳ || return false
    isempty(cache.steady_state_solution) && return false
    names = raw_model.constants.post_complete_parameters.nsss_sol_names
    indices = Dict(name => index for (index, name) in enumerate(names))
    for variable in cache.steady_state_variables
        intercept_name = Symbol(string(variable) * "ᴬ")
        growth_name = Symbol(string(variable) * "ᴳ")
        haskey(indices, intercept_name) && haskey(indices, growth_name) || return false
        intercept = cache.steady_state_solution[indices[intercept_name]]
        growth = cache.steady_state_solution[indices[growth_name]]
        isfinite(intercept) && isfinite(growth) && growth > 0 || return false
        abs(intercept) > 1e-8 && abs(growth - 1.0) > 1e-8 && return false
    end
    true
end

function direct_bgp_affine_coefficients(cache::affine_bgp_cache)
    raw_model = cache.steady_state_model
    names = raw_model.constants.post_complete_parameters.nsss_sol_names
    indices = Dict(name => index for (index, name) in enumerate(names))
    coefficients = Dict{Symbol, Tuple{Float64, Float64}}()
    for name in cache.steady_state_variables
        intercept_name = Symbol(string(name) * "ᴬ")
        growth_name = Symbol(string(name) * "ᴳ")
        haskey(indices, intercept_name) && haskey(indices, growth_name) ||
            throw(ArgumentError("Missing affine BGP coefficients for $(name)."))
        intercept = cache.steady_state_solution[indices[intercept_name]]
        growth = cache.steady_state_solution[indices[growth_name]]
        isfinite(intercept) && isfinite(growth) && growth > 0 ||
            throw(ArgumentError("Invalid affine BGP coefficients for $(name): ($(intercept), $(growth))."))
        affine = abs(intercept) > 1e-8 && abs(growth - 1.0) > 1e-8
        affine && throw(ArgumentError(
            "The solved path for $(name) is affine with both an intercept and " *
            "a non-unit multiplier; it is not a balanced-growth path."))
        coefficients[name] = (intercept, growth)
    end
    coefficients
end

function affine_detrended_reference(node,
                                    endogenous::Set{Symbol},
                                    coefficients::Dict{Symbol, Tuple{Float64, Float64}})
    node isa Expr && node.head == :ref && length(node.args) == 2 || return node
    name, timing = node.args
    name isa Symbol && name ∈ endogenous && timing isa Int || return node
    intercept, growth = coefficients[name]
    reference = Expr(:ref, name, timing)
    if abs(intercept) > 1e-8
        timing == 0 && return reference
        return Expr(:call, :+, reference, timing * intercept)
    end
    factor = growth ^ timing
    abs(factor - 1.0) <= 1e-12 && return reference
    Expr(:call, :*, factor, reference)
end

function affine_detrended_equation(equation::Expr,
                                   endogenous::Set{Symbol},
                                   coefficients::Dict{Symbol, Tuple{Float64, Float64}})
    sides = stationarization_equation_sides(equation)
    sides === nothing && throw(ArgumentError("Expected an equation, got $(equation)."))
    lhs, rhs = sides
    transform = node -> affine_detrended_reference(node, endogenous, coefficients)
    Expr(:(=), postwalk(transform, lhs), postwalk(transform, rhs))
end

function ensure_direct_bgp_perturbation_model!(𝓂::ℳ,
                                               cache::affine_bgp_cache,
                                               perturbation_order::Int)
    cache.perturbation_model isa ℳ && cache.perturbation_order >= perturbation_order &&
        return cache.perturbation_model

    coefficients = direct_bgp_affine_coefficients(cache)
    source_model = deepcopy(𝓂)
    raw_equations = copy(source_model.equations.original)
    endogenous = Set(source_model.constants.post_model_macro.var)
    transformed_equations = [affine_detrended_equation(equation, endogenous, coefficients)
                             for equation in raw_equations]

    T, equations_struct, constants, workspaces = process_model_equations(
        Expr(:block, transformed_equations...),
        source_model.constants.post_model_macro.max_obc_horizon,
        source_model.constants.post_parameters_macro.precompile;
        allow_single_variable_equations = true,
    )

    equations_struct.original = raw_equations
    equations_struct.ss_anchors = source_model.equations.ss_anchors
    equations_struct.bgp_detection = source_model.equations.bgp_detection
    equations_struct.stationarization = nothing
    equations_struct.calibration = source_model.equations.calibration
    equations_struct.calibration_no_var = source_model.equations.calibration_no_var
    equations_struct.calibration_parameters = source_model.equations.calibration_parameters
    equations_struct.calibration_original = source_model.equations.calibration_original

    source_model.constants = constants
    source_model.constants.post_parameters_macro = 𝓂.constants.post_parameters_macro
    source_model.constants.post_complete_parameters = update_post_complete_parameters(
        source_model.constants.post_complete_parameters;
        parameters = 𝓂.constants.post_complete_parameters.parameters,
        missing_parameters = 𝓂.constants.post_complete_parameters.missing_parameters,
    )
    source_model.equations = equations_struct
    source_model.workspaces = workspaces
    reset_solver_state!(source_model)
    source_model.functions.NSSS_custom = nothing
    set_up_steady_state_solver!(source_model;
                                verbose = false,
                                silent = true,
                                ss_symbolic_mode = :none)
    write_symbolic_derivatives!(source_model;
                                perturbation_order = perturbation_order,
                                silent = true)
    source_model.functions.functions_written = true

    cache.perturbation_model = source_model
    cache.perturbation_order = perturbation_order
    source_model
end

function direct_bgp_generic_perturbation_model(𝓂::ℳ, perturbation_order::Int)
    # A stationarized model already has the complete dynamic variable set
    # (including driver-growth equations). The affine raw clone is therefore
    # used only for models that remained in the raw representation; otherwise
    # the ordinary processed derivative model is dimensionally authoritative.
    𝓂.equations.stationarization === nothing || return nothing
    cache = 𝓂.direct_bgp_cache
    cache isa affine_bgp_cache || return nothing
    cache.steady_state_model isa ℳ || return nothing
    isempty(cache.steady_state_solution) && return nothing
    ensure_direct_bgp_perturbation_model!(𝓂, cache, perturbation_order)
end

function direct_bgp_perturbation_inputs(cache::affine_bgp_cache,
                                        direct_model::ℳ)
    isempty(cache.steady_state_solution) &&
        return nothing
    solved_names = direct_model.constants.post_complete_parameters.nsss_sol_names
    raw_names = cache.steady_state_model.constants.post_complete_parameters.nsss_sol_names
    raw_indices = Dict(name => index for (index, name) in enumerate(raw_names))
    output_names = solved_names[direct_model.constants.post_complete_parameters.nsss_output_indices]
    all(haskey(raw_indices, name) for name in output_names) || return nothing
    cache.steady_state_solution[[raw_indices[name] for name in output_names]]
end

function bgp_internal_source_indices(𝓂::ℳ)
    metadata = 𝓂.equations.stationarization
    metadata === nothing && return Int[]

    T = 𝓂.constants.post_model_macro
    public_names = filter(name -> !endswith(string(name), "ᴳ"), T.var)
    output_names = vcat(public_names, 𝓂.equations.calibration_parameters)
    output_index_by_name = Dict(name => index for (index, name) in enumerate(output_names))

    hidden_source = Dict{Symbol, Symbol}()
    for equation in metadata.stationary_equations
        sides = stationarization_equation_sides(equation)
        sides === nothing && continue
        lhs, rhs = sides
        for growth_variable in metadata.growth_variables
            if timed_reference(lhs, growth_variable, 0) &&
               rhs isa Expr && rhs.head == :ref &&
               length(rhs.args) == 2 && rhs.args[2] == 0
                hidden_source[growth_variable] = rhs.args[1]
            end
        end
    end

    source_indices = zeros(Int, length(T.var) + length(𝓂.equations.calibration_parameters))
    for (index, name) in enumerate(T.var)
        source_name = get(hidden_source, name, name)
        source_indices[index] = get(output_index_by_name, source_name, 0)
    end
    for (index, name) in enumerate(𝓂.equations.calibration_parameters)
        source_indices[length(T.var) + index] = output_index_by_name[name]
    end
    source_indices
end

function bgp_internal_steady_state_and_parameters(SS_and_pars::Vector{R},
                                                   𝓂::ℳ) where R <: Real
    internal_steady_state_and_parameters(SS_and_pars, 𝓂)
end

function calculate_bgp_jacobian(𝓂::ℳ,
                                parameters::Vector{M},
                                SS_and_pars::Vector{N};
                                caching::Bool = true)::Matrix{M} where {M,N}
    direct_model = direct_bgp_generic_perturbation_model(𝓂, 1)
    if direct_model isa ℳ
        direct_inputs = direct_bgp_perturbation_inputs(𝓂.direct_bgp_cache, direct_model)
        direct_inputs !== nothing &&
            return calculate_jacobian(parameters,
                                       direct_inputs,
                                       direct_model.caches,
                                       direct_model.functions.jacobian,
                                       direct_model.workspaces;
                                       caching = caching)
    end
    if 𝓂.equations.stationarization === nothing
        return calculate_jacobian(parameters,
                                  SS_and_pars,
                                  𝓂.caches,
                                  𝓂.functions.jacobian,
                                  𝓂.workspaces;
                                  caching = caching)
    end

    internal_SS_and_pars = bgp_internal_steady_state_and_parameters(SS_and_pars, 𝓂)
    calculate_jacobian(parameters,
                       internal_SS_and_pars,
                       𝓂.caches,
                       𝓂.functions.jacobian,
                       𝓂.workspaces;
                       caching = caching)
end

function calculate_bgp_hessian(𝓂::ℳ,
                               parameters::Vector{M},
                               SS_and_pars::Vector{N};
                               caching::Bool = true)::SparseMatrixCSC{M, Int} where {M,N}
    direct_model = direct_bgp_generic_perturbation_model(𝓂, 2)
    if direct_model isa ℳ
        direct_inputs = direct_bgp_perturbation_inputs(𝓂.direct_bgp_cache, direct_model)
        direct_inputs !== nothing &&
            return calculate_hessian(parameters,
                                      direct_inputs,
                                      direct_model.caches,
                                      direct_model.functions.hessian,
                                      direct_model.workspaces;
                                      caching = caching)
    end
    if 𝓂.equations.stationarization === nothing
        return calculate_hessian(parameters,
                                 SS_and_pars,
                                 𝓂.caches,
                                 𝓂.functions.hessian,
                                 𝓂.workspaces;
                                 caching = caching)
    end

    internal_SS_and_pars = bgp_internal_steady_state_and_parameters(SS_and_pars, 𝓂)
    calculate_hessian(parameters,
                      internal_SS_and_pars,
                      𝓂.caches,
                      𝓂.functions.hessian,
                      𝓂.workspaces;
                      caching = caching)
end

function calculate_bgp_third_order_derivatives(𝓂::ℳ,
                                               parameters::Vector{M},
                                               SS_and_pars::Vector{N};
                                               caching::Bool = true)::SparseMatrixCSC{M, Int} where {M,N}
    direct_model = direct_bgp_generic_perturbation_model(𝓂, 3)
    if direct_model isa ℳ
        direct_inputs = direct_bgp_perturbation_inputs(𝓂.direct_bgp_cache, direct_model)
        direct_inputs !== nothing &&
            return calculate_third_order_derivatives(parameters,
                                                      direct_inputs,
                                                      direct_model.caches,
                                                      direct_model.functions.third_order_derivatives,
                                                      direct_model.workspaces;
                                                      caching = caching)
    end
    if 𝓂.equations.stationarization === nothing
        return calculate_third_order_derivatives(parameters,
                                                 SS_and_pars,
                                                 𝓂.caches,
                                                 𝓂.functions.third_order_derivatives,
                                                 𝓂.workspaces;
                                                 caching = caching)
    end

    internal_SS_and_pars = bgp_internal_steady_state_and_parameters(SS_and_pars, 𝓂)
    calculate_third_order_derivatives(parameters,
                                      internal_SS_and_pars,
                                      𝓂.caches,
                                      𝓂.functions.third_order_derivatives,
                                      𝓂.workspaces;
                                      caching = caching)
end
