"""
Cached raw-equation representation used to solve a BGP NSSS.

The direct BGP route evaluates the original equations at two consecutive
points, assigns gross growth factors to timed endogenous variables, and
reuses the ordinary NSSS solver. Perturbation stays on the processed
stationary model and uses the ordinary derivative functions with the full
internal BGP steady-state vector.
"""
mutable struct direct_bgp_cache
    steady_state_model::Any
    raw_equations::Vector{Expr}
    steady_state_drivers::Vector{Symbol}
    steady_state_solution::Vector{Float64}
    steady_state_parameters::Vector{Float64}
    steady_state_error::Float64
end

function direct_bgp_cache_matches(cache, 𝓂::ℳ)
    cache isa direct_bgp_cache && cache.raw_equations == 𝓂.equations.original
end

function ensure_direct_bgp_cache!(𝓂::ℳ)
    if direct_bgp_cache_matches(𝓂.direct_bgp_cache, 𝓂) &&
       𝓂.direct_bgp_cache.steady_state_model isa ℳ
        return 𝓂.direct_bgp_cache
    end

    metadata = 𝓂.equations.stationarization
    metadata === nothing &&
        throw(ArgumentError("The direct BGP path requires an active balanced-growth representation."))

    𝓂.direct_bgp_cache = direct_bgp_cache(
        nothing,
        copy(𝓂.equations.original),
        copy(metadata.trend_drivers),
        Float64[],
        Float64[],
        Inf,
    )
    𝓂.direct_bgp_cache
end

function gross_bgp_growth_expression(name::Symbol)
    Expr(:ref, stationary_growth_symbol(name), 0)
end

function expand_gross_bgp_timed_power(node)
    node isa Expr && node.head == :call && node.args[1] == :^ || return node
    base, exponent = node.args[2:3]
    base isa Expr && base.head == :call && base.args[1] == :* || return node
    factors = [Expr(:call, :^, factor, exponent) for factor in base.args[2:end]]
    foldl((left, right) -> Expr(:call, :*, left, right), factors)
end

function gross_bgp_reference(node, exogenous::Set{Symbol}, shift::Int)
    node isa Expr && node.head == :ref && length(node.args) == 2 || return node
    name, timing = node.args
    timing == :x && return 0
    timing == :ss && return name ∈ exogenous ? 0 : Expr(:ref, name, 0)
    timing isa Int || return node
    name isa Symbol || return node
    name ∈ exogenous && return 0

    exponent = timing + shift
    level = Expr(:ref, name, 0)
    exponent == 0 && return level
    growth = gross_bgp_growth_expression(name)
    factor = exponent == 1 ? growth : Expr(:call, :^, growth, exponent)
    exponent > 0 ? Expr(:call, :*, level, factor) :
    Expr(:call, :/, level, Expr(:call, :^, growth, -exponent))
end

function gross_bgp_steady_state_equation(equation,
                                         exogenous::Set{Symbol},
                                         shift::Int)
    sides = stationarization_equation_sides(equation)
    sides === nothing && throw(ArgumentError("Expected an equation, got $(equation)."))
    lhs, rhs = sides
    transformed = Expr(:(=),
                       postwalk(node -> gross_bgp_reference(node, exogenous, shift), lhs),
                       postwalk(node -> gross_bgp_reference(node, exogenous, shift), rhs))
    postwalk(expand_gross_bgp_timed_power, transformed)
end

function ensure_direct_bgp_steady_state_model!(𝓂::ℳ,
                                               cache::direct_bgp_cache)
    metadata = 𝓂.equations.stationarization
    metadata === nothing &&
        throw(ArgumentError("The direct BGP steady-state path requires an active balanced-growth representation."))

    if cache.steady_state_model isa ℳ &&
       cache.steady_state_drivers == metadata.trend_drivers &&
       direct_bgp_cache_matches(cache, 𝓂)
        return cache.steady_state_model
    end

    raw_model = deepcopy(𝓂)
    restore_raw_model!(raw_model)
    raw_equations = copy(raw_model.equations.original)
    exogenous = Set(raw_model.constants.post_model_macro.exo)
    transformed_equations = Expr[]

    # The two points are t and t+1. A shifted driver law is redundant after
    # the driver level is anchored at one. All other shifted equations identify
    # the implied gross growth factors.
    for shift in (0, 1)
        for equation in raw_equations
            if shift == 1 && any(
                driver_growth_equation(equation, driver) !== nothing
                for driver in metadata.trend_drivers
            )
                continue
            end
            push!(transformed_equations,
                  gross_bgp_steady_state_equation(equation, exogenous, shift))
        end
    end

    for driver in metadata.trend_drivers
        push!(transformed_equations,
              Expr(:(=), Expr(:ref, driver, 0), 1))
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
    reset_solver_state!(raw_model)
    raw_model.functions.NSSS_custom = nothing
    raw_model.functions.functions_written = false
    set_up_steady_state_solver!(raw_model; verbose = false, silent = true)

    cache.steady_state_model = raw_model
    cache.steady_state_drivers = copy(metadata.trend_drivers)
    empty!(cache.steady_state_solution)
    empty!(cache.steady_state_parameters)
    cache.steady_state_error = Inf
    raw_model
end

function direct_bgp_nsss_solution!(𝓂::ℳ,
                                   parameters::Vector{Float64};
                                   opts::CalculationOptions,
                                   cold_start::Bool,
                                   caching::Bool)
    cache = ensure_direct_bgp_cache!(𝓂)
    raw_model = ensure_direct_bgp_steady_state_model!(𝓂, cache)
    parameter_values = Float64.(parameters)
    cache_hit = caching &&
                cache.steady_state_parameters == parameter_values &&
                !isempty(cache.steady_state_solution) &&
                isfinite(cache.steady_state_error) &&
                cache.steady_state_error <= opts.tol.nsss.acceptance_tol

    if !cache_hit
        _, (solution_error, iters) = get_NSSS_and_parameters(
            raw_model,
            parameter_values;
            opts = opts,
            cold_start = cold_start,
            caching = false,
        )
        raw_solution = copy(raw_model.workspaces.nsss_solver.sol_vec_buffer)
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
    for (index, name) in enumerate(active_names)
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
    if !cache_hit
        update_ss_counter!(𝓂.counters,
                           result[1] <= opts.tol.nsss.acceptance_tol,
                           estimation = estimation)
    end
    output, result
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
    𝓂.equations.stationarization === nothing &&
        return calculate_jacobian(parameters,
                                   SS_and_pars,
                                   𝓂.caches,
                                   𝓂.functions.jacobian,
                                   𝓂.workspaces;
                                   caching = caching)

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
    𝓂.equations.stationarization === nothing &&
        return calculate_hessian(parameters,
                                 SS_and_pars,
                                 𝓂.caches,
                                 𝓂.functions.hessian,
                                 𝓂.workspaces;
                                 caching = caching)

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
    𝓂.equations.stationarization === nothing &&
        return calculate_third_order_derivatives(parameters,
                                                 SS_and_pars,
                                                 𝓂.caches,
                                                 𝓂.functions.third_order_derivatives,
                                                 𝓂.workspaces;
                                                 caching = caching)

    internal_SS_and_pars = bgp_internal_steady_state_and_parameters(SS_and_pars, 𝓂)
    calculate_third_order_derivatives(parameters,
                                      internal_SS_and_pars,
                                      𝓂.caches,
                                      𝓂.functions.third_order_derivatives,
                                      𝓂.workspaces;
                                      caching = caching)
end
