"""
$(SIGNATURES)
Return the shock decomposition in absolute deviations from the relevant steady state. The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. The deviations are based on the Kalman smoother or filter (depending on the `smooth` keyword argument) or inversion filter using the provided data and solution of the model. When the defaults are used, the filter is selected automatically—Kalman for first order solutions and inversion otherwise—and smoothing is only enabled when the Kalman filter is active. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

In case of pruned second and pruned third order perturbation algorithms the decomposition additionally contains a term `Nonlinearities`. This term represents the nonlinear interaction between the states in the periods after the shocks arrived and in the case of pruned third order, the interaction between (pruned second order) states and contemporaneous shocks.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $FILTER®
- $ALGORITHM®
- $DATA_IN_LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, shocks in columns, and periods as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulation = simulate(RBC)

get_shock_decomposition(RBC,simulation([:c],:,:simulate))
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
◪   Periods ∈ 40-element UnitRange{Int64}
And data, 4×2×40 Array{Float64, 3}:
[showing 3 of 40 slices]
[:, :, 1] ~ (:, :, 1):
        (:eps_z₍ₓ₎)   (:Initial_values)
  (:c)   0.000407252  -0.00104779
  (:k)   0.00374808   -0.0104645
  (:q)   0.00415533   -0.000807161
  (:z)   0.000603617  -1.99957e-6

[:, :, 21] ~ (:, :, 21):
        (:eps_z₍ₓ₎)  (:Initial_values)
  (:c)   0.026511    -0.000433619
  (:k)   0.25684     -0.00433108
  (:q)   0.115858    -0.000328764
  (:z)   0.0150266    0.0

[:, :, 40] ~ (:, :, 40):
        (:eps_z₍ₓ₎)  (:Initial_values)
  (:c)   0.0437976   -0.000187505
  (:k)   0.4394      -0.00187284
  (:q)   0.00985518  -0.000142164
  (:z)  -0.00366442   8.67362e-19
```
"""
function get_shock_decomposition(𝓂::ℳ,
                                data::KeyedArray{Float64};
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)::KeyedArray
    # @nospecialize # reduce compile time

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                    lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, pruning, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts, 
            dynamics = true, 
            algorithm = algorithm)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))

    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂) |> sort

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth)
    
    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    exo_axis = 𝓂.constants.post_complete_parameters.exo_axis_with_subscript

    if pruning
        axis2 = vcat(exo_axis, :Nonlinearities, :Initial_values)
    else
        axis2 = vcat(exo_axis, :Initial_values)
    end

    if pruning
        decomposition[:,1:(end - 2 - pruning),:]    .+= SSS_delta
        decomposition[:,end - 2,:]                  .-= SSS_delta * (size(decomposition,2) - 4)
    end

    return KeyedArray(decomposition[:,1:end-1,:];  Variables = axis1, Shocks = axis2, Periods = 1:size(data,2))
end




"""
$(SIGNATURES)
Return the estimated shocks based on the inversion filter (depending on the `filter` keyword argument), or Kalman filter or smoother (depending on the `smooth` keyword argument) using the provided data and (non-)linear solution of the model. By default MacroModelling chooses the Kalman filter for first order solutions and the inversion filter for higher order ones, and only enables smoothing when the Kalman filter is used. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulation = simulate(RBC)

get_estimated_shocks(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Shocks ∈ 1-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 1×40 Matrix{Float64}:
               (1)          (2)         (3)         (4)         …  (37)         (38)        (39)         (40)
  (:eps_z₍ₓ₎)    0.0603617    0.614652   -0.519048    0.711454       -0.873774     1.27918    -0.929701    -0.2255
```
"""
function get_estimated_shocks(𝓂::ℳ,
                            data::KeyedArray{Float64};
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing,
                            algorithm::Symbol = DEFAULT_ALGORITHM, 
                            filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                            data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                            smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                            verbose::Bool = DEFAULT_VERBOSE,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)::KeyedArray
    # @nospecialize # reduce compile time

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts,
            dynamics = true)
    
    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂) |> sort

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth)
    
    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.exo_axis_with_subscript

    return KeyedArray(shocks;  Shocks = axis1, Periods = 1:size(data,2))
end






"""
$(SIGNATURES)
Return the estimated variables (in levels by default, see `levels` keyword argument) based on the inversion filter (depending on the `filter` keyword argument), or Kalman filter or smoother (depending on the `smooth` keyword argument) using the provided data and (non-)linear solution of the model. With the default options the Kalman filter is applied to first order solutions, while the inversion filter is used for higher order methods; smoothing is activated automatically only when the Kalman filter is available. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- `levels` [Default: `true`, Type: `Bool`]: $LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulation = simulate(RBC)

get_estimated_variables(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 4×40 Matrix{Float64}:
        (1)           (2)           (3)           (4)          …  (37)          (38)            (39)           (40)
  (:c)    5.92901       5.92797       5.92847       5.92048          5.95845       5.95697         5.95686        5.96173
  (:k)   47.3185       47.3087       47.3125       47.2392          47.6034       47.5969         47.5954        47.6402
  (:q)    6.87159       6.86452       6.87844       6.79352          7.00476       6.9026          6.90727        6.95841
  (:z)   -0.00109471   -0.00208056    4.43613e-5   -0.0123318        0.0162992     0.000445065     0.00119089     0.00863586
```
"""
function get_estimated_variables(𝓂::ℳ,
                                data::KeyedArray{Float64};
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                algorithm::Symbol = DEFAULT_ALGORITHM, 
                                filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                                warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                                data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                levels::Bool = DEFAULT_LEVELS,
                                smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)::KeyedArray
    # @nospecialize # reduce compile time                         

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    filter, smooth, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, smooth, algorithm, false, warmup_iterations)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts,
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂) |> sort

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(algorithm), Val(filter), 
                                                                                    warmup_iterations = warmup_iterations, 
                                                                                    opts = opts,
                                                                                    smooth = smooth)

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    return KeyedArray(levels ? variables .+ NSSS[1:length(𝓂.constants.post_model_macro.var)] : variables;  Variables = axis1, Periods = 1:size(data,2))
end


"""
$(SIGNATURES)
Return the vertical concatenation of `get_estimated_variables` and `get_estimated_shocks`
as a single `KeyedArray` with a common first axis named `Estimates` and the
second axis `Periods`. Variables appear first, followed by shocks.

All keyword arguments are forwarded to the respective functions. See the
docstrings of `get_estimated_variables` and `get_estimated_shocks` for details.

# Arguments
- $MODEL®
- $DATA®

# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- $DATA_IN_LEVELS®
- `levels` [Default: `true`, Type: `Bool`]: $LEVELS®
- $SMOOTH®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables followed by shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulation = simulate(RBC)

get_model_estimates(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_shocks ∈ 5-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 5×40 Matrix{Float64}:
               (1)          (2)           (3)           (4)          …  (37)           (38)           (39)           (40)
  (:c)           5.94335      5.94676       5.94474       5.95135          5.93773        5.94333        5.94915        5.95473
  (:k)          47.4603      47.4922       47.476        47.5356          47.4079        47.4567        47.514         47.5696
  (:q)           6.89873      6.92782       6.87844       6.96043          6.85055        6.9403         6.95556        6.96064
  (:z)           0.0014586    0.00561728   -0.00189203    0.0101896       -0.00543334     0.00798437     0.00968602     0.00981981
  (:eps_z₍ₓ₎)    0.12649      0.532556     -0.301549      1.0568     …    -0.746981       0.907104       0.808914       0.788261
```
"""
function get_model_estimates(𝓂::ℳ,
                             data::KeyedArray{Float64};
                             parameters::ParameterType = nothing,
                             steady_state_function::SteadyStateFunctionType = missing,
                             algorithm::Symbol = DEFAULT_ALGORITHM,
                             filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm),
                             warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS,
                             data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                             levels::Bool = DEFAULT_LEVELS,
                             smooth::Bool = DEFAULT_SMOOTH_SELECTOR(filter),
                             verbose::Bool = DEFAULT_VERBOSE,
                             tol::Tolerances = Tolerances(),
                             quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                             sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                             lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)::KeyedArray

    vars = get_estimated_variables(𝓂, data;
                                   parameters = parameters,
                                   steady_state_function = steady_state_function,
                                   algorithm = algorithm,
                                   filter = filter,
                                   warmup_iterations = warmup_iterations,
                                   data_in_levels = data_in_levels,
                                   levels = levels,
                                   smooth = smooth,
                                   verbose = verbose,
                                   tol = tol,
                                   quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                   sylvester_algorithm = sylvester_algorithm,
                                   lyapunov_algorithm = lyapunov_algorithm)

    shks = get_estimated_shocks(𝓂, data;
                                parameters = parameters,
                                steady_state_function = steady_state_function,
                                algorithm = algorithm,
                                filter = filter,
                                warmup_iterations = warmup_iterations,
                                data_in_levels = data_in_levels,
                                smooth = smooth,
                                verbose = verbose,
                                tol = tol,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm = sylvester_algorithm,
                                lyapunov_algorithm = lyapunov_algorithm)

    # Build unified first axis and concatenate data
    est_labels = vcat(collect(axiskeys(vars, 1)), collect(axiskeys(shks, 1)))
    est_data = vcat(Matrix(vars), Matrix(shks))

    return KeyedArray(est_data; Variables_and_shocks = est_labels, Periods = axiskeys(vars, 2))
end



"""
$(SIGNATURES)
Return the standard deviations of the Kalman smoother or filter (depending on the `smooth` keyword argument) estimates of the model variables based on the provided data and first order solution of the model. For the default settings this function relies on the Kalman filter and therefore keeps smoothing enabled. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $DATA_IN_LEVELS®
- $SMOOTH®
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with standard deviations in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulation = simulate(RBC)

get_estimated_variable_standard_deviations(RBC,simulation([:c],:,:simulate))
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Standard_deviations ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
And data, 4×40 Matrix{Float64}:
        (1)           (2)            (3)            (4)            …  (38)            (39)            (40)
  (:c)    1.23202e-9    1.84069e-10    8.23181e-11    8.23181e-11        8.23181e-11     8.23181e-11     0.0
  (:k)    0.00509299    0.000382934    2.87922e-5     2.16484e-6         1.6131e-9       9.31323e-10     1.47255e-9
  (:q)    0.0612887     0.0046082      0.000346483    2.60515e-5         1.31709e-9      1.31709e-9      9.31323e-10
  (:z)    0.00961766    0.000723136    5.43714e-5     4.0881e-6          3.08006e-10     3.29272e-10     2.32831e-10
```
"""
function get_estimated_variable_standard_deviations(𝓂::ℳ,
                                                    data::KeyedArray{Float64};
                                                    parameters::ParameterType = nothing,
                                                    steady_state_function::SteadyStateFunctionType = missing,
                                                    data_in_levels::Bool = DEFAULT_DATA_IN_LEVELS,
                                                    smooth::Bool = DEFAULT_SMOOTH_FLAG,
                                                    verbose::Bool = DEFAULT_VERBOSE,
                                                    tol::Tolerances = Tolerances(),
                                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)
    # @nospecialize # reduce compile time                                               

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    lyapunov_algorithm = lyapunov_algorithm)

    algorithm = :first_order

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂) |> sort

    if data_in_levels
        data_in_deviations = data .- NSSS[obs_idx]
    else
        data_in_deviations = data
    end

    variables, shocks, standard_deviations, decomposition = filter_data_with_model(𝓂, data_in_deviations, Val(:first_order), Val(:kalman), 
                                                                                    smooth = smooth, 
                                                                                    opts = opts)

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    return KeyedArray(standard_deviations;  Standard_deviations = axis1, Periods = 1:size(data,2))
end





"""
$(SIGNATURES)
Return the conditional forecast given restrictions on endogenous variables and shocks (optional). By default, the values represent absolute deviations from the relevant steady state (see `levels` for details). The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions. A constrained minimisation problem is solved to find the combination of shocks with the smallest squared magnitude fulfilling the conditions.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $CONDITIONS®
# Keyword Arguments
- $SHOCK_CONDITIONS®
- $INITIAL_STATE®
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $CONDITIONS_IN_LEVELS®
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables  and shocks in rows, and periods in columns.

# Examples
```jldoctest
using MacroModelling
using SparseArrays, AxisKeys

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 2
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,2),Variables = [:c,:y], Periods = 1:2)
conditions[1,1] = .01
conditions[2,2] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

get_conditional_forecast(RBC_CME, conditions, shocks = shocks, conditions_in_levels = false)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_shocks ∈ 9-element Vector{Symbol}
→   Periods ∈ 42-element UnitRange{Int64}
And data, 9×42 Matrix{Float64}:
                (1)            (2)           …  (41)            (42)
  (:A)            0.0313639      0.0134792         0.000221372     0.000199235
  (:Pi)           0.000780257    0.00020929       -0.000146071    -0.000140137
  (:R)            0.00117156     0.00031425       -0.000219325    -0.000210417
  (:c)            0.01           0.00600605        0.00213278      0.00203751
  (:k)            0.034584       0.0477482   …     0.0397631       0.0380482
  (:y)            0.0446375      0.02              0.00129544      0.001222
  (:z_delta)      0.00025        0.000225          3.69522e-6      3.3257e-6
  (:delta_eps)    0.05           0.0               0.0             0.0
  (:eps_z)        4.61234       -2.16887           0.0             0.0

# The same can be achieved with the other input formats:
# conditions = Matrix{Union{Nothing,Float64}}(undef,7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# using SparseArrays
# conditions = spzeros(7,2)
# conditions[4,1] = .01
# conditions[6,2] = .02

# shocks = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,1,1),Variables = [:delta_eps], Periods = [1])
# shocks[1,1] = .05

# using SparseArrays
# shocks = spzeros(2,1)
# shocks[1,1] = .05
```
"""
function get_conditional_forecast(𝓂::ℳ,
                                conditions::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}};
                                shocks::Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing} = nothing, 
                                initial_state::Union{Vector{Vector{Float64}},Vector{Float64}} = DEFAULT_INITIAL_STATE,
                                periods::Int = DEFAULT_PERIODS, 
                                parameters::ParameterType = nothing,
                                steady_state_function::SteadyStateFunctionType = missing,
                                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                                conditions_in_levels::Bool = DEFAULT_CONDITIONS_IN_LEVELS,
                                algorithm::Symbol = DEFAULT_ALGORITHM,
                                levels::Bool = false,
                                verbose::Bool = DEFAULT_VERBOSE,
                                tol::Tolerances = Tolerances(),
                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                                conditional_forecast_solver::Symbol = :LagrangeNewton)
    # @nospecialize # reduce compile time                        

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    periods += max(size(conditions,2), shocks isa Nothing ? 1 : size(shocks,2)) # isa Nothing needed otherwise JET tests fail

    if conditions isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.var)) * " variables (including auxiliary variables): " * repr(𝓂.constants.post_model_macro.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.var)) * " variables (including auxiliary variables): " * repr(𝓂.constants.post_model_macro.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        conditions_axis = collect(axiskeys(conditions,1))

        conditions_symbols = conditions_axis isa String_input ? conditions_axis .|> Meta.parse .|> replace_indices : conditions_axis

        @assert length(setdiff(conditions_symbols, 𝓂.constants.post_model_macro.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(conditions_symbols,𝓂.constants.post_model_macro.var))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.constants.post_model_macro.var),periods)
        cond_tmp[indexin(sort(conditions_symbols),𝓂.constants.post_model_macro.var),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end
    
    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.constants.post_model_macro.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.constants.post_model_macro.exo)) * " shocks: " * repr(𝓂.constants.post_model_macro.exo)

        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        shocks_axis = collect(axiskeys(shocks,1))

        shocks_symbols = shocks_axis isa String_input ? shocks_axis .|> Meta.parse .|> replace_indices : shocks_axis

        @assert length(setdiff(shocks_symbols,𝓂.constants.post_model_macro.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(shocks_symbols, 𝓂.constants.post_model_macro.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
        shocks_tmp[indexin(sort(shocks_symbols), 𝓂.constants.post_model_macro.exo), axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Number}}(nothing,length(𝓂.constants.post_model_macro.exo),periods)
    end

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            algorithm = algorithm)

    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta, zeros(𝓂.constants.post_model_macro.nVars)]
        else
            initial_state = zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.constants.post_model_macro.nVars], zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.constants.post_model_macro.nVars], zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta, zeros(𝓂.constants.post_model_macro.nVars)]
            else
                initial_state = initial_state - NSSS
            end
        else
            if algorithm ∉ [:pruned_second_order, :pruned_third_order]
                @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
            end
        end
    end

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    Y = zeros(size(𝓂.caches.first_order_solution_matrix,1),periods)

    cond_var_idx = findall(conditions[:,1] .!= nothing)
    
    free_shock_idx = findall(shocks[:,1] .== nothing)

    shocks[free_shock_idx,1] .= 0
    
    if conditions_in_levels
        conditions[cond_var_idx,1] .-= reference_steady_state[cond_var_idx] + SSS_delta[cond_var_idx]
    else
        conditions[cond_var_idx,1] .-= SSS_delta[cond_var_idx]
    end

    @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with at least as many free shocks than conditioned variables. Period 1 has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

    if algorithm ∈ [:second_order, :third_order, :pruned_second_order, :pruned_third_order]
        S₁ = 𝓂.caches.first_order_solution_matrix
        Ŝ₁ = [S₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] zeros(𝓂.constants.post_model_macro.nVars) S₁[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+1:end]]

        S₂ = nothing
        if size(𝓂.caches.second_order_solution, 2) > 0
            S₂ = 𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂
        end

        S₃ = nothing
        if algorithm ∈ [:third_order, :pruned_third_order] && size(𝓂.caches.third_order_solution, 2) > 0
            S₃ = 𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃
        end

        ensure_conditional_forecast_constants!(𝓂.constants; third_order = !isnothing(S₃))

        # Use Lagrange-Newton algorithm to find shocks
        x, matched = find_shocks_conditional_forecast(Val(conditional_forecast_solver),
                                                      initial_state,
                                                      Float64[shocks[:,1]...],
                                                      Float64[conditions[cond_var_idx,1]...],
                                                      cond_var_idx,
                                                      free_shock_idx,
                                                      state_update,
                                                      Ŝ₁,
                                                      S₂,
                                                      S₃,
                                                      𝓂.constants,
                                                      𝓂.workspaces.find_shocks;
                                                      verbose = verbose)

        @assert matched "Numerical stabiltiy issues for restrictions in period 1."
    
        shocks[free_shock_idx,1] .= x
                
        initial_state = state_update(initial_state, Float64[shocks[:,1]...])

        Y[:,1] = pruning ? sum(initial_state) : initial_state

        for i in 2:size(conditions,2)
            cond_var_idx = findall(conditions[:,i] .!= nothing)
            
            if conditions_in_levels
                conditions[cond_var_idx,i] .-= reference_steady_state[cond_var_idx] + SSS_delta[cond_var_idx]
            else
                conditions[cond_var_idx,i] .-= SSS_delta[cond_var_idx]
            end
    
            free_shock_idx = findall(shocks[:,i] .== nothing)

            shocks[free_shock_idx,i] .= 0
    
            @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with at least as many free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."
    
            if length(cond_var_idx) == 0
                # No conditions this period: set free shocks to zero
                shocks[free_shock_idx,i] .= 0
            else
                # Use Lagrange-Newton algorithm to find shocks
                x, matched = find_shocks_conditional_forecast(Val(conditional_forecast_solver),
                                                              pruning ? initial_state : Y[:,i-1],
                                                              Float64[shocks[:,i]...],
                                                              Float64[conditions[cond_var_idx,i]...],
                                                              cond_var_idx,
                                                              free_shock_idx,
                                                              state_update,
                                                              Ŝ₁,
                                                              S₂,
                                                              S₃,
                                                              𝓂.constants,
                                                              𝓂.workspaces.find_shocks;
                                                              verbose = verbose)

                @assert matched "Numerical stabiltiy issues for restrictions in period $i."

                shocks[free_shock_idx,i] .= x
            end

            initial_state = state_update(initial_state, Float64[shocks[:,i]...])

            Y[:,i] = pruning ? sum(initial_state) : initial_state
        end
    elseif algorithm == :first_order
        C = @views 𝓂.caches.first_order_solution_matrix[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+1:end]
    
        CC = C[cond_var_idx,free_shock_idx]

        if length(cond_var_idx) == 1
            @assert any(CC .!= 0) "Free shocks have no impact on conditioned variable in period 1."
        elseif length(free_shock_idx) == length(cond_var_idx)
            CC = RF.lu(CC, check = false)
    
            @assert ℒ.issuccess(CC) "Numerical stabiltiy issues for restrictions in period 1."
        end
    
        shocks[free_shock_idx,1] .= 0
    
        shocks[free_shock_idx,1] = CC \ (conditions[cond_var_idx,1] - state_update(initial_state, Float64[shocks[:,1]...])[cond_var_idx])
    
        Y[:,1] = state_update(initial_state, Float64[shocks[:,1]...])

        for i in 2:size(conditions,2)
            cond_var_idx = findall(conditions[:,i] .!= nothing)
            
            if conditions_in_levels
                conditions[cond_var_idx,i] .-= reference_steady_state[cond_var_idx]
            end
    
            free_shock_idx = findall(shocks[:,i] .== nothing)
            shocks[free_shock_idx,i] .= 0
    
            @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period " * repr(i) * " has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."
    
            CC = C[cond_var_idx,free_shock_idx]
    
            if length(cond_var_idx) == 1
                @assert any(CC .!= 0) "Free shocks have no impact on conditioned variable in period " * repr(i) * "."
            elseif length(free_shock_idx) == length(cond_var_idx)
    
            CC = RF.lu(CC, check = false)
    
            @assert ℒ.issuccess(CC) "Numerical stabiltiy issues for restrictions in period " * repr(i) * "."
            end
    
            shocks[free_shock_idx,i] = CC \ (conditions[cond_var_idx,i] - state_update(Y[:,i-1], Float64[shocks[:,i]...])[cond_var_idx])
    
            Y[:,i] = state_update(Y[:,i-1], Float64[shocks[:,i]...])
        end
    end

    axis1 = [𝓂.constants.post_model_macro.var[var_idx]; 𝓂.constants.post_model_macro.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
    else
        axis1 = [𝓂.constants.post_model_macro.var[var_idx]; map(x->Symbol(string(x) * "₍ₓ₎"), 𝓂.constants.post_model_macro.exo)]
    end

    return KeyedArray([Y[var_idx,:] .+ (levels ? reference_steady_state + SSS_delta : SSS_delta)[var_idx]; convert(Matrix{Float64}, shocks)];  Variables_and_shocks = axis1, Periods = 1:periods)
end


"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model.
Function to use when differentiating IRFs with respect to parameters.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $PARAMETER_VALUES®
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $PERIODS®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $SHOCKS®
- $NEGATIVE_SHOCK®
- $INITIAL_STATE®1
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $QME®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Array{<:AbstractFloat, 3}` with variables in rows, periods in columns, and shocks as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(RBC, RBC.parameter_values)
# output
4×40×1 Array{Float64, 3}:
[:, :, 1] =
 0.00674687  0.00729773  0.00715114  0.00687615  …  0.00146962   0.00140619
 0.0620937   0.0718322   0.0712153   0.0686381      0.0146789    0.0140453
 0.0688406   0.0182781   0.00797091  0.0057232      0.00111425   0.00106615
 0.01        0.002       0.0004      8.0e-5         2.74878e-29  5.49756e-30
```
"""
function get_irf(𝓂::ℳ,
                    parameters::Vector{S};
                    steady_state_function::SteadyStateFunctionType = missing,
                    periods::Int = DEFAULT_PERIODS,
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC,
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCK_SELECTION,
                    negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK, 
                    initial_state::Vector{Float64} = DEFAULT_INITIAL_STATE,
                    levels::Bool = false,
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances(),
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM) where S <: Real

    opts = merge_calculation_options(tol = tol, verbose = verbose,
        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)

    estimation = true

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
           steady_state_function = steady_state_function,
           opts = opts)

    shocks = 𝓂.constants.post_model_macro.nExo == 0 ? :none : shocks

    @assert shocks != :simulate "Use parameters as a known argument to simulate the model."

    shocks, negative_shock, _, periods, shock_idx, shock_history = process_shocks_input(shocks, negative_shock, 1.0, periods, 𝓂)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    reference_steady_state, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation)
    
    if (solution_error > tol.NSSS_acceptance_tol) || isnan(solution_error)
        return zeros(S, length(var_idx), periods, shocks == :none ? 1 : length(shock_idx))
    end

    ∇₁ = calculate_jacobian(parameters, reference_steady_state, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix

    sol_mat, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution)
    
    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        return zeros(S, length(var_idx), periods, shocks == :none ? 1 : length(shock_idx))
    end

    state_update = function(state::Vector, shock::Vector) sol_mat * [state[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx]; shock] end

    initial_state = initial_state == [0.0] ? zeros(𝓂.constants.post_model_macro.nVars) : initial_state - reference_steady_state[1:length(𝓂.constants.post_model_macro.var)]

    # Y = zeros(𝓂.constants.post_model_macro.nVars,periods,𝓂.constants.post_model_macro.nExo)
    Ŷ = []

    for ii in shock_idx
        Y = []

        if shocks isa Union{Symbol_input,String_input}
            shock_history = zeros(𝓂.constants.post_model_macro.nExo,periods)
            if shocks ≠ :none
                shock_history[ii,1] = negative_shock ? -1 : 1
            end
        end

        push!(Y, state_update(initial_state,shock_history[:,1]))

        for t in 1:periods-1
            push!(Y, state_update(Y[end],shock_history[:,t+1]))
        end

        push!(Ŷ, reduce(hcat,Y))
    end

    deviations = reshape(reduce(hcat,Ŷ),𝓂.constants.post_model_macro.nVars, periods, shocks == :none ? 1 : length(shock_idx))[var_idx,:,:]

    if levels
        return deviations .+ reference_steady_state[var_idx]
    else
        return deviations
    end
end




"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model. By default, the values represent absolute deviations from the relevant steady state (see `levels` for details). The non-stochastic steady state (NSSS) is relevant for first order solutions and the stochastic steady state for higher order solutions.

If the model contains occasionally binding constraints and `ignore_obc = false` they are enforced using shocks.

# Arguments
- $MODEL®
# Keyword Arguments
- $PERIODS®
- $ALGORITHM®
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $SHOCKS®
- $NEGATIVE_SHOCK®
- $GENERALISED_IRF®
- $GENERALISED_IRF_WARMUP_ITERATIONS®
- $GENERALISED_IRF_DRAWS®
- $INITIAL_STATE®
- `levels` [Default: `false`, Type: `Bool`]: $LEVELS®
- $SHOCK_SIZE®
- $IGNORE_OBC®
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, periods in columns, and shocks as the third dimension.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_irf(RBC)
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Periods ∈ 40-element UnitRange{Int64}
◪   Shocks ∈ 1-element Vector{Symbol}
And data, 4×40×1 Array{Float64, 3}:
[:, :, 1] ~ (:, :, :eps_z):
        (1)           (2)           …  (39)            (40)
  (:c)    0.00674687    0.00729773        0.00146962      0.00140619
  (:k)    0.0620937     0.0718322         0.0146789       0.0140453
  (:q)    0.0688406     0.0182781         0.00111425      0.00106615
  (:z)    0.01          0.002             2.74878e-29     5.49756e-30
```
"""
function get_irf(𝓂::ℳ; 
                periods::Int = DEFAULT_PERIODS, 
                algorithm::Symbol = DEFAULT_ALGORITHM, 
                parameters::ParameterType = nothing,
                steady_state_function::SteadyStateFunctionType = missing,
                variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = DEFAULT_SHOCKS_EXCLUDING_OBC,
                negative_shock::Bool = DEFAULT_NEGATIVE_SHOCK, 
                generalised_irf::Bool = DEFAULT_GENERALISED_IRF,
                generalised_irf_warmup_iterations::Int = DEFAULT_GENERALISED_IRF_WARMUP,
                generalised_irf_draws::Int = DEFAULT_GENERALISED_IRF_DRAWS,
                initial_state::Union{Vector{Vector{R}},Vector{R}} = DEFAULT_INITIAL_STATE,
                levels::Bool = false,
                shock_size::Real = DEFAULT_SHOCK_SIZE,
                ignore_obc::Bool = DEFAULT_IGNORE_OBC,
                # timer::TimerOutput = TimerOutput(),
                verbose::Bool = DEFAULT_VERBOSE,
                tol::Tolerances = Tolerances(),
                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)::KeyedArray where R <: Real
    # @nospecialize # reduce compile time            

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                                lyapunov_algorithm = lyapunov_algorithm)

    # @timeit_debug timer "Wrangling inputs" begin

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks, negative_shock, shock_size, periods, _, _ = process_shocks_input(shocks, negative_shock, shock_size, periods, 𝓂)
    
    ignore_obc, occasionally_binding_constraints, obc_shocks_included = process_ignore_obc_flag(shocks, ignore_obc, 𝓂)

    generalised_irf = adjust_generalised_irf_flag(generalised_irf, generalised_irf_warmup_iterations, generalised_irf_draws, algorithm, occasionally_binding_constraints, shocks)

    # end # timeit_debug
    
    # @timeit_debug timer "Solve model" begin

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            algorithm = algorithm,
            # timer = timer, 
            obc = occasionally_binding_constraints || obc_shocks_included)
    
    # end # timeit_debug

    # @timeit_debug timer "Get relevant steady state" begin

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm, opts = opts)
    
    # end # timeit_debug

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.constants.post_model_macro.nVars), zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta, zeros(𝓂.constants.post_model_macro.nVars)]
        else
            initial_state = zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.constants.post_model_macro.nVars], zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.constants.post_model_macro.nVars], zeros(𝓂.constants.post_model_macro.nVars) - SSS_delta, zeros(𝓂.constants.post_model_macro.nVars)]
            else
                initial_state = initial_state - NSSS
            end
        else
            if algorithm ∉ [:pruned_second_order, :pruned_third_order]
                @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
            end
        end
    end

    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks without enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end
    
    level = levels ? reference_steady_state + SSS_delta : SSS_delta

    responses = compute_irf_responses(𝓂,
                                        state_update,
                                        initial_state,
                                        level;
                                        periods = periods,
                                        shocks = shocks,
                                        variables = variables,
                                        shock_size = shock_size,
                                        negative_shock = negative_shock,
                                        generalised_irf = generalised_irf,
                                        generalised_irf_warmup_iterations = generalised_irf_warmup_iterations,
                                        generalised_irf_draws = generalised_irf_draws,
                                        enforce_obc = occasionally_binding_constraints,
                                        algorithm = algorithm)

    return responses

end



"""
See [`get_irf`](@ref)
"""
get_irfs = get_irf

"""
See [`get_irf`](@ref)
"""
get_IRF = get_irf

# """
# See [`get_irf`](@ref)
# """
# irfs = get_irf

# """
# See [`get_irf`](@ref)
# """
# irf = get_irf

# """
# See [`get_irf`](@ref)
# """
# IRF = get_irf

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
simulate(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
get_simulation(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
get_simulations(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., shocks = :simulate, levels = get(kwargs, :levels, true))#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `generalised_irf = true`.
"""
get_girf(𝓂::ℳ; kwargs...) =  get_irf(𝓂; kwargs..., generalised_irf = true)









"""
$(SIGNATURES)
Return the (non-stochastic) steady state, calibrated parameters, and derivatives with respect to model parameters.

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $DERIVATIVES®
- $PARAMETER_DERIVATIVES®
- `stochastic` [Default: `false`, Type: `Bool`]: return stochastic steady state using second order perturbation if no other higher order perturbation algorithm is provided in `algorithm`.
- `return_variables_only` [Default: `false`, Type: `Bool`]: return only variables and not calibrated parameters.
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows. The columns show the (non-stochastic) steady state and parameters for which derivatives are taken. 

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_steady_state(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables_and_calibrated_parameters ∈ 4-element Vector{Symbol}
→   Steady_state_and_∂steady_state∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Steady_state)  (:std_z)  (:ρ)     (:δ)      (:α)       (:β)
  (:c)   5.93625          0.0       0.0   -116.072    55.786     76.1014
  (:k)  47.3903           0.0       0.0  -1304.95    555.264   1445.93
  (:q)   6.88406          0.0       0.0    -94.7805   66.8912   105.02
  (:z)   0.0              0.0       0.0      0.0       0.0        0.0
```
"""
function get_steady_state(𝓂::ℳ; 
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing, 
                            derivatives::Bool = DEFAULT_DERIVATIVES_FLAG, 
                            stochastic::Bool = DEFAULT_STOCHASTIC_FLAG,
                            algorithm::Symbol = DEFAULT_ALGORITHM_SELECTOR(stochastic),
                            parameter_derivatives::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                            return_variables_only::Bool = DEFAULT_RETURN_VARIABLES_ONLY,
                            verbose::Bool = DEFAULT_VERBOSE,
                            silent::Bool = DEFAULT_SILENT_FLAG,
                            tol::Tolerances = Tolerances(),
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂))::KeyedArray
    # @nospecialize # reduce compile time
                            
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])
    
    if stochastic
        if algorithm == :first_order
            @info "Stochastic steady state requested but algorithm is $algorithm. Setting `algorithm = :second_order`." maxlog = DEFAULT_MAXLOG
            algorithm = :second_order
        end
    else
        if algorithm != :first_order
            @info "Non-stochastic steady state requested but algorithm is $algorithm. Setting `stochastic = true`." maxlog = DEFAULT_MAXLOG
            stochastic = true
        end
    end

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts)

    vars_in_ss_equations = 𝓂.constants.post_model_macro.vars_in_ss_equations_no_aux
    
    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

    if parameter_derivatives == :all
        length_par = length(𝓂.constants.post_complete_parameters.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.constants.post_complete_parameters.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.constants.post_complete_parameters.parameters)
        length_par = 1
    elseif length(parameter_derivatives) > 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.constants.post_complete_parameters.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.constants.post_complete_parameters.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    SS, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    if solution_error > tol.NSSS_acceptance_tol
        @warn "Could not find non-stochastic steady state. Solution error: $solution_error > $(tol.NSSS_acceptance_tol)"
    end

    if stochastic
        solve!(𝓂, 
                opts = opts, 
                steady_state_function = steady_state_function, 
                dynamics = true, 
                algorithm = algorithm, 
                silent = silent, 
                obc = length(𝓂.equations.obc_violation) > 0)

        if  algorithm == :third_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.third_order_stochastic_steady_state
        elseif  algorithm == :pruned_third_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.pruned_third_order_stochastic_steady_state
        elseif  algorithm == :pruned_second_order
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.pruned_second_order_stochastic_steady_state
        else
            SS[1:length(𝓂.constants.post_model_macro.var)] = 𝓂.caches.second_order_stochastic_steady_state#[indexin(sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.exo_present)),sort(union(𝓂.constants.post_model_macro.var,𝓂.constants.post_model_macro.aux,𝓂.constants.post_model_macro.exo_present)))]
        end
    end

    ms = ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    var_idx = ms.ss_var_idx_in_var_and_calib

    calib_idx = return_variables_only ? Int[] : ms.calib_idx_in_var_and_calib

    if length_par * length(var_idx) > 200 && derivatives
        @info "Most of the time is spent calculating derivatives wrt parameters. If they are not needed, add `derivatives = false` as an argument to the function call." maxlog = DEFAULT_MAXLOG
    #     derivatives = false
    end

    if parameter_derivatives != :all
        derivatives = true
    end

    ensure_name_display_constants!(𝓂)
    var_axis = 𝓂.constants.post_complete_parameters.var_axis
    calib_axis = 𝓂.constants.post_complete_parameters.calib_axis
    axis1 = return_variables_only ? var_axis[var_idx] : vcat(var_axis[var_idx], calib_axis)

    axis2 = vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters[param_idx])

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if derivatives 
        if stochastic
                n_tuple = algorithm ∈ (:third_order, :pruned_third_order) ? 10 : 8
                SSS_result, SSS_pb = rrule(calculate_stochastic_steady_state, Val(algorithm), 𝓂.parameter_values, 𝓂, opts = opts)
                n_sss = length(SSS_result[1])
                n_ss = length(SSS_result[3])
                nv = length(var_idx)
                nc = length(calib_idx)
                n_out = nv + nc
                np = length(𝓂.parameter_values)
                dSSS = zeros(n_out, np)
                for j in 1:n_out
                    if j ≤ nv
                        ∂sss = zeros(n_sss); ∂sss[var_idx[j]] = 1.0
                        seed = ntuple(k -> k == 1 ? ∂sss : NoTangent(), n_tuple)
                    else
                        ∂ss = zeros(n_ss); ∂ss[calib_idx[j - nv]] = 1.0
                        seed = ntuple(k -> k == 3 ? ∂ss : NoTangent(), n_tuple)
                    end
                    ∂p = SSS_pb(seed)[3]
                    if !(∂p isa AbstractZero); dSSS[j, :] .= ∂p; end
                end
                dSSS = dSSS[:, param_idx]

                return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
        else
            (nsss_result, nsss_pb) = rrule(get_NSSS_and_parameters, 𝓂, 𝓂.parameter_values, opts = opts)
            out_idx = [var_idx..., calib_idx...]
            n_ss = length(nsss_result[1])
            np = length(𝓂.parameter_values)
            n_out = length(out_idx)
            dSS = zeros(n_out, np)
            for j in 1:n_out
                ∂ss = zeros(n_ss); ∂ss[out_idx[j]] = 1.0
                ∂p = nsss_pb((∂ss, NoTangent()))[3]
                if !(∂p isa AbstractZero); dSS[j, :] .= ∂p; end
            end
            dSS = dSS[:, param_idx]

            # if length(𝓂.calibration_equations_parameters) == 0        
            #     return KeyedArray(hcat(collect(NSSS)[1:(end-1)],dNSSS);  Variables = [sort(union(𝓂.constants.post_model_macro.exo_present,var))...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters))
            # else
            # return ComponentMatrix(hcat(collect(NSSS), dNSSS)',Axis(vcat(:SS, 𝓂.constants.post_complete_parameters.parameters)),Axis([sort(union(𝓂.constants.post_model_macro.exo_present,var))...,𝓂.calibration_equations_parameters...]))
            # return NamedArray(hcat(collect(NSSS), dNSSS), ([sort(union(𝓂.constants.post_model_macro.exo_present,var))..., 𝓂.calibration_equations_parameters...], vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters)), ("Var. and par.", "∂x/∂y"))
            return KeyedArray(hcat(SS[[var_idx...,calib_idx...]],dSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
            # end
        end
    else
        # return ComponentVector(collect(NSSS),Axis([sort(union(𝓂.constants.post_model_macro.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(collect(NSSS), [sort(union(𝓂.constants.post_model_macro.exo_present,var))..., 𝓂.calibration_equations_parameters...], ("Variables and calibrated parameters"))
        return KeyedArray(SS[[var_idx...,calib_idx...]];  Variables_and_calibrated_parameters = axis1)
    end
    # ComponentVector(non_stochastic_steady_state = ComponentVector(NSSS.non_stochastic_steady_state, Axis(sort(union(𝓂.constants.post_model_macro.exo_present,var)))),
    #                 calibrated_parameters = ComponentVector(NSSS.non_stochastic_steady_state, Axis(𝓂.calibration_equations_parameters)),
    #                 stochastic = stochastic)

    # return (var .=> 𝓂.parameter_to_steady_state(𝓂.parameter_values...)[1:length(var)]),  (𝓂.par .=> 𝓂.parameter_to_steady_state(𝓂.parameter_values...)[length(var)+1:end])[getindex(1:length(𝓂.par),map(x->x ∈ collect(𝓂.calibration_equations_parameters),𝓂.par))]
end


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = false`.
"""
get_non_stochastic_steady_state(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = false)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
get_stochastic_steady_state(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
get_SSS(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
SSS(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)


"""
Wrapper for [`get_steady_state`](@ref) with `stochastic = true`.
"""
sss(args...; kwargs...) = get_steady_state(args...; kwargs..., stochastic = true)



"""
See [`get_steady_state`](@ref)
"""
SS(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
steady_state(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
get_SS(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
get_ss(args...; kwargs...) = get_steady_state(args...; kwargs...)

"""
See [`get_steady_state`](@ref)
"""
ss(args...; kwargs...) = get_steady_state(args...; kwargs...)




"""
$(SIGNATURES)
Return the solution of the model. In the linear case it returns the non-stochastic steady state (NSSS) followed by the linearised solution of the model. In the nonlinear case (higher order perturbation) the function returns a multidimensional array with the endogenous variables as the second dimension and the state variables, shocks, and perturbation parameter (:Volatility) as the other dimensions.

The values of the output represent the NSSS in the case of a linear solution and below it the effect that deviations from the NSSS of the respective past states, shocks, and perturbation parameter have (perturbation parameter = 1) on the present value (NSSS deviation) of the model variables.

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with the endogenous variables including the auxiliary endogenous and exogenous variables (due to leads and lags > 1) as columns. The rows and other dimensions (depending on the chosen perturbation order) include the NSSS for the linear case only, followed by the states, and exogenous shocks. Subscripts following variable names indicate the timing (e.g. `variable₍₋₁₎`  indicates the variable being in the past). Superscripts indicate leads or lags (e.g. `variableᴸ⁽²⁾` indicates the variable being in lead by two periods). If no super- or subscripts follow the variable name, the variable is in the present.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_solution(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Steady_state__States__Shocks ∈ 4-element Vector{Symbol}
→   Variables ∈ 4-element Vector{Symbol}
And data, 4×4 adjoint(::Matrix{Float64}) with eltype Float64:
                   (:c)         (:k)        (:q)        (:z)
  (:Steady_state)   5.93625     47.3903      6.88406     0.0
  (:k₍₋₁₎)          0.0957964    0.956835    0.0726316  -0.0
  (:z₍₋₁₎)          0.134937     1.24187     1.37681     0.2
  (:eps_z₍ₓ₎)       0.00674687   0.0620937   0.0688406   0.01
```
"""
function get_solution(𝓂::ℳ; 
                        parameters::ParameterType = nothing,
                        steady_state_function::SteadyStateFunctionType = missing,
                        algorithm::Symbol = DEFAULT_ALGORITHM, 
                        silent::Bool = DEFAULT_SILENT_FLAG,
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂))::KeyedArray
    # @nospecialize # reduce compile time      

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts,
            dynamics = true, 
            silent = silent, 
            algorithm = algorithm)

    if algorithm == :first_order
        solution_matrix = 𝓂.caches.first_order_solution_matrix
    end

    axis1 = [𝓂.constants.post_model_macro.past_not_future_and_mixed; :Volatility; 𝓂.constants.post_model_macro.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
        axis1[1:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)] = axis1[1:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)] .* "₍₋₁₎"
    else
        axis1 = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.constants.post_model_macro.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
    end

    axis2 = 𝓂.constants.post_model_macro.var

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if algorithm == :second_order
        return KeyedArray(permutedims(reshape(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :pruned_second_order
        return KeyedArray(permutedims(reshape(𝓂.caches.second_order_solution * 𝓂.constants.second_order.𝐔₂, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :third_order
        return KeyedArray(permutedims(reshape(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    elseif algorithm == :pruned_third_order
        return KeyedArray(permutedims(reshape(𝓂.caches.third_order_solution * 𝓂.constants.third_order.𝐔₃, 
                                    𝓂.constants.post_model_macro.nVars, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo, 
                                    𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    else
        axis1 = [:Steady_state; 𝓂.constants.post_model_macro.past_not_future_and_mixed; 𝓂.constants.post_model_macro.exo]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] = axis1[end-length(𝓂.constants.post_model_macro.exo)+1:end] .* "₍ₓ₎"
            axis1[2:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)+1] = axis1[2:length(𝓂.constants.post_model_macro.past_not_future_and_mixed)+1] .* "₍₋₁₎"
        else
            axis1 = [:Steady_state; map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.constants.post_model_macro.past_not_future_and_mixed); map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.constants.post_model_macro.exo)]
        end

        n_vars = length(𝓂.constants.post_model_macro.var)
        nsss = if length(𝓂.caches.non_stochastic_steady_state) >= n_vars
            𝓂.caches.non_stochastic_steady_state[1:n_vars]
        else
            get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)[1][1:n_vars]
        end

        return KeyedArray([nsss solution_matrix]';
                            Steady_state__States__Shocks = axis1,
                            Variables = axis2)
    end
end


"""
Wrapper for [`get_solution`](@ref) with `algorithm = :first_order`.
"""
get_first_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :first_order)

"""
Wrapper for [`get_solution`](@ref) with `algorithm = :second_order`.
"""
get_second_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :second_order)

"""
Wrapper for [`get_solution`](@ref) with `algorithm = :third_order`.
"""
get_third_order_solution(args...; kwargs...) = get_solution(args...; kwargs..., algorithm = :third_order)

"""
See [`get_solution`](@ref)
"""
get_perturbation_solution(args...; kwargs...) = get_solution(args...; kwargs...)




"""
$(SIGNATURES)
Return the components of the solution of the model: non-stochastic steady state (NSSS), and solution martrices corresponding to the order of the solution. Note that all returned objects have the variables in rows and the solution matrices have as columns the state variables followed by the perturbation/volatility parameter for higher order solution matrices and lastly the exogenous shocks. Higher order perturbation matrices are sparse and have the Kronecker product of the forementioned elements as columns. The last element, a Boolean indicates whether the solution is numerically accurate.
Function to use when differentiating IRFs with respect to parameters.

# Arguments
- $MODEL®
- $PARAMETERS®
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Tuple` consisting of a `Vector` containing the NSSS, followed by a `Matrix` containing the first order solution matrix. In case of higher order solutions, `SparseMatrixCSC` represent the higher order solution matrices. The last element is a `Bool` indicating the correctness of the solution provided.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_solution(RBC, RBC.parameter_values)
# output
([5.936252888048724, 47.39025414828808, 6.884057971014486, 0.0], 
 [0.09579643002421227 0.1349373930517757 0.006746869652588215; 
  0.9568351489231555 1.241874201151121 0.06209371005755664; 
  0.07263157894736819 1.376811594202897 0.06884057971014486; 
  0.0 0.19999999999999998 0.01], true)
```
"""
function get_solution(𝓂::ℳ, 
                        parameters::Vector{S}; 
                        steady_state_function::SteadyStateFunctionType = missing,
                        algorithm::Symbol = DEFAULT_ALGORITHM, 
                        verbose::Bool = DEFAULT_VERBOSE, 
                        tol::Tolerances = Tolerances(),
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂)) where S <: Real

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? :bicgstab : sylvester_algorithm[2])

    estimation = true

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
           opts = opts, 
           steady_state_function = steady_state_function,
           algorithm = algorithm)

    
    if length(𝓂.constants.post_parameters_macro.bounds) > 0
        for (k,v) in 𝓂.constants.post_parameters_macro.bounds
            if k ∈ 𝓂.constants.post_complete_parameters.parameters
                if min(max(parameters[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1], v[1]), v[2]) != parameters[indexin([k], 𝓂.constants.post_complete_parameters.parameters)][1]
                    return -Inf
                end
            end
        end
    end

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts, estimation = estimation)

    if solution_error > tol.NSSS_acceptance_tol || isnan(solution_error)
        if algorithm in [:second_order, :pruned_second_order]
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], zeros(length(𝓂.constants.post_model_macro.var),2), spzeros(length(𝓂.constants.post_model_macro.var),2), false
        elseif algorithm in [:third_order, :pruned_third_order]
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], zeros(length(𝓂.constants.post_model_macro.var),2), spzeros(length(𝓂.constants.post_model_macro.var),2), spzeros(length(𝓂.constants.post_model_macro.var),2), false
        else
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], zeros(length(𝓂.constants.post_model_macro.var),2), false
        end
    end

    ∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution)
    
    update_perturbation_counter!(𝓂.counters, solved, estimation = estimation, order = 1)

    if !solved
        if algorithm in [:second_order, :pruned_second_order]
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, spzeros(length(𝓂.constants.post_model_macro.var),2), false
        elseif algorithm in [:third_order, :pruned_third_order]
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, spzeros(length(𝓂.constants.post_model_macro.var),2), spzeros(length(𝓂.constants.post_model_macro.var),2), false
        else
            return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, false
        end
    end

    if algorithm in [:second_order, :pruned_second_order]
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)# * 𝓂.constants.second_order.𝐔∇₂
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                    initial_guess = 𝓂.caches.second_order_solution,
                                opts = opts)

        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        𝐒₂ *= 𝓂.constants.second_order.𝐔₂

        if !(typeof(𝐒₂) <: AbstractSparseMatrix)
            𝐒₂ = sparse(𝐒₂) # * 𝓂.constants.second_order.𝐔₂)
        end

        return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, 𝐒₂, true
    elseif algorithm in [:third_order, :pruned_third_order]
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.hessian)# * 𝓂.constants.second_order.𝐔∇₂
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.constants, 𝓂.workspaces, 𝓂.caches;
                                                    initial_guess = 𝓂.caches.second_order_solution,
                                opts = opts)
    
        update_perturbation_counter!(𝓂.counters, solved2, estimation = estimation, order = 2)

        𝐒₂ *= 𝓂.constants.second_order.𝐔₂

        if !(typeof(𝐒₂) <: AbstractSparseMatrix)
            𝐒₂ = sparse(𝐒₂) # * 𝓂.constants.second_order.𝐔₂)
        end

        ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂.caches, 𝓂.functions.third_order_derivatives)# * 𝓂.constants.third_order.𝐔∇₃
                
    𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 
                            𝐒₁, 𝐒₂,
                            𝓂.constants,
                            𝓂.workspaces,
                            𝓂.caches;
                            initial_guess = 𝓂.caches.third_order_solution,
                            opts = opts)

    update_perturbation_counter!(𝓂.counters, solved3, estimation = estimation, order = 3)

        𝐒₃ *= 𝓂.constants.third_order.𝐔₃

        if !(typeof(𝐒₃) <: AbstractSparseMatrix)
            𝐒₃ = sparse(𝐒₃) # * 𝓂.constants.third_order.𝐔₃)
        end

        return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, 𝐒₂, 𝐒₃, true
    else
        return SS_and_pars[1:length(𝓂.constants.post_model_macro.var)], 𝐒₁, true
    end
end


"""
$(SIGNATURES)
Return the conditional variance decomposition of endogenous variables with regards to the shocks using the linearised solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- `periods` [Default: `[1:20...,Inf]`, Type: `Union{Vector{Int},Vector{Float64},UnitRange{Int64}}`]: vector of periods for which to calculate the conditional variance decomposition. If the vector contains `Inf`, also the unconditional variance decomposition is calculated (same output as [`get_variance_decomposition`](@ref)).
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, shocks in columns, and periods as the third dimension.

# Examples
```jldoctest part1
using MacroModelling

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_conditional_variance_decomposition(RBC_CME)
# output
3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 7-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
◪   Periods ∈ 21-element Vector{Float64}
And data, 7×2×21 Array{Float64, 3}:
[showing 3 of 21 slices]
[:, :, 1] ~ (:, :, 1.0):
              (:delta_eps)  (:eps_z)
  (:A)         0.0           1.0
  (:Pi)        0.00158668    0.998413
  (:R)         0.00158668    0.998413
  (:c)         0.0277348     0.972265
  (:k)         0.00869568    0.991304
  (:y)         0.0           1.0
  (:z_delta)   1.0           0.0

[:, :, 11] ~ (:, :, 11.0):
              (:delta_eps)  (:eps_z)
  (:A)         5.88653e-32   1.0
  (:Pi)        0.0245641     0.975436
  (:R)         0.0245641     0.975436
  (:c)         0.0175249     0.982475
  (:k)         0.00869568    0.991304
  (:y)         7.63511e-5    0.999924
  (:z_delta)   1.0           0.0

[:, :, 21] ~ (:, :, Inf):
              (:delta_eps)  (:eps_z)
  (:A)         9.6461e-31    1.0
  (:Pi)        0.0156771     0.984323
  (:R)         0.0156771     0.984323
  (:c)         0.0134672     0.986533
  (:k)         0.00869568    0.991304
  (:y)         0.000313462   0.999687
  (:z_delta)   1.0           0.0
```
"""
function get_conditional_variance_decomposition(𝓂::ℳ; 
                                                periods::Union{Vector{Int},Vector{Float64},UnitRange{Int64}} = DEFAULT_CONDITIONAL_VARIANCE_PERIODS,
                                                parameters::ParameterType = nothing,
                                                steady_state_function::SteadyStateFunctionType = missing,  
                                                verbose::Bool = DEFAULT_VERBOSE,
                                                tol::Tolerances = Tolerances(),
                                                quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                                lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)
    # @nospecialize # reduce compile time                                            

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm)

    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
            opts = opts,
            steady_state_function = steady_state_function,  
            parameters = parameters)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix

    𝑺₁, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution)
    
    update_perturbation_counter!(𝓂.counters, solved, order = 1)

    A = @views 𝑺₁[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[indexin(𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,1:𝓂.constants.post_model_macro.nVars),:]
    
    sort!(periods)

    maxperiods = periods == [Inf] ? 0 : Int(maximum(periods[isfinite.(periods)]))

    var_container = zeros(size(𝑺₁)[1], 𝓂.constants.post_model_macro.nExo, length(periods))

    for i in 1:𝓂.constants.post_model_macro.nExo
        C = @views 𝑺₁[:,𝓂.constants.post_model_macro.nPast_not_future_and_mixed+i]
        CC = C * C'
        varr = zeros(size(C)[1],size(C)[1])
        for k in 1:maxperiods
            varr = A * varr * A' + CC
            if k ∈ periods
                var_container[:,i,indexin(k, periods)] = ℒ.diag(varr)
            end
        end
        if Inf in periods
            # Ensure lyapunov workspace is properly sized and get it
            lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, 𝓂.constants.post_model_macro.nVars, :first_order)

            covar_raw, _ = solve_lyapunov_equation(A, CC, lyap_ws,
                                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                                    tol = opts.tol.lyapunov_tol,
                                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                                    verbose = opts.verbose)

            var_container[:,i,indexin(Inf,periods)] = ℒ.diag(covar_raw) # numerically more stable
        end
    end

    sum_var_container = max.(sum(var_container, dims=2),eps())
    
    var_container[var_container .< opts.tol.lyapunov_acceptance_tol] .= 0
    
    cond_var_decomp = var_container ./ sum_var_container

    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

    KeyedArray(cond_var_decomp; Variables = axis1, Shocks = axis2, Periods = periods)
end


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
get_fevd = get_conditional_variance_decomposition


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
get_forecast_error_variance_decomposition = get_conditional_variance_decomposition


"""
See [`get_conditional_variance_decomposition`](@ref)
"""
fevd = get_conditional_variance_decomposition




"""
$(SIGNATURES)
Return the variance decomposition of endogenous variables with regards to the shocks using the linearised solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $QME®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows, and shocks in columns.

# Examples
```jldoctest part1
using MacroModelling

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

get_variance_decomposition(RBC_CME)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 7-element Vector{Symbol}
→   Shocks ∈ 2-element Vector{Symbol}
And data, 7×2 Matrix{Float64}:
              (:delta_eps)  (:eps_z)
  (:A)         9.78485e-31   1.0
  (:Pi)        0.0156771     0.984323
  (:R)         0.0156771     0.984323
  (:c)         0.0134672     0.986533
  (:k)         0.00869568    0.991304
  (:y)         0.000313462   0.999687
  (:z_delta)   1.0           0.0
```
"""
function get_variance_decomposition(𝓂::ℳ; 
                                    parameters::ParameterType = nothing,
                                    steady_state_function::SteadyStateFunctionType = missing,
                                    verbose::Bool = DEFAULT_VERBOSE,
                                    tol::Tolerances = Tolerances(),
                                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM)
    # @nospecialize # reduce compile time
                                    
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    lyapunov_algorithm = lyapunov_algorithm)
    
    # Initialize constants at entry point
    constants = initialise_constants!(𝓂)

    solve!(𝓂, 
            opts = opts, 
            steady_state_function = steady_state_function, 
            parameters = parameters)

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂.caches, 𝓂.functions.jacobian)# |> Matrix

    sol, qme_sol, solved = calculate_first_order_solution(∇₁,
                                                        constants,
                                                        𝓂.workspaces,
                                                        𝓂.caches;
                                                        opts = opts,
                                                        initial_guess = 𝓂.caches.qme_solution)

    update_perturbation_counter!(𝓂.counters, solved, order = 1)
    
    variances_by_shock = zeros(𝓂.constants.post_model_macro.nVars, 𝓂.constants.post_model_macro.nExo)

    A = @views sol[:, 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]

    for i in 1:𝓂.constants.post_model_macro.nExo
        C = @views sol[:, 𝓂.constants.post_model_macro.nPast_not_future_and_mixed + i]
        
        CC = C * C'

        # Ensure lyapunov workspace is properly sized and get it
        lyap_ws = ensure_lyapunov_workspace!(𝓂.workspaces, 𝓂.constants.post_model_macro.nVars, :first_order)

        covar_raw, _ = solve_lyapunov_equation(A, CC, lyap_ws,
                                                lyapunov_algorithm = opts.lyapunov_algorithm, 
                                                tol = opts.tol.lyapunov_tol,
                                                acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                                verbose = opts.verbose)

        variances_by_shock[:,i] = ℒ.diag(covar_raw)
    end

    sum_variances_by_shock = max.(sum(variances_by_shock, dims=2), eps())
    
    variances_by_shock[variances_by_shock .< opts.tol.lyapunov_acceptance_tol] .= 0
    
    var_decomp = variances_by_shock ./ sum_variances_by_shock
    
    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain

    KeyedArray(var_decomp; Variables = axis1, Shocks = axis2)
end



"""
See [`get_variance_decomposition`](@ref)
"""
get_var_decomp = get_variance_decomposition




"""
$(SIGNATURES)
Return the correlations of endogenous variables using the first, pruned second, or pruned third order perturbation solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows and columns.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_correlation(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 ∈ 4-element Vector{Symbol}
And data, 4×4 Matrix{Float64}:
        (:c)       (:k)       (:q)       (:z)
  (:c)   1.0        0.999812   0.550168   0.314562
  (:k)   0.999812   1.0        0.533879   0.296104
  (:q)   0.550168   0.533879   1.0        0.965726
  (:z)   0.314562   0.296104   0.965726   1.0
```
"""
function get_correlation(𝓂::ℳ; 
                        parameters::ParameterType = nothing,
                        steady_state_function::SteadyStateFunctionType = missing,  
                        algorithm::Symbol = DEFAULT_ALGORITHM,
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances())
    # @nospecialize # reduce compile time                    

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    @assert algorithm ∈ [:first_order, :pruned_second_order,:pruned_third_order] "Correlation can only be calculated for first order perturbation or second and third order pruned perturbation solutions."

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts, 
            algorithm = algorithm)

    if algorithm == :pruned_third_order
        covar_dcmp, state_μ, SS_and_pars, solved = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, opts = opts)
    elseif algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
    else
        covar_dcmp, sol, _, SS_and_pars, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)

        @assert solved "Could not find covariance matrix."
    end

    covar_dcmp[abs.(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol] .= 0

    std = sqrt.(max.(ℒ.diag(covar_dcmp),eps(Float64)))
    
    corr = covar_dcmp ./ (std * std')
    
    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    KeyedArray(collect(corr); Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1)
end

"""
See [`get_correlation`](@ref)
"""
get_corr = get_correlation


"""
See [`get_correlation`](@ref)
"""
corr = get_correlation




"""
$(SIGNATURES)
Return the autocorrelations of endogenous variables using the first, pruned second, or pruned third order perturbation solution. 

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- `autocorrelation_periods` [Default: `1:5`, Type: `UnitRange{Int}`]: periods for which to return the autocorrelation
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) with variables in rows and autocorrelation periods in columns.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_autocorrelation(RBC)
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Autocorrelation_periods ∈ 5-element UnitRange{Int64}
And data, 4×5 Matrix{Float64}:
        (1)         (2)         (3)         (4)         (5)
  (:c)    0.966974    0.927263    0.887643    0.849409    0.812761
  (:k)    0.971015    0.931937    0.892277    0.853876    0.817041
  (:q)    0.32237     0.181562    0.148347    0.136867    0.129944
  (:z)    0.2         0.04        0.008       0.0016      0.00032
```
"""
function get_autocorrelation(𝓂::ℳ; 
                            autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                            parameters::ParameterType = nothing,
                            steady_state_function::SteadyStateFunctionType = missing,  
                            algorithm::Symbol = DEFAULT_ALGORITHM,
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                            verbose::Bool = DEFAULT_VERBOSE,
                            tol::Tolerances = Tolerances())
    # @nospecialize # reduce compile time
    
    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] "Autocorrelation can only be calculated for first order perturbation or second and third order pruned perturbation solutions."

    solve!(𝓂, 
            opts = opts, 
            steady_state_function = steady_state_function, 
            parameters = parameters,
            algorithm = algorithm)

    if algorithm == :pruned_third_order
        covar_dcmp, state_μ, autocorr, SS_and_pars, solved = calculate_third_order_moments_with_autocorrelation(𝓂.parameter_values, 𝓂.constants.post_model_macro.var, 𝓂, 
                                                                                            opts = opts, 
                                                                                            autocorrelation_periods = autocorrelation_periods)

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol,:] .= 0
    elseif algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)

        ŝ_to_ŝ₂ⁱ = ℒ.diagm(ones(size(Σᶻ₂,1)))

        autocorr = zeros(size(covar_dcmp,1),length(autocorrelation_periods))

        covar_dcmp[abs.(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol] .= 0

        for i in autocorrelation_periods
            autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ ℒ.diag(covar_dcmp) 
            ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
        end

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol,:] .= 0
    else
        covar_dcmp, sol, _, SS_and_pars, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)

        @assert solved "Could not find covariance matrix."

        A = @views sol[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]
    
        autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in autocorrelation_periods])

        autocorr[ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol,:] .= 0
    end

    
    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis

    KeyedArray(collect(autocorr); Variables = axis1, Autocorrelation_periods = autocorrelation_periods)
end

"""
See [`get_autocorrelation`](@ref)
"""
get_autocorr(args...; kwargs...) = get_autocorrelation(args...; kwargs...)


"""
See [`get_autocorrelation`](@ref)
"""
autocorr(args...; kwargs...) = get_autocorrelation(args...; kwargs...)




"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using the first, pruned second, or pruned third order perturbation solution. By default returns: non-stochastic steady state (NSSS), and standard deviations, but can optionally return variances, and covariance matrix. Derivatives of the moments (except for covariance) can also be provided by setting `derivatives` to `true`.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- `non_stochastic_steady_state` [Default: `true`, Type: `Bool`]: switch to return SS of endogenous variables
- `mean` [Default: `false`, Type: `Bool`]: switch to return mean of endogenous variables (the mean for the linearised solutoin is the NSSS)
- `standard_deviation` [Default: `true`, Type: `Bool`]: switch to return standard deviation of endogenous variables
- `variance` [Default: `false`, Type: `Bool`]: switch to return variance of endogenous variables
- `covariance` [Default: `false`, Type: `Bool`]: switch to return covariance matrix of endogenous variables
- $(VARIABLES®(DEFAULT_VARIABLES_EXCLUDING_OBC))
- $DERIVATIVES®
- $PARAMETER_DERIVATIVES®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Dict{Symbol,KeyedArray}` containing the selected moments. All moments have variables as rows and the moment as the first column followed by partial derivatives wrt parameters. The `KeyedArray` type is provided by the `AxisKeys` package.

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

moments = get_moments(RBC);

moments[:non_stochastic_steady_state]
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Steady_state_and_∂steady_state∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Steady_state)  (:std_z)  (:ρ)     (:δ)      (:α)       (:β)
  (:c)   5.93625          0.0       0.0   -116.072    55.786     76.1014
  (:k)  47.3903           0.0       0.0  -1304.95    555.264   1445.93
  (:q)   6.88406          0.0       0.0    -94.7805   66.8912   105.02
  (:z)   0.0              0.0       0.0      0.0       0.0        0.0
```


```jldoctest part1
moments[:standard_deviation]
# output
2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Variables ∈ 4-element Vector{Symbol}
→   Standard_deviation_and_∂standard_deviation∂parameter ∈ 6-element Vector{Symbol}
And data, 4×6 Matrix{Float64}:
        (:Standard_deviation)  (:std_z)  …  (:δ)       (:α)       (:β)
  (:c)   0.0266642              2.66642     -0.384359   0.2626     0.144789
  (:k)   0.264677              26.4677      -5.74194    2.99332    6.30323
  (:q)   0.0739325              7.39325     -0.974722   0.726551   1.08
  (:z)   0.0102062              1.02062      0.0        0.0        0.0
```
"""
function get_moments(𝓂::ℳ; 
                    parameters::ParameterType = nothing,
                    steady_state_function::SteadyStateFunctionType = missing,  
                    non_stochastic_steady_state::Bool = DEFAULT_NON_STOCHASTIC_STEADY_STATE_FLAG, 
                    mean::Bool = DEFAULT_MEAN_FLAG,
                    standard_deviation::Bool = DEFAULT_STANDARD_DEVIATION_FLAG, 
                    variance::Bool = DEFAULT_VARIANCE_FLAG, 
                    covariance::Bool = DEFAULT_COVARIANCE_FLAG, 
                    variables::Union{Symbol_input,String_input} = DEFAULT_VARIABLES_EXCLUDING_OBC, 
                    derivatives::Bool = DEFAULT_DERIVATIVES_FLAG,
                    parameter_derivatives::Union{Symbol_input,String_input} = DEFAULT_VARIABLE_SELECTION,
                    algorithm::Symbol = DEFAULT_ALGORITHM,
                    silent::Bool = DEFAULT_SILENT_FLAG,
                    quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                    sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                    lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                    verbose::Bool = DEFAULT_VERBOSE,
                    tol::Tolerances = Tolerances())#limit output by selecting pars and vars like for plots and irfs!?
    # @nospecialize # reduce compile time          

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                    sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                    sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                    lyapunov_algorithm = lyapunov_algorithm)

    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            algorithm = algorithm, 
            opts = opts, 
            silent = silent)

    for (moment_name, condition) in [("Mean", mean), ("Standard deviation", standard_deviation), ("Variance", variance), ("Covariance", covariance)]
        if condition
            @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] moment_name * " only available for algorithms: `first_order`, `pruned_second_order`, and `pruned_third_order`."
        end
    end

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂) |> sort

    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives
    length_par = 0

    param_idx = 0:0
    
    if parameter_derivatives == :all
        length_par = length(𝓂.constants.post_complete_parameters.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.constants.post_complete_parameters.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.constants.post_complete_parameters.parameters)
        length_par = 1
    elseif length(parameter_derivatives) ≥ 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.constants.post_complete_parameters.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.constants.post_complete_parameters.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    NSSS, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    @assert solution_error < tol.NSSS_acceptance_tol "Could not find non-stochastic steady state."

    if length_par * length(NSSS) > 200 && derivatives
        @info "Most of the time is spent calculating derivatives wrt parameters. If they are not needed, add `derivatives = false` as an argument to the function call." maxlog = DEFAULT_MAXLOG
    end 

    if (!variance && !standard_deviation && !non_stochastic_steady_state && !mean && !covariance)
        derivatives = false
    end

    if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state || mean || covariance)
        derivatives = true
    end


    axis1 = 𝓂.constants.post_model_macro.var

    ensure_name_display_constants!(𝓂)
    axis1 = 𝓂.constants.post_complete_parameters.var_axis
    axis2 = 𝓂.constants.post_complete_parameters.exo_axis_plain


    if derivatives
        if non_stochastic_steady_state
            axis1 = [𝓂.constants.post_model_macro.var[var_idx]...,𝓂.equations.calibration_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            axis2 = vcat(:Steady_state, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            (nsss_d_result, nsss_d_pb) = rrule(get_NSSS_and_parameters, 𝓂, 𝓂.parameter_values, opts = opts)
            n_ss_full = length(nsss_d_result[1])
            np = length(𝓂.parameter_values)
            dNSSS = zeros(n_ss_full, np)
            for j in 1:n_ss_full
                ∂ss = zeros(n_ss_full); ∂ss[j] = 1.0
                ∂p = nsss_d_pb((∂ss, NoTangent()))[3]
                if !(∂p isa AbstractZero); dNSSS[j, :] .= ∂p; end
            end
            dNSSS = dNSSS[:, param_idx]
            
            if length(𝓂.equations.calibration_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.constants.post_model_macro.nVars .+ (1:length(𝓂.equations.calibration_parameters)))
            else
                var_idx_ext = var_idx
            end

            SS =  KeyedArray(hcat(collect(NSSS[var_idx_ext]),dNSSS[var_idx_ext,:]);  Variables = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
        end
        
        axis1 = 𝓂.constants.post_model_macro.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        # Hoist covariance rrule call for shared use across variance/std_dev/covariance
        if variance || standard_deviation || covariance
            if algorithm == :pruned_second_order
                _cov_result, _cov_pb = rrule(calculate_second_order_moments_with_covariance, 𝓂.parameter_values, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                _n_cov_tuple = 15
            elseif algorithm == :pruned_third_order
                _cov_obs = covariance ? :full_covar : variables
                _cov_result, _cov_pb = rrule(calculate_third_order_moments, 𝓂.parameter_values, _cov_obs, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                _n_cov_tuple = 4
            else
                _cov_result, _cov_pb = rrule(calculate_covariance, 𝓂.parameter_values, 𝓂, opts = opts)
                covar_dcmp = _cov_result[1]
                @assert _cov_result[5] "Could not find covariance matrix."
                _n_cov_tuple = 5
            end

            # Compute variance Jacobian via VJP (shared by variance & std_dev)
            if variance || standard_deviation
                _np_cov = length(𝓂.parameter_values)
                _nv_cov = size(covar_dcmp, 1)
                _dvariance_full = zeros(_nv_cov, _np_cov)
                for j in 1:_nv_cov
                    if covar_dcmp[j,j] > eps(Float64)
                        ∂Σ = zeros(_nv_cov, _nv_cov); ∂Σ[j,j] = 1.0
                        seed = ntuple(k -> k == 1 ? ∂Σ : NoTangent(), _n_cov_tuple)
                        ∂p = _cov_pb(seed)[2]
                        if !(∂p isa AbstractZero); _dvariance_full[j,:] .= ∂p; end
                    end
                end
            end
        end

        if variance
            axis2 = vcat(:Variance, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            dvariance = _dvariance_full[:, param_idx]

            vari = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))
            
            varrs =  KeyedArray(hcat(vari[var_idx],dvariance[var_idx,:]);  Variables = axis1, Variance_and_∂variance∂parameter = axis2)

            if standard_deviation
                axis2 = vcat(:Standard_deviation, 𝓂.constants.post_complete_parameters.parameters[param_idx])
            
                if any(x -> contains(string(x), "◖"), axis2)
                    axis2_decomposed = decompose_name.(axis2)
                    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
                end
    
                standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))
                # Analytical: d(sqrt(v))/d(params) = dv/d(params) / (2*sqrt(v))
                dst_dev = _dvariance_full[:, param_idx] ./ (2 .* standard_dev)

                st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
            end
        end

        if standard_deviation
            axis2 = vcat(:Standard_deviation, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))
            # Analytical: d(sqrt(v))/d(params) = dv/d(params) / (2*sqrt(v))
            dst_dev = _dvariance_full[:, param_idx] ./ (2 .* standard_dev)

            st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
        end


        if covariance
            axis3 = vcat(:Covariance, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis3)
                axis3_decomposed = decompose_name.(axis3)
                axis3 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis3_decomposed]
            end

            # Compute full covariance Jacobian via VJP from hoisted rrule
            _np_cov2 = length(𝓂.parameter_values)
            _nv_cov2 = size(covar_dcmp, 1)
            dcovariance = zeros(_nv_cov2 * _nv_cov2, _np_cov2)
            for j in 1:(_nv_cov2 * _nv_cov2)
                r = mod1(j, _nv_cov2)
                c = div(j - 1, _nv_cov2) + 1
                ∂Σ = zeros(_nv_cov2, _nv_cov2); ∂Σ[r,c] = 1.0
                seed = ntuple(k -> k == 1 ? ∂Σ : NoTangent(), _n_cov_tuple)
                ∂p = _cov_pb(seed)[2]
                if !(∂p isa AbstractZero); dcovariance[j,:] .= ∂p; end
            end
            dcovariance = dcovariance[:, param_idx]
        end

        if mean && algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            axis2 = vcat(:Mean, 𝓂.constants.post_complete_parameters.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            (mean_result, mean_pb) = rrule(calculate_mean, 𝓂.parameter_values, 𝓂, algorithm = algorithm, opts = opts)
            state_μ = mean_result[1]
            
            @assert mean_result[2] "Mean not found."

            n_mean = length(state_μ)
            np_mean = length(𝓂.parameter_values)
            state_μ_dev = zeros(n_mean, np_mean)
            for j in 1:n_mean
                ∂mean = zeros(n_mean); ∂mean[j] = 1.0
                ∂p = mean_pb((∂mean, NoTangent()))[2]
                if !(∂p isa AbstractZero); state_μ_dev[j,:] .= ∂p; end
            end
            state_μ_dev = state_μ_dev[:, param_idx]
            
            var_means =  KeyedArray(hcat(state_μ[var_idx], state_μ_dev[var_idx, :]);  Variables = axis1, Mean_and_∂mean∂parameter = axis2)
        end
    else
        if non_stochastic_steady_state
            axis1 = [𝓂.constants.post_model_macro.var[var_idx]...,𝓂.equations.calibration_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            if length(𝓂.equations.calibration_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.constants.post_model_macro.nVars .+ (1:length(𝓂.equations.calibration_parameters)))
            else
                var_idx_ext = var_idx
            end

            if mean && algorithm == :first_order
                var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
            end

            SS =  KeyedArray(collect(NSSS)[var_idx_ext];  Variables = axis1)
        end

        axis1 = 𝓂.constants.post_model_macro.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if mean && !(variance || standard_deviation || covariance)
            state_μ, solved = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, opts = opts)

            @assert solved "Mean not found."

            var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
        end

        if variance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                
                @assert solved "Could not find covariance matrix."

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end
            end

            varr = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

            varrs = KeyedArray(varr[var_idx];  Variables = axis1)

            if standard_deviation
                st_dev = KeyedArray(sqrt.(varr)[var_idx];  Variables = axis1)
            end
        end

        if standard_deviation
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                
                @assert solved "Could not find covariance matrix."

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end
            end
            st_dev = KeyedArray(sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))[var_idx];  Variables = axis1)
        end

        if covariance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _, solved = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, opts = opts)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _, solved = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)
                
                @assert solved "Could not find covariance matrix."

                if mean && algorithm == :first_order
                    var_means = KeyedArray(collect(NSSS)[var_idx];  Variables = 𝓂.constants.post_model_macro.var[var_idx])
                end
            end
        end
    end

    
    ret = Dict{Symbol,KeyedArray}()
    if non_stochastic_steady_state
        # push!(ret,SS)
        ret[:non_stochastic_steady_state] = SS
    end
    if mean
        # push!(ret,var_means)
        ret[:mean] = var_means
    end
    if standard_deviation
        # push!(ret,st_dev)
        ret[:standard_deviation] = st_dev
    end
    if variance
        # push!(ret,varrs)
        ret[:variance] = varrs
    end
    if covariance
        axis1 = 𝓂.constants.post_model_macro.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if derivatives
            # Determine dimensions
            n_full_vars = size(covar_dcmp, 1)        # Full number of variables (n)
            n_reduced_vars = length(var_idx)         # Reduced number of variables (k)
            n_params = length(param_idx)             # Number of parameters (p)

            # Pre-allocate array to hold reduced covariance and derivatives: k x k x (1 + p)
            covar_with_derivs = zeros(n_reduced_vars, n_reduced_vars, 1 + n_params)

            # First slice is the reduced covariance matrix
            # Take the slice of the covariance matrix
            covar_with_derivs[:, :, 1] = covar_dcmp[var_idx, var_idx]

            # Subsequent slices are reduced derivatives wrt each parameter
            # The key is to reshape the full n_full_vars x n_full_vars derivative
            # and then take the slice [var_idx, var_idx]
            for i in 1:n_params
                # dcovariance[:,i] is the vectorized full derivative (n_full_vars^2 length)
                # 1. Reshape to the full n_full_vars x n_full_vars derivative matrix
                full_deriv_matrix = reshape(dcovariance[:, i], n_full_vars, n_full_vars)

                # 2. Take the reduced slice [var_idx, var_idx] and assign to the pre-allocated array
                covar_with_derivs[:, :, i+1] = full_deriv_matrix[var_idx, var_idx]
            end

            # ---
            # Create axis names (unchanged from original)
            if !@isdefined axis3
                axis3 = vcat(:Covariance, 𝓂.constants.post_complete_parameters.parameters[param_idx])

                if any(x -> contains(string(x), "◖"), axis3)
                    axis3_decomposed = decompose_name.(axis3)
                    axis3 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis3_decomposed]
                end
            end
            # ---

            # Assign the result
            # The array is already sliced, so no need for covar_with_derivs[var_idx, var_idx, :]
            ret[:covariance] = KeyedArray(covar_with_derivs;
                Variables = axis1,         # Assuming axis1 holds the full variable names
                𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1,
                Covariance_and_∂covariance∂parameter = axis3
            )
        else
            # push!(ret,KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1))
            ret[:covariance] = KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1)
        end
    end

    return ret
end

"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
get_variance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = true, non_stochastic_steady_state = false, standard_deviation = false, covariance = false)[:variance]


"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
get_var = get_variance


"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
var = get_variance


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
get_standard_deviation(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = true, covariance = false)[:standard_deviation]


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
get_std =  get_standard_deviation

"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
get_stdev =  get_standard_deviation


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
stdev =  get_standard_deviation


"""
Wrapper for [`get_moments`](@ref) with `standard_deviation = true` and `non_stochastic_steady_state = false, variance = false, covariance = false`.
"""
std =  get_standard_deviation

"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, derivatives = false`.
"""
get_covariance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = true, derivatives = false)[:covariance]


"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false`.
"""
get_cov = get_covariance


"""
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false`.
"""
cov = get_covariance


"""
Wrapper for [`get_moments`](@ref) with `mean = true`, and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, covariance = false`
"""
get_mean(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = false, mean = true)[:mean]


# """
# Wrapper for [`get_moments`](@ref) with `mean = true`, the default algorithm being `:pruned_second_order`, and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, covariance = false`
# """
# mean(𝓂::ℳ; kwargs...) = get_mean(𝓂; kwargs...)



"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using either the linearised solution or the pruned second or pruned third order perturbation solution. By default returns a `Dict` with: non-stochastic steady state (NSSS), and standard deviations, but can also return variances, and covariance matrix. Values are returned in the order given for the specific moment.
Function to use when differentiating model moments with respect to parameters.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- `parameter_values` [Type: `Vector`]: Parameter values. If `parameter_names` is not explicitly defined, `parameter_values` are assumed to correspond to the parameters and the order of the parameters declared in the `@parameters` block.
# Keyword Arguments
- `parameters` [Type: `Vector{Symbol}`]: Corresponding names in the same order as `parameter_values`.
- `non_stochastic_steady_state` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the NSSS of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `mean` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the mean of selected variables (the mean for the linearised solution is the NSSS). Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `standard_deviation` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the standard deviation of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `variance` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the variance of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `covariance` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the covariance of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. For grouped covariance computation, pass a `Vector` of `Vector`s (e.g. `[[:y, :c], [:k, :i]]`) to compute covariances only within each group, returning a single covariance matrix where cross-group covariances are set to zero. This allows more granular control over which covariances to compute. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all variables less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all variables less those related to occasionally binding constraints. `:all` will contain all variables.
- `autocorrelation` [Default: `Symbol[]`, Type: `Union{Symbol_input,String_input}`]: variables for which to show the autocorrelation of selected variables. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or `\"y\"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` contains all shocks less those related to auxiliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxiliary variables. `:all` will contain all variables.
- `autocorrelation_periods` [Default: `1:5`, Type = `UnitRange{Int}`]: periods for which to return the autocorrelation of selected variables
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $QME®
- $LYAPUNOV®
- $SYLVESTER®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `Dict` with the name of the statistics and the corresponding vectors (NSSS, mean, standard deviation, variance) or matrices (covariance, autocorrelation).

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

get_statistics(RBC, RBC.parameter_values, parameters = get_parameters(RBC), standard_deviation = RBC.var)
# output
Dict{Symbol, AbstractArray{Float64}} with 1 entry:
  :standard_deviation => [0.0266642, 0.264677, 0.0739325, 0.0102062]

# For grouped covariance (computing covariances only within specified groups):
get_statistics(RBC, RBC.parameter_values, covariance = [[:c, :k], [:y, :i]])
# output
Dict{Symbol, AbstractArray{Float64}} with 1 entry:
  :covariance => [...4x4 matrix with c-k covariances filled, y-i covariances filled, and cross-group elements set to zero...]
```
"""
function get_statistics(𝓂::ℳ,
                        parameter_values::Vector{T};
                        parameters::Union{Vector{Symbol},Vector{String}} = 𝓂.constants.post_complete_parameters.parameters,
                        steady_state_function::SteadyStateFunctionType = missing, 
                        non_stochastic_steady_state::Union{Symbol_input,String_input} = Symbol[],
                        mean::Union{Symbol_input,String_input} = Symbol[],
                        standard_deviation::Union{Symbol_input,String_input} = Symbol[],
                        variance::Union{Symbol_input,String_input} = Symbol[],
                        covariance::Union{Symbol_input,String_input, Vector{Vector{Symbol}},Vector{Tuple{Symbol,Vararg{Symbol}}},Vector{Vector{Symbol}},Tuple{Tuple{Symbol,Vararg{Symbol}},Vararg{Tuple{Symbol,Vararg{Symbol}}}}, Vector{Vector{String}},Vector{Tuple{String,Vararg{String}}},Vector{Vector{String}},Tuple{Tuple{String,Vararg{String}},Vararg{Tuple{String,Vararg{String}}}}} = Symbol[],
                        autocorrelation::Union{Symbol_input,String_input} = Symbol[],
                        autocorrelation_periods::UnitRange{Int} = DEFAULT_AUTOCORRELATION_PERIODS,
                        algorithm::Symbol = DEFAULT_ALGORITHM,
                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                        sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                        lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM,
                        verbose::Bool = DEFAULT_VERBOSE,
                        tol::Tolerances = Tolerances()) where T

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                        sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                        sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                        lyapunov_algorithm = lyapunov_algorithm)

    @assert length(parameter_values) == length(parameters) "Vector of `parameters` must correspond to `parameter_values` in length and order. Define the parameter names in the `parameters` keyword argument."
    
    @assert algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] || !(!(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[])) "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = parse_variables_input_to_index(non_stochastic_steady_state, 𝓂)

    mean_var_idx = parse_variables_input_to_index(mean, 𝓂)

    std_var_idx = parse_variables_input_to_index(standard_deviation, 𝓂)

    var_var_idx = parse_variables_input_to_index(variance, 𝓂)

    covar_var_idx = parse_variables_input_to_index(covariance, 𝓂)
    
    # Parse covariance groups if input is grouped format
    covar_groups = is_grouped_covariance_input(covariance) ? parse_covariance_groups(covariance, 𝓂.constants) : nothing

    autocorr_var_idx = parse_variables_input_to_index(autocorrelation, 𝓂)


    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters)]

    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.constants.post_complete_parameters.parameters, parameters), 𝓂.constants.post_complete_parameters.parameters), indexin(parameters, 𝓂.constants.post_complete_parameters.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]

    solved = true

    if algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]))
        algorithm = :pruned_second_order
    end

    solve!(𝓂, 
           algorithm = algorithm, 
           steady_state_function = steady_state_function,
           opts = opts)

    if !(non_stochastic_steady_state == Symbol[]) && (standard_deviation == Symbol[]) && (variance == Symbol[]) && (covariance == Symbol[]) && (autocorrelation == Symbol[])
        SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, all_parameters, opts = opts) # timer = timer, 
        
        SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

        ret = Dict{Symbol,AbstractArray{T}}()

        ret[:non_stochastic_steady_state] = solution_error < opts.tol.NSSS_acceptance_tol ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))

        return ret
    end

    if algorithm == :pruned_third_order

        if !(autocorrelation == Symbol[])
            second_mom_third_order = union(autocorr_var_idx, std_var_idx, var_var_idx)

            covar_dcmp, state_μ, autocorr, SS_and_pars, solved = calculate_third_order_moments_with_autocorrelation(all_parameters, 𝓂.constants.post_model_macro.var[second_mom_third_order], 𝓂, covariance = 𝓂.constants.post_model_macro.var[covar_var_idx], opts = opts, autocorrelation_periods = autocorrelation_periods)

        elseif !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[])

            covar_dcmp, state_μ, SS_and_pars, solved = calculate_third_order_moments(all_parameters, 𝓂.constants.post_model_macro.var[union(std_var_idx, var_var_idx)], 𝓂, covariance = 𝓂.constants.post_model_macro.var[covar_var_idx], opts = opts)

        end

    elseif algorithm == :pruned_second_order

        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[])
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments_with_covariance(all_parameters, 𝓂, opts = opts)
        else
            state_μ, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂, solved = calculate_second_order_moments(all_parameters, 𝓂, opts = opts)
        end

    else
        covar_dcmp, sol, _, SS_and_pars, solved = calculate_covariance(all_parameters, 𝓂, opts = opts)

        # @assert solved "Could not find covariance matrix."
    end

    SS = SS_and_pars[1:end - length(𝓂.equations.calibration)]

    if !(variance == Symbol[])
        varrs = convert(Vector{T},max.(ℒ.diag(covar_dcmp),eps(Float64)))
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(varrs)
        end
    end

    if !(autocorrelation == Symbol[])
        if algorithm == :pruned_second_order
            ŝ_to_ŝ₂ⁱ = zero(ŝ_to_ŝ₂)
            ŝ_to_ŝ₂ⁱ += ℒ.diagm(ones(size(ŝ_to_ŝ₂,1)))

            autocorr = zeros(T,size(covar_dcmp,1),length(autocorrelation_periods))

            for i in autocorrelation_periods
                autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ max.(ℒ.diag(covar_dcmp),eps(Float64))
                ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
            end
            
            autocorr[ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol,:] .= 0
        elseif !(algorithm == :pruned_third_order)
            A = @views sol[:,1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.constants.post_model_macro.nVars))[𝓂.constants.post_model_macro.past_not_future_and_mixed_idx,:]
        
            autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ max.(ℒ.diag(covar_dcmp),eps(Float64))) for i in autocorrelation_periods])

            autocorr[ℒ.diag(covar_dcmp) .< opts.tol.lyapunov_acceptance_tol,:] .= 0
        end
    end

    if !(standard_deviation == Symbol[])
        st_dev = sqrt.(abs.(convert(Vector{T}, max.(ℒ.diag(covar_dcmp),eps(Float64)))))
    end
        

    # ret = AbstractArray{T}[]
    ret = Dict{Symbol,AbstractArray{T}}()

    if !(non_stochastic_steady_state == Symbol[])
        # push!(ret,SS[SS_var_idx])
        ret[:non_stochastic_steady_state] = solved ? SS[SS_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(SS_var_idx) ? 0 : length(SS_var_idx))
    end
    if !(mean == Symbol[])
        if algorithm ∉ [:pruned_second_order,:pruned_third_order]
            # push!(ret,SS[mean_var_idx])
            ret[:mean] = solved ? SS[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        else
            # push!(ret,state_μ[mean_var_idx])
            ret[:mean] = solved ? state_μ[mean_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(mean_var_idx) ? 0 : length(mean_var_idx))
        end
    end
    if !(standard_deviation == Symbol[])
        # push!(ret,st_dev[std_var_idx])
        ret[:standard_deviation] = solved ? st_dev[std_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(std_var_idx) ? 0 : length(std_var_idx))
    end
    if !(variance == Symbol[])
        # push!(ret,varrs[var_var_idx])
        ret[:variance] = solved ? varrs[var_var_idx] : fill(Inf * sum(abs2,parameter_values), isnothing(var_var_idx) ? 0 : length(var_var_idx))
    end
    if !(covariance == Symbol[])
        covar_dcmp_sp = (ℒ.triu(covar_dcmp))

        # droptol!(covar_dcmp_sp,eps(Float64))

        if !isnothing(covar_groups)
            # Extract only the specified covariance groups (block diagonal structure)
            # Return a single matrix with zeros for non-computed covariances
            if solved
                # Initialize matrix with zeros
                covar_result = zeros(T, length(covar_var_idx), length(covar_var_idx))
                
                # Fill in only the specified groups
                for group in covar_groups
                    for (i_idx, i) in enumerate(group)
                        for (j_idx, j) in enumerate(group)
                            # Find position in covar_var_idx
                            i_pos = findfirst(==(i), covar_var_idx)
                            j_pos = findfirst(==(j), covar_var_idx)
                            if !isnothing(i_pos) && !isnothing(j_pos)
                                covar_result[i_pos, j_pos] = covar_dcmp_sp[i, j]
                            end
                        end
                    end
                end
                
                ret[:covariance] = covar_result
            else
                # Return matrix with Inf-filled diagonal and zeros elsewhere
                covar_result = fill(Inf * sum(abs2,parameter_values), length(covar_var_idx), length(covar_var_idx))
                ret[:covariance] = covar_result
            end
        else
            # Original behavior for non-grouped input
            # push!(ret,covar_dcmp_sp[covar_var_idx,covar_var_idx])
            ret[:covariance] = solved ? covar_dcmp_sp[covar_var_idx,covar_var_idx] : fill(Inf * sum(abs2,parameter_values),isnothing(covar_var_idx) ? 0 : length(covar_var_idx), isnothing(covar_var_idx) ? 0 : length(covar_var_idx))
        end
    end
    if !(autocorrelation == Symbol[]) 
        # push!(ret,autocorr[autocorr_var_idx,:] )
        ret[:autocorrelation] = solved ? autocorr[autocorr_var_idx,:] : fill(Inf * sum(abs2,parameter_values), isnothing(autocorr_var_idx) ? 0 : length(autocorr_var_idx), isnothing(autocorrelation_periods) ? 0 : length(autocorrelation_periods))
    end

    return ret
end




"""
$(SIGNATURES)
Return the loglikelihood of the model given the data and parameters provided. The loglikelihood is either calculated based on the inversion or the Kalman filter (depending on the `filter` keyword argument). By default the package selects the Kalman filter for first order solutions and the inversion filter for nonlinear (higher order) solution algorithms. The data must be provided as a `KeyedArray{Float64}` with the names of the variables to be matched in rows and the periods in columns. The `KeyedArray` type is provided by the `AxisKeys` package.

This function is differentiable (so far for the Kalman filter only) and can be used in gradient based sampling or optimisation.

If occasionally binding constraints are present in the model, they are not taken into account here. 

# Arguments
- $MODEL®
- $DATA®
- `parameter_values` [Type: `Vector`]: Parameter values.
# Keyword Arguments
- $STEADY_STATE_FUNCTION®
- $ALGORITHM®
- $FILTER®
- `presample_periods` [Default: `0`, Type: `Int`]: periods at the beginning of the data for which the loglikelihood is discarded.
- `initial_covariance` [Default: `:theoretical`, Type: `Symbol`]: defines the method to initialise the Kalman filters covariance matrix. It can be initialised with the theoretical long run values (option `:theoretical`) or large values (10.0) along the diagonal (option `:diagonal`).
- `on_failure_loglikelihood` [Default: `-Inf`, Type: `AbstractFloat`]: value to return if the loglikelihood calculation fails. Setting this to a finite value can avoid errors in codes that rely on finite loglikelihood values, such as e.g. slice samplers (in Pigeons.jl).
- $QME®
- $SYLVESTER®
- $LYAPUNOV®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `<:AbstractFloat` loglikelihood 

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

simulated_data = simulate(RBC)

get_loglikelihood(RBC, simulated_data([:k], :, :simulate), RBC.parameter_values)
# output
58.24780188977981
```
"""
function get_loglikelihood(𝓂::ℳ, 
                            data::KeyedArray{Float64}, 
                            parameter_values::Vector{S}; 
                            steady_state_function::SteadyStateFunctionType = missing, 
                            algorithm::Symbol = DEFAULT_ALGORITHM, 
                            filter::Symbol = DEFAULT_FILTER_SELECTOR(algorithm), 
                            on_failure_loglikelihood::U = -Inf,
                            warmup_iterations::Int = DEFAULT_WARMUP_ITERATIONS, 
                            presample_periods::Int = DEFAULT_PRESAMPLE_PERIODS,
                            initial_covariance::Symbol = :theoretical,
                            filter_algorithm::Symbol = :LagrangeNewton,
                            tol::Tolerances = Tolerances(), 
                            quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM, 
                            lyapunov_algorithm::Symbol = DEFAULT_LYAPUNOV_ALGORITHM, 
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = DEFAULT_SYLVESTER_SELECTOR(𝓂),
                            verbose::Bool = DEFAULT_VERBOSE)::S where {S <: Real, U <: AbstractFloat}
                            # timer::TimerOutput = TimerOutput(),

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.constants.post_model_macro.nPast_not_future_and_mixed + 1 + 𝓂.constants.post_model_macro.nExo) > DEFAULT_SYLVESTER_THRESHOLD ? DEFAULT_LARGE_SYLVESTER_ALGORITHM : DEFAULT_SYLVESTER_ALGORITHM : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    estimation = true

    # if algorithm ∈ [:third_order,:pruned_third_order]
    #     sylvester_algorithm = :bicgstab
    # end

    @assert length(parameter_values) == length(𝓂.constants.post_complete_parameters.parameters) "The number of parameter values provided does not match the number of parameters in the model. If this function is used in the context of estimation and not all parameters are estimated, the estimated parameters need to be combined with the other model parameters in one `Vector`. Ensure they have the same order they were declared in the `@parameters` block (check by calling `get_parameters`)."

    # checks to avoid errors further down the line and inform the user
    @assert initial_covariance ∈ [:theoretical, :diagonal] "Invalid method to initialise the Kalman filters covariance matrix. Supported methods are: the theoretical long run values (option `:theoretical`) or large values (10.0) along the diagonal (option `:diagonal`)."

    filter, _, algorithm, _, _, warmup_iterations = normalize_filtering_options(filter, false, algorithm, false, warmup_iterations)

    observables = get_and_check_observables(𝓂.constants.post_model_macro, data)

    solve!(𝓂, 
           opts = opts,
           steady_state_function = steady_state_function,
           # timer = timer, 
           algorithm = algorithm)

    bounds_violated = check_bounds(parameter_values, 𝓂)

    if bounds_violated 
        # println("Bounds violated")
        return on_failure_loglikelihood
    end

    SS_and_pars_names = 𝓂.constants.post_complete_parameters.SS_and_pars_names

    obs_indices = convert(Vector{Int}, indexin(observables, SS_and_pars_names))

    # @timeit_debug timer "Get relevant steady state and solution" begin

    constants_obj, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, opts = opts, estimation = estimation)
                                                                                    # timer = timer,

    # end # timeit_debug

    if !solved 
        # println("Main call: 1st order solution not found")
        return on_failure_loglikelihood 
    end
 
    if collect(axiskeys(data,1)) isa Vector{String}
        data = rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
    end

    dt = collect(data(observables))

    # prepare data
    data_in_deviations = dt .- SS_and_pars[obs_indices]

    # @timeit_debug timer "Filter" begin

    llh = calculate_loglikelihood(Val(filter),
                                Val(algorithm),
                                obs_indices,
                                𝐒,
                                data_in_deviations,
                                constants_obj,
                                state,
                                𝓂.workspaces,
                                warmup_iterations = warmup_iterations,
                                presample_periods = presample_periods,
                                initial_covariance = initial_covariance,
                                filter_algorithm = filter_algorithm,
                                opts = opts,
                                on_failure_loglikelihood = on_failure_loglikelihood) # timer = timer

    # end # timeit_debug

    return llh
end


"""
$(SIGNATURES)
Calculate the residuals of the non-stochastic steady state equations of the model for a given set of values. Values not provided, will be filled with the non-stochastic steady state values corresponding to the current parameters.

# Arguments
- $MODEL®
- `values` [Type: `Union{Vector{Float64}, Dict{Symbol, Float64}, Dict{String, Float64}, KeyedArray{Float64, 1}}`]: A Vector, Dict, or KeyedArray containing the values of the variables and calibrated parameters in the non-stochastic steady state equations (including calibration equations). The `KeyedArray` type is provided by the `AxisKeys` package.

# Keyword Arguments
- $PARAMETERS®
- $STEADY_STATE_FUNCTION®
- $TOLERANCES®
- $VERBOSE®

# Returns
- `KeyedArray` (from the `AxisKeys` package) containing the absolute values of the residuals of the non-stochastic steady state equations.

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    k[ss] / q[ss] = 2.5 | α
    β = 0.95
end

steady_state = SS(RBC, derivatives = false)

get_non_stochastic_steady_state_residuals(RBC, steady_state)
# output
1-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Equation ∈ 5-element Vector{Symbol}
And data, 5-element Vector{Float64}:
 (:Equation₁)             0.0
 (:Equation₂)             0.0
 (:Equation₃)             0.0
 (:Equation₄)             0.0
 (:CalibrationEquation₁)  0.0

get_non_stochastic_steady_state_residuals(RBC, [1.1641597, 3.0635781, 1.2254312, 0.0, 0.18157895])
# output
1-dimensional KeyedArray(NamedDimsArray(...)) with keys:
↓   Equation ∈ 5-element Vector{Symbol}
And data, 5-element Vector{Float64}:
 (:Equation₁)             2.7360991250446887e-10
 (:Equation₂)             6.199999980083248e-8
 (:Equation₃)             2.7897102183871425e-8
 (:Equation₄)             0.0
 (:CalibrationEquation₁)  8.160392850342646e-8
```
"""
function get_non_stochastic_steady_state_residuals(𝓂::ℳ, 
                                                    values::Union{Vector{Float64}, Dict{Symbol, Float64}, Dict{String, Float64}, KeyedArray{Float64, 1}}; 
                                                    parameters::ParameterType = nothing,
                                                    steady_state_function::SteadyStateFunctionType = missing,
                                                    tol::Tolerances = Tolerances(),
                                                    verbose::Bool = DEFAULT_VERBOSE)
    # @nospecialize # reduce compile time                                             

    opts = merge_calculation_options(tol = tol, verbose = verbose)
    
    solve!(𝓂, 
            parameters = parameters,
            steady_state_function = steady_state_function, 
            opts = opts)

    SS_and_pars, _ = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values, opts = opts)

    axis1 = vcat(𝓂.constants.post_model_macro.var, 𝓂.equations.calibration_parameters)

    vars_in_ss_equations = sort(collect(setdiff(reduce(union, get_symbols.(𝓂.equations.steady_state)), union(𝓂.constants.post_model_macro.parameters_in_equations))))

    unknowns = vcat(vars_in_ss_equations, 𝓂.equations.calibration_parameters)

    combined_values = Dict(unknowns .=> SS_and_pars[indexin(unknowns, axis1)])

    if isa(values, Vector)
        @assert length(values) == length(unknowns) "Invalid input. Expected a vector of length $(length(unknowns))."
        for (i, value) in enumerate(values)
            combined_values[unknowns[i]] = value
        end
    elseif isa(values, Dict)
        for (key, value) in values
            if key isa String
                key = replace_indices(key)
            end
            combined_values[key] = value
        end
    elseif isa(values, KeyedArray)
        for (key, value) in Dict(axiskeys(values, 1) .=> collect(values))
            if key isa String
                key = replace_indices(key)
            end
            combined_values[key] = value
        end
    end

    vals = [combined_values[i] for i in unknowns]

    axis1 = vcat([Symbol("Equation" * sub(string(i))) for i in 1:length(vars_in_ss_equations)], [Symbol("CalibrationEquation" * sub(string(i))) for i in 1:length(𝓂.equations.calibration_parameters)])
    
    residual = zeros(length(vals))

    𝓂.functions.NSSS_check(residual, 𝓂.parameter_values, vals)

    KeyedArray(abs.(residual), Equation = axis1)
end

"""
See [`get_non_stochastic_steady_state_residuals`](@ref)
"""
get_residuals = get_non_stochastic_steady_state_residuals

"""
See [`get_non_stochastic_steady_state_residuals`](@ref)
"""
check_residuals = get_non_stochastic_steady_state_residuals