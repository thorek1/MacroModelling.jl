"""
$(SIGNATURES)
Return the shock decomposition in absolute deviations from the non stochastic steady state based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- $DATA
# Keyword Arguments
- $PARAMETERS
- `data_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the data is provided in levels. If `true` the input to the data argument will have the non stochastic steady state substracted.
- `smooth` [Default: `true`, Type: `Bool`]: whether to return smoothed (`true`) or filtered (`false`) shocks.
- $VERBOSE

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
    parameters = nothing,
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    data = data(sort(axiskeys(data,1)))

    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; verbose = verbose)

    axis1 = 𝓂.timings.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = vcat(𝓂.timings.exo, :Initial_values)

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        axis2[1:length(𝓂.timings.exo)] = axis2[1:length(𝓂.timings.exo)] .* "₍ₓ₎"
    else
        axis2 = vcat(map(x->Symbol(string(x) * "₍ₓ₎"), 𝓂.timings.exo), :Initial_values)
    end

    return KeyedArray(filtered_and_smoothed[smooth ? 4 : 8][:,1:end-1,:];  Variables = axis1, Shocks = axis2, Periods = 1:size(data,2))
end




"""
$(SIGNATURES)
Return the estimated shocks based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- $DATA
# Keyword Arguments
- $PARAMETERS
- `data_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the data is provided in levels. If `true` the input to the data argument will have the non stochastic steady state substracted.
- `smooth` [Default: `true`, Type: `Bool`]: whether to return smoothed (`true`) or filtered (`false`) shocks.
- $VERBOSE

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
    parameters = nothing,
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; verbose = verbose)

    axis1 = 𝓂.timings.exo

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1 = axis1 .* "₍ₓ₎"
    else
        axis1 = map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo)
    end

    return KeyedArray(filtered_and_smoothed[smooth ? 3 : 7];  Shocks = axis1, Periods = 1:size(data,2))
end






"""
$(SIGNATURES)
Return the estimated variables based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- $DATA
# Keyword Arguments
- $PARAMETERS
- `data_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the data is provided in levels. If `true` the input to the data argument will have the non stochastic steady state substracted.
- $LEVELS
- `smooth` [Default: `true`, Type: `Bool`]: whether to return smoothed (`true`) or filtered (`false`) shocks.
- $VERBOSE

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
        (1)            (2)           (3)            (4)           …  (37)           (38)          (39)           (40)
  (:c)   -0.000640535    0.00358475    0.000455785    0.00490466        0.0496719      0.055509      0.0477877      0.0436101
  (:k)   -0.00671639     0.0324867     0.00663736     0.0456383         0.500217       0.548478      0.481045       0.437527
  (:q)    0.00334817     0.0426535    -0.0247438      0.0440383        -0.0114766      0.113775     -0.00867574     0.00971302
  (:z)    0.000601617    0.00626684   -0.00393712     0.00632712       -0.00771079     0.0112496    -0.00704709    -0.00366442
```
"""
function get_estimated_variables(𝓂::ℳ,
    data::KeyedArray{Float64};
    parameters = nothing,
    data_in_levels::Bool = true,
    levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    data = data(sort(axiskeys(data,1)))

    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; verbose = verbose)

    axis1 = 𝓂.timings.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    return KeyedArray(levels ? filtered_and_smoothed[smooth ? 1 : 5] .+ reference_steady_state[1:length(𝓂.var)] : filtered_and_smoothed[smooth ? 1 : 5];  Variables = axis1, Periods = 1:size(data,2))
end





"""
$(SIGNATURES)
Return the standard deviations of the Kalman smoother or filter (depending on the `smooth` keyword argument) estimates of the model variables based on the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- $DATA
# Keyword Arguments
- $PARAMETERS
- `data_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the data is provided in levels. If `true` the input to the data argument will have the non stochastic steady state substracted.
- `smooth` [Default: `true`, Type: `Bool`]: whether to return smoothed (`true`) or filtered (`false`) shocks.
- $VERBOSE

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
    parameters = nothing,
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    data = data(sort(axiskeys(data,1)))
    
    obs_axis = collect(axiskeys(data,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    obs_idx = parse_variables_input_to_index(obs_symbols, 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; verbose = verbose)

    axis1 = 𝓂.timings.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    return KeyedArray(filtered_and_smoothed[smooth ? 2 : 6];  Standard_deviations = axis1, Periods = 1:size(data,2))
end





"""
$(SIGNATURES)
Return the conditional forecast given restrictions on endogenous variables and shocks (optional) in a 2-dimensional array. The algorithm finds the combinations of shocks with the smallest magnitude to match the conditions.

Limited to the first order perturbation solution of the model.

# Arguments
- $MODEL
- $CONDITIONS
# Keyword Arguments
- $SHOCK_CONDITIONS
- $INITIAL_STATE
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS
- $VARIABLES
- `conditions_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the conditions are provided in levels. If `true` the input to the conditions argument will have the non stochastic steady state substracted.
- $LEVELS
- $VERBOSE

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

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
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
    initial_state::Vector{Float64} = [0.0],
    periods::Int = 40, 
    parameters = nothing,
    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
    conditions_in_levels::Bool = true,
    levels::Bool = false,
    verbose::Bool = false)

    periods += max(size(conditions,2), shocks isa Nothing ? 1 : size(shocks,2))

    if conditions isa SparseMatrixCSC{Float64}
        @assert length(𝓂.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.var)) * " variables (including auxilliary variables): " * repr(𝓂.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.var),periods)
        nzs = findnz(conditions)
        for i in 1:length(nzs[1])
            cond_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        conditions = cond_tmp
    elseif conditions isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.var) == size(conditions,1) "Number of rows of condition argument and number of model variables must match. Input to conditions has " * repr(size(conditions,1)) * " rows but the model has " * repr(length(𝓂.var)) * " variables (including auxilliary variables): " * repr(𝓂.var)

        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.var),periods)
        cond_tmp[:,axes(conditions,2)] = conditions
        conditions = cond_tmp
    elseif conditions isa KeyedArray{Union{Nothing,Float64}} || conditions isa KeyedArray{Float64}
        conditions_axis = collect(axiskeys(conditions,1))

        conditions_symbols = conditions_axis isa String_input ? conditions_axis .|> Meta.parse .|> replace_indices : conditions_axis

        @assert length(setdiff(conditions_symbols, 𝓂.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(conditions_symbols,𝓂.var))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.var),periods)
        cond_tmp[indexin(sort(conditions_symbols),𝓂.var),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
        conditions = cond_tmp
    end
    
    if shocks isa SparseMatrixCSC{Float64}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        nzs = findnz(shocks)
        for i in 1:length(nzs[1])
            shocks_tmp[nzs[1][i],nzs[2][i]] = nzs[3][i]
        end
        shocks = shocks_tmp
    elseif shocks isa Matrix{Union{Nothing,Float64}}
        @assert length(𝓂.exo) == size(shocks,1) "Number of rows of shocks argument and number of model variables must match. Input to shocks has " * repr(size(shocks,1)) * " rows but the model has " * repr(length(𝓂.exo)) * " shocks: " * repr(𝓂.exo)

        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[:,axes(shocks,2)] = shocks
        shocks = shocks_tmp
    elseif shocks isa KeyedArray{Union{Nothing,Float64}} || shocks isa KeyedArray{Float64}
        shocks_axis = collect(axiskeys(shocks,1))

        shocks_symbols = shocks_axis isa String_input ? shocks_axis .|> Meta.parse .|> replace_indices : shocks_axis

        @assert length(setdiff(shocks_symbols,𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(shocks_symbols, 𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(shocks_symbols), 𝓂.exo), axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true)

    state_update, pruning = parse_algorithm_to_state_update(:first_order, 𝓂)

    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - reference_steady_state[1:length(𝓂.var)]

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    C = @views 𝓂.solution.perturbation.first_order.solution_matrix[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    Y = zeros(size(C,1),periods)

    cond_var_idx = findall(conditions[:,1] .!= nothing)
    
    free_shock_idx = findall(shocks[:,1] .== nothing)

    if conditions_in_levels
        conditions[cond_var_idx,1] .-= reference_steady_state[cond_var_idx]
    end

    @assert length(free_shock_idx) >= length(cond_var_idx) "Exact matching only possible with more free shocks than conditioned variables. Period 1 has " * repr(length(free_shock_idx)) * " free shock(s) and " * repr(length(cond_var_idx)) * " conditioned variable(s)."

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

    axis1 = [𝓂.timings.var[var_idx]; 𝓂.timings.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.timings.exo)+1:end] = axis1[end-length(𝓂.timings.exo)+1:end] .* "₍ₓ₎"
    else
        axis1 = [𝓂.timings.var[var_idx]; map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo)]
    end

    return KeyedArray([levels ? (Y[var_idx,:] .+ reference_steady_state[var_idx]) : Y[var_idx,:]; convert(Matrix{Float64},shocks)];  Variables_and_shocks = axis1, Periods = 1:periods)
end


"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model in a 3-dimensional array.
Function to use when differentiating IRFs with repect to parameters.

# Arguments
- $MODEL
- $PARAMETER_VALUES
# Keyword Arguments
- $PERIODS
- $VARIABLES
- $SHOCKS
- $NEGATIVE_SHOCK
- $INITIAL_STATE
- $LEVELS
- $VERBOSE

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
                    parameters::Vector; 
                    periods::Int = 40, 
                    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
                    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
                    negative_shock::Bool = false, 
                    initial_state::Vector{Float64} = [0.0],
                    levels::Bool = false,
                    verbose::Bool = false)

    solve!(𝓂, verbose = verbose)

    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    @assert shocks != :simulate "Use parameters as a known argument to simulate the model."

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods += size(shocks)[2]

        shock_history = zeros(𝓂.timings.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shocks_axis = collect(axiskeys(shocks,1))

        shocks_symbols = shocks_axis isa String_input ? shocks_axis .|> Meta.parse .|> replace_indices : shocks_axis

        shock_input = map(x->Symbol(replace(string(x), "₍ₓ₎" => "")), shocks_symbols)

        periods += size(shocks)[2]

        @assert length(setdiff(shock_input, 𝓂.timings.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(𝓂.timings.nExo, periods)

        shock_history[indexin(shock_input,𝓂.timings.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)
    end

    reference_steady_state, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
	∇₁ = calculate_jacobian(parameters, reference_steady_state, 𝓂) |> Matrix
								
    sol_mat, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    state_update = function(state::Vector, shock::Vector) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - reference_steady_state[1:length(𝓂.var)]

    # Y = zeros(𝓂.timings.nVars,periods,𝓂.timings.nExo)
    Ŷ = []
    for ii in shock_idx
        Y = []

        if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
            shock_history = zeros(𝓂.timings.nExo,periods)
            shock_history[ii,1] = negative_shock ? -1 : 1
        end

        if shocks == :none
            shock_history = zeros(𝓂.timings.nExo,periods)
        end

        push!(Y, state_update(initial_state,shock_history[:,1]))

        for t in 1:periods-1
            push!(Y, state_update(Y[end],shock_history[:,t+1]))
        end

        push!(Ŷ, reduce(hcat,Y))
    end

    deviations = reshape(reduce(hcat,Ŷ),𝓂.timings.nVars,periods,length(shock_idx))[var_idx,:,:]

    if levels
        return deviations .+ reference_steady_state[var_idx]
    else
        return deviations
    end
end




"""
$(SIGNATURES)
Return impulse response functions (IRFs) of the model in a 3-dimensional KeyedArray. Values are returned in absolute deviations from the (non) stochastic steady state by default.

# Arguments
- $MODEL
# Keyword Arguments
- $PERIODS
- $ALGORITHM
- $PARAMETERS
- $VARIABLES
- $SHOCKS
- $NEGATIVE_SHOCK
- $GENERALISED_IRF
- $INITIAL_STATE
- $LEVELS
- $VERBOSE

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
    periods::Int = 40, 
    algorithm::Symbol = :first_order, 
    parameters = nothing,
    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all_excluding_obc, 
    negative_shock::Bool = false, 
    generalised_irf::Bool = false,
    initial_state::Vector{Float64} = [0.0],
    levels::Bool = false,
    ignore_obc::Bool = false,
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true, algorithm = algorithm)
    
    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    @assert !(shocks == :none && generalised_irf) "Cannot compute generalised IRFs for model without shocks."

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods += size(shocks)[2]

        shock_history = zeros(𝓂.timings.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        periods += size(shocks)[2]

        @assert length(setdiff(shock_input, 𝓂.timings.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(𝓂.timings.nExo, periods + 1)

        shock_history[indexin(shock_input,𝓂.timings.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)
    end


    reference_steady_state, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    if algorithm == :second_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :pruned_second_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.pruned_second_order.stochastic_steady_state
    elseif algorithm == :third_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.third_order.stochastic_steady_state
    elseif algorithm == :pruned_third_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.pruned_third_order.stochastic_steady_state
    else
        SSS_delta = zeros(length(𝓂.var))
    end

    if levels
        if algorithm == :second_order
            reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state
        elseif algorithm == :pruned_second_order
            reference_steady_state = 𝓂.solution.perturbation.pruned_second_order.stochastic_steady_state
        elseif algorithm == :third_order
            reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state
        elseif algorithm == :pruned_third_order
            reference_steady_state = 𝓂.solution.perturbation.pruned_third_order.stochastic_steady_state
        end
    end

    unspecified_initial_state = initial_state == [0.0]

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) - SSS_delta : initial_state - reference_steady_state[1:𝓂.timings.nVars]
    
    if ignore_obc
        occasionally_binding_constraints = false
    else
        occasionally_binding_constraints = length(𝓂.obc_violation_equations) > 0
    end

    stochastic_model = length(𝓂.timings.exo) > 0

    obc_shocks_included = stochastic_model && (length(𝓂.obc_violation_equations) > 0) && (intersect(((shock_idx isa Vector) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])

    if occasionally_binding_constraints || obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraints only compatible with first order perturbation solutions."
        
        solve!(𝓂, parameters = :activeᵒᵇᶜshocks => 1, verbose = false, dynamics = true, algorithm = algorithm)
    end
    
    state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂)

    if generalised_irf
        girfs =  girf(state_update,
                        SSS_delta,
                        levels ? reference_steady_state : SSS_delta,
                        pruning,
                        unspecified_initial_state,
                        𝓂.timings; 
                        algorithm = algorithm,
                        periods = periods, 
                        shocks = shocks, 
                        variables = variables, 
                        negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
        return girfs
    else
        if occasionally_binding_constraints
            function obc_state_update(present_states::Vector{R}, present_shocks::Vector{R}, state_update::Function) where R <: Float64
                unconditional_forecast_horizon = 𝓂.max_obc_horizon

                reference_steady_state = 𝓂.solution.non_stochastic_steady_state

                obc_shock_idx = contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")

                periods_per_shock = 𝓂.max_obc_horizon + 1
                
                num_shocks = sum(obc_shock_idx) ÷ periods_per_shock
                
                p = (present_states, state_update, reference_steady_state, 𝓂, unconditional_forecast_horizon, present_shocks)

                constraints_violated = any(𝓂.obc_violation_function(zeros(num_shocks*periods_per_shock), p) .> eps(Float32))

                if constraints_violated
                    opt = NLopt.Opt(NLopt.:LD_SLSQP, num_shocks*periods_per_shock)

                    opt.min_objective = obc_objective_optim_fun

                    opt.xtol_rel = eps()
                    
                    # Adding constraints
                    upper_bounds = zeros(1 + 2*(num_shocks*periods_per_shock-1))
                    
                    NLopt.inequality_constraint!(opt, (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p), upper_bounds)

                    (minf,x,ret) = NLopt.optimize(opt, zeros(num_shocks*periods_per_shock))
                    
                    solved = ret ∈ Symbol.([
                        NLopt.SUCCESS,
                        NLopt.STOPVAL_REACHED,
                        NLopt.FTOL_REACHED,
                        NLopt.XTOL_REACHED,
                        NLopt.ROUNDOFF_LIMITED,
                    ])
                    
                    present_shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")] .= x

                else
                    solved = true
                end

                present_states = state_update(present_states, present_shocks)

                return present_states, present_shocks, solved
            end

            irfs =  irf(state_update,
                        obc_state_update, 
                        initial_state, 
                        levels ? reference_steady_state : SSS_delta,
                        pruning,
                        unspecified_initial_state,
                        𝓂.timings; 
                        algorithm = algorithm,
                        periods = periods, 
                        shocks = shocks, 
                        variables = variables, 
                        negative_shock = negative_shock)
        else
            irfs =  irf(state_update, 
                        initial_state, 
                        levels ? reference_steady_state : SSS_delta,
                        pruning,
                        unspecified_initial_state,
                        𝓂.timings; 
                        algorithm = algorithm,
                        periods = periods, 
                        shocks = shocks, 
                        variables = variables, 
                        negative_shock = negative_shock)
        end

        if occasionally_binding_constraints || obc_shocks_included #&& algorithm ∈ [:pruned_second_order, :second_order, :pruned_third_order, :third_order]
            solve!(𝓂, parameters = :activeᵒᵇᶜshocks => 0, verbose = false, dynamics = true, algorithm = algorithm)
        end

        return irfs
    end
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
simulate(args...; kwargs...) =  get_irf(args...; levels = true, kwargs..., shocks = :simulate)#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
get_simulation(args...; kwargs...) =  get_irf(args...; levels = true, kwargs..., shocks = :simulate)#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`. Function returns values in levels by default.
"""
get_simulations(args...; kwargs...) =  get_irf(args...; levels = true, kwargs..., shocks = :simulate)#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`.
"""
get_girf(args...; kwargs...) =  get_irf(args...; kwargs..., generalised_irf = true)









"""
$(SIGNATURES)
Return the (non stochastic) steady state and derivatives with respect to model parameters.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $DERIVATIVES
- `stochastic` [Default: `false`, Type: `Bool`]: return stochastic steady state using second order perturbation. No derivatives are calculated.
- $ALGORITHM
- $PARAMETER_DERIVATIVES
- $VERBOSE

The columns show the SS and parameters for which derivatives are taken. The rows show the variables.
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
    parameters = nothing, 
    derivatives::Bool = true, 
    stochastic::Bool = false,
    algorithm::Symbol = :first_order,
    parameter_derivatives::Union{Symbol_input,String_input} = :all,
    verbose::Bool = false,
    silent::Bool = true,
    tol::AbstractFloat = eps())

    solve!(𝓂, parameters = parameters, verbose = verbose)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))
    
    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

    if parameter_derivatives == :all
        length_par = length(𝓂.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.parameters)
        length_par = 1
    elseif length(parameter_derivatives) > 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    SS, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters, false, 𝓂.solver_parameters)
    # SS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    if solution_error > tol
        @warn "Could not find non-stochastic steady state."
    end

    if stochastic
        if  algorithm == :third_order
            solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm, silent = silent)
            SS[1:length(𝓂.var)] = 𝓂.solution.perturbation.third_order.stochastic_steady_state
        elseif  algorithm == :pruned_third_order
            solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm, silent = silent)
            SS[1:length(𝓂.var)] = 𝓂.solution.perturbation.pruned_third_order.stochastic_steady_state
        elseif  algorithm == :pruned_second_order
            solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm, silent = silent)
            SS[1:length(𝓂.var)] = 𝓂.solution.perturbation.pruned_second_order.stochastic_steady_state
        else
            solve!(𝓂, verbose = verbose, dynamics = true, algorithm = :second_order, silent = silent)
            SS[1:length(𝓂.var)] = 𝓂.solution.perturbation.second_order.stochastic_steady_state#[indexin(sort(union(𝓂.var,𝓂.exo_present)),sort(union(𝓂.var,𝓂.aux,𝓂.exo_present)))]
        end
    end

    var_idx = indexin([vars_in_ss_equations...], [𝓂.var...,𝓂.calibration_equations_parameters...])

    calib_idx = indexin([𝓂.calibration_equations_parameters...], [𝓂.var...,𝓂.calibration_equations_parameters...])

    if length_par * length(var_idx) > 200 
        derivatives = false
    end

    if parameter_derivatives != :all
        derivatives = true
    end

    axis1 = [vars_in_ss_equations...,𝓂.calibration_equations_parameters...]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = vcat(:Steady_state, 𝓂.parameters[param_idx])

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if derivatives 
        if stochastic
                if algorithm == :third_order

                    dSSS = ℱ.jacobian(x->begin 
                                SSS = SSS_third_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)

                elseif algorithm == :pruned_third_order

                    dSSS = ℱ.jacobian(x->begin 
                                SSS = SSS_third_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose, pruning = true)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
                
                elseif algorithm == :pruned_second_order

                    dSSS = ℱ.jacobian(x->begin 
                                SSS  = SSS_second_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose, pruning = true)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)

                else

                    dSSS = ℱ.jacobian(x->begin 
                                SSS  = SSS_second_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)

                end
        else
            # dSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
            dSS = ℱ.jacobian(x->collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1])[[var_idx...,calib_idx...]], 𝓂.parameter_values[param_idx])

            # if length(𝓂.calibration_equations_parameters) == 0        
            #     return KeyedArray(hcat(collect(NSSS)[1:(end-1)],dNSSS);  Variables = [sort(union(𝓂.exo_present,var))...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters))
            # else
            # return ComponentMatrix(hcat(collect(NSSS), dNSSS)',Axis(vcat(:SS, 𝓂.parameters)),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
            # return NamedArray(hcat(collect(NSSS), dNSSS), ([sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], vcat(:Steady_state, 𝓂.parameters)), ("Var. and par.", "∂x/∂y"))
            return KeyedArray(hcat(SS[[var_idx...,calib_idx...]],dSS);  Variables_and_calibrated_parameters = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
            # end
        end
    else
        # return ComponentVector(collect(NSSS),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(collect(NSSS), [sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], ("Variables and calibrated parameters"))
        return KeyedArray(SS[[var_idx...,calib_idx...]];  Variables_and_calibrated_parameters = axis1)
    end
    # ComponentVector(non_stochastic_steady_state = ComponentVector(NSSS.non_stochastic_steady_state, Axis(sort(union(𝓂.exo_present,var)))),
    #                 calibrated_parameters = ComponentVector(NSSS.non_stochastic_steady_state, Axis(𝓂.calibration_equations_parameters)),
    #                 stochastic = stochastic)

    # return 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂) : 𝓂.solution.non_stochastic_steady_state
    # return 𝓂.SS_solve_func(𝓂)
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
SS = get_steady_state

"""
See [`get_steady_state`](@ref)
"""
steady_state = get_steady_state

"""
See [`get_steady_state`](@ref)
"""
get_SS = get_steady_state

"""
See [`get_steady_state`](@ref)
"""
get_ss = get_steady_state

"""
See [`get_steady_state`](@ref)
"""
ss(args...; kwargs...) = get_steady_state(args...; kwargs...)




"""
$(SIGNATURES)
Return the solution of the model. In the linear case it returns the linearised solution and the non stochastic steady state (SS) of the model. In the nonlinear case (higher order perturbation) the function returns a multidimensional array with the endogenous variables as the second dimension and the state variables and shocks as the other dimensions.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- `algorithm` [Default: `:first_order`, Type: `Symbol`]: algorithm to solve for the dynamics of the model. Only linear algorithms allowed.
- $VERBOSE

The returned `KeyedArray` shows as columns the endogenous variables inlcuding the auxilliary endogenous and exogenous variables (due to leads and lags > 1). The rows and other dimensions (depending on the chosen perturbation order) include the SS for the linear case only, followed by the states, and exogenous shocks. 
Subscripts following variable names indicate the timing (e.g. `variable₍₋₁₎`  indicates the variable being in the past). Superscripts indicate leads or lags (e.g. `variableᴸ⁽²⁾` indicates the variable being in lead by two periods). If no super- or subscripts follow the variable name, the variable is in the present.
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
    parameters = nothing,
    algorithm::Symbol = :first_order, 
    verbose::Bool = false)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)
    
    # @assert algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran] "This function only works for linear solutions. Choose a respective algorithm."

    solve!(𝓂, parameters = parameters, verbose = verbose, dynamics = true, algorithm = algorithm)

    if algorithm == :linear_time_iteration
        solution_matrix = 𝓂.solution.perturbation.linear_time_iteration.solution_matrix
    elseif algorithm ∈ [:riccati, :first_order]
        solution_matrix = 𝓂.solution.perturbation.first_order.solution_matrix
    elseif algorithm ∈ [:quadratic_iteration, :binder_pesaran]
        solution_matrix = 𝓂.solution.perturbation.quadratic_iteration.solution_matrix
    end

    axis1 = [𝓂.timings.past_not_future_and_mixed; :Volatility; 𝓂.exo]

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        axis1[end-length(𝓂.timings.exo)+1:end] = axis1[end-length(𝓂.timings.exo)+1:end] .* "₍ₓ₎"
        axis1[1:length(𝓂.timings.past_not_future_and_mixed)] = axis1[1:length(𝓂.timings.past_not_future_and_mixed)] .* "₍₋₁₎"
    else
        axis1 = [map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); :Volatility;map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)]
    end

    axis2 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    if algorithm == :second_order
        return KeyedArray(permutedims(reshape(𝓂.solution.perturbation.second_order.solution_matrix, 
                                    𝓂.timings.nVars, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :pruned_second_order
        return KeyedArray(permutedims(reshape(𝓂.solution.perturbation.pruned_second_order.solution_matrix, 
                                    𝓂.timings.nVars, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo),
                                [2,1,3]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1)
    elseif algorithm == :third_order
        return KeyedArray(permutedims(reshape(𝓂.solution.perturbation.third_order.solution_matrix, 
                                    𝓂.timings.nVars, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    elseif algorithm == :pruned_third_order
        return KeyedArray(permutedims(reshape(𝓂.solution.perturbation.pruned_third_order.solution_matrix, 
                                    𝓂.timings.nVars, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo, 
                                    𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo),
                                [2,1,3,4]);
                            States__Shocks¹ = axis1,
                            Variables = axis2,
                            States__Shocks² = axis1,
                            States__Shocks³ = axis1)
    else
        axis1 = [:Steady_state; 𝓂.timings.past_not_future_and_mixed; 𝓂.exo]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            axis1[end-length(𝓂.timings.exo)+1:end] = axis1[end-length(𝓂.timings.exo)+1:end] .* "₍ₓ₎"
            axis1[2:length(𝓂.timings.past_not_future_and_mixed)+1] = axis1[2:length(𝓂.timings.past_not_future_and_mixed)+1] .* "₍₋₁₎"
        else
            axis1 = [:Steady_state; map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)]
        end

        return KeyedArray([𝓂.solution.non_stochastic_steady_state[1:length(𝓂.var)] solution_matrix]';
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




function get_solution(𝓂::ℳ, 
                        parameters::Vector{<: Real}; 
                        algorithm::Symbol = :first_order, 
                        verbose::Bool = false, 
                        tol::AbstractFloat = eps())
    @ignore_derivatives solve!(𝓂, verbose = verbose, algorithm = algorithm)

    ub = @ignore_derivatives fill(1e12+rand(),length(𝓂.parameters))
    lb = @ignore_derivatives -ub

    for (i,v) in enumerate(𝓂.bounded_vars)
        if v ∈ 𝓂.parameters
            @ignore_derivatives lb[i] = 𝓂.lower_bounds[i]
            @ignore_derivatives ub[i] = 𝓂.upper_bounds[i]
        end
    end

    if min(max(parameters,lb),ub) != parameters 
        return -Inf
    end

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(parameters, 𝓂, verbose, false, 𝓂.solver_parameters)
    
    if solution_error > tol || isnan(solution_error)
        if algorithm == :second_order
            return SS_and_pars[1:length(𝓂.var)], zeros(length(𝓂.var),2), spzeros(length(𝓂.var),2), false
        elseif algorithm == :third_order
            return SS_and_pars[1:length(𝓂.var)], zeros(length(𝓂.var),2), spzeros(length(𝓂.var),2), spzeros(length(𝓂.var),2), false
        else
            return SS_and_pars[1:length(𝓂.var)], zeros(length(𝓂.var),2), false
        end
    end

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂) |> Matrix

    𝐒₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    if !solved
        if algorithm == :second_order
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, spzeros(length(𝓂.var),2), false
        elseif algorithm == :third_order
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, spzeros(length(𝓂.var),2), spzeros(length(𝓂.var),2), false
        else
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, false
        end
    end

    if algorithm == :second_order
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, 𝐒₂, true
    elseif algorithm == :third_order
        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
        𝐒₂, solved2 = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁, 
        𝓂.solution.perturbation.second_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)
    
        ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)
                
        𝐒₃, solved3 = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂, 
        𝓂.solution.perturbation.second_order_auxilliary_matrices, 
        𝓂.solution.perturbation.third_order_auxilliary_matrices; T = 𝓂.timings, tol = tol)

        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, 𝐒₂, 𝐒₃, true
    else
        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, true
    end
end



"""
$(SIGNATURES)
Return the conditional variance decomposition of endogenous variables with regards to the shocks using the linearised solution. 

# Arguments
- $MODEL
# Keyword Arguments
- `periods` [Default: `[1:20...,Inf]`, Type: `Union{Vector{Int},Vector{Float64},UnitRange{Int64}}`]: vector of periods for which to calculate the conditional variance decomposition. If the vector conatins `Inf`, also the unconditional variance decomposition is calculated (same output as [`get_variance_decomposition`](@ref)).
- $PARAMETERS
- $VERBOSE

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
    periods::Union{Vector{Int},Vector{Float64},UnitRange{Int64}} = [1:20...,Inf],
    parameters = nothing,  
    verbose::Bool = false)

    solve!(𝓂, parameters = parameters, verbose = verbose)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    SS_and_pars, _ = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

    𝑺₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    A = @views 𝑺₁[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[indexin(𝓂.timings.past_not_future_and_mixed_idx,1:𝓂.timings.nVars),:]
    
    sort!(periods)

    maxperiods = Int(maximum(periods[isfinite.(periods)]))

    var_container = zeros(size(𝑺₁)[1], 𝓂.timings.nExo, length(periods))

    for i in 1:𝓂.timings.nExo
        C = @views 𝑺₁[:,𝓂.timings.nPast_not_future_and_mixed+i]
        CC = C * C'
        varr = zeros(size(C)[1],size(C)[1])
        for k in 1:maxperiods
            varr = A * varr * A' + CC
            if k ∈ periods
                var_container[:,i,indexin(k, periods)] = ℒ.diag(varr)
            end
        end
        if Inf in periods
            sylvester = LinearOperators.LinearOperator(Float64, length(CC), length(CC), false, false, 
            (sol,𝐱) -> begin 
                𝐗 = sparse(reshape(𝐱, size(CC)))
                sol .= vec(A * 𝐗 * A' - 𝐗)
                return sol
            end)
        
            𝐂, info = Krylov.bicgstab(sylvester, sparsevec(collect(-CC)))
        
            if !info.solved
                𝐂, info = Krylov.gmres(sylvester, sparsevec(collect(-CC)))
            end

            var_container[:,i,indexin(Inf,periods)] = ℒ.diag(reshape(𝐂, size(CC))) # numerically more stable
        end
    end

    cond_var_decomp = var_container ./ sum(var_container,dims=2)

    axis1 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = 𝓂.timings.exo

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

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

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $VERBOSE

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
    parameters = nothing,  
    verbose::Bool = false)
    
    solve!(𝓂, parameters = parameters, verbose = verbose)

    SS_and_pars, (solution_error, iters) = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂) |> Matrix

    sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
    variances_by_shock = zeros(𝓂.timings.nVars, 𝓂.timings.nExo)

    for i in 1:𝓂.timings.nExo
        A = @views sol[:, 1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

        C = @views sol[:, 𝓂.timings.nPast_not_future_and_mixed + i]
        
        CC = C * C'

        coordinates = Tuple{Vector{Int}, Vector{Int}}[]
    
        dimensions = Tuple{Int, Int}[]
        push!(dimensions,size(A))
        push!(dimensions,size(CC))
        
        values = vcat(vec(A), vec(collect(-CC)))
    
        covar_raw, _ = solve_matrix_equation_AD(values, coords = coordinates, dims = dimensions, solver = :doubling)

        variances_by_shock[:,i] = ℒ.diag(covar_raw)
    end
    
    var_decomp = variances_by_shock ./ sum(variances_by_shock, dims=2)

    axis1 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = 𝓂.timings.exo

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    KeyedArray(var_decomp; Variables = axis1, Shocks = axis2)
end



"""
See [`get_variance_decomposition`](@ref)
"""
get_var_decomp = get_variance_decomposition




"""
$(SIGNATURES)
Return the correlations of endogenous variables using the first, pruned second, or pruned third order perturbation solution. 

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $ALGORITHM
- $VERBOSE

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
    parameters = nothing,  
    algorithm::Symbol = :first_order,
    verbose::Bool = false)
    
    @assert algorithm ∈ [:first_order,:linear_time_iteration,:quadratic_iteration,:pruned_second_order,:pruned_third_order] "Correlation can only be calculated for first order perturbation or second and third order pruned perturbation solutions."

    solve!(𝓂, parameters = parameters, algorithm = algorithm, verbose = verbose)

    if algorithm == :pruned_third_order
        covar_dcmp, state_μ, SS_and_pars = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, verbose = verbose)
    elseif algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
    else
        covar_dcmp, sol, _, SS_and_pars = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
    end

    std = sqrt.(ℒ.diag(covar_dcmp))

    corr = covar_dcmp ./ (std * std')
    
    axis1 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

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

# Arguments
- $MODEL
# Keyword Arguments
- `autocorrelation_periods` [Default: `1:5`]: periods for which to return the autocorrelation
- $PARAMETERS
- $ALGORITHM
- $VERBOSE

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
→   Autocorrelation_orders ∈ 5-element UnitRange{Int64}
And data, 4×5 Matrix{Float64}:
        (1)         (2)         (3)         (4)         (5)
  (:c)    0.966974    0.927263    0.887643    0.849409    0.812761
  (:k)    0.971015    0.931937    0.892277    0.853876    0.817041
  (:q)    0.32237     0.181562    0.148347    0.136867    0.129944
  (:z)    0.2         0.04        0.008       0.0016      0.00032
```
"""
function get_autocorrelation(𝓂::ℳ; 
    autocorrelation_periods = 1:5,
    parameters = nothing,  
    algorithm::Symbol = :first_order,
    verbose::Bool = false)
    
    @assert algorithm ∈ [:first_order,:linear_time_iteration,:quadratic_iteration,:pruned_second_order,:pruned_third_order] "Autocorrelation can only be calculated for first order perturbation or second and third order pruned perturbation solutions."

    solve!(𝓂, parameters = parameters, algorithm = algorithm, verbose = verbose)

    if algorithm == :pruned_third_order
        covar_dcmp, state_μ, autocorr, SS_and_pars = calculate_third_order_moments(𝓂.parameter_values, 𝓂.timings.var, 𝓂, verbose = verbose, autocorrelation = true)
    elseif algorithm == :pruned_second_order
        covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

        ŝ_to_ŝ₂ⁱ = ℒ.diagm(ones(size(Σᶻ₂,1)))

        autocorr = zeros(size(covar_dcmp,1),length(autocorrelation_periods))

        for i in autocorrelation_periods
            autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ ℒ.diag(covar_dcmp) 
            ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
        end
    else
        covar_dcmp, sol, _, SS_and_pars = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

        A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
    
        autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in autocorrelation_periods])
    end

    
    axis1 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    KeyedArray(collect(autocorr); Variables = axis1, Autocorrelation_orders = autocorrelation_periods)
end

"""
See [`get_autocorrelation`](@ref)
"""
get_autocorr = get_autocorrelation


"""
See [`get_autocorrelation`](@ref)
"""
autocorr = get_autocorrelation




"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using the first, pruned second, or pruned third order perturbation solution. By default returns: non stochastic steady state (SS), and standard deviations, but can optionally return variances, and covariance matrix.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- `non_stochastic_steady_state` [Default: `true`, Type: `Bool`]: switch to return SS of endogenous variables
- `mean` [Default: `false`, Type: `Bool`]: switch to return mean of endogenous variables (the mean for the linearised solutoin is the NSSS)
- `standard_deviation` [Default: `true`, Type: `Bool`]: switch to return standard deviation of endogenous variables
- `variance` [Default: `false`, Type: `Bool`]: switch to return variance of endogenous variables
- `covariance` [Default: `false`, Type: `Bool`]: switch to return covariance matrix of endogenous variables
- $VARIABLES
- $DERIVATIVES
- $PARAMETER_DERIVATIVES
- $ALGORITHM
- `dependencies_tol` [Default: `1e-12`, Type: `AbstractFloat`]: tolerance for the effect of a variable on the variable of interest when isolating part of the system for calculating covariance related statistics
- $VERBOSE

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

moments[1]
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
moments[2]
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
    parameters = nothing,  
    non_stochastic_steady_state::Bool = true, 
    mean::Bool = false,
    standard_deviation::Bool = true, 
    variance::Bool = false, 
    covariance::Bool = false, 
    variables::Union{Symbol_input,String_input} = :all_excluding_obc, 
    derivatives::Bool = true,
    parameter_derivatives::Union{Symbol_input,String_input} = :all,
    algorithm::Symbol = :first_order,
    dependencies_tol::AbstractFloat = 1e-12,
    verbose::Bool = false,
    silent::Bool = true)#limit output by selecting pars and vars like for plots and irfs!?
    
    solve!(𝓂, parameters = parameters, algorithm = algorithm, verbose = verbose, silent = silent)

    # write_parameters_input!(𝓂,parameters, verbose = verbose)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    parameter_derivatives = parameter_derivatives isa String_input ? parameter_derivatives .|> Meta.parse .|> replace_indices : parameter_derivatives

    if parameter_derivatives == :all
        length_par = length(𝓂.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.parameters string(parameter_derivatives) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.parameters)
        length_par = 1
    elseif length(parameter_derivatives) > 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    NSSS, (solution_error, iters) = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose, false, 𝓂.solver_parameters) : (copy(𝓂.solution.non_stochastic_steady_state), (eps(), 0))

    if length_par * length(NSSS) > 200 || (!variance && !standard_deviation && !non_stochastic_steady_state && !mean)
        derivatives = false
    end

    if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state || mean)
        derivatives = true
    end


    axis1 = 𝓂.var

    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = 𝓂.timings.exo

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end





    if derivatives
        if non_stochastic_steady_state
            axis1 = [𝓂.var[var_idx]...,𝓂.calibration_equations_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            axis2 = vcat(:Steady_state, 𝓂.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            dNSSS = ℱ.jacobian(x -> collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1]), 𝓂.parameter_values[param_idx])
            
            if length(𝓂.calibration_equations_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.timings.nVars .+ (1:length(𝓂.calibration_equations_parameters)))
            else
                var_idx_ext = var_idx
            end

            # dNSSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
            SS =  KeyedArray(hcat(collect(NSSS[var_idx_ext]),dNSSS[var_idx_ext,:]);  Variables = axis1, Steady_state_and_∂steady_state∂parameter = axis2)
        end
        
        axis1 = 𝓂.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if variance
            axis2 = vcat(:Variance, 𝓂.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

                dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose), 𝓂.parameter_values[param_idx])

                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, verbose = verbose)

                dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose), 𝓂.parameter_values[param_idx])

                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

                dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose), 𝓂.parameter_values[param_idx])
            end

            vari = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

            # dvariance = ℱ.jacobian(x-> convert(Vector{Number},max.(ℒ.diag(calculate_covariance(x, 𝓂)),eps(Float64))), Float64.(𝓂.parameter_values))
            
            
            varrs =  KeyedArray(hcat(vari[var_idx],dvariance[var_idx,:]);  Variables = axis1, Variance_and_∂variance∂parameter = axis2)

            if standard_deviation
                axis2 = vcat(:Standard_deviation, 𝓂.parameters[param_idx])
            
                if any(x -> contains(string(x), "◖"), axis2)
                    axis2_decomposed = decompose_name.(axis2)
                    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
                end
    
                standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

                if algorithm == :pruned_second_order
                    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
                elseif algorithm == :pruned_third_order
                    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)), 𝓂.parameter_values[param_idx])
                else
                    dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
                end

                st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
            end
        end

        if standard_deviation
            axis2 = vcat(:Standard_deviation, 𝓂.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)

                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_second_order(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])

                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, verbose = verbose)

                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives_third_order(x, variables, param_idx, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)), 𝓂.parameter_values[param_idx])

                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
                
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
            end

            standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

            st_dev =  KeyedArray(hcat(standard_dev[var_idx], dst_dev[var_idx, :]);  Variables = axis1, Standard_deviation_and_∂standard_deviation∂parameter = axis2)
        end


        if mean && !(variance || standard_deviation || covariance)
            axis2 = vcat(:Mean, 𝓂.parameters[param_idx])
        
            if any(x -> contains(string(x), "◖"), axis2)
                axis2_decomposed = decompose_name.(axis2)
                axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
            end

            state_μ, ___ = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, verbose = verbose)

            state_μ_dev = ℱ.jacobian(x -> mean_parameter_derivatives(x, param_idx, 𝓂, algorithm = algorithm, verbose = verbose), 𝓂.parameter_values[param_idx])
            
            var_means =  KeyedArray(hcat(state_μ[var_idx], state_μ_dev[var_idx, :]);  Variables = axis1, Mean_and_∂mean∂parameter = axis2)
        end


    else
        if non_stochastic_steady_state
            axis1 = [𝓂.var[var_idx]...,𝓂.calibration_equations_parameters...]
    
            if any(x -> contains(string(x), "◖"), axis1)
                axis1_decomposed = decompose_name.(axis1)
                axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
            end

            if length(𝓂.calibration_equations_parameters) > 0
                var_idx_ext = vcat(var_idx, 𝓂.timings.nVars .+ (1:length(𝓂.calibration_equations_parameters)))
            else
                var_idx_ext = var_idx
            end

            SS =  KeyedArray(collect(NSSS)[var_idx_ext];  Variables = axis1)
        end

        axis1 = 𝓂.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        if mean && !(variance || standard_deviation || covariance)
            state_μ, ___ = calculate_mean(𝓂.parameter_values, 𝓂, algorithm = algorithm, verbose = verbose)
            var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
        end

        if variance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
            end

            varr = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

            varrs = KeyedArray(varr[var_idx];  Variables = axis1)

            if standard_deviation
                st_dev = KeyedArray(sqrt.(varr)[var_idx];  Variables = axis1)
            end
        end

        if standard_deviation
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, variables, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
            end
            st_dev = KeyedArray(sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))[var_idx];  Variables = axis1)
        end

        if covariance
            if algorithm == :pruned_second_order
                covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(𝓂.parameter_values, 𝓂, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            elseif algorithm == :pruned_third_order
                covar_dcmp, state_μ, _ = calculate_third_order_moments(𝓂.parameter_values, :full_covar, 𝓂, dependencies_tol = dependencies_tol, verbose = verbose)
                if mean
                    var_means = KeyedArray(state_μ[var_idx];  Variables = axis1)
                end
            else
                covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)
            end
        end
    end

    
    ret = []
    if non_stochastic_steady_state
        push!(ret,SS)
    end
    if mean
        push!(ret,var_means)
    end
    if standard_deviation
        push!(ret,st_dev)
    end
    if variance
        push!(ret,varrs)
    end
    if covariance
        axis1 = 𝓂.var[var_idx]

        if any(x -> contains(string(x), "◖"), axis1)
            axis1_decomposed = decompose_name.(axis1)
            axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
        end

        push!(ret,KeyedArray(covar_dcmp[var_idx, var_idx]; Variables = axis1, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = axis1))
    end

    return ret
end

"""
Wrapper for [`get_moments`](@ref) with `variance = true` and `non_stochastic_steady_state = false, standard_deviation = false, covariance = false`.
"""
get_variance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = true, non_stochastic_steady_state = false, standard_deviation = false, covariance = false)[1]


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
get_standard_deviation(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = true, covariance = false)[1]


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
Wrapper for [`get_moments`](@ref) with `covariance = true` and `non_stochastic_steady_state = false, variance = false, standard_deviation = false`.
"""
get_covariance(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = true)[1]


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
get_mean(args...; kwargs...) =  get_moments(args...; kwargs..., variance = false, non_stochastic_steady_state = false, standard_deviation = false, covariance = false, mean = true)[1]


# """
# Wrapper for [`get_moments`](@ref) with `mean = true`, the default algorithm being `:pruned_second_order`, and `non_stochastic_steady_state = false, variance = false, standard_deviation = false, covariance = false`
# """
# mean(𝓂::ℳ; kwargs...) = get_mean(𝓂; kwargs...)



"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using either the linearised solution or the pruned second or third order perturbation solution. By default returns: non stochastic steady state (SS), and standard deviations, but can also return variances, and covariance matrix.
Function to use when differentiating model moments with repect to parameters.

# Arguments
- $MODEL
- `parameter_values` [Type: `Vector`]: Parameter values.
# Keyword Arguments
- `parameters` [Type: `Vector{Symbol}`]: Corresponding names of parameters values.
- `non_stochastic_steady_state` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the SS of endogenous variables
- `mean` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the mean of endogenous variables (the mean for the linearised solutoin is the NSSS)
- `standard_deviation` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the standard deviation of the mentioned variables
- `variance` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the variance of the mentioned variables
- `covariance` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the covariance of the mentioned variables
- `autocorrelation` [Default: `Symbol[]`, Type: `Vector{Symbol}`]: if values are provided the function returns the autocorrelation of the mentioned variables
- `autocorrelation_periods` [Default: `1:5`]: periods for which to return the autocorrelation of the mentioned variables
- $ALGORITHM
- $VERBOSE

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

get_statistics(RBC, RBC.parameter_values, parameters = RBC.parameters, standard_deviation = RBC.var)
# output
1-element Vector{Any}:
 [0.02666420378525503, 0.26467737291221793, 0.07393254045396483, 0.010206207261596574]
```
"""
function get_statistics(𝓂, 
    parameter_values::Vector{T}; 
    parameters::Vector{Symbol} = Symbol[], 
    non_stochastic_steady_state::Vector{Symbol} = Symbol[],
    mean::Vector{Symbol} = Symbol[],
    standard_deviation::Vector{Symbol} = Symbol[],
    variance::Vector{Symbol} = Symbol[],
    covariance::Vector{Symbol} = Symbol[],
    autocorrelation::Vector{Symbol} = Symbol[],
    autocorrelation_periods::U = 1:5,
    algorithm::Symbol = :first_order,
    verbose::Bool = false) where {U,T}


    @assert algorithm ∈ [:first_order,:linear_time_iteration,:quadratic_iteration,:pruned_second_order,:pruned_third_order] "Statistics can only be provided for first order perturbation or second and third order pruned perturbation solutions."

    @assert !(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]) "Provide variables for at least one output."

    SS_var_idx = indexin(non_stochastic_steady_state, 𝓂.var)

    mean_var_idx = indexin(mean, 𝓂.var)

    std_var_idx = indexin(standard_deviation, 𝓂.var)

    var_var_idx = indexin(variance, 𝓂.var)

    covar_var_idx = indexin(covariance, 𝓂.var)

    autocorr_var_idx = indexin(autocorrelation, 𝓂.var)

    other_parameter_values = 𝓂.parameter_values[indexin(setdiff(𝓂.parameters, parameters), 𝓂.parameters)]

    sort_idx = sortperm(vcat(indexin(setdiff(𝓂.parameters, parameters), 𝓂.parameters), indexin(parameters, 𝓂.parameters)))

    all_parameters = vcat(other_parameter_values, parameter_values)[sort_idx]

    if algorithm == :pruned_third_order && !(!(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]))
        algorithm = :pruned_second_order
    end

    solve!(𝓂, algorithm = algorithm, verbose = verbose)

    if algorithm == :pruned_third_order

        if !(autocorrelation == Symbol[])
            second_mom_third_order = union(autocorrelation, standard_deviation, variance, covariance)

            covar_dcmp, state_μ, autocorr, SS_and_pars = calculate_third_order_moments(all_parameters, second_mom_third_order, 𝓂, verbose = verbose, autocorrelation = true, autocorrelation_periods = autocorrelation_periods)

        elseif !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[])

            covar_dcmp, state_μ, SS_and_pars = calculate_third_order_moments(all_parameters, union(variance,covariance,standard_deviation), 𝓂, verbose = verbose)

        end

    elseif algorithm == :pruned_second_order

        if !(standard_deviation == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[])
            covar_dcmp, Σᶻ₂, state_μ, Δμˢ₂, autocorr_tmp, ŝ_to_ŝ₂, ŝ_to_y₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(all_parameters, 𝓂, verbose = verbose)
        else
            state_μ, Δμˢ₂, Σʸ₁, Σᶻ₁, SS_and_pars, 𝐒₁, ∇₁, 𝐒₂, ∇₂ = calculate_second_order_moments(all_parameters, 𝓂, verbose = verbose, covariance = false)
        end

    else
        covar_dcmp, sol, _, SS_and_pars = calculate_covariance(all_parameters, 𝓂, verbose = verbose)
    end

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]

    if !(variance == Symbol[])
        varrs = convert(Vector{T},ℒ.diag(covar_dcmp))
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(varrs)
        end
    elseif !(autocorrelation == Symbol[])
        if algorithm == :pruned_second_order
            ŝ_to_ŝ₂ⁱ = zero(ŝ_to_ŝ₂)
            ŝ_to_ŝ₂ⁱ += ℒ.diagm(ones(size(ŝ_to_ŝ₂,1)))

            autocorr = zeros(T,size(covar_dcmp,1),length(autocorrelation_periods))

            for i in autocorrelation_periods
                autocorr[:,i] .= ℒ.diag(ŝ_to_y₂ * ŝ_to_ŝ₂ⁱ * autocorr_tmp) ./ ℒ.diag(covar_dcmp) 
                ŝ_to_ŝ₂ⁱ *= ŝ_to_ŝ₂
            end
        elseif !(algorithm == :pruned_third_order)
            A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]
        
            autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in autocorrelation_periods])
        end

        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(abs.(convert(Vector{T},ℒ.diag(covar_dcmp))))
        end
    else
        if !(standard_deviation == Symbol[])
            st_dev = sqrt.(abs.(convert(Vector{T},ℒ.diag(covar_dcmp))))
        end
    end

    ret = AbstractArray{T}[]
    if !(non_stochastic_steady_state == Symbol[])
        push!(ret,SS[SS_var_idx])
    end
    if !(mean == Symbol[])
        if algorithm ∉ [:pruned_second_order,:pruned_third_order]
            push!(ret,SS[mean_var_idx])
        else
            push!(ret,state_μ[mean_var_idx])
        end
    end
    if !(standard_deviation == Symbol[])
        push!(ret,st_dev[std_var_idx])
    end
    if !(variance == Symbol[])
        push!(ret,varrs[var_var_idx])
    end
    if !(covariance == Symbol[])
        covar_dcmp_sp = sparse(ℒ.triu(covar_dcmp))

        droptol!(covar_dcmp_sp,eps(Float64))

        push!(ret,covar_dcmp_sp[covar_var_idx,covar_var_idx])
    end
    if !(autocorrelation == Symbol[]) 
        push!(ret,autocorr[autocorr_var_idx,:] )
    end

    return ret
end
