"""
$(SIGNATURES)
Return the shock decomposition in absolute deviations from the non stochastic steady state based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- `data` [Type: `KeyedArray`]: data matrix with variables in rows and time in columns
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

simulation = simulate(RBC);

get_shock_decomposition(RBC,simulation([:c],:,:simulate), data_in_levels = false)
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
    data::AbstractArray{Float64};
    parameters = nothing,
    # variables::Symbol_input = :all_including_auxilliary, 
    # shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    write_parameters_input!(𝓂, parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

    # var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    return KeyedArray(filtered_and_smoothed[smooth ? 4 : 8][:,1:end-1,:];  Variables = 𝓂.timings.var, Shocks = vcat(map(x->Symbol(string(x) * "₍ₓ₎"), 𝓂.timings.exo), :Initial_values), Periods = 1:size(data,2))
end




"""
$(SIGNATURES)
Return the estimated shocks based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- `data` [Type: `KeyedArray`]: data matrix with variables in rows and time in columns
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

simulation = simulate(RBC);

get_estimated_shocks(RBC,simulation([:c],:,:simulate), data_in_levels = false)
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
    data::AbstractArray{Float64};
    parameters = nothing,
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    write_parameters_input!(𝓂, parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

    return KeyedArray(filtered_and_smoothed[smooth ? 3 : 7];  Shocks = map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo), Periods = 1:size(data,2))
end






"""
$(SIGNATURES)
Return the estimated variables based on the Kalman smoother or filter (depending on the `smooth` keyword argument) using the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- `data` [Type: `KeyedArray`]: data matrix with variables in rows and time in columns
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

simulation = simulate(RBC);

get_estimated_variables(RBC,simulation([:c],:,:simulate), data_in_levels = false)
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
    data::AbstractArray{Float64};
    parameters = nothing,
    # variables::Symbol_input = :all_including_auxilliary, 
    data_in_levels::Bool = true,
    levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    write_parameters_input!(𝓂, parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

    return KeyedArray(levels ? filtered_and_smoothed[smooth ? 1 : 5] .+ reference_steady_state[1:length(𝓂.var)] : filtered_and_smoothed[smooth ? 1 : 5];  Variables = 𝓂.timings.var, Periods = 1:size(data,2))
end





"""
$(SIGNATURES)
Return the standard deviations of the Kalman smoother or filter (depending on the `smooth` keyword argument) estimates of the model variables based on the provided data and first order solution of the model. Data is by default assumed to be in levels unless `data_in_levels` is set to `false`.

# Arguments
- $MODEL
- `data` [Type: `KeyedArray`]: data matrix with variables in rows and time in columns
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

simulation = simulate(RBC);

get_estimated_variable_standard_deviations(RBC,simulation([:c],:,:simulate), data_in_levels = false)
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
    data::AbstractArray{Float64};
    parameters = nothing,
    # variables::Symbol_input = :all_including_auxilliary, 
    data_in_levels::Bool = true,
    smooth::Bool = true,
    verbose::Bool = false)

    write_parameters_input!(𝓂, parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    obs_idx = parse_variables_input_to_index(collect(axiskeys(data)[1]), 𝓂.timings)

    if data_in_levels
        data_in_deviations = data .- reference_steady_state[obs_idx]
    else
        data_in_deviations = data
    end

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, sort(axiskeys(data)[1]); verbose = verbose)

    return KeyedArray(filtered_and_smoothed[smooth ? 2 : 6];  Standard_deviations = 𝓂.timings.var, Periods = 1:size(data,2))
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
- `periods` [Default: `40`, Type: `Int`]: the total number of periods is the sum of the argument provided here and the maximum of periods of the shocks or conditions argument.
- $PARAMETERS
- $VARIABLES
- `conditions_in_levels` [Default: `false`, Type: `Bool`]: indicator whether the conditions are provided in levels. If `true` the input to the conditions argument will have the non stochastic steady state substracted.
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

get_conditional_forecast(RBC_CME, conditions, shocks = shocks)
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
    periods::Int = 40, 
    parameters = nothing,
    variables::Symbol_input = :all_including_auxilliary, 
    conditions_in_levels::Bool = false,
    levels::Bool = false,
    verbose::Bool = false)

    periods += max(size(conditions,2), isnothing(shocks) ? 1 : size(shocks,2))

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
        @assert length(setdiff(axiskeys(conditions,1),𝓂.var)) == 0 "The following symbols in the first axis of the conditions matrix are not part of the model: " * repr(setdiff(axiskeys(conditions,1),𝓂.var))
        
        cond_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.var),periods)
        cond_tmp[indexin(sort(axiskeys(conditions,1)),𝓂.var),axes(conditions,2)] .= conditions(sort(axiskeys(conditions,1)))
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
        @assert length(setdiff(axiskeys(shocks,1),𝓂.exo)) == 0 "The following symbols in the first axis of the shocks matrix are not part of the model: " * repr(setdiff(axiskeys(shocks,1),𝓂.exo))
        
        shocks_tmp = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
        shocks_tmp[indexin(sort(axiskeys(shocks,1)),𝓂.exo),axes(shocks,2)] .= shocks(sort(axiskeys(shocks,1)))
        shocks = shocks_tmp
    elseif isnothing(shocks)
        shocks = Matrix{Union{Nothing,Float64}}(undef,length(𝓂.exo),periods)
    end

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    state_update = parse_algorithm_to_state_update(:first_order, 𝓂)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())


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

    shocks[free_shock_idx,1] = CC \ (conditions[cond_var_idx,1] - state_update(zeros(size(C,1)), Float64[shocks[:,1]...])[cond_var_idx])

    Y[:,1] = state_update(zeros(size(C,1)), Float64[shocks[:,1]...])

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

    return KeyedArray([levels ? (Y[var_idx,:] .+ reference_steady_state[var_idx]) : Y[var_idx,:]; convert(Matrix{Float64},shocks)];  Variables_and_shocks = [𝓂.timings.var[var_idx]; map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.timings.exo)], Periods = 1:periods)
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
                    variables::Symbol_input = :all_including_auxilliary, 
                    shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
                    negative_shock::Bool = false, 
                    initial_state::Vector{Float64} = [0.0],
                    levels::Bool = false,
                    verbose::Bool = false)

    solve!(𝓂, verbose = verbose)

    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    @assert shocks != :simulate "Use parameters as a known argument to simulate the model."

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        periods += size(shocks)[2]

        shock_history = zeros(𝓂.timings.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = axiskeys(shocks)[1]

        periods += size(shocks)[2]

        @assert length(setdiff(shock_input, 𝓂.timings.exo)) == 0 "Provided shocks which are not part of the model."

        shock_history = zeros(𝓂.timings.nExo, periods)

        shock_history[indexin(shock_input,𝓂.timings.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)
    end

    reference_steady_state, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
	∇₁ = calculate_jacobian(parameters, reference_steady_state, 𝓂)
								
    sol_mat, success = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    state_update = function(state::Vector, shock::Vector) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - reference_steady_state[1:length(𝓂.var)]

    # Y = zeros(𝓂.timings.nVars,periods,𝓂.timings.nExo)
    Ŷ = []
    for ii in shock_idx
        Y = []

        if shocks != :simulate && shocks isa Symbol_input
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
Return impulse response functions (IRFs) of the model in a 3-dimensional KeyedArray

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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
    variables::Symbol_input = :all_including_auxilliary, 
    shocks::Union{Symbol_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    negative_shock::Bool = false, 
    generalised_irf::Bool = false,
    initial_state::Vector{Float64} = [0.0],
    levels::Bool = false,
    verbose::Bool = false)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm)
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    @assert !(shocks == :none && generalised_irf) "Cannot compute generalised IRFs for model without shocks."

    state_update = parse_algorithm_to_state_update(algorithm, 𝓂)

    reference_steady_state, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    if algorithm == :second_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.second_order.stochastic_steady_state
    elseif algorithm == :third_order
        SSS_delta = reference_steady_state[1:length(𝓂.var)] - 𝓂.solution.perturbation.third_order.stochastic_steady_state
    else
        SSS_delta = zeros(length(𝓂.var))
    end

    if levels
        if algorithm == :second_order
            reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state#[indexin(full_SS,sort(union(𝓂.var,𝓂.exo_present)))]
        elseif algorithm == :third_order
            reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state#[indexin(full_SS,sort(union(𝓂.var,𝓂.exo_present)))]
        end
    end

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) - SSS_delta : initial_state - reference_steady_state[1:length(𝓂.var)]

    if generalised_irf
        girfs =  girf(state_update,
                        SSS_delta,
                        levels ? reference_steady_state : SSS_delta,
                        𝓂.timings; 
                        periods = periods, 
                        shocks = shocks, 
                        variables = variables, 
                        negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
        return girfs
    else
        irfs =  irf(state_update, 
                    initial_state, 
                    levels ? reference_steady_state : SSS_delta,
                    𝓂.timings; 
                    periods = periods, 
                    shocks = shocks, 
                    variables = variables, 
                    negative_shock = negative_shock)
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

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`.
"""
simulate(args...; kwargs...) =  get_irf(args...; kwargs..., shocks = :simulate)#[:,:,1]

"""
Wrapper for [`get_irf`](@ref) with `shocks = :simulate`.
"""
get_simulation(args...; kwargs...) =  get_irf(args...; kwargs..., shocks = :simulate)#[:,:,1]

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
- `parameter_derivatives` [Default: :all]: parameters for which to calculate partial derivatives. Inputs can be either a `Symbol` (e.g. `:alpha`, or `:all`), `Tuple{Symbol, Vararg{Symbol}}`, `Matrix{Symbol}` or `Vector{Symbol}`.
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
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
    parameter_derivatives::Symbol_input = :all,
    verbose::Bool = false,
    silent::Bool = true)

    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    vars_in_ss_equations = sort(collect(setdiff(reduce(union,get_symbols.(𝓂.ss_aux_equations)),union(𝓂.parameters_in_equations,𝓂.➕_vars))))
    
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

    SS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    if stochastic
        if  algorithm == :third_order
            solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm, silent = silent)
            SS[1:length(𝓂.var)] = 𝓂.solution.perturbation.third_order.stochastic_steady_state
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

    if derivatives 
        if stochastic
                if algorithm == :third_order
                    dSSS = ℱ.jacobian(x->begin 
                                SSS = SSS_third_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = [vars_in_ss_equations...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))
                else
                    dSSS = ℱ.jacobian(x->begin 
                                SSS  = SSS_second_order_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)
                                [collect(SSS[1])[var_idx]...,collect(SSS[3])[calib_idx]...]
                            end, 𝓂.parameter_values[param_idx])

                    return KeyedArray(hcat(SS[[var_idx...,calib_idx...]], dSSS);  Variables_and_calibrated_parameters = [vars_in_ss_equations...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))
                end
        else
            # dSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
            dSS = ℱ.jacobian(x->collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1])[[var_idx...,calib_idx...]], 𝓂.parameter_values[param_idx])

            # if length(𝓂.calibration_equations_parameters) == 0        
            #     return KeyedArray(hcat(collect(NSSS)[1:(end-1)],dNSSS);  Variables = [sort(union(𝓂.exo_present,var))...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters))
            # else
            # return ComponentMatrix(hcat(collect(NSSS), dNSSS)',Axis(vcat(:SS, 𝓂.parameters)),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
            # return NamedArray(hcat(collect(NSSS), dNSSS), ([sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], vcat(:Steady_state, 𝓂.parameters)), ("Var. and par.", "∂x/∂y"))
            return KeyedArray(hcat(SS[[var_idx...,calib_idx...]],dSS);  Variables_and_calibrated_parameters = [vars_in_ss_equations...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))
            # end
        end
    else
        # return ComponentVector(collect(NSSS),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(collect(NSSS), [sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], ("Variables and calibrated parameters"))
        return KeyedArray(SS[[var_idx...,calib_idx...]];  Variables_and_calibrated_parameters = [vars_in_ss_equations...,𝓂.calibration_equations_parameters...])
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
$(SIGNATURES)
Return the linearised solution and the non stochastic steady state (SS) of the model.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- `algorithm` [Default: `:first_order`, Type: `Symbol`]: algorithm to solve for the dynamics of the model. Only linear algorithms allowed.
- $VERBOSE

The returned `KeyedArray` shows the SS, policy and transition functions of the model. The columns show the varibales including auxilliary endogenous and exogenous variables (due to leads and lags > 1). The rows are the SS, followed by the states, and exogenous shocks. 
Subscripts following variable names indicate the timing (e.g. `variable₍₋₁₎`  indicates the variable being in the past). Superscripts indicate leads or lags (e.g. `variableᴸ⁽²⁾` indicates the variable being in lead by two periods). If no super- or subscripts follow the variable name, the variable is in the present.
# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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

    write_parameters_input!(𝓂,parameters, verbose = verbose)
    
    @assert algorithm ∈ [:linear_time_iteration, :riccati, :first_order, :quadratic_iteration, :binder_pesaran] "This function only works for linear solutions. Choose a respective algorithm."

    solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm)

    if algorithm == :linear_time_iteration
        solution_matrix = 𝓂.solution.perturbation.linear_time_iteration.solution_matrix
    elseif algorithm ∈ [:riccati, :first_order]
        solution_matrix = 𝓂.solution.perturbation.first_order.solution_matrix
    elseif algorithm ∈ [:quadratic_iteration, :binder_pesaran]
        solution_matrix = 𝓂.solution.perturbation.quadratic_iteration.solution_matrix
    end

    KeyedArray([𝓂.solution.non_stochastic_steady_state[1:length(𝓂.var)] solution_matrix]';
    Steady_state__States__Shocks = [:Steady_state; map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)],
    Variables = 𝓂.var)
end


"""
See [`get_solution`](@ref)
"""
get_first_order_solution = get_solution

"""
See [`get_solution`](@ref)
"""
get_perturbation_solution = get_solution




function get_solution(𝓂::ℳ, parameters::Vector{<: Real}; algorithm::Symbol = :first_order, verbose::Bool = false, tol::AbstractFloat = eps())
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

    SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

    𝐒₁, success = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    if algorithm == :second_order
        if !success
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, spzeros(𝓂.timings.nVars,2), success
        end

        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
        𝐒₂ = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁; T = 𝓂.timings)

        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, 𝐒₂, success
    elseif algorithm == :third_order
        if !success
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, spzeros(𝓂.timings.nVars,2), spzeros(𝓂.timings.nVars,2), success
        end

        ∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)
    
        𝐒₂ = calculate_second_order_solution(∇₁, ∇₂, 𝐒₁; T = 𝓂.timings)
    
        ∇₃ = calculate_third_order_derivatives(parameters, SS_and_pars, 𝓂)
                
        𝐒₃ = calculate_third_order_solution(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂; T = 𝓂.timings)

        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, 𝐒₂, 𝐒₃, success
    else
        if !success
            return SS_and_pars[1:length(𝓂.var)], 𝐒₁, success
        end

        return SS_and_pars[1:length(𝓂.var)], 𝐒₁, success
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
  (:A)         1.29651e-31   1.0
  (:Pi)        0.0245641     0.975436
  (:R)         0.0245641     0.975436
  (:c)         0.0175249     0.982475
  (:k)         0.00869568    0.991304
  (:y)         7.63511e-5    0.999924
  (:z_delta)   1.0           0.0

[:, :, 21] ~ (:, :, Inf):
              (:delta_eps)  (:eps_z)
  (:A)         2.47454e-30   1.0
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

    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    SS_and_pars, _ = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

    𝑺₁, success = calculate_first_order_solution(∇₁; T = 𝓂.timings)
    
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
            lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

            # var_container[:,i,indexin(Inf,periods)] = ℒ.diag(reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))) # faster
            var_container[:,i,indexin(Inf,periods)] = ℒ.diag(reshape(ℐ.gmres(lm, vec(-CC)), size(CC))) # numerically more stable
        end
    end

    cond_var_decomp = var_container ./ sum(var_container,dims=2)

    KeyedArray(cond_var_decomp; Variables = 𝓂.var, Shocks = 𝓂.timings.exo, Periods = periods)
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
  (:A)         1.69478e-29   1.0
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
    
    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    SS_and_pars, solution_error = 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose)
    
	∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

    sol, success = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    variances_by_shock = reduce(hcat,[ℒ.diag(calculate_covariance_forward(sol[:,[1:𝓂.timings.nPast_not_future_and_mixed..., 𝓂.timings.nPast_not_future_and_mixed+i]], T = 𝓂.timings, subset_indices = collect(1:𝓂.timings.nVars))) for i in 1:𝓂.timings.nExo])

    var_decomp = variances_by_shock ./ sum(variances_by_shock,dims=2)

    KeyedArray(var_decomp; Variables = 𝓂.var, Shocks = 𝓂.timings.exo)
end



"""
See [`get_variance_decomposition`](@ref)
"""
get_var_decomp = get_variance_decomposition




"""
$(SIGNATURES)
Return the correlations of endogenous variables using the linearised solution. 

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $VERBOSE

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
    verbose::Bool = false)
    
    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    covar_dcmp, ___, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

    std = sqrt.(ℒ.diag(covar_dcmp))

    corr = covar_dcmp ./ (std * std')
    
    KeyedArray(collect(corr); Variables = 𝓂.var, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = 𝓂.var)
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
Return the autocorrelations of endogenous variables using the linearised solution. 

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $VERBOSE

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
    parameters = nothing,  
    verbose::Bool = false)
    
    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    covar_dcmp, sol, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp)) for i in 1:5])
    
    KeyedArray(collect(autocorr); Variables = 𝓂.var, Autocorrelation_orders = 1:5)
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
Return the first and second moments of endogenous variables using the linearised solution. By default returns: non stochastic steady state (SS), and standard deviations, but can also return variances, and covariance matrix.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- `non_stochastic_steady_state` [Default: `true`, Type: `Bool`]: switch to return SS of endogenous variables
- `standard_deviation` [Default: `true`, Type: `Bool`]: switch to return standard deviation of endogenous variables
- `variance` [Default: `false`, Type: `Bool`]: switch to return variance of endogenous variables
- `covariance` [Default: `false`, Type: `Bool`]: switch to return covariance matrix of endogenous variables
- $DERIVATIVES
- `parameter_derivatives` [Default: :all]: parameters for which to calculate partial derivatives. Inputs can be either a `Symbol` (e.g. `:alpha`, or `:all`), `Tuple{Symbol, Vararg{Symbol}}`, `Matrix{Symbol}` or `Vector{Symbol}`.
- $VERBOSE

# Examples
```jldoctest part1
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

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
    standard_deviation::Bool = true, 
    variance::Bool = false, 
    covariance::Bool = false, 
    derivatives::Bool = true,
    parameter_derivatives::Symbol_input = :all,
    verbose::Bool = false)#limit output by selecting pars and vars like for plots and irfs!?
    
    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

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

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, verbose) : (copy(𝓂.solution.non_stochastic_steady_state), eps())

    if length_par * length(NSSS) > 200 || (!variance && !standard_deviation && !non_stochastic_steady_state)
        derivatives = false
    end

    if parameter_derivatives != :all && (variance || standard_deviation || non_stochastic_steady_state)
        derivatives = true
    end

    if derivatives
        if non_stochastic_steady_state
            dNSSS = ℱ.jacobian(x -> collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1]), 𝓂.parameter_values[param_idx])
            
            # dNSSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
            SS =  KeyedArray(hcat(collect(NSSS),dNSSS);  Variables = [𝓂.var...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))
        end
        
        if variance
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

            vari = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))

            # dvariance = ℱ.jacobian(x-> convert(Vector{Number},max.(ℒ.diag(calculate_covariance(x, 𝓂)),eps(Float64))), Float64.(𝓂.parameter_values))
            dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose), 𝓂.parameter_values[param_idx])
            
            
            varrs =  KeyedArray(hcat(vari,dvariance);  Variables = 𝓂.var, Variance_and_∂variance∂parameter = vcat(:Variance, 𝓂.parameters[param_idx]))

            if standard_deviation
                standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
                

                st_dev =  KeyedArray(hcat(standard_dev,dst_dev);  Variables = 𝓂.var, Standard_deviation_and_∂standard_deviation∂parameter = vcat(:Standard_deviation, 𝓂.parameters[param_idx]))
            end
        end

        if standard_deviation
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

            standard_dev = sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))))

            dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)), 𝓂.parameter_values[param_idx])
            

            st_dev =  KeyedArray(hcat(standard_dev,dst_dev);  Variables = 𝓂.var, Standard_deviation_and_∂standard_deviation∂parameter = vcat(:Standard_deviation, 𝓂.parameters[param_idx]))
        end
    else
        if non_stochastic_steady_state
            SS =  KeyedArray(collect(NSSS);  Variables = [𝓂.var...,𝓂.calibration_equations_parameters...])
        end

        if variance
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]
            varr = convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64)))
            varrs = KeyedArray(varr;  Variables = 𝓂.var)
            if standard_deviation
                st_dev = KeyedArray(sqrt.(varr);  Variables = 𝓂.var)
            end
        end

        if standard_deviation
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]
            st_dev = KeyedArray(sqrt.(convert(Vector{Real},max.(ℒ.diag(covar_dcmp),eps(Float64))));  Variables = 𝓂.var)
        end

        if covariance
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]
        end
    end

    
    ret = []
    if non_stochastic_steady_state
        push!(ret,SS)
    end
    if standard_deviation
        push!(ret,st_dev)
    end
    if variance
        push!(ret,varrs)
    end
    if covariance
        push!(ret,KeyedArray(covar_dcmp; Variables = 𝓂.var, 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠 = 𝓂.var))
    end

    return ret
end




"""
$(SIGNATURES)
Return the first and second moments of endogenous variables using the linearised solution. By default returns: non stochastic steady state (SS), and standard deviations, but can also return variances, and covariance matrix.
Function to use when differentiating model moments with repect to parameters.

# Arguments
- $MODEL
- $PARAMETER_VALUES
# Keyword Arguments
- `non_stochastic_steady_state` [Default: `true`, Type: `Bool`]: switch to return SS of endogenous variables
- `standard_deviation` [Default: `true`, Type: `Bool`]: switch to return standard deviation of endogenous variables
- `variance` [Default: `false`, Type: `Bool`]: switch to return variance of endogenous variables
- `covariance` [Default: `false`, Type: `Bool`]: switch to return covariance matrix of endogenous variables
- $VERBOSE

# Examples
```jldoctest
using MacroModelling

@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

get_moments(RBC, RBC.parameter_values)
# output
2-element Vector{Any}:
 [5.936252888048724, 47.39025414828808, 6.884057971014486, 0.0]
 [0.026664203785255254, 0.26467737291222343, 0.07393254045396497, 0.010206207261596576]
```
"""
function get_moments(𝓂::ℳ, parameters::Vector; 
    non_stochastic_steady_state::Bool = true, 
    standard_deviation::Bool = true, 
    variance::Bool = false, 
    covariance::Bool = false,
    verbose::Bool = false)

    solve!(𝓂, verbose = verbose)

    covar_dcmp, __, _, SS_and_pars = calculate_covariance(parameters,𝓂, verbose = verbose)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]

    if variance
        varrs = convert(Vector{Real},ℒ.diag(covar_dcmp))
        if standard_deviation
            st_dev = sqrt.(varrs)
        end
    else
        if standard_deviation
            st_dev = sqrt.(convert(Vector{Real},ℒ.diag(covar_dcmp)))
        end
    end

    ret = []
    if non_stochastic_steady_state
        push!(ret,SS)
    end
    if standard_deviation
        push!(ret,st_dev)
    end
    if variance
        push!(ret,varrs)
    end
    if covariance
        covar_dcmp_sp = sparse(ℒ.triu(covar_dcmp))

        droptol!(covar_dcmp_sp,eps(Float64))

        push!(ret,covar_dcmp_sp)
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
