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
                    variables::Symbol_input = :all, 
                    shocks::Symbol_input = :all, 
                    negative_shock::Bool = false, 
                    initial_state::Vector{Float64} = [0.0],
                    levels::Bool = false,
                    verbose = false)

    solve!(𝓂, verbose = verbose)

    NSSS, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, false, verbose)
    
	∇₁ = calculate_jacobian(parameters, NSSS, 𝓂)
								
    sol_mat = calculate_first_order_solution(∇₁; T = 𝓂.timings)

    state_update = function(state::Vector, shock::Vector) sol_mat * [state[𝓂.timings.past_not_future_and_mixed_idx]; shock] end
    
    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

    var_idx = parse_variables_input_to_index(variables, 𝓂.timings)
    
    SS = collect(NSSS[1:end - length(𝓂.calibration_equations)])

    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - SS

    # Y = zeros(𝓂.timings.nVars,periods,𝓂.timings.nExo)
    Ŷ = []
    for ii in shock_idx
        Y = []
        if shocks != :simulate
            ET = zeros(𝓂.timings.nExo,periods)
            ET[ii,1] = negative_shock ? -1 : 1
        end

        push!(Y, state_update(initial_state,ET[:,1]))

        for t in 1:periods-1
            push!(Y, state_update(Y[end],ET[:,t+1]))
        end
        push!(Ŷ, reduce(hcat,Y))
    end

    deviations = reshape(reduce(hcat,Ŷ),𝓂.timings.nVars,periods,𝓂.timings.nExo)[var_idx,:,shock_idx]

    if levels
        return deviations .+ SS[var_idx]
    else
        return deviations
    end
    # return KeyedArray(Y[var_idx,:,shock_idx];  Variables = T.var[var_idx], Period = 1:periods, Shock = T.exo[shock_idx])
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
→   Period ∈ 40-element UnitRange{Int64}
◪   Shock ∈ 1-element Vector{Symbol}
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
    variables::Symbol_input = :all, 
    shocks::Symbol_input = :all, 
    negative_shock::Bool = false, 
    generalised_irf::Bool = false,
    initial_state::Vector{Float64} = [0.0],
    levels::Bool = false,
    verbose = false)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true, algorithm = algorithm)
    
    state_update = parse_algorithm_to_state_update(algorithm, 𝓂)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    full_SS = sort(union(𝓂.var,𝓂.aux,𝓂.exo_present))
    full_SS[indexin(𝓂.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  𝓂.aux)

    reference_steady_state = [NSSS[s] for s in full_SS]#collect(NSSS[1:end - length(𝓂.calibration_equations)])

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    if levels
        if algorithm == :second_order
            reference_steady_state = 𝓂.solution.perturbation.second_order.stochastic_steady_state[indexin(full_SS,sort(union(𝓂.var,𝓂.exo_present)))]
        elseif algorithm == :third_order
            reference_steady_state = 𝓂.solution.perturbation.third_order.stochastic_steady_state[indexin(full_SS,sort(union(𝓂.var,𝓂.exo_present)))]
        end
        var_idx = parse_variables_input_to_index(variables, 𝓂.timings)
    end
    
    initial_state = initial_state == [0.0] ? zeros(𝓂.timings.nVars) : initial_state - reference_steady_state

    shocks = 𝓂.timings.nExo == 0 ? :none : shocks

    if shocks == :none && generalised_irf
        @error "Cannot compute generalised IRFs for model without shocks."
    end
    
    if generalised_irf
        girfs =  girf(state_update, 
                        𝓂.timings; 
                        periods = periods, 
                        shocks = shocks, 
                        variables = variables, 
                        negative_shock = negative_shock)#, warmup_periods::Int = 100, draws::Int = 50, iterations_to_steady_state::Int = 500)
        if levels
            return girfs .+ reference_steady_state
        else
            return girfs
        end
    else
        irfs =  irf(state_update, 
                    initial_state, 
                    𝓂.timings; 
                    periods = periods, 
                    shocks = shocks, 
                    variables = variables, 
                    negative_shock = negative_shock)
        if levels
            return irfs .+ reference_steady_state
        else
            return irfs
        end
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
$(SIGNATURES)
Return the (non stochastic) steady state and derivatives with respect to model parameters.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
- $DERIVATIVES
- `stochastic` [Default: `false`, Type: `Bool`]: return stochastic steady state using second order perturbation. No derivatives are calculated.
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
    parameter_derivatives::Symbol_input = :all,
    verbose = false)

    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    if parameter_derivatives == :all
        length_par = length(𝓂.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.parameters)
        length_par = 1
    elseif length(parameter_derivatives) > 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    SS = collect(NSSS)

    if stochastic
        solve!(𝓂, verbose = verbose, dynamics = true, algorithm = :second_order)
        SS[1:length(union(𝓂.exo_present,var))] = 𝓂.solution.perturbation.second_order.stochastic_steady_state
    end

    NSSS_labels = labels(NSSS) .|> Symbol
    var_idx = indexin(vcat(var,𝓂.calibration_equations_parameters),NSSS_labels)

    if length_par * length(var_idx) > 200
        derivatives = false
    end

    if derivatives && !stochastic
        # dSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
        dSS = ℱ.jacobian(x->collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1])[var_idx], Float64.(𝓂.parameter_values[param_idx]))
        𝓂.parameter_values = ℱ.value.(𝓂.parameter_values)

        # if length(𝓂.calibration_equations_parameters) == 0        
        #     return KeyedArray(hcat(collect(NSSS)[1:(end-1)],dNSSS);  Variables = [sort(union(𝓂.exo_present,var))...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters))
        # else
        # return ComponentMatrix(hcat(collect(NSSS), dNSSS)',Axis(vcat(:SS, 𝓂.parameters)),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(hcat(collect(NSSS), dNSSS), ([sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], vcat(:Steady_state, 𝓂.parameters)), ("Var. and par.", "∂x/∂y"))
        return KeyedArray(hcat(SS[var_idx],dSS);  Variables_and_calibrated_parameters = [sort(var)...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))
        # end
    else
        # return ComponentVector(collect(NSSS),Axis([sort(union(𝓂.exo_present,var))...,𝓂.calibration_equations_parameters...]))
        # return NamedArray(collect(NSSS), [sort(union(𝓂.exo_present,var))..., 𝓂.calibration_equations_parameters...], ("Variables and calibrated parameters"))
        return KeyedArray(SS[var_idx];  Variables_and_calibrated_parameters = [sort(var)...,𝓂.calibration_equations_parameters...])
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
See [`get_steady_state`](@ref)
"""
get_SS = get_steady_state




"""
$(SIGNATURES)
Return the linearised solution and the non stochastic steady state (SS) of the model.

# Arguments
- $MODEL
# Keyword Arguments
- $PARAMETERS
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
→   Variable ∈ 4-element Vector{Symbol}
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
    verbose = false)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    solve!(𝓂, verbose = verbose, dynamics = true)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    KeyedArray([𝓂.solution.non_stochastic_steady_state[[indexin(sort([var; map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  union(𝓂.aux,𝓂.exo_present))]), sort(union(var,𝓂.exo_present)))...]] 𝓂.solution.perturbation.first_order.solution_matrix]';
    Steady_state__States__Shocks = [:Steady_state; map(x->Symbol(string(x) * "₍₋₁₎"),𝓂.timings.past_not_future_and_mixed); map(x->Symbol(string(x) * "₍ₓ₎"),𝓂.exo)],
    Variable = sort([var; 𝓂.aux; 𝓂.exo_present]))
end


"""
See [`get_solution`](@ref)
"""
get_first_order_solution = get_solution

"""
See [`get_solution`](@ref)
"""
get_perturbation_solution = get_solution




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
→   Autocorrelation_order ∈ 5-element UnitRange{Int64}
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
    verbose = false)
    
    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    covar_dcmp, sol, __, _ = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    autocorr = reduce(hcat,[ℒ.diag(A ^ i * covar_dcmp ./ ℒ.diag(covar_dcmp))[indexin(sort(var),sort([var; 𝓂.aux; 𝓂.exo_present]))] for i in 1:5])
    
    KeyedArray(collect(autocorr); Variables = sort(var), Autocorrelation_order = 1:5)
end





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
- `parameter_derivatives` [Default: :all]: parameters for which to calculate derivatives of the SS. Inputs can be either a `Symbol` (e.g. `:alpha`, or `:all`), `Tuple{Symbol, Vararg{Symbol}}`, `Matrix{Symbol}` or `Vector{Symbol}`.
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
    verbose = false)#limit output by selecting pars and vars like for plots and irfs!?
    
    solve!(𝓂, verbose = verbose)

    write_parameters_input!(𝓂,parameters, verbose = verbose)

    var = setdiff(𝓂.var,𝓂.nonnegativity_auxilliary_vars)

    if parameter_derivatives == :all
        length_par = length(𝓂.parameters)
        param_idx = 1:length_par
    elseif isa(parameter_derivatives,Symbol)
        @assert parameter_derivatives ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."

        param_idx = indexin([parameter_derivatives], 𝓂.parameters)
        length_par = 1
    elseif length(parameter_derivatives) > 1
        for p in vec(collect(parameter_derivatives))
            @assert p ∈ 𝓂.parameters string(p) * " is not part of the free model parameters."
        end
        param_idx = indexin(parameter_derivatives |> collect |> vec, 𝓂.parameters) |> sort
        length_par = length(parameter_derivatives)
    end

    NSSS, solution_error = 𝓂.solution.outdated_NSSS ? 𝓂.SS_solve_func(𝓂.parameter_values, 𝓂, false, verbose) : (𝓂.solution.non_stochastic_steady_state, eps())

    NSSS_labels = labels(NSSS) .|> Symbol
    var_idx = indexin(var,NSSS_labels)
    var_idx_SS = indexin(vcat(var,𝓂.calibration_equations_parameters),NSSS_labels)

    if length_par * length(var_idx_SS) > 200
        derivatives = false
    end

    if derivatives
        dNSSS = ℱ.jacobian(x -> collect(SS_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[1])[var_idx_SS], Float64.(𝓂.parameter_values[param_idx]))
        𝓂.parameter_values[param_idx] = ℱ.value.(𝓂.parameter_values[param_idx])
        # dNSSS = ℱ.jacobian(x->𝓂.SS_solve_func(x, 𝓂),𝓂.parameter_values)
        SS =  KeyedArray(hcat(collect(NSSS)[var_idx_SS],dNSSS);  Variables = [sort(var)...,𝓂.calibration_equations_parameters...], Steady_state_and_∂steady_state∂parameter = vcat(:Steady_state, 𝓂.parameters[param_idx]))

        if variance
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

            vari = convert(Vector{Number},max.(ℒ.diag(covar_dcmp),eps(Float64)))

            # dvariance = ℱ.jacobian(x-> convert(Vector{Number},max.(ℒ.diag(calculate_covariance(x, 𝓂)),eps(Float64))), Float64.(𝓂.parameter_values))
            dvariance = ℱ.jacobian(x -> covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose)[var_idx], Float64.(𝓂.parameter_values[param_idx]))
            𝓂.parameter_values[param_idx] = ℱ.value.(𝓂.parameter_values[param_idx])
            
            varrs =  KeyedArray(hcat(vari[var_idx],dvariance);  Variables = sort(var), Variance_and_∂variance∂parameter = vcat(:Variance, 𝓂.parameters[param_idx]))

            if standard_deviation
                standard_dev = sqrt.(convert(Vector{Number},max.(ℒ.diag(covar_dcmp),eps(Float64))))
                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose))[var_idx], Float64.(𝓂.parameter_values[param_idx]))
                𝓂.parameter_values[param_idx] = ℱ.value.(𝓂.parameter_values[param_idx])

                st_dev =  KeyedArray(hcat(standard_dev[var_idx],dst_dev);  Variables = sort(var), Standard_deviation_and_∂standard_deviation∂parameter = vcat(:Standard_deviation, 𝓂.parameters[param_idx]))
            end
        else
            if standard_deviation
                covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]

                standard_dev = sqrt.(convert(Vector{Number},max.(ℒ.diag(covar_dcmp),eps(Float64))))

                dst_dev = ℱ.jacobian(x -> sqrt.(covariance_parameter_derivatives(x, param_idx, 𝓂, verbose = verbose))[var_idx], Float64.(𝓂.parameter_values[param_idx]))
                𝓂.parameter_values[param_idx] = ℱ.value.(𝓂.parameter_values[param_idx])

                st_dev =  KeyedArray(hcat(standard_dev[var_idx],dst_dev);  Variables = sort(var), Standard_deviation_and_∂standard_deviation∂parameter = vcat(:Standard_deviation, 𝓂.parameters[param_idx]))
            end
        end

    else
        SS =  KeyedArray(collect(NSSS)[var_idx_SS];  Variables = [sort(var)...,𝓂.calibration_equations_parameters...])

        if variance
            covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]
            varr = convert(Vector{Number},max.(ℒ.diag(covar_dcmp),eps(Float64)))
            varrs = KeyedArray(varr[var_idx];  Variables = sort(var))
            if standard_deviation
                st_dev = KeyedArray(sqrt.(varr[var_idx]);  Variables = sort(var))
            end
        else
            if standard_deviation
                covar_dcmp = calculate_covariance(𝓂.parameter_values, 𝓂, verbose = verbose)[1]
                st_dev = KeyedArray(sqrt.(convert(Vector{Number},max.(ℒ.diag(covar_dcmp)[var_idx],eps(Float64))));  Variables = sort(var))
            end
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
        push!(ret,covar_dcmp)
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
    verbose = false)

    solve!(𝓂, verbose = verbose)

    covar_dcmp, __, _, SS_and_pars = calculate_covariance(parameters,𝓂, verbose = verbose)

    SS = SS_and_pars[1:end - length(𝓂.calibration_equations)]

    if variance
        varrs = convert(Vector{Number},ℒ.diag(covar_dcmp))
        if standard_deviation
            st_dev = sqrt.(varrs)
        end
    else
        if standard_deviation
            st_dev = sqrt.(convert(Vector{Number},ℒ.diag(covar_dcmp)))
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

