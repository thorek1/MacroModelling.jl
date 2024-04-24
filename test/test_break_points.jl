
using MacroModelling

function transform_break_points(break_points)
    dict = Dict{Int, Dict{Symbol, Float64}}()
    
    for i in 1:size(break_points,2)
        time = axiskeys(break_points,2)[i]

        if !haskey(dict, time)
            dict[time] = Dict{Symbol, Float64}()
        end
        
        for k in 1:size(break_points,1)
            parameter = axiskeys(break_points,1)[k]
            value = break_points[k,i]
            if !(isnothing(value) || !isfinite(value))
                dict[time][parameter] = value
            end
        end
    end
    
    if !haskey(dict, 0)
        dict[0] = Dict{Symbol, Float64}()
    end

    return dict
end

include("../models/Smets_Wouters_2007.jl")

irfs = get_irf(Smets_Wouters_2007)

irfs = get_irf(Smets_Wouters_2007, shocks = :none, levels = true)
get_parameters(Smets_Wouters_2007, values = true)

break_points = KeyedArray([.4,.3,.25]', Variable =[:ctrend], Time = [3,15,30])

irfs = get_irf(Smets_Wouters_2007, parameters = break_points, shocks = :none, levels = true)

starting_vals = get_irf(Smets_Wouters_2007, shocks = :none, levels = true, periods = 1)
irfalt = get_irf(Smets_Wouters_2007, parameters = :ctrend => .4, shocks = :none, levels = true, initial_state = vec(starting_vals))





parameters = KeyedArray(
    [nothing .3 .25 
    .77 NaN .9], 
    Variable = [:ctrend, :constepinf], 
    Time = [2, 15, 30])



# translate this to the combined parameter vector
import MacroModelling: get_relevant_steady_states, parse_shocks_input_to_index, parse_algorithm_to_state_update, obc_objective_optim_fun, obc_constraint_optim_fun, String_input, Symbol_input, timings, parse_variables_input_to_index, ParameterType

𝓂 = Smets_Wouters_2007
T = 𝓂.timings
periods = 100
algorithm = :first_order

# parameters = nothing

variables = :all_excluding_obc
shocks = :none # :all_excluding_obc
negative_shock = false
generalised_irf = false
initial_state = [0.0]
levels = false
ignore_obc = false
verbose = false





shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

shocks = 𝓂.timings.nExo == 0 ? :none : shocks

@assert !(shocks == :none && generalised_irf) "Cannot compute generalised IRFs for model without shocks."

stochastic_model = length(𝓂.timings.exo) > 0

obc_model = length(𝓂.obc_violation_equations) > 0

if ignore_obc
    occasionally_binding_constraints = false
else
    occasionally_binding_constraints = length(𝓂.obc_violation_equations) > 0
end

if parameters isa ParameterType
    break_points_dict = Dict{Int, Dict{Symbol, Float64}}(0 => Dict{Symbol, Float64}())
    periods_of_change = [0]
else
    break_points_dict = transform_break_points(parameters)
    periods_of_change = break_points_dict |> keys |> collect |> sort
end

if shocks isa Matrix{Float64}
    @assert size(shocks)[1] == 𝓂.timings.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

    periods += size(shocks)[2]

    shock_idx = [1]
    
    shock_history = zeros(𝓂.timings.nExo, periods + periods_of_change[end], length(shock_idx))

    shock_history[:, 1:size(shocks)[2], 1] = shocks

    obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ"),:]) > 1e-10
elseif shocks isa KeyedArray{Float64}
    shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

    periods += size(shocks)[2]

    @assert length(setdiff(shock_input, 𝓂.timings.exo)) == 0 "Provided shocks which are not part of the model."

    shock_idx = [1]

    shock_history = zeros(𝓂.timings.nExo, periods + periods_of_change[end], length(shock_idx))

    shock_history[indexin(shock_input,𝓂.timings.exo), 1:size(shocks)[2], 1] = shocks

    obc_shocks_included = stochastic_model && obc_model && sum(abs2,shocks(intersect(𝓂.timings.exo,axiskeys(shocks,1)),:)) > 1e-10
elseif shocks == :simulate
    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

    shock_history = zeros(𝓂.timings.nExo, periods + periods_of_change[end], length(shock_idx))

    shock_history[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ"),:] .= 0

    obc_shocks_included = stochastic_model && obc_model && (intersect((((shock_idx isa Vector) || (shock_idx isa UnitRange)) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])
elseif shocks == :none
    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

    shock_history = zeros(𝓂.timings.nExo, periods + periods_of_change[end], length(shock_idx))

    obc_shocks_included = stochastic_model && obc_model && (intersect((((shock_idx isa Vector) || (shock_idx isa UnitRange)) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])
elseif shocks isa Union{Symbol_input,String_input}
    shock_idx = parse_shocks_input_to_index(shocks,𝓂.timings)

    shock_history = zeros(𝓂.timings.nExo, periods + periods_of_change[end], length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        shock_history[ii,1,i] = negative_shock ? -1 : 1
    end

    obc_shocks_included = stochastic_model && obc_model && (intersect((((shock_idx isa Vector) || (shock_idx isa UnitRange)) && (length(shock_idx) > 0)) ? 𝓂.timings.exo[shock_idx] : [𝓂.timings.exo[shock_idx]], 𝓂.timings.exo[contains.(string.(𝓂.timings.exo),"ᵒᵇᶜ")]) != [])
end

solve!(𝓂, parameters = parameters isa ParameterType ? parameters : nothing, verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

var_idx = parse_variables_input_to_index(variables, 𝓂.timings)

axis1 = 𝓂.timings.var[var_idx]
    
if any(x -> contains(string(x), "◖"), axis1)
    axis1_decomposed = decompose_name.(axis1)
    axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
end
            

if shocks == :simulate
    axis2 = [:simulate]
elseif shocks == :none
    axis2 = [:none]
else
    axis2 = shocks isa Union{Symbol_input,String_input} ? 
        shock_idx isa Int ? 
            [𝓂.timings.exo[shock_idx]] : 
        𝓂.timings.exo[shock_idx] : 
    [:Shock_matrix]

    if any(x -> contains(string(x), "◖"), axis2)
    axis2_decomposed = decompose_name.(axis2)
    axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end
end

reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)

unspecified_initial_state = initial_state == [0.0]

if unspecified_initial_state
    if algorithm == :pruned_second_order
        initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta]
    elseif algorithm == :pruned_third_order
        initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
    else
        initial_state = zeros(𝓂.timings.nVars) - SSS_delta
    end
else
    if initial_state isa Vector{Float64}
        if algorithm == :pruned_second_order
            initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
        else
            initial_state = initial_state - NSSS
        end
    else
        if algorithm ∉ [:pruned_second_order, :pruned_third_order]
            @assert initial_state isa Vector{Float64} "The solution algorithm has one state vector: initial_state must be a Vector{Float64}."
        end
    end
end


steady_states_and_state_update = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Function}}()

for p in periods_of_change
    solve!(𝓂, parameters = break_points_dict[p], verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

    if occasionally_binding_constraints
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    elseif obc_shocks_included
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks witout enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)

    steady_states_and_state_update[p] = (reference_steady_state, NSSS, SSS_delta, state_update)
end


periods += periods_of_change[end]
# function irf(steady_states_and_state_update::Dict{Int, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Function}},
#     initial_state::Union{Vector{Vector{Float64}},Vector{Float64}},
#     shock_history::Matrix{Float64},
#     T::timings; 
#     periods::Int = 40, 
#     negative_shock::Bool = false)


Y = zeros(T.nVars, periods, length(shock_idx))

pruning = initial_state isa Vector{Vector{Float64}}

periods_of_change = steady_states_and_state_update |> keys |> collect |> sort

initial_state_copy = [deepcopy(initial_state) for _ in shock_idx]

for i in shock_idx
    for (k,p) in enumerate(periods_of_change)
        concerned_periods = (k == 1 ? 1 : periods_of_change[k]):(k == length(periods_of_change) ? periods : periods_of_change[k+1] - 1)

        println(concerned_periods)
        reference_steady_state, NSSS, SSS_delta, state_update = steady_states_and_state_update[p]

        if k > 1
            ΔSS = steady_states_and_state_update[periods_of_change[k]][2] - steady_states_and_state_update[periods_of_change[k-1]][2]

            if pruning
                for j in initial_state_copy[i]
                    j += ΔSS
                end
            else
                initial_state_copy[i] += ΔSS
            end
        end

        for t in concerned_periods
            initial_state_copy[i] = state_update(initial_state_copy[i], shock_history[:,t,i])

            Y[:,t,i] = pruning ? sum(initial_state_copy[i]) : initial_state_copy[i]
        end
    end
end


Y[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:,:]

(Y .+ reference_steady_state)[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:,:]

reference_steady_states[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:]



Y = zeros(T.nVars,periods,length(shock_idx))

for (i,ii) in enumerate(shock_idx)
    initial_state_copy = deepcopy(initial_state)
    
    if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
        shock_history = zeros(T.nExo,periods)
        shock_history[ii,1] = negative_shock ? -1 : 1
    end

    initial_state_copy = state_update(initial_state_copy, shock_history[:,1])

    Y[:,1,i] = pruning ? sum(initial_state_copy) : initial_state_copy

    for t in 1:periods-1
        initial_state_copy = state_update(initial_state_copy, shock_history[:,t+1])

        Y[:,t+1,i] = pruning ? sum(initial_state_copy) : initial_state_copy
    end
end








get_parameters(Smets_Wouters_2007, values = true)

# solve!(𝓂, parameters = break_points_dict[30], verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)
    
# reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)


initial_state = [0.0]

pruning = initial_state isa Vector{Vector{Float64}}

Y = zeros(T.nVars,periods + periods_of_change[end],length(shock_idx));

reference_steady_states = zeros(T.nVars,periods + periods_of_change[end]);

for (i,p) in enumerate(periods_of_change)
# i = 1
# p  = periods_of_change[i]

    concerned_periods = (i == 1 ? 1 : periods_of_change[i]):(i == length(periods_of_change) ? p + periods : periods_of_change[i+1] - 1)
    # periods_with_these_parameters = (i == length(periods_of_change) ? p + periods : periods_of_change[i+1]) - (i == 1 ? 1 : periods_of_change[i])

    # if periods_with_these_parameters == 0 continue end

    solve!(𝓂, parameters = break_points_dict[p], verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)
    
    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)

    reference_steady_states[:,concerned_periods] .= reference_steady_state

    Δreference_steady_state = reference_steady_state - reference_steady_stateꜜ
    ΔNSSS = NSSS - NSSSꜜ
    ΔSSS_delta = SSS_delta - SSS_deltaꜜ

    # println(maximum(abs, Δreference_steady_state), maximum(abs, ΔNSSS), maximum(abs, ΔSSS_delta))

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta]
            Δinitial_state = [Δreference_steady_state[1:𝓂.timings.nVars], Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
            Δinitial_state = [Δreference_steady_state[1:𝓂.timings.nVars], Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta, Δreference_steady_state[1:𝓂.timings.nVars]]
        else
            initial_state = zeros(𝓂.timings.nVars) - SSS_delta
            Δinitial_state = Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta]
                Δinitial_state = [Δreference_steady_state[1:𝓂.timings.nVars], Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
                Δinitial_state = [Δreference_steady_state[1:𝓂.timings.nVars], Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta, Δreference_steady_state[1:𝓂.timings.nVars]]
            else
                initial_state = initial_state - NSSS
                Δinitial_state = Δreference_steady_state[1:𝓂.timings.nVars] - ΔSSS_delta
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
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks witout enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end
    
    if 1 ∈ concerned_periods
        initial_state_copy = [deepcopy(initial_state) for _ in shock_idx]
    end

    initial_state_copy = [s - Δinitial_state for s in initial_state_copy]

    for (i,ii) in enumerate(shock_idx)
        for t in concerned_periods .- 1
            initial_state_copy[i] = state_update(initial_state_copy[i], shock_history[:,t+1,i])

            Y[:,t+1,i] = pruning ? sum(initial_state_copy[i]) : initial_state_copy[i]
        end
    end
    reference_steady_stateꜜ, NSSSꜜ, SSS_deltaꜜ = reference_steady_state, NSSS, SSS_delta
    # println("concerned_periods: ", concerned_periods)
    # println("periods_with_these_parameters: ", periods_with_these_parameters)
    # println("parameters: ", break_points_dict[p])
end


(Y .+ reference_steady_states)[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:,:]

reference_steady_states[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:]





# check against alternative way

solve!(𝓂, parameters = previous_parameter_values, verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)

iirrff = get_irf(𝓂, initial_state = reference_steady_state[1:𝓂.timings.nVars], parameters = break_points_dict[2], shocks = :none)

iirrff([:dinve,:pinfobs,:robs,:dwobs,:labobs],:,:)

Y[indexin([:dinve,:pinfobs,:robs,:dwobs,:labobs],T.var),:,:]



solve!(𝓂, parameters = previous_parameter_values, verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)
    



solve!(𝓂, parameters = previous_parameter_values, verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

solve!(𝓂, parameters = break_points_dict[1], verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

i = 2

concerned_periods = (i == 1 ? 1 : periods_of_change[i-1]):periods_of_change[i]
periods_with_these_parameters = periods_of_change[i] - (i == 1 ? 1 : periods_of_change[i-1])

# for (i,p) in enumerate(periods_of_change)
#     concerned_periods = (i == 1 ? 1 : periods_of_change[i-1]):periods_of_change[i]
#     periods_with_these_parameters = periods_of_change[i+1] - (i == 1 ? 1 : periods_of_change[i])
# if p == 1
#     pars = break_points_dict[p]
# else
#     pars = nothing
# end

# write a function that goes from breakpoint to breakpoint, solves the model with the new parameters, and then computes the IRFs for as many periods until the next breakpoint


    solve!(𝓂, parameters = pars, verbose = verbose, dynamics = true, algorithm = algorithm, obc = occasionally_binding_constraints || obc_shocks_included)

    reference_steady_state, NSSS, SSS_delta = get_relevant_steady_states(𝓂, algorithm)

    unspecified_initial_state = initial_state == [0.0]

    if unspecified_initial_state
        if algorithm == :pruned_second_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta]
        elseif algorithm == :pruned_third_order
            initial_state = [zeros(𝓂.timings.nVars), zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
        else
            initial_state = zeros(𝓂.timings.nVars) - SSS_delta
        end
    else
        if initial_state isa Vector{Float64}
            if algorithm == :pruned_second_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta]
            elseif algorithm == :pruned_third_order
                initial_state = [initial_state - reference_steady_state[1:𝓂.timings.nVars], zeros(𝓂.timings.nVars) - SSS_delta, zeros(𝓂.timings.nVars)]
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
        @assert algorithm ∉ [:pruned_second_order, :second_order, :pruned_third_order, :third_order] "Occasionally binding constraint shocks witout enforcing the constraint is only compatible with first order perturbation solutions."

        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, true)
    else
        state_update, pruning = parse_algorithm_to_state_update(algorithm, 𝓂, false)
    end

    irfs1 =  irff(state_update, 
    initial_state, 
    levels ? reference_steady_state + SSS_delta : SSS_delta,
    𝓂.timings; 
    periods = p, 
    shocks = shocks, 
    variables = variables, 
    negative_shock = negative_shock)

# end

irfs1 =  irff(state_update, 
            initial_state, 
            levels ? reference_steady_state + SSS_delta : SSS_delta,
            𝓂.timings; 
            periods = periods, 
            shocks = shocks, 
            variables = variables, 
            negative_shock = negative_shock)

irfs1[:,1,:]











function irff(state_update::Function, 
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64},Matrix{Float64}}, 
    level::Vector{Float64},
    T::timings; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    negative_shock::Bool = false)

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = [1]
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks which are not part of the model."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = [1]
    else
        shock_idx = parse_shocks_input_to_index(shocks,T)
    end


    if shocks == :simulate
        shock_history = randn(T.nExo,periods)

        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        Y = zeros(T.nVars,periods,1)

        initial_state = state_update(initial_state,shock_history[:,1])

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state,shock_history[:,t+1])

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)

        initial_state = state_update(initial_state, shck)

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state, shck)

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for (i,ii) in enumerate(shock_idx)
            initial_state_copy = deepcopy(initial_state)
            
            if shocks != :simulate && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -1 : 1
            end

            initial_state_copy = state_update(initial_state_copy, shock_history[:,1])

            Y[:,1,i] = pruning ? sum(initial_state_copy) : initial_state_copy

            for t in 1:periods-1
                initial_state_copy = state_update(initial_state_copy, shock_history[:,t+1])

                Y[:,t+1,i] = pruning ? sum(initial_state_copy) : initial_state_copy
            end
        end
    end

    return Y
end

