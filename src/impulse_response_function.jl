
@unstable function compute_irf_responses(𝓂::ℳ,
                                state_update::Function,
                                initial_state::Union{Vector{Vector{Float64}},Vector{Float64}},
                                level::Vector{Float64};
                                periods::Int,
                                shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}},
                                variables::Union{Symbol_input,String_input},
                                shock_size::Real,
                                negative_shock::Bool,
                                generalised_irf::Bool,
                                generalised_irf_warmup_iterations::Int,
                                generalised_irf_draws::Int,
                                enforce_obc::Bool,
                                algorithm::Symbol)

    if enforce_obc
        obc_update = (present_states, present_shocks, state_update) -> obc_state_update(present_states, present_shocks, state_update, 𝓂, algorithm)

        if generalised_irf
            return girf(state_update,
                        obc_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock,
                        warmup_periods = generalised_irf_warmup_iterations,
                        draws = generalised_irf_draws)
        else
            return irf(state_update,
                        obc_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock)
        end
    else
        if generalised_irf
            return girf(state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock,
                        warmup_periods = generalised_irf_warmup_iterations,
                        draws = generalised_irf_draws)
        else
            return irf(state_update,
                        initial_state,
                        level,
                        𝓂.constants;
                        periods = periods,
                        shocks = shocks,
                        shock_size = shock_size,
                        variables = variables,
                        negative_shock = negative_shock)
        end
    end
end


function irf(state_update::Function, 
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    always_solved = true

    if shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        Y = zeros(T.nVars,periods,1)

        past_states = initial_state
        
        for t in 1:periods
            past_states, past_shocks, solved  = obc_state_update(past_states, shock_history[:,t], state_update)

            if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

            always_solved = always_solved && solved

            if !always_solved break end

            Y[:,t,1] = pruning ? sum(past_states) : past_states

            shock_history[:,t] = past_shocks
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)
        
        past_states = initial_state
        
        for t in 1:periods
            past_states, _, solved  = obc_state_update(past_states, shck, state_update)

            if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

            always_solved = always_solved && solved

            if !always_solved break end

            Y[:,t,1] = pruning ? sum(past_states) : past_states
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:none])
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for (i,ii) in enumerate(shock_idx)
            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
            end

            past_states = initial_state
            
            for t in 1:periods
                past_states, past_shocks, solved = obc_state_update(past_states, shock_history[:,t], state_update)

                if !solved @warn "No solution in period: $t" end#. Possible reasons: 1. infeasability 2. too long spell of binding constraint. To address the latter try setting max_obc_horizon to a larger value (default: 40): @model <name> max_obc_horizon=40 begin ... end" end

                always_solved = always_solved && solved

                if !always_solved break end

                Y[:,t,i] = pruning ? sum(past_states) : past_states

                shock_history[:,t] = past_shocks
            end
        end

        axis2 = shocks isa Union{Symbol_input,String_input} ? 
                    shock_idx isa Int ? 
                        [T.exo[shock_idx]] : 
                    T.exo[shock_idx] : 
                [:Shock_matrix]

        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end
    
        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
    end
end




@unstable function irf(state_update::Function, 
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64},
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."
        
        shock_history = zeros(T.nExo, periods)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    if shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        Y = zeros(T.nVars,periods,1)

        initial_state = state_update(initial_state,shock_history[:,1])

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state,shock_history[:,t+1])

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:simulate])
    elseif shocks == :none
        Y = zeros(T.nVars,periods,1)

        shck = T.nExo == 0 ? Vector{Float64}(undef, 0) : zeros(T.nExo)

        initial_state = state_update(initial_state, shck)

        Y[:,1,1] = pruning ? sum(initial_state) : initial_state

        for t in 1:periods-1
            initial_state = state_update(initial_state, shck)

            Y[:,t+1,1] = pruning ? sum(initial_state) : initial_state
        end

        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = [:none])
    else
        Y = zeros(T.nVars,periods,length(shock_idx))

        for (i,ii) in enumerate(shock_idx)
            initial_state_copy = deepcopy(initial_state)
            
            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
            end

            initial_state_copy = state_update(initial_state_copy, shock_history[:,1])

            Y[:,1,i] = pruning ? sum(initial_state_copy) : initial_state_copy

            for t in 1:periods-1
                initial_state_copy = state_update(initial_state_copy, shock_history[:,t+1])

                Y[:,t+1,i] = pruning ? sum(initial_state_copy) : initial_state_copy
            end
        end

        axis2 = shocks isa Union{Symbol_input,String_input} ? 
                    shock_idx isa Int ? 
                        [T.exo[shock_idx]] : 
                    T.exo[shock_idx] : 
                [:Shock_matrix]
        
        if any(x -> contains(string(x), "◖"), axis2)
            axis2_decomposed = decompose_name.(axis2)
            axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
        end
    
        return KeyedArray(Y[var_idx,:,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
    end
end



function girf(state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64}, 
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model (model has $(T.nExo) shocks)."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."

        shock_history = zeros(T.nExo, periods + 1)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    Y = zeros(T.nVars, periods + 1, length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        initial_state_copy = deepcopy(initial_state)

        accepted_draws = 0

        for draw in 1:draws
            ok = true

            initial_state_copy² = deepcopy(initial_state_copy)

            for i in 1:warmup_periods
                initial_state_copy² = state_update(initial_state_copy², randn(T.nExo))
                if any(!isfinite, [x for v in initial_state_copy² for x in v])
                    # @warn "No solution in warmup period: $i"
                    ok = false
                    break
                end
            end
            
            if !ok continue end

            Y₁ = zeros(T.nVars, periods + 1)
            Y₂ = zeros(T.nVars, periods + 1)

            baseline_noise = randn(T.nExo)

            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input,String_input}
                shock_history = zeros(T.nExo,periods)
                shock_history[ii,1] = negative_shock ? -shock_size : shock_size
            end

            if pruning
                initial_state_copy² = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, [x for v in initial_state_copy² for x in v]) continue end

                initial_state₁ = deepcopy(initial_state_copy²)
                initial_state₂ = deepcopy(initial_state_copy²)

                Y₁[:,1] = initial_state_copy² |> sum
                Y₂[:,1] = initial_state_copy² |> sum
            else
                Y₁[:,1] = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, Y₁[:,1]) continue end

                Y₂[:,1] = state_update(initial_state_copy², baseline_noise)
                
                if any(!isfinite, Y₂[:,1]) continue end
            end

            for t in 1:periods
                baseline_noise = randn(T.nExo)

                if pruning
                    initial_state₁ = state_update(initial_state₁, baseline_noise)
                
                    if any(!isfinite, [x for v in initial_state₁ for x in v])
                        ok = false
                        break
                    end

                    initial_state₂ = state_update(initial_state₂, baseline_noise + shock_history[:,t])
                
                    if any(!isfinite, [x for v in initial_state₂ for x in v])
                        ok = false
                        break
                    end

                    Y₁[:,t+1] = initial_state₁ |> sum
                    Y₂[:,t+1] = initial_state₂ |> sum
                else
                    Y₁[:,t+1] = state_update(Y₁[:,t],baseline_noise)

                    if any(!isfinite, Y₁[:,t+1])
                        ok = false
                        break
                    end

                    Y₂[:,t+1] = state_update(Y₂[:,t],baseline_noise + shock_history[:,t])

                    if any(!isfinite, Y₂[:,t+1])
                        ok = false
                        break
                    end
                end
            end

            if !ok continue end

            Y[:,:,i] += Y₂ - Y₁

            accepted_draws += 1
        end
        
        if accepted_draws == 0
            @warn "No draws accepted. Results are empty."
        elseif accepted_draws < draws
            # average over accepted draws, if desired
            @info "$accepted_draws of $draws draws accepted for shock: $(shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input} ? T.exo[ii] : :Shock_matrix)"
            Y[:, :, i] ./= accepted_draws
        else
            Y[:, :, i] ./= accepted_draws
        end
    end
    
    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = shocks isa Union{Symbol_input,String_input} ? 
                shock_idx isa Int ? 
                    [T.exo[shock_idx]] : 
                T.exo[shock_idx] : 
            [:Shock_matrix]

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    return KeyedArray(Y[var_idx,2:end,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
end


function girf(state_update::Function,
    obc_state_update::Function,
    initial_state::Union{Vector{Vector{Float64}},Vector{Float64}}, 
    level::Vector{Float64}, 
    constants::constants; 
    periods::Int = 40, 
    shocks::Union{Symbol_input,String_input,Matrix{Float64},KeyedArray{Float64}} = :all, 
    variables::Union{Symbol_input,String_input} = :all, 
    shock_size::Real = 1,
    negative_shock::Bool = false, 
    warmup_periods::Int = 100, 
    draws::Int = 50)::Union{KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{String}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{String},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{Symbol}}},   KeyedArray{Float64, 3, NamedDimsArray{(:Variables, :Periods, :Shocks), Float64, 3, Array{Float64, 3}}, Tuple{Vector{Symbol},UnitRange{Int},Vector{String}}}}
    T = constants.post_model_macro

    pruning = initial_state isa Vector{Vector{Float64}}

    shocks = shocks isa KeyedArray ? axiskeys(shocks,1) isa Vector{String} ? rekey(shocks, 1 => axiskeys(shocks,1) .|> Meta.parse .|> replace_indices) : shocks : shocks

    shocks = shocks isa String_input ? shocks .|> Meta.parse .|> replace_indices : shocks

    if shocks isa Matrix{Float64}
        @assert size(shocks)[1] == T.nExo "Number of rows of provided shock matrix does not correspond to number of shocks. Please provide matrix with as many rows as there are shocks in the model."

        # periods += size(shocks)[2]

        shock_history = zeros(T.nExo, periods)

        shock_history[:,1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks isa KeyedArray{Float64}
        shock_input = map(x->Symbol(replace(string(x),"₍ₓ₎" => "")),axiskeys(shocks)[1])

        # periods += size(shocks)[2]

        @assert length(setdiff(shock_input, T.exo)) == 0 "Provided shocks are not part of the model. Use `get_shocks(𝓂)` to list valid shock names."

        shock_history = zeros(T.nExo, periods + 1)

        shock_history[indexin(shock_input,T.exo),1:size(shocks)[2]] = shocks

        shock_idx = 1
    elseif shocks == :simulate
        shock_history = randn(T.nExo,periods) * shock_size
        
        shock_history[contains.(string.(T.exo),"ᵒᵇᶜ"),:] .= 0

        shock_idx = 1
    else
        shock_idx = parse_shocks_input_to_index(shocks,constants)
    end

    var_idx = parse_variables_input_to_index(variables, constants) |> sort

    Y = zeros(T.nVars, periods + 1, length(shock_idx))

    for (i,ii) in enumerate(shock_idx)
        initial_state_copy = deepcopy(initial_state)

        accepted_draws = 0

        for draw in 1:draws
            ok = true

            initial_state_copy² = deepcopy(initial_state_copy)

            warmup_shocks = randn(T.nExo)
            warmup_shocks[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

            # --- warmup ---
            for i_w in 1:warmup_periods
                initial_state_copy², _, solved = obc_state_update(initial_state_copy², warmup_shocks, state_update)
                if !solved
                    # @warn "No solution in warmup period: $i_w"
                    ok = false
                    break
                end
            end
            
            if !ok continue end

            Y₁ = zeros(T.nVars, periods + 1)
            Y₂ = zeros(T.nVars, periods + 1)

            baseline_noise = randn(T.nExo)
            baseline_noise[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

            if shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input}
                shock_history = zeros(T.nExo, periods)
                shock_history[ii, 1] = negative_shock ? -shock_size : shock_size
            end

            # --- period 1 ---
            if pruning
                initial_state_copy², _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end

                initial_state₁ = deepcopy(initial_state_copy²)
                initial_state₂ = deepcopy(initial_state_copy²)

                Y₁[:, 1] = initial_state_copy² |> sum
                Y₂[:, 1] = initial_state_copy² |> sum
            else
                Y₁[:, 1], _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end

                Y₂[:, 1], _, solved = obc_state_update(initial_state_copy², baseline_noise, state_update)
                if !solved continue end
            end

            # --- remaining periods ---
            for t in 1:periods
                baseline_noise = randn(T.nExo)
                baseline_noise[contains.(string.(T.exo), "ᵒᵇᶜ")] .= 0

                if pruning
                    initial_state₁, _, solved = obc_state_update(initial_state₁, baseline_noise, state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    initial_state₂, _, solved = obc_state_update(initial_state₂, baseline_noise + shock_history[:, t], state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    Y₁[:, t + 1] = initial_state₁ |> sum
                    Y₂[:, t + 1] = initial_state₂ |> sum
                else
                    Y₁[:, t + 1], _, solved = obc_state_update(Y₁[:, t], baseline_noise, state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end

                    Y₂[:, t + 1], _, solved = obc_state_update(Y₂[:, t], baseline_noise + shock_history[:, t], state_update)
                    if !solved
                        # @warn "No solution in period: $t"
                        ok = false
                        break
                    end
                end
            end

            if !ok continue end

            # Note: replace `i` if your outer scope uses another index
            Y[:, :, i] .+= (Y₂ .- Y₁)
            accepted_draws += 1
        end

        if accepted_draws == 0
            @warn "No draws accepted. Results are empty."
        elseif accepted_draws < draws
            # average over accepted draws, if desired
            @info "$accepted_draws of $draws draws accepted for shock: $(shocks ∉ [:simulate, :none] && shocks isa Union{Symbol_input, String_input} ? T.exo[ii] : :Shock_matrix)"
            Y[:, :, i] ./= accepted_draws
        else
            Y[:, :, i] ./= accepted_draws
        end
    end
    
    axis1 = T.var[var_idx]
        
    if any(x -> contains(string(x), "◖"), axis1)
        axis1_decomposed = decompose_name.(axis1)
        axis1 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis1_decomposed]
    end

    axis2 = shocks isa Union{Symbol_input,String_input} ? 
                shock_idx isa Int ? 
                    [T.exo[shock_idx]] : 
                T.exo[shock_idx] : 
            [:Shock_matrix]

    if any(x -> contains(string(x), "◖"), axis2)
        axis2_decomposed = decompose_name.(axis2)
        axis2 = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in axis2_decomposed]
    end

    return KeyedArray(Y[var_idx,2:end,:] .+ level[var_idx];  Variables = axis1, Periods = 1:periods, Shocks = axis2)
end

