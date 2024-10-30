using Revise
using MacroModelling
using Flux
# using FluxKAN
using ParameterSchedulers
using Optim
using FluxOptTools
using StatsPlots
using Sobol

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())

# include("../models/Smets_Wouters_2007.jl")

# model = Smets_Wouters_2007


@model RBC_baseline begin
	c[0] ^ (-σ) = β * c[1] ^ (-σ) * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ * c[0] ^ σ / (1 - l[0]) = w[0]

	k[0] = (1 - δ) * k[-1] + z[0] * k[-1] ^ α * l[0] ^ (1 - α) - g[0] - c[0] 

	w[0] = z[0] * (1 - α) * (k[-1] / l[0]) ^ α

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]
end


@parameters RBC_baseline begin
	σᶻ = 0.066

	σᵍ = .104

	σ = 1

	α = 1/3

	i_y = 0.25

	k_y = 10.4

	ρᶻ = 0.97

	ρᵍ = 0.989

	g_y = 0.2038

	ḡ | ḡ = g_y * k[ss] ^ α * l[ss] ^ (1 - α)

    δ = i_y / k_y

    β = 1 / (α / k_y + (1 - δ))

	ψ | l[ss] = 1/3
end

model = RBC_baseline

normalise = true  # use asinh and tanh at the beginning if there is no normalisation
recurrent = false # the internal state needs to be reset but carries a lot of information

model_parameters = Symbol.(get_parameters(model))

n_model_parameters = length(model_parameters)

n_shocks = length(get_shocks(model))

n_vars = length(get_variables(model))

n_hidden = max(64, n_vars * 2)

n_inputs = n_vars + n_shocks + n_model_parameters

if recurrent
    neural_net = Chain( Dense(n_inputs, n_hidden, asinh),
                        Flux.LSTM(n_hidden, n_hidden ÷ 2),
                        Flux.GRU(n_hidden ÷ 2, n_hidden ÷ 2), # optional
                        Dense(n_hidden ÷ 2, n_hidden ÷ 2, celu),
                        Dense(n_hidden ÷ 2, n_hidden, celu),
                        Dense(n_hidden, n_hidden, celu), # optional
                        Dense(n_hidden, n_vars))   
else
    if normalise
        neural_net = Chain( Dense(n_inputs, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_vars))
    else
        neural_net = Chain( Dense(n_inputs, n_hidden, asinh),
                            Dense(n_hidden, n_hidden, asinh),
                            Dense(n_hidden, n_hidden, tanh),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_vars))
    end
end

optim = Flux.setup(Flux.Adam(), neural_net)
# optim = Flux.setup(Flux.AdamW(.001,(.9,.999),.01), neural_net)

nn_params = sum(length.(Flux.params(neural_net)))

n_internal_loop = 10
# n_batches = 100
n_simul_per_batch = 40 # nn_params ÷ (n_vars * 10)
n_burnin = 500
n_epochs = 15000

# n_periods_batch_stays_in_sample = 200
# n_batches_in_total = 3000
# Hannos settings
n_periods_batch_stays_in_sample = 1000
n_batches_in_total = 1000

n_batches = n_periods_batch_stays_in_sample * n_batches_in_total ÷ n_epochs
new_batch_every_n_periods = n_epochs ÷ n_batches_in_total

# total_obs_seen = n_epochs * n_simul_per_batch * n_batches_in_total

s = ParameterSchedulers.Stateful(CosAnneal(.001, 1e-8, n_epochs))
# s = ParameterSchedulers.Stateful(SinDecay2(.001, 1e-6, 500))


# Parameter draws (Sobol)
sob = SobolSeq(length(model_parameters))
lower_bounds_par = model.parameter_values .- 0.01
upper_bounds_par = model.parameter_values .+ 0.01

lower_bounds_par = [0
                    0
                    -2
                    0.2
                    8
                    0.25
                    0.25
                    0.15
                    0.1] .+ eps()
upper_bounds_par = [1
                    1
                    3
                    0.3
                    12
                    1
                    1
                    0.25
                    0.5] .- eps()

bounds_range = upper_bounds_par .- lower_bounds_par

get_irf(model)
get_mean(model, derivatives = false)
get_std(model, derivatives = false)

get_parameters(model, values = true)
function generate_new_data(sob::SobolSeq, n_batches::Int, n_simul_per_batch::Int, n_burnin::Int, n_shocks::Int)
    training_data = Tuple{Matrix{Float32}, Matrix{Float32}}[]

    for i in 1:n_batches
        draw = next!(sob)

        transformed_draw = draw .* bounds_range .+ lower_bounds_par
        
        normalised_draw = (draw .- 0.5) .* sqrt(12)

        shcks = randn(n_shocks, n_burnin + n_simul_per_batch)
        
        solved = false
        
        while !solved
            irf_succeeded = true

            sims = try get_irf(model, 
                                shocks = shcks, 
                                parameters = (model_parameters .=> transformed_draw),
                                periods = 0, 
                                levels = true)
            catch
                draw = next!(sob)
                
                transformed_draw = draw .* bounds_range .+ lower_bounds_par
                
                normalised_draw = (draw .- 0.5) .* sqrt(12)

                continue
            end

            if normalise
                mn = get_mean(model, 
                                # parameters = (pararmeters .=> draw),
                                derivatives = false)
                
                stddev = get_std(model, 
                                # parameters = (pararmeters .=> draw),
                                # verbose = true,
                                derivatives = false)
                
                normalised_sims = collect((sims[:,n_burnin:end,1] .- mn) ./ stddev)
    
                if maximum(abs.(normalised_sims)) > 10 || any(!isfinite, normalised_sims)
                    draw = next!(sob)
                    
                    transformed_draw = draw .* bounds_range .+ lower_bounds_par
    
                    normalised_draw = (draw .- 0.5) .* sqrt(12)

                    continue 
                end
                
                inputs = Float32.(vcat(normalised_sims[:,1:end - 1], shcks[:,n_burnin + 1:n_burnin + n_simul_per_batch], reshape(repeat(normalised_draw, n_simul_per_batch), length(normalised_draw), n_simul_per_batch)))
    
                outputs = Float32.(normalised_sims[:,2:end])
            else
                inputs = Float32.(vcat(collect(sims[:,n_burnin:n_burnin + n_simul_per_batch - 1,1]), shcks[:,n_burnin + 1:n_burnin + n_simul_per_batch], reshape(repeat(normalised_draw, n_simul_per_batch), length(normalised_draw), n_simul_per_batch)))
                
                outputs = Float32.(collect(sims[:,n_burnin+1:n_burnin + n_simul_per_batch,1]))  
            end

            push!(training_data, (outputs, inputs))

            solved = true
        end
    end

    return training_data
end


training_data = generate_new_data(sob, n_batches, n_simul_per_batch, n_burnin, n_shocks)

out = mapreduce(x -> x[1], hcat, training_data)
inp = mapreduce(x -> x[2], hcat, training_data)

losses = []
# Training loop
for epoch in 1:n_epochs
    if epoch % new_batch_every_n_periods == 0
        training_dat = generate_new_data(sob, 1, n_simul_per_batch, n_burnin, n_shocks)
        popfirst!(training_data)
        push!(training_data, training_dat[1])

        out = mapreduce(x -> x[1], hcat, training_data)
        inp = mapreduce(x -> x[2], hcat, training_data)   
    end

    # if epoch % 100 == 0
    #     training_data = generate_new_data(sob, n_batches, n_simul_per_batch, n_burnin, n_shocks)

    #     out = mapreduce(x -> x[1], hcat, training_data)
    #     inp = mapreduce(x -> x[2], hcat, training_data)        
    # end

    for i in 1:n_internal_loop
    # for (out,in) in training_data
        lss, grads = Flux.withgradient(neural_net) do nn
            sqrt(Flux.mse(out, nn(inp)))
        end

        Flux.update!(optim, neural_net, grads[1])

        push!(losses, lss)  # logging, outside gradient context
    end

    Flux.adjust!(optim, ParameterSchedulers.next!(s))

    if epoch % 100 == 0 println("Epoch: $epoch; Loss: $(sum(losses[end-99:end])/100); Opt state: $(optim.layers[1].weight.rule)") end
end

plot(losses[500:end], yaxis=:log)

outputs = training_data[1][1]
inputs = training_data[1][2]

# norm((outputs - neural_net(inputs)) .* stddev) / norm(outputs .* stddev .+ mn)

norm(outputs - neural_net(inputs)) / norm(outputs)

# maximum((outputs[:,1] .* stddev - neural_net(inputs[:,1]) .* stddev))


# does it converge to a steady state
stt = Float32.(zero(outputs[:,1]))
shck = zeros(Float32,n_shocks)
for i in 1:100000
    stt = neural_net(vcat(stt, shck))
end