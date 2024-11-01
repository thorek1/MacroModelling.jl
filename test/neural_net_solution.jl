using Revise
using MacroModelling
using Flux
using ParameterSchedulers
using Optim
using FluxOptTools
using StatsPlots
using Sobol
using HDF5
using BSON
using Random

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())

## Settings
normalise = true  # use asinh and tanh at the beginning if there is no normalisation
recurrent = false # the internal state needs to be reset but carries a lot of information
pretrain = false

## Generate dataset

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

model_parameters = Symbol.(get_parameters(model))

n_model_parameters = length(model_parameters)

n_points_per_parameter_dimension = 100

n_parameter_draws = n_points_per_parameter_dimension ^ (n_model_parameters ÷ 4)

n_time_steps = 20

n_burnin = 500

n_shocks = length(get_shocks(model))

n_vars = length(get_variables(model))

n_inputs = n_vars + n_shocks + n_model_parameters

sob = SobolSeq(n_model_parameters)

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


outputs = zeros(Float32, n_vars, n_time_steps * n_parameter_draws)
inputs = zeros(Float32, n_inputs, n_time_steps * n_parameter_draws)

Rnadom.seed!(14124)

for i in 1:n_parameter_draws
    draw = next!(sob)

    transformed_draw = draw .* bounds_range .+ lower_bounds_par
    
    normalised_draw = (draw .- 0.5) .* sqrt(12)

    shcks = randn(n_shocks, n_burnin + n_time_steps)
    
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
            
            normalised_sims = (sims[:,n_burnin:end,1] .- mn) ./ stddev

            if maximum(abs.(normalised_sims)) > 10 || any(!isfinite, normalised_sims)
                draw = next!(sob)
                
                transformed_draw = draw .* bounds_range .+ lower_bounds_par

                normalised_draw = (draw .- 0.5) .* sqrt(12)

                continue 
            end
            
            inputs[1:n_vars,1+(i-1)*n_time_steps:i*n_time_steps] = normalised_sims[:,1:end - 1]
            inputs[n_vars+1:n_vars+n_shocks,1+(i-1)*n_time_steps:i*n_time_steps] = shcks[:,n_burnin + 1:n_burnin + n_time_steps]
            inputs[n_vars+n_shocks+1:end,1+(i-1)*n_time_steps:i*n_time_steps] = reshape(repeat(normalised_draw, n_time_steps), length(normalised_draw), n_time_steps)

            outputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = normalised_sims[:,2:end]
        else
            inputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = vcat(collect(sims[:,n_burnin:n_burnin + n_time_steps - 1,1]), shcks[:,n_burnin + 1:n_burnin + n_time_steps], reshape(repeat(normalised_draw, n_time_steps), length(normalised_draw), n_time_steps))
            
            outputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = sims[:,n_burnin+1:n_burnin + n_time_steps,1]
        end

        # push!(training_data, (outputs, inputs))

        solved = true
    end
end

outputs /= 6
# outputs .+= .5

inputs /= 6
# inputs .+= .5

# h5write("data.h5", "inputs", inputs)
# h5write("data.h5", "outputs", outputs)


## Create Neural Network
n_hidden = max(256, n_vars * 2)
n_hidden_small = max(256, n_vars * 2)

Random.seed!(6794)

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
        neural_net = Chain( Dense(n_inputs, n_hidden),
                            Dense(n_hidden, n_hidden, leakyrelu), # going to 256 brings it down to .0016
                            Dense(n_hidden, n_hidden_small, tanh_fast), # without these i get to .0032 and relnorm .0192
                            Dense(n_hidden_small, n_hidden_small, leakyrelu), # without these i get to .0032 and relnorm .0192, with these it goes to .002 and .0123
                            Dense(n_hidden_small, n_hidden_small, tanh_fast),
                            Dense(n_hidden_small, n_hidden_small, leakyrelu),
                            Dense(n_hidden_small, n_vars))
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

# Pretrain with L-BFGS
if pretrain
    n_pretrain_epochs = 2000
    n_pretrain_batches = 128

    pretrain_loader = Flux.DataLoader((outputs, inputs), batchsize = (n_time_steps * n_parameter_draws) ÷ n_pretrain_batches, shuffle = true)

    for (out,inp) in pretrain_loader
        loss_func() = sqrt(Flux.mse(out, neural_net(inp)))
        pars   = Flux.params(neural_net)
        lossfun, gradfun, fg!, p0 = optfuns(loss_func, pars)
        res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=n_pretrain_epochs, show_trace=true))
        if loss_func() < 1e-2 break end
    end

    # Save and load model

    BSON.@save "post_LBFGS.bson" neural_net

    # BSON.@load "post_LBFGS.bson" neural_net
end



# Setup optimiser

n_epochs = 100 # 1000 goes to .0016; 300 goes to .0023

# optim = Flux.setup(Flux.Adam(), neural_net)
optim = Flux.setup(Flux.Optimiser(Flux.ClipNorm(1), Flux.AdamW()), neural_net)

# lr_start = 1e-3
# lr_end = 1e-10

# eta_sched = ParameterSchedulers.Stateful(CosAnneal(lr_start, lr_end, n_epochs * n_batches))

lr_start = 1e-3
lr_end   = 1e-10

# degree = (log(lr_start) - log(lr_end)) / log((1 - (n_epochs - 1) / n_epochs))

eta_sched = ParameterSchedulers.Stateful(CosAnneal(lr_start, lr_end, n_epochs))
# eta_sched = ParameterSchedulers.Stateful(Exp(start = lr_start, decay = (lr_end / lr_start) ^ (1 / n_epochs)))
# eta_sched = ParameterSchedulers.Stateful(Poly(start = lr_start, degree = 3, max_iter = n_epochs))

# decay_sched = ParameterSchedulers.Stateful(CosAnneal(.00001, 1e-10, n_epochs * n_batches))
# s = ParameterSchedulers.Stateful(Sequence([  CosAnneal(.001, 1e-5, 5000), 
#                                             Exp(start = 1e-5, decay = .9995), 
#                                             Exp(start = 1e-6, decay = .999)],
#                                 [scheduler_period ÷ 3, scheduler_period ÷ 3, scheduler_period ÷ 3]))


# Training loop

batchsize = 512

train_loader = Flux.DataLoader((outputs, inputs), batchsize = batchsize, shuffle = true)

n_batches = length(train_loader)

print_every = 10
# print_every = 100000 ÷ batchsize 

losses = []
for epoch in 1:n_epochs
    for (out,inp) in train_loader
        lss, grads = Flux.withgradient(neural_net) do nn
            sqrt(Flux.mse(out, nn(inp)))
        end

        Flux.update!(optim, neural_net, grads[1])

        push!(losses, lss)  # logging, outside gradient context

        # if length(losses) % print_every == 0 println("Epoch: $epoch; Loss: $(sum(losses[end-print_every+1:end])/print_every); η: $(optim.layers[1].weight.rule.opts[2].opts[1].eta); λ: $(optim.layers[1].weight.rule.opts[2].opts[2].lambda)") end
    end

    if epoch % print_every == 0 println("Epoch: $epoch; Loss: $(sum(losses[end-n_batches * print_every+1:end])/(n_batches*print_every)); η: $(optim.layers[1].weight.rule.opts[2].opts[1].eta); λ: $(optim.layers[1].weight.rule.opts[2].opts[2].lambda)") end

    sched_update = ParameterSchedulers.next!(eta_sched)

    Flux.adjust!(optim; eta = sched_update)
    Flux.adjust!(optim; lambda = sched_update * 0.01)
end


# BSON.@save "post_ADAM.bson" neural_net

# BSON.@load "post_ADAM.bson" neural_net


plot(losses[500:end], yaxis=:log)

eta_sched_plot = ParameterSchedulers.Stateful(CosAnneal(lr_start, lr_end, n_epochs * length(train_loader)))
# eta_sched_plot = ParameterSchedulers.Stateful(Exp(start = lr_start, decay = (lr_end / lr_start) ^ (1 / (n_epochs * length(train_loader)))))
# eta_sched_plot = ParameterSchedulers.Stateful(Poly(start = lr_start, degree = 2, max_iter = n_epochs * length(train_loader)))

lr = [ParameterSchedulers.next!(eta_sched_plot) for i in 1:n_epochs*length(train_loader)]

plot!(twinx(),lr[500:end], yaxis=:log, label = "Learning rate", lc = "black")


# [lr_start * (1 - (t - 1) / (n_epochs))^1.5 for t in 1:n_epochs]

# plot([lr_start * (1 - (t - 1) / (n_epochs))^1.5 for t in 1:n_epochs]
# , yaxis=:log, label = "Learning rate", lc = "black")

# plot!([lr_start * (1 - (t - 1) / (n_epochs))^3 for t in 1:n_epochs]
# , yaxis=:log, label = "Learning rate", lc = "black")



# norm((outputs - neural_net(inputs)) .* stddev) / norm(outputs .* stddev .+ mn)

norm(outputs - neural_net(inputs)) / norm(outputs)

maximum(abs, outputs - neural_net(inputs))
sum(abs, outputs - neural_net(inputs)) / length(outputs)
sum(abs2, outputs - neural_net(inputs)) / length(outputs)
# maximum((outputs[:,1] .* stddev - neural_net(inputs[:,1]) .* stddev))

model_state = Flux.state(model)

jldsave("post_ADAM.jld2"; model_state)


# does it converge to a steady state
stt = Float32.(zero(outputs[:,1]))
shck = zeros(Float32,n_shocks)
for i in 1:100000
    stt = neural_net(vcat(stt, shck))
end




### old code

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
