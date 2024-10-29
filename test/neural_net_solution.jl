using MacroModelling
using Flux
# using FluxKAN
using ParameterSchedulers
using Optim
using FluxOptTools
using StatsPlots

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())

include("../models/Smets_Wouters_2007.jl")


normalise = true  # use asinh and tanh at the beginning if there is no normalisation
recurrent = false # the internal state needs to be reset but carries a lot of information

n_shocks = length(get_shocks(Smets_Wouters_2007))

n_vars = length(get_variables(Smets_Wouters_2007))

n_hidden = n_vars * 2

if recurrent
    neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, asinh),
                        Flux.LSTM(n_hidden, n_hidden ÷ 2),
                        Flux.GRU(n_hidden ÷ 2, n_hidden ÷ 2), # optional
                        Dense(n_hidden ÷ 2, n_hidden ÷ 2, celu),
                        Dense(n_hidden ÷ 2, n_hidden, celu),
                        Dense(n_hidden, n_hidden, celu), # optional
                        Dense(n_hidden, n_vars))   
else
    if normalise
        neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_hidden, celu),
                            Dense(n_hidden, n_vars))
    else
        neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, asinh),
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

n_batches = 10
n_simul = n_batches * nn_params ÷ (n_vars * 10)
n_burnin = 500
scheduler_period = 15000

s = ParameterSchedulers.Stateful(CosAnneal(.001, 1e-8, scheduler_period))
# s = ParameterSchedulers.Stateful(SinDecay2(.001, 1e-6, 500))

shcks = randn(n_shocks, n_burnin + n_simul)

sims = get_irf(Smets_Wouters_2007, shocks = shcks, periods = 0, levels = true)

if normalise
    mn = get_mean(Smets_Wouters_2007, derivatives = false)
    
    stddev = get_std(Smets_Wouters_2007, derivatives = false)
    
    normalised_sims = collect((sims[:,n_burnin:end,1] .- mn) ./ stddev)

    inputs = Float32.(vcat(normalised_sims[:,1:end - 1], shcks[:,n_burnin + 1:n_burnin + n_simul]))

    outputs = Float32.(normalised_sims[:,2:end])
else
    inputs = Float32.(vcat(collect(sims[:,n_burnin:n_burnin + n_simul - 1,1]), shcks[:,n_burnin + 1:n_burnin + n_simul]))
    
    outputs = Float32.(collect(sims[:,n_burnin+1:n_burnin + n_simul,1]))  
end

train_loader = Flux.DataLoader((outputs, inputs), batchsize = n_simul ÷ n_batches, shuffle = true)

losses = []
# Training loop
for epoch in 1:scheduler_period
    for (out,in) in train_loader
        lss, grads = Flux.withgradient(neural_net) do nn
            sqrt(Flux.mse(out, nn(in)))
        end

        Flux.update!(optim, neural_net, grads[1])

        push!(losses, lss)  # logging, outside gradient context
    end

    Flux.adjust!(optim, ParameterSchedulers.next!(s))

    if epoch % 100 == 0 println("Epoch: $epoch; Loss: $(sum(losses[end-99:end])/100); Opt state: $(optim.layers[1].weight.rule)") end
end

plot(losses[500:end], yaxis=:log)

norm((outputs - neural_net(inputs)) .* stddev) / norm(outputs .* stddev .+ mn)

norm(outputs - neural_net(inputs)) / norm(outputs)

maximum((outputs[:,1] .* stddev - neural_net(inputs[:,1]) .* stddev))


# does it converge to a steady state
stt = Float32.(zero(outputs[:,1]))
shck = zeros(Float32,n_shocks)
for i in 1:100000
    stt = neural_net(vcat(stt, shck))
end