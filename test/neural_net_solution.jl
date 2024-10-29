using MacroModelling
using Flux
# using FluxKAN
using ParameterSchedulers
using Optim
using FluxOptTools
using LinearAlgebra

include("../models/Smets_Wouters_2007.jl")

n_shocks = length(get_shocks(Smets_Wouters_2007))

n_vars = length(get_variables(Smets_Wouters_2007))

n_hidden = 64

neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, asinh),
                    Dense(n_hidden, n_hidden, asinh),
                    Dense(n_hidden, n_hidden, tanh),
                    # Dense(n_hidden, n_hidden, celu),
                    # Dense(n_hidden, n_hidden, celu),
                    # Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_vars))

n_hidden = 64

neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, asinh),
                    Flux.LSTM(n_hidden, n_hidden ÷ 2),
                    Flux.GRU(n_hidden ÷ 2, n_hidden ÷ 2), # optional
                    Dense(n_hidden ÷ 2, n_hidden ÷ 2, celu),
                    Dense(n_hidden ÷ 2, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu), # optional
                    Dense(n_hidden, n_vars))
  
s = ParameterSchedulers.Stateful(CosAnneal(.001, 1e-6, 500))
# s = ParameterSchedulers.Stateful(SinDecay2(.001, 1e-6, 500))

# optim = Flux.setup(Flux.AdamW(0.001, (0.9, 0.999), 0.01), neural_net)  # will store optimiser momentum, etc.

optim = Flux.setup(Flux.Adam(), neural_net)  # will store optimiser momentum, etc.


# for i in 1:10
n_simul = 1000
n_burnin = 500

shcks = randn(n_shocks, n_burnin + n_simul)

sims = get_irf(Smets_Wouters_2007, shocks = shcks, periods = 0, levels = true)

normalised_sims = Flux.normalise(collect(sims[:,n_burnin:end,1]), dims=1)

normalised_sim_slices = Float32.(vcat(normalised_sims[:,1:end - 1], shcks[:,n_burnin + 1:n_burnin + n_simul]))

normalised_out_slices = Float32.(normalised_sims[:,2:end])

# loss() = sqrt(sum(abs2, out_slices - neural_net(sim_slices)))
# loss() = Flux.mse(neural_net(sim_slices), out_slices)
  
# Training loop, using the whole data set 1000 times:
losses = []
for epoch in 1:5000
    # for (x, y) in loader
        lss, grads = Flux.withgradient(neural_net) do nn
            # Evaluate model and loss inside gradient context:
            sqrt(Flux.mse(nn(normalised_sim_slices), normalised_out_slices))
        end
        Flux.adjust!(optim, ParameterSchedulers.next!(s))
        Flux.update!(optim, neural_net, grads[1])
        push!(losses, loss)  # logging, outside gradient context
        if epoch % 100 == 0 println("Epoch: $epoch; Loss: $lss; Opt state: $(optim.layers[1].weight.rule)") end
    # end
end


sqrt(sum(abs2, normalised_out_slices - neural_net(normalised_sim_slices)) / (n_simul * n_vars)) # RMSE
sqrt(Flux.mse(neural_net(normalised_sim_slices), normalised_out_slices))

pars   = Flux.params(neural_net)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=100, show_trace=true))
# end



sim_slices = Float32.(vcat(collect(sims[:,n_burnin:n_burnin + n_simul - 1,1]), shcks[:,n_burnin + 1:n_burnin + n_simul]))

out_slices = Float32.(collect(sims[:,n_burnin+1:n_burnin + n_simul,1]))


maximum((normalised_out_slices - neural_net(normalised_sim_slices))[:,1])


maximum(neural_net(sim_slices[:,1]))
maximum(out_slices[:,1])

norm(normalised_out_slices - neural_net(normalised_sim_slices)) / norm(normalised_out_slices)