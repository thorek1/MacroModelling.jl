using MacroModelling
using Flux
using Optim
using FluxOptTools
using LinearAlgebra
include("../models/Smets_Wouters_2007.jl")

n_shocks = length(get_shocks(Smets_Wouters_2007))

n_vars = length(get_variables(Smets_Wouters_2007))

n_hidden = 128

neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, asinh),
                    Dense(n_hidden, n_hidden, asinh),
                    Dense(n_hidden, n_hidden, tanh),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_hidden, celu),
                    Dense(n_hidden, n_vars))

# neural_net = Chain( Dense(n_vars + n_shocks, n_hidden, celu),
#                     Dense(n_hidden, n_hidden, celu),
#                     Dense(n_hidden, n_hidden, celu),
#                     Dense(n_hidden, n_hidden, celu),
#                     Dense(n_hidden, n_hidden, celu),
#                     Dense(n_hidden, n_hidden, celu),
#                     Dense(n_hidden, n_vars))

# for i in 1:10
    n_simul = 10000
    n_burnin = 500

    shcks = randn(n_shocks, n_burnin + n_simul)

    sims = get_irf(Smets_Wouters_2007, shocks = shcks, periods = 0, levels = true)

    
    sim_slices = vcat(collect(sims[:,n_burnin:n_burnin + n_simul - 1, 1]), shcks[:,n_burnin + 1:n_burnin + n_simul])

    out_slices = collect(sims[:,n_burnin + 1:n_burnin + n_simul,1])
    
    loss() = sqrt(sum(abs2, out_slices - neural_net(sim_slices)))

    

    pars   = Flux.params(neural_net)
    lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, show_trace=true))
# end

out_slices[:,1] - neural_net(sim_slices[:,1])


minimum(neural_net(sim_slices[:,1]))
minimum(out_slices[:,1])

norm(out_slices[:,1] - neural_net(sim_slices[:,1])) / norm(out_slices[:,1])