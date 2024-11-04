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
using HyperbolicCrossApprox
using Zygote

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

algorithm = :pruned_second_order

model_parameters = Symbol.(get_parameters(model))

n_model_parameters = length(model_parameters)

n_points_per_parameter_dimension = 100

n_parameter_draws = n_points_per_parameter_dimension ^ (n_model_parameters ÷ 4)

n_time_steps = 20

n_burnin = 500

n_shocks = length(get_shocks(model))

n_vars = length(get_variables(model))

n_inputs = n_vars + n_shocks + n_model_parameters# + length(model.par_calib_list)

n_gradient_evals = 100000

scaling_factor = 1

sob = SobolSeq(n_model_parameters)

lower_bounds_par = model.parameter_values .- 0.01
upper_bounds_par = model.parameter_values .+ 0.01

# lower_bounds_par = [0
#                     0
#                     -2
#                     0.2
#                     8
#                     0.25
#                     0.25
#                     0.15
#                     0.1] .+ eps()

# upper_bounds_par = [1
#                     1
#                     3
#                     0.3
#                     12
#                     1
#                     1
#                     0.25
#                     0.5] .- eps()


lower_bounds_par = [0.01
                    0.01
                    .5
                    0.2
                    9
                    0.5
                    0.5
                    0.15
                    0.2] .+ eps()

upper_bounds_par = [.5
                    .5
                    1.5
                    0.3
                    11
                    .99
                    .99
                    0.25
                    0.4] .- eps()

bounds_range = upper_bounds_par .- lower_bounds_par


outputs = zeros(Float32, n_vars, n_time_steps * n_parameter_draws)
inputs = zeros(Float32, n_inputs, n_time_steps * n_parameter_draws)

# orig_sims = zeros(Float32, n_vars, n_time_steps * n_parameter_draws)

calibration_parameters = zeros(Float32, length(model.par_calib_list), n_time_steps * n_parameter_draws)

variables_scale = zeros(Float32, n_vars, n_time_steps * n_parameter_draws)
variables_bias = zeros(Float32, n_vars, n_time_steps * n_parameter_draws)

Random.seed!(14124)

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
                            algorithm = algorithm,
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
            stst = get_ss(model, 
                            algorithm = algorithm,
                            derivatives = false)

            mn = get_mean(model, 
                            algorithm = algorithm,
                            derivatives = false)
            
            stddev = get_std(model, 
                            algorithm = algorithm,
                            derivatives = false)
            
            normalised_sims = (sims[:,n_burnin:end,1] .- mn) ./ stddev

            variables_scale[:,1+(i-1)*n_time_steps:i*n_time_steps] = repeat(stddev, n_time_steps)
            variables_bias[:,1+(i-1)*n_time_steps:i*n_time_steps]  = repeat(mn, n_time_steps)

            if maximum(abs.(normalised_sims)) > 10 || any(!isfinite, normalised_sims) || any(sims[[1,3,4],:,:] .< 0)
                draw = next!(sob)
                
                transformed_draw = draw .* bounds_range .+ lower_bounds_par

                normalised_draw = (draw .- 0.5) .* sqrt(12)

                continue 
            end
            
            # orig_sims[:,1+(i-1)*n_time_steps:i*n_time_steps] = sims[:,n_burnin:end-1,1]

            inputs[1:n_vars,1+(i-1)*n_time_steps:i*n_time_steps] = normalised_sims[:,1:end - 1]

            inputs[n_vars+1:n_vars+n_shocks,1+(i-1)*n_time_steps:i*n_time_steps] = shcks[:,n_burnin + 1:n_burnin + n_time_steps]

            inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,1+(i-1)*n_time_steps:i*n_time_steps] = reshape(repeat(normalised_draw, n_time_steps), length(normalised_draw), n_time_steps)

            calibration_parameters[:,1+(i-1)*n_time_steps:i*n_time_steps] = reshape(repeat(stst[n_vars+1:end], n_time_steps), length(stst)-n_vars, n_time_steps)

            outputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = normalised_sims[:,2:end]
        else
            inputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = vcat(collect(sims[:,n_burnin:n_burnin + n_time_steps - 1,1]), shcks[:,n_burnin + 1:n_burnin + n_time_steps], reshape(repeat(normalised_draw, n_time_steps), length(normalised_draw), n_time_steps))
            
            outputs[:,1+(i-1)*n_time_steps:i*n_time_steps] = sims[:,n_burnin+1:n_burnin + n_time_steps,1]
        end

        # push!(training_data, (outputs, inputs))

        solved = true
    end
end

outputs /= scaling_factor
# outputs .+= .5

inputs /= scaling_factor
# inputs .+= .5

# h5write("data.h5", "inputs", inputs)
# h5write("data.h5", "outputs", outputs)




lr_start = 1e-3
lr_end   = 1e-10
n_hidden = 128
batchsize = 128

n_epochs = n_gradient_evals * batchsize ÷ (n_parameter_draws * n_time_steps)

activation = :gelu
schedule = :cos
optimiser = :adam
n_layers = 5

results = []
Random.seed!(6794)

if activation == :relu
    act = leakyrelu
elseif activation == :tanh
    act = tanh_fast
elseif activation == :celu
    act = celu
elseif activation == :gelu
    act = gelu
elseif activation == :swish
    act = swish
elseif activation == :mish
    act = mish
end

intermediate_layers = [Dense(n_hidden, n_hidden, act) for i in 1:n_layers]
neural_net_approx = Chain( Dense(n_inputs, n_hidden), intermediate_layers..., Dense(n_hidden, n_vars))

# Setup optimiser

# n_epochs = 100 # 1000 goes to .0016; 300 goes to .0023

if optimiser == :adam
    optim = Flux.setup(Flux.Adam(), neural_net_approx)
elseif optimiser == :adamw
    optim = Flux.setup(Flux.Optimiser(Flux.ClipNorm(1), Flux.AdamW()), neural_net_approx)
end

# lr_start = 1e-3
# lr_end   = 1e-10


# Training loop

# batchsize = 512

train_loader = Flux.DataLoader((outputs, inputs), batchsize = batchsize, shuffle = true)

n_batches = length(train_loader)

n_epochs = n_gradient_evals ÷ n_batches

if schedule == :cos
    eta_sched = ParameterSchedulers.Stateful(CosAnneal(lr_start, lr_end, n_epochs * n_batches))
elseif schedule == :exp
    eta_sched = ParameterSchedulers.Stateful(Exp(start = lr_start, decay = (lr_end / lr_start) ^ (1 / (n_epochs * n_batches))))
elseif schedule == :poly
    eta_sched = ParameterSchedulers.Stateful(Poly(start = lr_start, degree = 3, max_iter = n_epochs * n_batches))
end

start_time = time()
losses = []
for epoch in 1:n_epochs
    for (out,inp) in train_loader
        lss, grads = Flux.withgradient(neural_net_approx) do nn
            sqrt(Flux.mse(out, nn(inp)))
        end

        Flux.update!(optim, neural_net_approx, grads[1])

        sched_update = ParameterSchedulers.next!(eta_sched)

        Flux.adjust!(optim; eta = sched_update)
        Flux.adjust!(optim; lambda = sched_update * 0.01)

        push!(losses, lss)  # logging, outside gradient context

        if length(losses) % 100 == 0 && length(losses) > 100  println("Epoch: $epoch - Gradient calls: $(length(losses)) - Loss: $(sum(losses[end-100:end])/(100))") end
    end
end
end_time = time()  # Record end time
elapsed_time = end_time - start_time

relnorm = norm(outputs - neural_net_approx(inputs)) / norm(outputs)

# 0.0044 - 10
# 0.0029 - 3
# 0.0024 - 1

function calculate_loss(variables₍₋₁₎::Matrix{R}, 
                        variables₍₀₎::Matrix{R}, 
                        variables₍₁₎::Matrix{R}, 
                        shocks₍ₓ₎::Matrix{R}, 
                        model_parameters::Matrix{R}, 
                        calibration_parameters::Matrix{R}) where R <: Real
    c₍₋₁₎   = variables₍₋₁₎[1,:]
    g₍₋₁₎   = variables₍₋₁₎[2,:]
    k₍₋₁₎   = variables₍₋₁₎[3,:]
    l₍₋₁₎   = variables₍₋₁₎[4,:]
    w₍₋₁₎   = variables₍₋₁₎[5,:]
    z₍₋₁₎   = variables₍₋₁₎[6,:]

    c₍₀₎    = variables₍₀₎[1,:]
    g₍₀₎    = variables₍₀₎[2,:]
    k₍₀₎    = variables₍₀₎[3,:]
    l₍₀₎    = variables₍₀₎[4,:]
    w₍₀₎    = variables₍₀₎[5,:]
    z₍₀₎    = variables₍₀₎[6,:]

    c₍₁₎    = variables₍₁₎[1,:]
    g₍₁₎    = variables₍₁₎[2,:]
    k₍₁₎    = variables₍₁₎[3,:]
    l₍₁₎    = variables₍₁₎[4,:]
    w₍₁₎    = variables₍₁₎[5,:]
    z₍₁₎    = variables₍₁₎[6,:]

    ϵᵍ₍ₓ₎   = shocks₍ₓ₎[1,:]
    ϵᶻ₍ₓ₎   = shocks₍ₓ₎[2,:]

    σᶻ      = model_parameters[1,:]
    σᵍ      = model_parameters[2,:]
    σ       = model_parameters[3,:]
    i_y     = model_parameters[4,:]
    k_y     = model_parameters[5,:]
    ρᶻ      = model_parameters[6,:]
    ρᵍ      = model_parameters[7,:]
    g_y     = model_parameters[8,:]
    α       = model_parameters[9,:]

    ḡ       = calibration_parameters[1,:]
    ψ       = calibration_parameters[2,:]

    δ = @.(i_y / k_y)
    β = @.(1 / (α / k_y + (1 - δ)))

    loss = zero(eltype(inputs))

    loss += sum(abs2, min.(eps(eltype(inputs)), c₍₋₁₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), k₍₋₁₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), l₍₋₁₎))

    loss += sum(abs2, min.(eps(eltype(inputs)), c₍₀₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), k₍₀₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), l₍₀₎))

    loss += sum(abs2, min.(eps(eltype(inputs)), c₍₁₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), k₍₁₎))
    loss += sum(abs2, min.(eps(eltype(inputs)), l₍₁₎))

    c₍₋₁₎ = max.(eps(eltype(inputs)), c₍₋₁₎)
    k₍₋₁₎ = max.(eps(eltype(inputs)), k₍₋₁₎)
    l₍₋₁₎ = max.(eps(eltype(inputs)), l₍₋₁₎)

    c₍₀₎ = max.(eps(eltype(inputs)), c₍₀₎)
    k₍₀₎ = max.(eps(eltype(inputs)), k₍₀₎)
    l₍₀₎ = max.(eps(eltype(inputs)), l₍₀₎)

    c₍₁₎ = max.(eps(eltype(inputs)), c₍₁₎)
    k₍₁₎ = max.(eps(eltype(inputs)), k₍₁₎)
    l₍₁₎ = max.(eps(eltype(inputs)), l₍₁₎)


    loss += sum(abs2, @.(c₍₀₎ ^ -σ - β * c₍₁₎ ^ -σ * ((α * z₍₁₎ * (k₍₀₎ / l₍₁₎) ^ (α - 1) + 1) - δ)))
    loss += sum(abs2, @.(((ψ * c₍₀₎ ^ σ) / (1 - l₍₀₎) - w₍₀₎)))
    loss += sum(abs2, @.((k₍₀₎ - ((((1 - δ) * k₍₋₁₎ + z₍₀₎ * k₍₋₁₎ ^ α * l₍₀₎ ^ (1 - α)) - g₍₀₎) - c₍₀₎))))
    loss += sum(abs2, @.((w₍₀₎ - z₍₀₎ * (1 - α) * (k₍₋₁₎ / l₍₀₎) ^ α)))
    loss += sum(abs2, @.((z₍₀₎ - ((1 - ρᶻ) + ρᶻ * z₍₋₁₎ + σᶻ * ϵᶻ₍ₓ₎))))
    loss += sum(abs2, @.((g₍₀₎ - ((1 - ρᵍ) * ḡ + ρᵍ * g₍₋₁₎ + σᵍ * ϵᵍ₍ₓ₎))))

    return loss
end


domain = fill(3.0,n_shocks,2)
domain[2,:] *= -1

shock_grid, _ = hyperbolic_cross_grid(chebyshev_nodes, n_shocks, 2, domain)
shock_grid = Float32.(shock_grid)
# scatter(shock_grid[:,1], shock_grid[:,2])







if activation == :relu
    act = leakyrelu
elseif activation == :tanh
    act = tanh_fast
elseif activation == :celu
    act = celu
elseif activation == :gelu
    act = gelu
elseif activation == :swish
    act = swish
end

intermediate_layers = [Dense(n_hidden, n_hidden, act) for i in 1:n_layers]
neural_net = Chain( Dense(n_inputs, n_hidden), intermediate_layers..., Dense(n_hidden, n_vars))

# Setup optimiser
if optimiser == :adam
    optim = Flux.setup(Flux.Adam(), neural_net)
elseif optimiser == :adamw
    optim = Flux.setup(Flux.Optimiser(Flux.ClipNorm(1), Flux.AdamW()), neural_net)
end

# Training loop

train_loader_nonlinear = Flux.DataLoader((inputs, variables_scale, variables_bias, calibration_parameters), batchsize = batchsize, shuffle = true)

n_batches = length(train_loader_nonlinear)

n_epochs = n_gradient_evals ÷ n_batches

if schedule == :cos
    eta_sched = ParameterSchedulers.Stateful(CosAnneal(lr_start, lr_end, n_epochs * n_batches))
elseif schedule == :exp
    eta_sched = ParameterSchedulers.Stateful(Exp(start = lr_start, decay = (lr_end / lr_start) ^ (1 / (n_epochs * n_batches))))
elseif schedule == :poly
    eta_sched = ParameterSchedulers.Stateful(Poly(start = lr_start, degree = 3, max_iter = n_epochs * n_batches))
end

start_time = time()
losses = []

for epoch in 1:n_epochs
    for (inp, var_scale, var_bias, calib_pars) in train_loader_nonlinear
        lss, grads = Flux.withgradient(neural_net) do nn
            variables₍₋₁₎ = inp[1:n_vars,:] * scaling_factor .* var_scale .+ var_bias

            shocks₍ₓ₎ = inp[n_vars+1:n_vars+n_shocks,:]

            normalised_parameters = inp[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:]

            model_parameters = (normalised_parameters / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)

            normalised_variables₍₀₎ = nn(inp)

            variables₍₀₎ = normalised_variables₍₀₎ * scaling_factor .* var_scale .+ var_bias

            normalised_variables₍₁₎ = nn(vcat(normalised_variables₍₀₎, repeat(randn(Float32,2),1,size(inp,2)), normalised_parameters))

            for shck in eachrow(shock_grid)
                normalised_variables₍₁₎ += nn(vcat(normalised_variables₍₀₎, repeat(shck,1,size(inp,2)), normalised_parameters))
            end

            variables₍₁₎ = normalised_variables₍₁₎ / (size(shock_grid,1) + 1) * scaling_factor .* var_scale .+ var_bias
            
            return sqrt(calculate_loss(variables₍₋₁₎, variables₍₀₎, variables₍₁₎, shocks₍ₓ₎, model_parameters, calib_pars) / length(variables₍₀₎))
        end

        Flux.update!(optim, neural_net, grads[1])

        sched_update = ParameterSchedulers.next!(eta_sched)

        Flux.adjust!(optim; eta = sched_update)
        Flux.adjust!(optim; lambda = sched_update * 0.01)

        push!(losses, lss)  # logging, outside gradient context
        if length(losses) % 100 == 0 && length(losses) > 100  println("Epoch: $epoch - Gradient calls: $(length(losses)) - Loss: $(sum(losses[end-100:end])/(100))") end
    end
    # println("Epoch: $epoch - Loss: $(sum(losses[end-100:end])/(100))")
end

end_time = time()  # Record end time
elapsed_time = end_time - start_time

relnorm = norm(outputs - neural_net(inputs)) / norm(outputs)

relnorm = norm(outputs - neural_net_approx(inputs)) / norm(outputs)

sqrt(sum(abs2,outputs - neural_net_approx(inputs)) / length(outputs))

neural_net_approx(inputs)
neural_net(inputs)

variables₍₋₁₎ = inputs[1:n_vars,:] * scaling_factor .* variables_scale .+ variables_bias

shocks₍ₓ₎ = inputs[n_vars+1:n_vars+n_shocks,:]

normalised_parameters = inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:]

model_parameters = (normalised_parameters / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)

normalised_variables₍₀₎ = neural_net(inputs)

variables₍₀₎ = normalised_variables₍₀₎ * scaling_factor .* variables_scale .+ variables_bias

variables₍₁₎ = zeros(eltype(inputs), n_vars, size(inputs,2))

for shck in eachrow(shock_grid)
    variables₍₁₎ += neural_net(vcat(normalised_variables₍₀₎, repeat(shck,1,size(inputs,2)), normalised_parameters))
end

variables₍₁₎ = variables₍₁₎ / size(shock_grid,1) * scaling_factor .* variables_scale .+ variables_bias

sqrt(calculate_loss(variables₍₋₁₎, variables₍₀₎, variables₍₁₎, shocks₍ₓ₎, model_parameters, calibration_parameters) / length(variables₍₀₎))


sqrt(calculate_loss(variables₍₋₁₎, outputs, variables₍₁₎, shocks₍ₓ₎, model_parameters, calibration_parameters) / length(variables₍₀₎))


iter_inputs = copy(inputs)
iter_inputs[7:8,:] *= 0
for i in 1:100
    iter_inputs = vcat(neural_net(iter_inputs),iter_inputs[7:end,:])
end
iter_inputs[1:6,:] * scaling_factor .* variables_scale .+ variables_bias


iter_inputs = copy(inputs)
iter_inputs[7:8,:] *= 0
for i in 1:100
    iter_inputs = vcat(neural_net_approx(iter_inputs),iter_inputs[7:end,:])
end
iter_inputs[1:6,:] * scaling_factor .* variables_scale .+ variables_bias


iter_inputs[1:6,:] |> maximum
iter_inputs[1:6,:] |> minimum

iter_inputs[1:6,:] * scaling_factor .* variables_scale .+ variables_bias

inputs[1:n_vars,:]|>maximum
inputs[1:n_vars,:]|>minimum

inputs[n_vars+1:n_vars+n_shocks,:]|>maximum
inputs[n_vars+1:n_vars+n_shocks,:]|>minimum

inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:]|>maximum
inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:]|>minimum

inputs[n_vars+n_shocks+n_model_parameters:end,:]|>maximum
inputs[n_vars+n_shocks+n_model_parameters:end,:]|>minimum

outputs|>maximum


# inputs[1:n_vars,:] .* variables_scale .+ variables_bias

vars_trans = inputs[1:n_vars,:] * scaling_factor .* variables_scale .+ variables_bias

shocks = inputs[n_vars+1:n_vars+n_shocks,:]

pars_trans = (inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)

calib_pars = inputs[n_vars+n_shocks+n_model_parameters+1:end,:]

inputs₍₋₁₎ = vcat(vars_trans, shocks, pars_trans, calib_pars)

normalised_parameters = inputs[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:]

normalised_variables₍₀₎ = neural_net(inputs)

variables₍₀₎ = normalised_variables₍₀₎ * scaling_factor .* variables_scale .+ variables_bias

# norm(outputs * 6 .* variables_scale .+ variables_bias .- variables₍₀₎) / norm(variables₍₀₎)

# norm(outputs * 6 .* variables_scale .+ variables_bias .- variables₍₀₎) / norm(variables₍₀₎)


variables₍₁₎ = zeros(eltype(inputs), n_vars, size(inputs₍₀₎,2))

for shck in eachrow(shock_grid)
    inputs = vcat(normalised_variables₍₀₎, repeat(shck,1,size(inputs₍₀₎,2)), normalised_parameters)
    
    variables₍₁₎ += neural_net(inputs)
end

variables₍₀₎ = normalised_variables₍₀₎ * scaling_factor .* variables_scale .+ variables_bias

variables₍₁₎ = variables₍₁₎ * scaling_factor / size(shock_grid,1) .* variables_scale .+ variables_bias


sqrt(calculate_loss(inputs₍₋₁₎, calibration_parameters, variables₍₀₎, variables₍₁₎) / length(variables₍₀₎))





ststst = SS(model, algorithm = :pruned_second_order, derivatives = false)
variables₍₋₁₎ = collect(ststst[1:n_vars])[:,:]
variables₍₀₎ = collect(ststst[1:n_vars])[:,:]
variables₍₁₎ = collect(ststst[1:n_vars])[:,:] # this is different from (0) because the nonlinearities will push this away from the stochastic steady state. in essence you need to integrate over the shocks.
shocks₍ₓ₎ = zeros(n_shocks,1)
model_parameters = model.parameter_values[:,:]
calib_pars = collect(ststst[n_vars+1:end])[:,:]
calculate_loss(variables₍₋₁₎, variables₍₀₎, variables₍₁₎, shocks₍ₓ₎, model_parameters, calib_pars)





c₍₋₁₎   = variables₍₋₁₎[1,:]
g₍₋₁₎   = variables₍₋₁₎[2,:]
k₍₋₁₎   = variables₍₋₁₎[3,:]
l₍₋₁₎   = variables₍₋₁₎[4,:]
w₍₋₁₎   = variables₍₋₁₎[5,:]
z₍₋₁₎   = variables₍₋₁₎[6,:]

c₍₀₎    = variables₍₀₎[1,:]
g₍₀₎    = variables₍₀₎[2,:]
k₍₀₎    = variables₍₀₎[3,:]
l₍₀₎    = variables₍₀₎[4,:]
w₍₀₎    = variables₍₀₎[5,:]
z₍₀₎    = variables₍₀₎[6,:]

c₍₁₎    = variables₍₁₎[1,:]
g₍₁₎    = variables₍₁₎[2,:]
k₍₁₎    = variables₍₁₎[3,:]
l₍₁₎    = variables₍₁₎[4,:]
w₍₁₎    = variables₍₁₎[5,:]
z₍₁₎    = variables₍₁₎[6,:]

ϵᵍ₍ₓ₎   = shocks₍ₓ₎[1,:]
ϵᶻ₍ₓ₎   = shocks₍ₓ₎[2,:]

σᶻ      = model_parameters[1,:]
σᵍ      = model_parameters[2,:]
σ       = model_parameters[3,:]
i_y     = model_parameters[4,:]
k_y     = model_parameters[5,:]
ρᶻ      = model_parameters[6,:]
ρᵍ      = model_parameters[7,:]
g_y     = model_parameters[8,:]
α       = model_parameters[9,:]

ḡ       = calib_pars[1,:]
ψ       = calib_pars[2,:]

δ = @.(i_y / k_y)
β = @.(1 / (α / k_y + (1 - δ)))

loss = zero(eltype(inputs))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₋₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₋₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₋₁₎))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₀₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₀₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₀₎))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₁₎))

# loss *= 100000

c₍₋₁₎ = max.(eps(eltype(inputs)), c₍₋₁₎)
k₍₋₁₎ = max.(eps(eltype(inputs)), k₍₋₁₎)
l₍₋₁₎ = max.(eps(eltype(inputs)), l₍₋₁₎)

c₍₀₎ = max.(eps(eltype(inputs)), c₍₀₎)
k₍₀₎ = max.(eps(eltype(inputs)), k₍₀₎)
l₍₀₎ = max.(eps(eltype(inputs)), l₍₀₎)

c₍₁₎ = max.(eps(eltype(inputs)), c₍₁₎)
k₍₁₎ = max.(eps(eltype(inputs)), k₍₁₎)
l₍₁₎ = max.(eps(eltype(inputs)), l₍₁₎)


loss += sum(abs2, @.(c₍₀₎ ^ -σ - β * c₍₁₎ ^ -σ * ((α * z₍₁₎ * (k₍₀₎ / l₍₁₎) ^ (α - 1) + 1) - δ)))
loss += sum(abs2, @.(((ψ * c₍₀₎ ^ σ) / (1 - l₍₀₎) - w₍₀₎)))
loss += sum(abs2, @.((k₍₀₎ - ((((1 - δ) * k₍₋₁₎ + z₍₀₎ * k₍₋₁₎ ^ α * l₍₀₎ ^ (1 - α)) - g₍₀₎) - c₍₀₎))))
loss += sum(abs2, @.((w₍₀₎ - z₍₀₎ * (1 - α) * (k₍₋₁₎ / l₍₀₎) ^ α)))
loss += sum(abs2, @.((z₍₀₎ - ((1 - ρᶻ) + ρᶻ * z₍₋₁₎ + σᶻ * ϵᶻ₍ₓ₎))))
loss += sum(abs2, @.((g₍₀₎ - ((1 - ρᵍ) * ḡ + ρᵍ * g₍₋₁₎ + σᵍ * ϵᵍ₍ₓ₎))))





domain = fill(3.0,n_shocks,2)
domain[2,:] *= -1

shock_grid, _ = hyperbolic_cross_grid(chebyshev_nodes, n_shocks, 3, domain)
shock_grid = Float32.(shock_grid)
# scatter(shock_grid[:,1], shock_grid[:,2])


inputs₍₋₁₎ = copy(inputs)

inputs₍₋₁₎[1:n_vars,:] = inputs₍₀₎[1:n_vars,:] * 6 .* variables_scale .+ variables_bias

inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] = (inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)


inputs₍₀₎ = copy(inputs)
inputs₍₀₎[1:n_vars,:] = neural_net(inputs)

variables₍₁₎ = zero(variables₍₀₎)

for shck in eachrow(shock_grid)
    inputs₍₀₎[n_vars+1:n_vars+n_shocks,:] .= shck
    variables₍₁₎ += neural_net(inputs₍₀₎)
end

variables₍₀₎ = inputs₍₀₎[1:n_vars,:] * 6 .* variables_scale .+ variables_bias

variables₍₁₎ = variables₍₁₎ * 6 / size(shock_grid,1) .* variables_scale .+ variables_bias


calculate_loss(inputs₍₋₁₎, variables₍₀₎, variables₍₁₎)

inp, var_scale, var_bias = collect(train_loader_nonlinear)[1]

# for (inp, var_scale, var_bias) in train_loader_nonlinear

        inputs₍₋₁₎ = copy(inp)
        inputs₍₋₁₎ = Zygote.Buffer(inputs₍₋₁₎)

        inputs₍₋₁₎[1:n_vars,:] = inputs₍₋₁₎[1:n_vars,:] * 6 .* var_scale .+ var_bias

        inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] = (inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)

        inputs₍₋₁₎ = copy(inputs₍₋₁₎)

        inputs₍₀₎ = copy(inp)
        inputs₍₀₎ = Zygote.Buffer(inputs₍₀₎)
        inputs₍₀₎[1:n_vars,:] = neural_net(inp)
        inputs₍₀₎ = copy(inputs₍₀₎)

        variables₍₁₎ = zeros(eltype(inp), n_vars, batchsize)

        for shck in eachrow(shock_grid)
            # inputs₍₀₎ = Zygote.Buffer(inputs₍₀₎)
            # inputs₍₀₎[n_vars+1:n_vars+n_shocks,:] .= shck
            # inputs₍₀₎ = copy(inputs₍₀₎)
            inputs = vcat(inputs₍₀₎[1:n_vars,:], repeat(shck,1,batchsize), inputs₍₀₎[n_vars+n_shocks+1:end,:])
            # inputs = vcat(inputs[1:n_vars,:], inputs[n_vars+n_shocks+1:end,:])
            variables₍₁₎ += neural_net(inputs)
        end

        variables₍₀₎ = inputs₍₀₎[1:n_vars,:] * 6 .* var_scale .+ var_bias

        variables₍₁₎ = variables₍₁₎ * 6 / size(shock_grid,1) .* var_scale .+ var_bias
        
        return calculate_loss(inputs₍₋₁₎, variables₍₀₎, variables₍₁₎)
    # end


inputs₍₋₁₎ = copy(inputs)

inputs₍₋₁₎[1:n_vars,:] = inputs₍₀₎[1:n_vars,:] * 6 .* variables_scale .+ variables_bias

inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] = (inputs₍₋₁₎[n_vars+n_shocks+1:n_vars+n_shocks+n_model_parameters,:] / Float32(sqrt(12)) .+ 0.5f0) .* Float32.(bounds_range) .+ Float32.(lower_bounds_par)


inputs₍₀₎ = copy(inputs)
inputs₍₀₎[1:n_vars,:] = neural_net(inputs)

variables₍₁₎ = zero(variables₍₀₎)

for shck in eachrow(shock_grid)
    inputs₍₀₎[n_vars+1:n_vars+n_shocks,:] .= shck
    variables₍₁₎ += neural_net(inputs₍₀₎)
end

variables₍₀₎ = inputs₍₀₎[1:n_vars,:] * 6 .* variables_scale .+ variables_bias

variables₍₁₎ = variables₍₁₎ * 6 / size(shock_grid,1) .* variables_scale .+ variables_bias




c₍₋₁₎   = inputs₍₋₁₎[1,:]
g₍₋₁₎   = inputs₍₋₁₎[2,:]
k₍₋₁₎   = inputs₍₋₁₎[3,:]
l₍₋₁₎   = inputs₍₋₁₎[4,:]
w₍₋₁₎   = inputs₍₋₁₎[5,:]
z₍₋₁₎   = inputs₍₋₁₎[6,:]
ϵᵍ₍ₓ₎   = inputs₍₋₁₎[7,:]
ϵᶻ₍ₓ₎   = inputs₍₋₁₎[8,:]
σᶻ      = inputs₍₋₁₎[9,:]
σᵍ      = inputs₍₋₁₎[10,:]
σ       = inputs₍₋₁₎[11,:]
i_y     = inputs₍₋₁₎[12,:]
k_y     = inputs₍₋₁₎[13,:]
ρᶻ      = inputs₍₋₁₎[14,:]
ρᵍ      = inputs₍₋₁₎[15,:]
g_y     = inputs₍₋₁₎[16,:]
α       = inputs₍₋₁₎[17,:]
ḡ       = inputs₍₋₁₎[18,:]
ψ       = inputs₍₋₁₎[19,:]

c₍₀₎ = variables₍₀₎[1,:]
g₍₀₎ = variables₍₀₎[2,:]
k₍₀₎ = variables₍₀₎[3,:]
l₍₀₎ = variables₍₀₎[4,:]
w₍₀₎ = variables₍₀₎[5,:]
z₍₀₎ = variables₍₀₎[6,:]

c₍₁₎ = variables₍₁₎[1,:]
g₍₁₎ = variables₍₁₎[2,:]
k₍₁₎ = variables₍₁₎[3,:]
l₍₁₎ = variables₍₁₎[4,:]
w₍₁₎ = variables₍₁₎[5,:]
z₍₁₎ = variables₍₁₎[6,:]

δ = @.(i_y / k_y)
β = @.(1 / (α / k_y + (1 - δ)))

loss = zero(eltype(inputs))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₋₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₋₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₋₁₎))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₀₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₀₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₀₎))

loss += sum(abs2, min.(eps(eltype(inputs)), c₍₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), k₍₁₎))
loss += sum(abs2, min.(eps(eltype(inputs)), l₍₁₎))


c₍₋₁₎ .= max.(eps(eltype(inputs)), c₍₋₁₎)
k₍₋₁₎ .= max.(eps(eltype(inputs)), k₍₋₁₎)
l₍₋₁₎ .= max.(eps(eltype(inputs)), l₍₋₁₎)

c₍₀₎ .= max.(eps(eltype(inputs)), c₍₀₎)
k₍₀₎ .= max.(eps(eltype(inputs)), k₍₀₎)
l₍₀₎ .= max.(eps(eltype(inputs)), l₍₀₎)

c₍₁₎ .= max.(eps(eltype(inputs)), c₍₁₎)
k₍₁₎ .= max.(eps(eltype(inputs)), k₍₁₎)
l₍₁₎ .= max.(eps(eltype(inputs)), l₍₁₎)


loss += sum(abs2, @.(c₍₀₎ ^ -σ - β * c₍₁₎ ^ -σ * ((α * z₍₁₎ * (k₍₀₎ / l₍₁₎) ^ (α - 1) + 1) - δ)))
loss += sum(abs2, @.(((ψ * c₍₀₎ ^ σ) / (1 - l₍₀₎) - w₍₀₎)))
loss += sum(abs2, @.((k₍₀₎ - ((((1 - δ) * k₍₋₁₎ + z₍₀₎ * k₍₋₁₎ ^ α * l₍₀₎ ^ (1 - α)) - g₍₀₎) - c₍₀₎))))
loss += sum(abs2, @.((w₍₀₎ - z₍₀₎ * (1 - α) * (k₍₋₁₎ / l₍₀₎) ^ α)))
loss += sum(abs2, @.((z₍₀₎ - ((1 - ρᶻ) + ρᶻ * z₍₋₁₎ + σᶻ * ϵᶻ₍ₓ₎))))
loss += sum(abs2, @.((g₍₀₎ - ((1 - ρᵍ) * ḡ + ρᵍ * g₍₋₁₎ + σᵍ * ϵᵍ₍ₓ₎))))

# Finished [1.0e-8, 256.0, 100.0, 758.8324751853943, 0.0017534949583932757]
# Finished [1.0e-8, 256.0, 150.0, 1088.4795179367065, 0.0014873802429065108]
# Finished [1.0e-8, 256.0, 300.0, 2198.8943860530853, 0.0011740062618628144]
# Finished [1.0e-8, 512.0, 100.0, 517.70987200737, 0.0019748075865209103]
# Finished [1.0e-8, 512.0, 150.0, 752.7117650508881, 0.001620362396351993]

# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 64, :cos, 87.80793190002441, 0.0077648805f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 64, :cos, 118.03787803649902, 0.009695039f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 64, :cos, 126.16368412971497, 0.013013127f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 64, :cos, 226.33820700645447, 0.0055024438f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 64, :cos, 553.1387948989868, 0.006574779f0]
# Finished Any[0.001, 1.0e-10, :relu, 1024, 100, 64, :cos, 1023.6392850875854, 0.0074812747f0]
# Finished Any[0.001, 1.0e-10, :celu, 256, 100, 64, :cos, 117.92614006996155, 0.0063016475f0]
# Finished Any[0.001, 1.0e-10, :celu, 512, 100, 64, :cos, 1207.0850257873535, 0.008859483f0]
# Finished Any[0.001, 1.0e-10, :celu, 1024, 100, 64, :cos, 152.76296281814575, 0.01197234f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 128, :cos, 2269.02383184433, 0.0046920013f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 128, :cos, 216.41274404525757, 0.00684004f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 128, :cos, 178.79252886772156, 0.008927653f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 128, :cos, 304.0812249183655, 0.0028577521f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 128, :cos, 216.76893019676208, 0.0034144435f0]
# Finished Any[0.001, 1.0e-10, :relu, 1024, 100, 128, :cos, 172.68909406661987, 0.004311412f0]
# Finished Any[0.001, 1.0e-10, :celu, 256, 100, 128, :cos, 1578.1288549900055, 0.0035333529f0]
# Finished Any[0.001, 1.0e-10, :celu, 512, 100, 128, :cos, 1232.3901619911194, 0.0057994383f0]
# Finished Any[0.001, 1.0e-10, :celu, 1024, 100, 128, :cos, 276.539302110672, 0.008073823f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 256, :cos, 760.0306468009949, 0.0045519243f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 256, :cos, 755.0039529800415, 0.005255837f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 256, :cos, 453.7042829990387, 0.007580776f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 256, :cos, 736.4661679267883, 0.0016993907f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 256, :cos, 532.9272980690002, 0.0019593309f0]
# Finished Any[0.001, 1.0e-10, :relu, 1024, 100, 256, :cos, 417.4027919769287, 0.0025009874f0]
# Finished Any[0.001, 1.0e-10, :celu, 256, 100, 256, :cos, 964.1439228057861, 0.002458275f0]
# Finished Any[0.001, 1.0e-10, :celu, 512, 100, 256, :cos, 1650.427062034607, 0.0031529295f0]
# Finished Any[0.001, 1.0e-10, :celu, 1024, 100, 256, :cos, 1615.785187959671, 0.0053527183f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 64, :poly, 1009.7159638404846, 0.010393971f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 64, :poly, 167.65252709388733, 0.012821631f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 64, :poly, 121.7584228515625, 0.015461803f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 64, :poly, 90.5167019367218, 0.0073260977f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 64, :poly, 177.72605991363525, 0.00796308f0]
# Finished Any[0.001, 1.0e-10, :relu, 1024, 100, 64, :poly, 132.531240940094, 0.009469025f0]
# Finished Any[0.001, 1.0e-10, :celu, 256, 100, 64, :poly, 123.05211400985718, 0.008425333f0]
# Finished Any[0.001, 1.0e-10, :celu, 512, 100, 64, :poly, 255.57281684875488, 0.011553445f0]
# Finished Any[0.001, 1.0e-10, :celu, 1024, 100, 64, :poly, 184.74529004096985, 0.01403696f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 128, :poly, 485.4806730747223, 0.006409546f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 128, :poly, 335.3510570526123, 0.009440255f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 128, :poly, 255.51969480514526, 0.011993765f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 128, :poly, 436.82732701301575, 0.003490948f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 128, :poly, 305.7830169200897, 0.0043039513f0]
# Finished Any[0.001, 1.0e-10, :relu, 1024, 100, 128, :poly, 244.33404994010925, 0.00542185f0]
# Finished Any[0.001, 1.0e-10, :celu, 256, 100, 128, :poly, 722.4821009635925, 0.0054242457f0]
# Finished Any[0.001, 1.0e-10, :celu, 512, 100, 128, :poly, 311.80046796798706, 0.007692121f0]
# Finished Any[0.001, 1.0e-10, :celu, 1024, 100, 128, :poly, 278.63029313087463, 0.010396528f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 256, :poly, 1656.0463230609894, 0.0056912606f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 256, :poly, 500.66837191581726, 0.00709356f0]
# Finished Any[0.001, 1.0e-10, :tanh, 1024, 100, 256, :poly, 400.3649890422821, 0.009719013f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 256, :poly, 649.3027341365814, 0.0020020679f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 256, :cos, :adam, 874.7590310573578, 0.008674952f0, 0.053340144f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 256, :cos, :adam, 1479.008378982544, 0.0042581027f0, 0.025902053f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 256, :cos, :adam, 657.4577000141144, 0.005456611f0, 0.032977317f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 256, :poly, :adam, 1065.2488479614258, 0.0068564145f0, 0.042251226f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 256, :poly, :adam, 690.5816850662231, 0.0050768475f0, 0.030870058f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 256, :poly, :adam, 655.0082411766052, 0.0072127706f0, 0.04362028f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 256, :exp, :adam, 957.6404728889465, 0.01103832f0, 0.06775959f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 256, :exp, :adam, 707.8961498737335, 0.012567502f0, 0.07634977f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 256, :exp, :adam, 485.78916001319885, 0.012558071f0, 0.0759405f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 128, :cos, :adam, 441.0436019897461, 0.005521825f0, 0.034186002f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 128, :cos, :adam, 307.0251669883728, 0.005009134f0, 0.030602196f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 128, :cos, :adam, 209.77864789962769, 0.0074116797f0, 0.044822957f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 128, :poly, :adam, 404.1711390018463, 0.006917475f0, 0.042661365f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 128, :poly, :adam, 293.3614249229431, 0.0066384356f0, 0.040544663f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 128, :poly, :adam, 233.64928793907166, 0.009719833f0, 0.05880016f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 128, :exp, :adam, 457.6904640197754, 0.013690149f0, 0.083735645f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 128, :exp, :adam, 307.8617420196533, 0.013544772f0, 0.082225695f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 128, :exp, :adam, 228.92059087753296, 0.015151696f0, 0.0915573f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 384, :cos, :adam, 1548.1928508281708, 0.004702253f0, 0.029134441f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 384, :cos, :adam, 2695.2564520835876, 0.005433802f0, 0.033156622f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 384, :cos, :adam, 878.5811469554901, 0.0055878004f0, 0.033750694f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 384, :poly, :adam, 1666.9311110973358, 0.004909168f0, 0.030593151f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 384, :poly, :adam, 1250.2043538093567, 0.006147527f0, 0.037540067f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 384, :poly, :adam, 1013.6991958618164, 0.007157812f0, 0.043307714f0]
# Finished Any[0.001, 1.0e-10, :tanh, 128, 100, 384, :exp, :adam, 1612.9655721187592, 0.01080064f0, 0.066040665f0]
# Finished Any[0.001, 1.0e-10, :tanh, 256, 100, 384, :exp, :adam, 1105.5232849121094, 0.012695885f0, 0.07717036f0]
# Finished Any[0.001, 1.0e-10, :tanh, 512, 100, 384, :exp, :adam, 883.1924521923065, 0.0130463075f0, 0.07885451f0]
# Finished Any[0.001, 1.0e-10, :relu, 128, 100, 256, :cos, :adam, 947.8783750534058, 0.0017637294f0, 0.010659676f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 256, :cos, :adam, 666.8837270736694, 0.001751316f0, 0.010552594f0]
# Finished Any[0.001, 1.0e-10, :relu, 512, 100, 256, :cos, :adam, 472.91423892974854, 0.0019897788f0, 0.011983875f0]
# Finished Any[0.001, 1.0e-10, :relu, 128, 100, 256, :poly, :adam, 972.5033791065216, 0.0018560188f0, 0.01122801f0]
# Finished Any[0.001, 1.0e-10, :relu, 256, 100, 256, :poly, :adam, 657.7711980342865, 0.0020595042f0, 0.012445164f0]
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
