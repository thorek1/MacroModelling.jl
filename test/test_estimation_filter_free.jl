using MacroModelling
import Turing, Pigeons, Zygote
import Turing: NUTS, sample, logpdf, Beta, Normal, InverseGamma
import ADTypes: AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys, LinearAlgebra
import DynamicPPL

include("../models/FS2000.jl")

# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)



Ω = 0.001

# Handling distributions with varying parameters using arraydist
dists = [
    Beta(0.356, 0.02, Val(:μσ)),           # alp
    Beta(0.993, 0.002, Val(:μσ)),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, Val(:μσ)),          # rho
    Beta(0.65, 0.05, Val(:μσ)),            # psi
    Beta(0.01, 0.005, Val(:μσ)),           # del
    InverseGamma(0.035449, Inf, Val(:μσ)), # z_e_a
    InverseGamma(0.008862, Inf, Val(:μσ)), # z_e_m
    Turing.MvNormal(I(length(data)))  # shocks
]

Turing.@model function FS2000_loglikelihood_function(data, m)
    all_params ~ Turing.arraydist(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(m, data, all_params[1:end-1])
    end
end









function get_loglikelihood(𝓂::ℳ, 
                            data::KeyedArray{Float64}, 
                            parameter_values::Vector{S},
                            shocks::Vector{S}; 
                            Ω::Union{S, Vector{S}} = 1e-4,
                            algorithm::Symbol = :first_order, 
                            # filter::Symbol = :kalman, 
                            warmup_iterations::Int = 0, 
                            presample_periods::Int = 0,
                            initial_covariance::Symbol = :theoretical,
                            # filter_algorithm::Symbol = :LagrangeNewton,
                            tol::Tolerances = Tolerances(), 
                            quadratic_matrix_equation_algorithm::Symbol = :schur, 
                            lyapunov_algorithm::Symbol = :doubling, 
                            sylvester_algorithm::Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}} = sum(1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling,
                            verbose::Bool = false)::S where S <: Real
                            # timer::TimerOutput = TimerOutput(),

    opts = merge_calculation_options(tol = tol, verbose = verbose,
                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                            sylvester_algorithm² = isa(sylvester_algorithm, Symbol) ? sylvester_algorithm : sylvester_algorithm[1],
                            sylvester_algorithm³ = (isa(sylvester_algorithm, Symbol) || length(sylvester_algorithm) < 2) ? sum(k * (k + 1) ÷ 2 for k in 1:𝓂.timings.nPast_not_future_and_mixed + 1 + 𝓂.timings.nExo) > 1000 ? :bicgstab : :doubling : sylvester_algorithm[2],
                            lyapunov_algorithm = lyapunov_algorithm)

    # if algorithm ∈ [:third_order,:pruned_third_order]
    #     sylvester_algorithm = :bicgstab
    # end

    # TODO: throw error for bounds violations, suggesting this might be due to wrong parameter ordering
    @assert length(parameter_values) == length(𝓂.parameters) "The number of parameter values provided does not match the number of parameters in the model. If this function is used in the context of estimation and not all parameters are estimated, you need to combine the estimated parameters with the other model parameters in one `Vector`. Make sure they have the same order they were declared in the `@parameters` block (check by calling `get_parameters`)."

    # checks to avoid errors further down the line and inform the user
    @assert filter ∈ [:kalman, :inversion] "Currently only the Kalman filter (:kalman) for linear models and the inversion filter (:inversion) for linear and nonlinear models are supported."

    # checks to avoid errors further down the line and inform the user
    @assert initial_covariance ∈ [:theoretical, :diagonal] "Invalid method to initialise the Kalman filters covariance matrix. Supported methods are: the theoretical long run values (option `:theoretical`) or large values (10.0) along the diagonal (option `:diagonal`)."

    if algorithm ∈ [:second_order,:pruned_second_order,:third_order,:pruned_third_order]
        filter = :inversion
    end

    observables = @ignore_derivatives get_and_check_observables(𝓂, data)

    @ignore_derivatives solve!(𝓂, 
                                opts = opts,
                                # timer = timer, 
                                algorithm = algorithm)

    bounds_violated = @ignore_derivatives check_bounds(parameter_values, 𝓂)

    if bounds_violated 
        # println("Bounds violated")
        return -Inf 
    end

    NSSS_labels = @ignore_derivatives [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives convert(Vector{Int}, indexin(observables, NSSS_labels))

    # @timeit_debug timer "Get relevant steady state and solution" begin

    TT, SS_and_pars, 𝐒, state, solved = get_relevant_steady_state_and_state_update(Val(algorithm), parameter_values, 𝓂, opts = opts)
                                                                                    # timer = timer,

    # end # timeit_debug

    if !solved 
        # println("Main call: 1st order solution not found")
        return -Inf 
    end
 
    if collect(axiskeys(data,1)) isa Vector{String}
        data = @ignore_derivatives rekey(data, 1 => axiskeys(data,1) .|> Meta.parse .|> replace_indices)
    end

    dt = @ignore_derivatives collect(data(observables))

    # prepare data
    data_in_deviations = dt .- SS_and_pars[obs_indices]

    # @timeit_debug timer "Filter" begin

    llh = calculate_loglikelihood(Val(filter), algorithm, observables, 𝐒, data_in_deviations, TT, presample_periods, initial_covariance, state, warmup_iterations, filter_algorithm, opts) # timer = timer

    # end # timeit_debug

    return llh
end

S = Float64
T = FS2000.timings

observables_index = convert(Vector{Int},indexin(observables, sort(union(T.aux,T.var,T.exo_present))))

observables_and_states = sort(union(T.past_not_future_and_mixed_idx, observables_index))

A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(S, length(observables_and_states)))[(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]












FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000)


n_samples = 1000

# using Zygote
# Turing.setadbackend(:zygote)
samps = @time sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true, initial_params = FS2000.parameter_values)

println("Mean variable values (ForwardDiff): $(mean(samps).nt.mean)")

samps = @time sample(FS2000_loglikelihood, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = FS2000.parameter_values)

println("Mean variable values (Zygote): $(mean(samps).nt.mean)")

sample_nuts = mean(samps).nt.mean


# generate a Pigeons log potential
FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000))

# find a feasible starting point
pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1)

replica = pt.replicas[end]
XMAX = deepcopy(replica.state)
LPmax = FS2000_lp(XMAX)

i = 0

while !isfinite(LPmax) && i < 1000
    Pigeons.sample_iid!(FS2000_lp, replica, pt.shared)
    new_LP = FS2000_lp(replica.state)
    if new_LP > LPmax
        LPmax = new_LP
        XMAX  = deepcopy(replica.state)
    end
    i += 1
end

# define a specific initialization for this model
Pigeons.initialization(::Pigeons.TuringLogPotential{typeof(FS2000_loglikelihood_function)}, ::AbstractRNG, ::Int64) = deepcopy(XMAX)

pt = @time Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 10,
            multithreaded = true)

samps = MCMCChains.Chains(pt)

println("Mean variable values (Pigeons): $(mean(samps).nt.mean)")

sample_pigeons = mean(samps).nt.mean


modeFS2000 = Turing.maximum_a_posteriori(FS2000_loglikelihood, 
                                        # Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 2)), 
                                        Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                                        # Optim.NelderMead(), 
                                        adtype = AutoZygote(), 
                                        # maxiters = 100,
                                        # lb = [0,0,-10,-10,0,0,0,0,0], 
                                        # ub = [1,1,10,10,1,1,1,100,100], 
                                        initial_params = FS2000.parameter_values)

println("Mode variable values: $(modeFS2000.values); Mode loglikelihood: $(modeFS2000.lp)")

@testset "Estimation results" begin
    # @test isapprox(modeFS2000.lp, 1281.669108730447, rtol = eps(Float32))
    @test isapprox(sample_nuts, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
    @test isapprox(sample_pigeons[1:length(sample_nuts)], [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end



plot_model_estimates(FS2000, data, parameters = sample_nuts)
plot_shock_decomposition(FS2000, data)

FS2000 = nothing
m = nothing
# @profview sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)


# chain_NUTS  = sample(FS2000_loglikelihood, NUTS(), n_samples, init_params = FS2000.parameter_values, progress = true)#, init_params = FS2000.parameter_values)#init_theta = FS2000.parameter_values)

# StatsPlots.plot(chain_NUTS)

# parameter_mean = mean(chain_NUTS)

# pars = ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1]))

# logjoint(FS2000_loglikelihood, pars)

# function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
#     orig_pars[pars_syms] = [par1, par2]
#     logjoint(model, orig_pars)
# end

# granularity = 32;

# par1 = :del;
# par2 = :gam;
# par_range1 = collect(range(minimum(chain_NUTS[par1]), stop = maximum(chain_NUTS[par1]), length = granularity));
# par_range2 = collect(range(minimum(chain_NUTS[par2]), stop = maximum(chain_NUTS[par2]), length = granularity));

# p = surface(par_range1, par_range2, 
#             (x,y) -> calculate_log_probability(x, y, [par1, par2], pars, FS2000_loglikelihood),
#             camera=(30, 65),
#             colorbar=false,
#             color=:inferno);


# joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray(reduce(hcat, get(chain_NUTS, FS2000.parameters)[FS2000.parameters])[s,:], Axis(FS2000.parameters))) for s in 1:length(chain_NUTS)]

# scatter3d!(vec(collect(chain_NUTS[par1])),
#            vec(collect(chain_NUTS[par2])),
#            joint_loglikelihood,
#             mc = :viridis, 
#             marker_z = collect(1:length(chain_NUTS)), 
#             msw = 0,
#             legend = false, 
#             colorbar = false, 
#             xlabel = string(par1),
#             ylabel = string(par2),
#             zlabel = "Log probability",
#             alpha = 0.5);

# p
