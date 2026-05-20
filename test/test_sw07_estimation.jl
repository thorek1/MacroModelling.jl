using Test
using MacroModelling
import Mooncake
import ADTypes
import ADTypes: AutoMooncake
import DifferentiationInterface
import FiniteDifferences
import Turing
import Turing: NUTS
import LinearAlgebra as ℒ
using Random, DelimitedFiles, AxisKeys

using FlexiChains
include("test_helpers.jl")

# load data
dat, header = readdlm("data/usmodel.csv", ',', header = true)
dat = Float64.(dat)
names = vec(Symbol.(strip.(header)))

# load data
data = KeyedArray(dat', Variable = names, Time = axes(dat, 1))

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

# Subsample
# subset observables in data
sample_idx = 47:230 # 1960Q1-2004Q4

data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs] # note that :dw was renamed to :dwobs in linear model in order to avoid confusion with nonlinear model

data = rekey(data, :Variable => observables)


# Handling distributions with varying parameters using arraydist
dists = [
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ea
InverseGamma(0.1, 2.0, 0.025,5.0, μσ = true),   # z_eb
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eg
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_eqs
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_em
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_epinf
InverseGamma(0.1, 2.0, 0.01, 3.0, μσ = true),   # z_ew
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoa
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhob
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhog
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoqs
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhoms
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # crhopinf
Beta(0.5, 0.2, 0.001,0.9999, μσ = true),        # crhow
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmap
Beta(0.5, 0.2, 0.01, 0.9999, μσ = true),        # cmaw
Normal(4.0, 1.5,   2.0, 15.0),                  # csadjcost
Normal(1.50,0.375, 0.25, 3.0),                  # csigma
Beta(0.7, 0.1, 0.001, 0.99, μσ = true),         # chabb
Beta(0.5, 0.1, 0.3, 0.95, μσ = true),           # cprobw
Normal(2.0, 0.75, 0.25, 10.0),                  # csigl
Beta(0.5, 0.10, 0.5, 0.95, μσ = true),          # cprobp
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindw
Beta(0.5, 0.15, 0.01, 0.99, μσ = true),         # cindp
Beta(0.5, 0.15, 0.01, 0.99999, μσ = true),      # czcap
Normal(1.25, 0.125, 1.0, 3.0),                  # cfc
Normal(1.5, 0.25, 1.0, 3.0),                    # crpi
Beta(0.75, 0.10, 0.5, 0.975, μσ = true),        # crr
Normal(0.125, 0.05, 0.001, 0.5),                # cry
Normal(0.125, 0.05, 0.001, 0.5),                # crdy
Gamma(0.625, 0.1, 0.1, 2.0, μσ = true),         # constepinf
Gamma(0.25, 0.1, 0.01, 2.0, μσ = true),         # constebeta
Normal(0.0, 2.0, -10.0, 10.0),                  # constelab
Normal(0.4, 0.10, 0.1, 0.8),                    # ctrend
Normal(0.5, 0.25, 0.01, 2.0),                   # cgy
Normal(0.3, 0.05, 0.01, 1.0),                   # calfa
]

Turing.@model function SW07_loglikelihood_function(data, m, observables, fixed_parameters, filter)
    all_params ~ Turing.product_distribution(dists)

    z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = all_params

    ctou, clandaw, cg, curvp, curvw = fixed_parameters

    parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

    llh = get_loglikelihood(m, data(observables), parameters_combined, presample_periods = 4, initial_covariance = :diagonal, filter = filter)

    Turing.@addlogprob! llh
end

# estimate linear model

include("../models/Smets_Wouters_2007_linear.jl")

fixed_parameters = Smets_Wouters_2007_linear.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007_linear.constants.post_complete_parameters.parameters)]

SS(Smets_Wouters_2007_linear, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007_linear, observables, fixed_parameters, :kalman)
# inversion filter delivers similar results

# par_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# inits = [Dict(get_parameters(Smets_Wouters_2007_linear, values = true))[string(i)] for i in par_names]

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.SimulatedAnnealing())#,
# #                                         initial_params = inits)

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
#                                         initial_params = modeSW2007.params)

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.NelderMead())

# println("Mode variable values (linear): $(modeSW2007.params); Mode loglikelihood: $(modeSW2007.lp)")

# LLH = Turing.logjoint(SW07_loglikelihood, (all_params = inits,))

n_samples = 1000

samps = @time Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, 
                            # initial_params = inits,
                            progress = true)

posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (linear): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

@testset "Mooncake vs FiniteDifferences gradient (SW07 linear)" begin
    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(Smets_Wouters_2007_linear, data(observables), x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), ADTypes.AutoMooncake(config = nothing), Smets_Wouters_2007_linear.parameter_values)
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(Smets_Wouters_2007_linear, data(observables), x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), Smets_Wouters_2007_linear.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end

# estimate nonlinear model

include("../models/Smets_Wouters_2007.jl")

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou, :clandaw, :cg, :curvp, :curvw], Smets_Wouters_2007.constants.post_complete_parameters.parameters)]

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01, :cmap => 0.01, :cmaw => 0.01])(observables,:)

SW07_loglikelihood = SW07_loglikelihood_function(data, Smets_Wouters_2007, observables, fixed_parameters, :kalman)

# par_names = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in par_names]

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.SimulatedAnnealing())#,
# #                                         initial_params = inits)

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)),
#                                         initial_params = modeSW2007.params)

# modeSW2007 = Turing.maximum_a_posteriori(SW07_loglikelihood, 
#                                         Optim.NelderMead(),
#                                         initial_params = modeSW2007.params)

# println("Mode variable values (linear): $(modeSW2007.params); Mode loglikelihood: $(modeSW2007.lp)")

n_samples = 1000

samps = @time Turing.sample(SW07_loglikelihood, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples, 
                            # initial_params = inits,
                            progress = true)

posterior_summary = FlexiChains.summarystats(samps)
show(stdout, MIME"text/plain"(), posterior_summary)
println()
println("Mean variable values (nonlinear): $(collect(values(FlexiChains.mean(samps); parameters_only = true)))")

@testset "Mooncake vs FiniteDifferences gradient (SW07 nonlinear)" begin
    back_grad = DifferentiationInterface.gradient(x -> get_loglikelihood(Smets_Wouters_2007, data(observables), x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), ADTypes.AutoMooncake(config = nothing), Smets_Wouters_2007.parameter_values)
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(Smets_Wouters_2007, data(observables), x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), Smets_Wouters_2007.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end


# ---------------------------------------------------------------------------
# Replicate the estimation problems on data with missing observations.
# ---------------------------------------------------------------------------
data_missing = inject_missing_observations(data(observables))

# Linear model under missing observations
SW07_loglikelihood_linear_missing = SW07_loglikelihood_function(data_missing, Smets_Wouters_2007_linear, observables, fixed_parameters, :kalman)

samps_missing_linear = @time Turing.sample(SW07_loglikelihood_linear_missing, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples,
                            progress = true)

posterior_summary_missing_linear = FlexiChains.summarystats(samps_missing_linear)
show(stdout, MIME"text/plain"(), posterior_summary_missing_linear)
println()
println("Mean variable values (linear, missing data): $(collect(values(FlexiChains.mean(samps_missing_linear); parameters_only = true)))")

sample_nuts_linear_missing = collect(values(FlexiChains.mean(samps_missing_linear); parameters_only = true))

@testset "SW07 linear estimation results (missing data)" begin
    @test all(isfinite, sample_nuts_linear_missing)
    @test length(sample_nuts_linear_missing) == length(dists)
end

@testset "Mooncake vs FiniteDifferences gradient (SW07 linear, missing data)" begin
    # Constant contexts avoid Mooncake's __verify_const NaN-array failure.
    loglik_target(x, m, d) = get_loglikelihood(m, d, x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)
    back_grad = DifferentiationInterface.gradient(loglik_target,
        ADTypes.AutoMooncake(config = nothing), Smets_Wouters_2007_linear.parameter_values,
        DifferentiationInterface.Constant(Smets_Wouters_2007_linear),
        DifferentiationInterface.Constant(data_missing))
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(Smets_Wouters_2007_linear, data_missing, x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), Smets_Wouters_2007_linear.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end

# Nonlinear model under missing observations
SW07_loglikelihood_nonlinear_missing = SW07_loglikelihood_function(data_missing, Smets_Wouters_2007, observables, fixed_parameters, :kalman)

samps_missing_nonlinear = @time Turing.sample(SW07_loglikelihood_nonlinear_missing, NUTS(adtype = AutoMooncake(; config=nothing)), n_samples,
                            progress = true)

posterior_summary_missing_nonlinear = FlexiChains.summarystats(samps_missing_nonlinear)
show(stdout, MIME"text/plain"(), posterior_summary_missing_nonlinear)
println()
println("Mean variable values (nonlinear, missing data): $(collect(values(FlexiChains.mean(samps_missing_nonlinear); parameters_only = true)))")

sample_nuts_nonlinear_missing = collect(values(FlexiChains.mean(samps_missing_nonlinear); parameters_only = true))

@testset "SW07 nonlinear estimation results (missing data)" begin
    @test all(isfinite, sample_nuts_nonlinear_missing)
    @test length(sample_nuts_nonlinear_missing) == length(dists)
end

@testset "Mooncake vs FiniteDifferences gradient (SW07 nonlinear, missing data)" begin
    # Constant contexts avoid Mooncake's __verify_const NaN-array failure.
    loglik_target(x, m, d) = get_loglikelihood(m, d, x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman)
    back_grad = DifferentiationInterface.gradient(loglik_target,
        ADTypes.AutoMooncake(config = nothing), Smets_Wouters_2007.parameter_values,
        DifferentiationInterface.Constant(Smets_Wouters_2007),
        DifferentiationInterface.Constant(data_missing))
    @test !isnothing(back_grad)
    @test all(isfinite, back_grad)

    for i in 1:100
        local fin_grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), x -> get_loglikelihood(Smets_Wouters_2007, data_missing, x, presample_periods = 4, initial_covariance = :diagonal, filter = :kalman), Smets_Wouters_2007.parameter_values)
        if isfinite(ℒ.norm(fin_grad))
            println("Finite differences converged after $i iterations")
            @test isapprox(back_grad, fin_grad[1], rtol = 1e-4)
            break
        end
    end
end