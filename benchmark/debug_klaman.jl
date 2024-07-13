using MacroModelling
import Turing, Pigeons
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
import Zygote

include("../models/Smets_Wouters_2007_linear.jl")
Smets_Wouters_2007 = Smets_Wouters_2007_linear

# include("../models/Smets_Wouters_2007.jl")
# load data
dat = CSV.read("test/data/usmodel.csv", DataFrame)

# load data
data = KeyedArray(Array(dat)',Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

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

SS(Smets_Wouters_2007, parameters = [:crhoms => 0.01, :crhopinf => 0.01, :crhow => 0.01,:cmap => 0.01,:cmaw => 0.01])

fixed_parameters = Smets_Wouters_2007.parameter_values[indexin([:ctou,:clandaw,:cg,:curvp,:curvw],Smets_Wouters_2007.parameters)]

inits = [Dict(get_parameters(Smets_Wouters_2007, values = true))[string(i)] for i in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]]


z_ea, z_eb, z_eg, z_eqs, z_em, z_epinf, z_ew, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, csadjcost, csigma, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, cfc, crpi, crr, cry, crdy, constepinf, constebeta, constelab, ctrend, cgy, calfa = inits

ctou, clandaw, cg, curvp, curvw = fixed_parameters

parameters_combined = [ctou, clandaw, cg, curvp, curvw, calfa, csigma, cfc, cgy, csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw, constelab, constepinf, constebeta, ctrend, z_ea, z_eb, z_eg, z_em, z_ew, z_eqs, z_epinf]

get_loglikelihood(Smets_Wouters_2007, data, parameters_combined, verbose = false, presample_periods = 4, initial_covariance = :diagonal) |> println

grads = Zygote.gradient(x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined)

using BenchmarkTools

@benchmark get_loglikelihood(Smets_Wouters_2007, data, parameters_combined, verbose = false, presample_periods = 4, initial_covariance = :diagonal)
# BenchmarkTools.Trial: 1815 samples with 1 evaluation.
#  Range (min … max):  2.068 ms …   7.873 ms  ┊ GC (min … max):  0.00% … 69.88%
#  Time  (median):     2.389 ms               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   2.751 ms ± 910.168 μs  ┊ GC (mean ± σ):  12.62% ± 17.82%

#       ▅█▇▄▂                                          ▁▁ ▁▂     
#   ▆▅████████▆▅▇▆▆▃▄▄▄▃▁▁▄▄▄▄▁▁▁▁▃▁▃▃▁▁▃▄▁▁▁▁▁▄▅▃▃▆▇▆▇███████▇ █
#   2.07 ms      Histogram: log(frequency) by time      5.26 ms <

#  Memory estimate: 8.41 MiB, allocs estimate: 7334.

# without write_reduced_block_solutionBenchmarkTools.Trial: 1865 samples with 1 evaluation.
#  Range (min … max):  2.037 ms …   7.480 ms  ┊ GC (min … max):  0.00% … 69.95%
#  Time  (median):     2.385 ms               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   2.678 ms ± 773.663 μs  ┊ GC (mean ± σ):  11.27% ± 16.76%

#       ▂▆██▆▄▂                                      ▁▁▂▂▂▁     ▁
#   ▅▆▇▆███████▇▄▁▄▁▄▁▄▁▄▄▄▁▄▁▄▄▁▁▁▁▁▁▁▁▁▄▁▁▄▄▁▄▄▄▆▆█████████▇▇ █
#   2.04 ms      Histogram: log(frequency) by time      4.86 ms <

#  Memory estimate: 8.34 MiB, allocs estimate: 6686.

@benchmark get_loglikelihood(Smets_Wouters_2007, data, parameters_combined, verbose = false, presample_periods = 4, filter = :inversion)
# BenchmarkTools.Trial: 99 samples with 1 evaluation.
#  Range (min … max):  47.738 ms … 58.452 ms  ┊ GC (min … max): 4.79% … 7.71%
#  Time  (median):     50.329 ms              ┊ GC (median):    5.14%
#  Time  (mean ± σ):   50.576 ms ±  2.227 ms  ┊ GC (mean ± σ):  6.95% ± 2.27%

#      ▂█           ▂   ▃                                        
#   ▅▅████▅▇▇▇▁▁▄▅█▅███▅█▇▄▇▄▄▄▅▅▄▅▄▅▁▅▁▁▁▄▅▁▁▁▁▄▁▁▁▁▁▇▁▁▄▁▁▁▁▄ ▁
#   47.7 ms         Histogram: frequency by time          57 ms <

#  Memory estimate: 98.28 MiB, allocs estimate: 458194.

@profview for i in 1:100 get_loglikelihood(Smets_Wouters_2007, data, parameters_combined, verbose = false, presample_periods = 4, filter = :inversion) end




@benchmark grads = Zygote.gradient(x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined)
@profview for i in 1:100 Zygote.gradient(x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined) end


# with reduced smaller system
# BenchmarkTools.Trial: 95 samples with 1 evaluation.
#  Range (min … max):  48.919 ms … 63.667 ms  ┊ GC (min … max): 0.00% … 7.69%
#  Time  (median):     52.799 ms              ┊ GC (median):    5.67%
#  Time  (mean ± σ):   52.642 ms ±  2.658 ms  ┊ GC (mean ± σ):  4.67% ± 2.78%

#   ▂▅          ▁ ▅ ▂█▅▂▁                                        
#   ██▆▅▅▃▁▁▁▁▁▅███▆█████▅▁▃▁▆▃▁▃▃▁▁▁▁▃▃▃▃▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▃ ▁
#   48.9 ms         Histogram: frequency by time        62.6 ms <

# with larger systemBenchmarkTools.Trial: 73 samples with 1 evaluation.
#  Range (min … max):  65.542 ms … 81.523 ms  ┊ GC (min … max): 0.00% … 3.04%
#  Time  (median):     68.170 ms              ┊ GC (median):    3.59%
#  Time  (mean ± σ):   68.891 ms ±  2.328 ms  ┊ GC (mean ± σ):  3.82% ± 1.46%

#              █▁▆▁                                              
#   ▃▁▁▁▁▁▁▁▁▁█████▄▇▃▃▄▁▃▁▁▁▁▃▁▄▃▁▁▁▁▁▃▃▁▁▁▁▁▁▃▁▁▁▁▁▃▁▁▁▁▁▁▁▁▃ ▁
#   65.5 ms         Histogram: frequency by time          77 ms <

#  Memory estimate: 90.82 MiB, allocs estimate: 160931.

#  Memory estimate: 72.86 MiB, allocs estimate: 169874.
# using ForwardDiff
# grads = ForwardDiff.gradient(x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined)

# @benchmark grads = ForwardDiff.gradient(x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined)

# using FiniteDifferences


# fin_grad = FiniteDifferences.grad(central_fdm(5,1),x->get_loglikelihood(Smets_Wouters_2007, data, x, verbose = false, presample_periods = 4, initial_covariance = :diagonal), parameters_combined)[1]

# maximum(abs,(grads[1] - fin_grad) ./ fin_grad)