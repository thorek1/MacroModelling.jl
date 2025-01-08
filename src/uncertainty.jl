using Revise
using MacroModelling
using StatsPlots
using Optim, LineSearches
using Optimization, OptimizationNLopt, OptimizationOptimJL
using Zygote, ForwardDiff, FiniteDifferences
using BenchmarkTools

include("./Smets_Wouters_2007 copy.jl")



# estimation results
## 2nd order
algo = :pruned_second_order
### 1990-2024
full_sample = [0.6647831723963108, 0.1696409521786583, 0.5580454054112424, 1.4107875417460551, 0.27150800318229096, 0.2982789062192042, 0.9608926135337186, 0.9968070538427887, 0.9355375595019946, 0.8857554713134486, 0.6671161668424413, 0.363490876566295, 0.9844516855340667, 0.9548251912875737, 0.2672038505621581, 0.4710818007355946, 2.3199691360451973, 1.647356911213051, 0.2681908246948425, 0.4216832168949589, -0.15640416813475932, 0.5091965632941383, 0.6795174040555909, 0.3819940049374918, 0.8953965508435049, 1.0811321641690619, 2.763652185541633, 0.8578346355200306, 0.08963808582038361, 0.5101854590802727, 0.4527575703228302, -0.8627011833961279, 0.5462134869630935, 0.1813725165517729, 0.25860912552471754, 0.02283530676171379, 2.256424512963501, 0.1922713275188999, 8.218707368321041, 5.906699475772892]


### no pandemic
## 1990-2020
no_pandemic = [0.6214776253642815, 0.09180625969728813, 0.48695111652187245, 1.7462778722662238, 0.22762104671667938, 0.2841560574382372, 0.42421425521650646, 0.9968070538427887, 0.9355375595019946, 0.8857554713134486, 0.6671161668424413, 0.363490876566295, 0.9844516855340667, 0.9548251912875737, 0.2672038505621581, 0.4710818007355946, 2.3199691360451973, 1.647356911213051, 0.2681908246948425, 0.4216832168949589, -0.15640416813475932, 0.5091965632941383, 0.6795174040555909, 0.3819940049374918, 0.8953965508435049, 1.0811321641690619, 2.763652185541633, 0.8578346355200306, 0.08963808582038361, 0.5101854590802727, 0.4527575703228302, -0.8627011833961279, 0.5462134869630935, 0.1813725165517729, 0.25860912552471754, 0.02283530676171379, 2.256424512963501, 0.1922713275188999, 8.218707368321041, 5.906699475772892]

pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

std_full_sample = get_std(Smets_Wouters_2007, parameters = pars .=> full_sample, algorithm = algo, derivatives = false)

std_no_pandemic = get_std(Smets_Wouters_2007, parameters = pars .=> no_pandemic, algorithm = algo, derivatives = false)

round.(std_full_sample([:a,:b,:gy,:qs,:ms,:spinf,:sw,:pinfobs,:drobs,:ygap]), digits = 2)
# (:a)      0.02426349697813647
# (:b)      0.005998118330574482
# (:gy)     0.0241028522046634
# (:qs)     0.08951131382441423
# (:ms)     0.008085447272957411
# (:spinf)  0.0335251067024813
# (:sw)     8.324446250434056

round.(std_no_pandemic([:a,:b,:gy,:qs,:ms,:spinf,:sw,:pinfobs,:drobs,:ygap]), digits = 2)
# (:a)      0.05649332222233191
# (:b)      0.009154771190994018
# (:gy)     0.015818645868635427
# (:qs)     0.13897662836032265
# (:ms)     0.007814003846949065
# (:spinf)  0.047651064538222695
# (:sw)     2.0020346229399255

# Interpretation:
# estimating the model on the pre-pandemic sample, then fixing the model parameters and reestimating on the full sample only the AR1 shock process related parameters the impact of the pandemic and inflaiton surge period of the variance of the shock processes is higher variance on the investment and government spending shock, unchanged variance on the wage markup shock and all other variances decrease


std_full_sample = get_std(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

std_no_pandemic = get_std(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)

round.(std_no_pandemic([:a,:b,:gy,:qs,:ms,:spinf,:sw,:pinfobs,:drobs,:ygap]), digits = 2)

var_full_sample = get_var(Smets_Wouters_2007, parameters = pars .=> full_sample, algorithm = algo, derivatives = false)

round.(var_full_sample([:pinfobs,:drobs,:ygap]), digits = 2)

var_full_sample_1st = get_var(Smets_Wouters_2007, parameters = pars .=> full_sample, algorithm = :first_order, derivatives = false)

round.(var_full_sample_1st([:pinfobs,:drobs,:ygap]), digits = 2)

# var_full_sample_3rd = get_var(Smets_Wouters_2007, parameters = pars .=> full_sample, algorithm = :pruned_third_order, derivatives = false)

# round.(var_full_sample_3rd([:pinfobs,:drobs,:ygap]), digits = 2)

var_no_pandemic = get_var(Smets_Wouters_2007, parameters = pars .=> no_pandemic, algorithm = algo, derivatives = false)

var_no_pandemic_1st = get_var(Smets_Wouters_2007, parameters = pars .=> no_pandemic, algorithm = :first_order, derivatives = false)

round.(var_no_pandemic_1st([:pinfobs,:drobs,:ygap]), digits = 2)

# var_no_pandemic_3rd = get_var(Smets_Wouters_2007, parameters = pars .=> no_pandemic, algorithm = :pruned_third_order, derivatives = false)

# round.(var_no_pandemic_3rd([:pinfobs,:drobs,:ygap]), digits = 2)


# calculate third order standard deviations
SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)

optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

out = get_statistics(Smets_Wouters_2007,   
                    optimal_taylor_coefficients,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    standard_deviation = [:pinfobs, :ygap, :drobs],
                    algorithm = :pruned_third_order,
                    verbose = true)


SS(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

out = get_statistics(Smets_Wouters_2007,   
                    optimal_taylor_coefficients,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    standard_deviation = [:pinfobs, :ygap, :drobs],
                    algorithm = :pruned_second_order,
                    verbose = true)



# historical shock decomposition
using StatsPlots
include("../test/download_EA_data.jl")
data = rekey(data, :Variable => [:dy, :dc, :dinve, :labobs, :dlabobs, :pinfobs, :dwobs, :r̃obs, :robs])
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = data(observables)


plot_shock_decomposition(Smets_Wouters_2007, 
                        data[:,78:end],
                        parameters = pars .=> full_sample,
                        # parameters = pars .=> no_pandemic,
                        variables = [:ygap,:pinfobs,:robs],
                        algorithm = algo,
                        plots_per_page = 3,
                        # filter = :kalman,
                        # smooth = true,
                        save_plots_format = :png,
                        save_plots = true)

shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, 
                                    data[:,78:end],
                                    parameters = pars .=> full_sample,
                                    # parameters = pars .=> no_pandemic,
);#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)


shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, 
                                    data[:,78:end],
                                    # parameters = pars .=> full_sample,
                                    parameters = pars .=> no_pandemic,
                                    # parameters = pars .=> full_sample_shock_pandemic,
);#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)



SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)


# find optimal loss coefficients
optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

# taylor_coef_stds = [0.2739, 0.0113, 0.0467]
taylor_coef_stds = [0.2123, 0.0102, 0.0435]

function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    out = get_statistics(Smets_Wouters_2007,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = algo,
                    verbose = verbose)

    res = ([1,1,1] .* out[:variance])' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
    println(res)

    return res
end

function find_weights(loss_function_weights_regularisation, optimal_taylor_coefficients)
    loss_function_weights = loss_function_weights_regularisation[1:2]
    regularisation = 1 / loss_function_weights_regularisation[3]
    res = sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,loss_function_weights), regularisation)), optimal_taylor_coefficients))

    println(res)

    return res
end

# find_weights(vcat(loss_function_wts, 1 / regularisation[1]), optimal_taylor_coefficients)

# get_parameters(Smets_Wouters_2007, values = true)
lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())

prob = OptimizationProblem(f, [0.11627083536113818, 0.30115648430336367, 15.581312752315547], optimal_taylor_coefficients, ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

# no pandemic
# 0.11627083536113818
# 0.30115648430336367
# 15.581312752315547

# no pandemic (annual inflation and interest rate)
# 1.8603333657780987
# 0.301156484297508
# 0.9738320470198245

# full sample
# 0.10758325002719157
# 0.0
# 1.1206730563516494


# full sample (annual inflation and interest rate)
# 1.7213632689776661
# 0.0
# 0.07013786896845288
SS(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

prob = OptimizationProblem(f, fill(.5, 3), optimal_taylor_coefficients, ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results





# calculate optimal taylor coefficient for full sample given loss coefficients assuming pre pandemic taylor rule coefficients are optimal
function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    out = get_statistics(Smets_Wouters_2007,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = algo,
                    verbose = verbose)

    res = out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
    # println(res)

    return res
end

lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

# no pandemic weights
loss_function_weights = [1, 0.11627083536113818, 0.30115648430336367]
regularisation = 1 / 15.581312752315547 

f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())

prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation, Smets_Wouters_2007), ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results


sol = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order=2)), maxiters = 10000) # this seems to achieve best results

round.(no_pandemic[1:7], digits=2)

# optimal_taylor_coefficients no pnademic
# 2.763652185541633
# 0.08963808582038361
# 0.8578346355200306

(1-0.8578346355200306) *  2.763652185541633
(1-0.8578346355200306) *  0.08963808582038361

round.(full_sample[1:7], digits=2)

# optimal_taylor_coefficients full sample
# 3.8945759865189147
# 0.19577912225186558
# 0.963753544670934

(1-0.963753544670934) * 3.8945759865189147
(1-0.963753544670934) * 0.19577912225186558







# plots optimal policy for different values of the shock standard deviations

function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    out = get_statistics(Smets_Wouters_2007,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = algo,
                    verbose = verbose)

    res = out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
    # println(res)

    return res
end

f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())

lbs = fill(0.0, 3)
ubs = fill(1e6, 3)


SS(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)

optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

# no pandemic weights
loss_function_weights = [1, 0.11627083536113818, 0.30115648430336367]
regularisation = 1 / 15.581312752315547 

stds = pars
# std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])
std_vals = no_pandemic
# std_vals = full_sample[1:7]

k_range = [.5] # .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.1 * vl, length = n_σ_range)

        # combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)

            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            # soll = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order=2)), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective) $(soll.u)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_2nd_$(nm).pdf")

    ii += 1
end






find_weights(sol.u, optimal_taylor_coefficients)

ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,sol.u[1:2]), 1 / sol.u[3])), optimal_taylor_coefficients)


derivs = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                                    x -> begin
                                        prob = OptimizationProblem(f, fill(0.5, 3), x, ub = ubs, lb = lbs)
                                        sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)
                                        return sol.u
                                    end, optimal_taylor_coefficients)

loss_function_weights_lower = copy(sol.u) - derivs[1] * [0.2123, 0.0102, -0.0435] # taylor_coef_stds
loss_function_weights_upper = copy(sol.u) + derivs[1] * [0.2123, 0.0102, -0.0435] # taylor_coef_stds


loss_function_weights = vcat(1, copy(sol.u[1:2]))

lbs = [eps(),eps(),eps()] #,eps()]
ubs = [1e6, 1e6, 1] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
regularisation = 1 / copy(sol.u[3])  #,1e-5]


get_statistics(Smets_Wouters_2007,   
                    optimal_taylor_coefficients,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = :first_order,
                    verbose = true)


calculate_cb_loss(optimal_taylor_coefficients, (loss_function_weights, regularisation), verbose = true)

# SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000)

# sol = solve(prob, Optimization.LBFGS(), maxiters = 10000)

# sol = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

# sol = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 1000)

calculate_cb_loss(sol.u, (loss_function_weights, regularisation))

# abs2.(sol.u)' * regularisation

# sol.objective
# loop across different levels of std

get_parameters(Smets_Wouters_2007, values = true)

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007)#, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0))
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw],[:crr, :crpi, :cry]))'
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Standard_deviation_and_∂standard_deviation∂parameter ∈ 19-element view(::Vector{Symbol},...)
# →   Variables ∈ 4-element view(::Vector{Symbol},...)
# And data, 19×4 adjoint(view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 23, 24, 25, 26, 27, 28, 29, 30, 31, 20, 19, 21])) with eltype Float64:
#                 (:ygap)      (:pinfobs)   (:drobs)      (:robs)
#   (:z_ea)        0.431892     0.0337031    0.0209319     0.0399137
#   (:z_eb)        0.498688     7.68657      0.2897       18.1822
#   (:z_eg)        0.00730732   0.00281355   0.00143873    0.00418418
#   (:z_em)        1.23788      0.168688     0.73829       0.561376
#   (:z_ew)       15.9711       0.535026     0.104924      0.710824
#   (:z_eqs)       0.067932     0.00306293   0.000692842   0.00469514
#   (:z_epinf)    24.2758       0.535294     0.297274      0.504287
#   (:crhoa)       0.0784198   -0.156896    -0.0374588    -0.270447
#   (:crhob)      -0.537715    29.0378       0.0992404    69.119
#   (:crhog)      -0.201827    -0.00292154   0.00139387   -0.00340863
#   (:crhoqs)      0.113207     0.00511452   0.0019416     0.00756271
#   (:crhoms)      2.09109      0.369009    -0.45944      -0.0168982
#   (:crhopinf)   73.8881       0.226225    -0.0511477    -0.0245807
#   (:crhow)      94.8585       2.11075      0.078647      2.75274
#   (:cmap)       -5.89078     -0.0945698   -0.0291701    -0.0982907
#   (:cmaw)      -57.2493      -1.54286     -0.146276     -2.06334
#   (:crr)         2.58469      0.271689    -0.330182     -0.128754
#   (:crpi)        0.607836    -0.624988    -0.0260916    -0.539138
#   (:cry)       -20.0359       3.69087      0.464149      2.69922



# ↑ z_ea    ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.6078559133318278
# ↑ z_eb    ⟹   ↑ inflation, ↑ output, ↑ persistence   -    0.06836618238325545
# ↑ z_eg    ⟹   → inflation, → output, → persistence   -    0.4203898197505046
# ↑ z_em    ⟹   ↑ inflation, ↑ output, ↓ persistence   -    0.6541387441075293
# ↑ z_ew    ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.4528480216201151
# ↑ z_eqs   ⟹   → inflation, ↑ output, → persistence   -    1.1088241818556892
# ↑ z_epinf ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.1267035923202942

# do the same for the shock autocorrelation

rhos = Smets_Wouters_2007.parameters[end-19:end-11]
rho_vals = copy(Smets_Wouters_2007.parameter_values[end-19:end-11])


k_range = [.5] # .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(rhos), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(rhos,rho_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.0 * vl, length = n_σ_range)

        # combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective) $(soll.u)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_2nd_$(nm).png")

    ii += 1
end


# Interpretation:
# Shock persistence:
# ↑ crhob     ⟹ ↑ inflation, ↑ output, ↑ persistence    -   0.9941945010996004
# ↑ crhoa     ⟹ ↑ inflation, ↑ output, ↓ persistence   -   0.9900258724661307
# ↑ crhog     ⟹ → inflation, → output, → persistence    -   0.944447821651772
# ↑ crhoqs    ⟹ → inflation, → output, → persistence    -   0.09136974979681929
# ↑ crhoms    ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.5469941169752605
# ↑ crhopinf  ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.9839879182859345
# ↑ crhow     ⟹ ↓ inflation, ↓ output, ↑ persistence    -   0.8176542834158012
# ↑ cmap      ⟹ ↑ inflation, ↑ output, ↓ persistence    -   0.46404242788618344
# ↑ cmaw      ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.7277828188461039

# Interpretation:
# starting from the implied optimal weights by the taylor rule coefficients resulting from the estimation to be optimal on the pre-pandemic period and setting the crdy coefficient to 0 we change the shock standard deviations and recalculate the optimal Taylor rule coefficients given the implied optimal loss weights
# (:a)     ↓ stronger reaction to inflation and output
# (:b)     ↓ weaker reaction to inflation and output
# (:gy)    ↑ no impact on optimal reaction function 
# (:qs)    ↑ slightly weaker response on inflaiton, stronger response to output
# (:ms)    ↓ weaker reaction to inflation and output
# (:spinf) ↓ stronger reaction to inflation and output
# (:sw)    →

# doing the same but letting only the standard deviations vary the following changes apply:
# (:a)     ↓ 0.048876448357432746   (0.05649332222233191)   stronger reaction to inflation and output, less persistence
# (:b)     ↑ 0.017434807203149848   (0.009154771190994018)  stronger reaction to inflation and output, more persistence - dominates
# (:gy)    ↑ 0.024531928343334983   (0.015818645868635427)  no impact on optimal reaction function 
# (:qs)    ↑ 0.1633892331129693     (0.13897662836032265)   slightly weaker response on inflaiton, slightly less persistence, stronger response to output
# (:ms)    ↓ 0.0031954169476854496  (0.007814003846949065)  weaker reaction to inflation and output, more persistence - compensates
# (:spinf) ↑ 0.05720222121202543    (0.047651064538222695)  weaker reaction to inflation and output, more persistence
# (:sw)    ↑ 2.1905984089710966     (2.0020346229399255)    weaker reaction to inflation and output, more persistence



# Demand shocks (Y↑ - pi↑ - R↑)
# z_eb	# risk-premium shock
# z_eg	# government shock
# z_eqs	# investment-specific shock


# Monetary policy (Y↓ - pi↓ - R↑)
# z_em	# interest rate shock

# Supply shock (Y↓ - pi↑ - R↑)
# z_ea	# technology shock

## Mark-up/cost-push shocks (Y↓ - pi↑ - R↑)
# z_ew	# wage mark-up shock
# z_epinf	# price mark-up shock



## Analysis of other sources of uncertainty
include("../models/Smets_Wouters_2007_ext.jl")

model = Smets_Wouters_2007_ext


# EA tighter priors (no crdy)
# estimated_par_vals = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(model, parameters = :crdy => 0, derivatives = false)

SS(model, parameters = pars .=> no_pandemic, derivatives = false)

# SS(model, parameters = :crdy => 0, derivatives = false)

# SS(model, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)


optimal_taylor_coefficients = [Dict(get_parameters(model, values = true))[i] for i in ["crpi", "cry", "crr"]]

# taylor_coef_stds = [0.2739, 0.0113, 0.0467]
taylor_coef_stds = [0.2123, 0.0102, 0.0435]

function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation, model = p

    out = get_statistics(model,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = algo,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
end

function find_weights(loss_function_weights_regularisation, p)
    optimal_taylor_coefficients, model = p
    loss_function_weights = loss_function_weights_regularisation[1:2]
    regularisation = 1 / loss_function_weights_regularisation[3]
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,loss_function_weights), regularisation, model)), optimal_taylor_coefficients))
end

lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())

prob = OptimizationProblem(f, fill(0.5, 3), (optimal_taylor_coefficients, model), ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)




loss_function_weights = vcat(1, copy(sol.u[1:2]))

lbs = [eps(),eps(),eps()] #,eps()]
ubs = [1e6, 1e6, 1] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
regularisation = 1 / copy(sol.u[3])  #,1e-5]

f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())

prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation, model), ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results



# std cprobw: 0.08
# std cprobp: 0.0321

stds = [:σ_pinf, :σ_cprobp, :σ_cprobw]
std_vals = [0.015, 0.0321, 0.08]
std_vals = [0.02, 0.028, 0.8]

# stdderivs = get_std(model)#, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0))
# stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))



k_range = [.5] # .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);

ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(0 * vl, .1 * vl, length = n_σ_range)

        # combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation, model), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(model, parameters = nm => σ, derivatives = false)
            
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            # soll = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order=2)), maxiters = 10000) # this seems to achieve best results
            # soll = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective) $(soll.u)")
        end

        
        SS(model, parameters = nm => 0, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    # push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "crr", colorbar=false))
    # push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * crpi", colorbar=false))
    # push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * cry", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_2nd_$(nm)_ext.pdf")

    # plots = []
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    
    # p = plot(plots...) # , plot_title = string(nm))
    # savefig(p,"OSR_$(nm)_ext.png")

    ii += 1
end


# Interpretation:
# increase in uncertainty regarding the calvo probabilities, or a measurement error of inflation in the Taylor rule all trigger a weaker inflation response and a stronger output response
