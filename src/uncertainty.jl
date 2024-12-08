using Revise
using MacroModelling
using StatsPlots
using Optim, LineSearches
using Optimization, OptimizationNLopt#, OptimizationOptimJL
using Zygote, ForwardDiff, FiniteDifferences
using BenchmarkTools

include("../models/Smets_Wouters_2007 copy.jl")

# US SW07 sample estims
# estimated_par_vals = [0.4818650901000989, 0.24054470291311028, 0.5186956692202958, 0.4662413867655003, 0.23136135922950385, 0.13132950287219664, 0.2506090809487915, 0.9776707755474057, 0.2595790622654468, 0.9727418060187103, 0.687330720531337, 0.1643636762401503, 0.9593771388356938, 0.9717966717403557, 0.8082505346152592, 0.8950643861525535, 5.869499350284732, 1.4625899840952736, 0.724649200081708, 0.7508616008157103, 2.06747381157293, 0.647865359908012, 0.585642549132298, 0.22857733002230182, 0.4476375712834215, 1.6446238878581076, 2.0421854715489007, 0.8196744223749656, 0.10480818163546246, 0.20376610336806866, 0.7312462829038883, 0.14032972276989308, 1.1915345520903131, 0.47172181998770146, 0.5676468533218533, 0.2071701728019517]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# EA long sample
# estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# EA tighter priors (no crdy)
estimated_par_vals = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]

estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# find optimal loss coefficients
# Problem definition, find the loss coefficients such that the derivatives of the Taylor rule coefficients wrt the loss are 0
# lbs = [0,0]
# ubs = [1e6, 1e6] #, 1e6]
# initial_values = [.3 ,.3] # ,0.2347]

# var = get_variance(Smets_Wouters_2007, derivatives = false)



# Define the given vector
# SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

# loss_function_weights = [1, 1, .1]

# lbs = [eps(),eps(),eps()] #,eps()]
# ubs = [1-eps(), 1e6, 1e6] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
# regularisation = [1e-7,1e-5,1e-5]  #,1e-5]

# optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

# out = get_statistics(Smets_Wouters_2007,   
#                     optimal_taylor_coefficients,
#                     parameters = [:crr, :crpi, :cry],#, :crdy],
#                     variance = [:ygap, :pinfobs, :drobs],
#                     algorithm = :first_order,
#                     verbose = true)


# out[:variance]' * loss_function_weights + abs2.(initial_values)' * regularisation





# function calculate_loss(loss_function_weights,regularisation; verbose = false)
#     out = get_statistics(Smets_Wouters_2007,   
#                     [0.824085387718046, 1.9780022172135707, 4.095695818850862],
#                     # [0.935445005053725, 1.0500350793944067, 0.14728806935911198, 0.05076653598648485, 0],
#                     parameters = [:crr, :crpi, :cry, :crdy],
#                     variance = [:ygap, :pinfobs, :drobs],
#                     algorithm = :first_order,
#                     verbose = verbose)

#     return out[:variance]' * loss_function_weights + abs2.([0.824085387718046, 1.9780022172135707, 4.095695818850862,0])' * regularisation
# end

optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

taylor_coef_stds = [0.2739, 0.0113, 0.0467]


function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    out = get_statistics(Smets_Wouters_2007,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
end

function find_weights(loss_function_weights_regularisation, optimal_taylor_coefficients)
    loss_function_weights = loss_function_weights_regularisation[1:2]
    regularisation = 1 / loss_function_weights_regularisation[3]
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,loss_function_weights), regularisation)), optimal_taylor_coefficients))
end

# find_weights(vcat(loss_function_wts, 1 / regularisation[1]), optimal_taylor_coefficients)

# get_parameters(Smets_Wouters_2007, values = true)
lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())

prob = OptimizationProblem(f, fill(0.5, 3), optimal_taylor_coefficients, ub = ubs, lb = lbs)

# sol = solve(prob, NLopt.LD_TNEWTON(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

find_weights(sol.u, optimal_taylor_coefficients)

ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,sol.u[1:2]), 1 / sol.u[3])), optimal_taylor_coefficients)


derivs = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                                    x -> begin
                                        prob = OptimizationProblem(f, fill(0.5, 3), x, ub = ubs, lb = lbs)
                                        sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)
                                        return sol.u
                                    end, optimal_taylor_coefficients)

loss_function_weights_lower = copy(sol.u) - derivs[1] * [.2739,0.0113,-0.0467] # taylor_coef_stds
loss_function_weights_upper = copy(sol.u) + derivs[1] * [.2739,0.0113,-0.0467] # taylor_coef_stds


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
# sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000)

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
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 4-element view(::Vector{Symbol},...)
# →   Standard_deviation_and_∂standard_deviation∂parameter ∈ 10-element view(::Vector{Symbol},...)
# And data, 4×10 view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 20, 19, 21]) with eltype Float64:
#               (:z_ea)     (:z_eb)    (:z_eg)      (:z_em)    (:z_ew)     (:z_eqs)     (:z_epinf)  (:crr)     (:crpi)     (:cry)
#   (:ygap)      1.5512      1.43771    0.0386122    2.92771    9.07026     0.165315    22.7405      8.57412    0.183427  -41.4294
#   (:pinfobs)   0.0441266   2.33557    0.00261732   0.157377   0.225685    0.00387798   0.878799    0.343056  -0.19993     1.93685
#   (:drobs)     0.0142478   0.117477   0.00070993   0.768874   0.0522678   0.00104345   0.412586   -0.367055  -0.013684    0.381459
#   (:robs)      0.0576051   7.14347    0.0042997    0.787224   0.286545    0.00582915   1.00055     0.102156  -0.162961    1.65897

# Zygote.gradient(x->calculate_cb_loss(x,regularisation), sol.u)[1]


# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)

# Zygote.gradient(x->calculate_cb_loss(x,regularisation * 1),sol.u)[1]

# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1] * 1.05, derivatives = false)

# using FiniteDifferences

# FiniteDifferences.hessian(x->calculate_cb_loss(x,regularisation * 0),sol.u)[1]


# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)



# SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

# nms = []

k_range = .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.1 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "crr", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "(1 - crr) * cry", colorbar=false))

    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm)_surface.png")

    # plots = []
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Std($nm)", ylabel = "crr", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Std($nm)", ylabel = "(1 - crr) * crpi", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Std($nm)", ylabel = "(1 - crr) * cry", colorbar=false))
    
    # p = plot(plots...) # , plot_title = string(nm))
    # savefig(p,"OSR_$(nm).png")

    ii += 1
end



ii = 1
for (nm,vl) in zip(stds,std_vals)
    plots = []
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "crr", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "(1 - crr) * crpi", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "(1 - crr) * cry", colorbar=true))

    p = plot(plots...) # , plot_title = string(nm))

    savefig(p,"OSR_$(nm)_contour.png")
    ii += 1
end


xs = [string("x", i) for i = 1:10]
ys = [string("y", i) for i = 1:4]
z = float((1:4) * reshape(1:10, 1, :))
heatmap(xs, ys, z, aspect_ratio = 1)


contour(x,y,Z)

ii = 2

heatmap((coeff[1,ii,:,4]), (coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]))#, label = "", xlabel = "Loss weight: Δr", ylabel = "Std()", zlabel = "crr", colorbar=false)
heatmap(vec(coeff[:,1,:,4]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "crpi")
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "cry")

shck = 7
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec((1 .- coeff[:,1,:,3]) .* coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * crpi")
surface(vec(coeff[:,shck,:,1]), vec(coeff[:,shck,:,2]), vec((1 .- coeff[:,shck,:,3]) .* coeff[:,shck,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * cry")


surface(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr")

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [i[2] for i in coeffs], label = "", ylabel = "crpi"))
    push!(plots, plot(σ_range, [i[3] for i in coeffs], label = "", ylabel = "cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_direct_$nm.png")
    # display(p)
end

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



# demand shock (Y↑ - pi↑ - R↑): more aggressive on all three measures
# irf: GDP and inflation in same direction so you can neutralise this shocks at the cost of higher rate volatility

# supply shocks (Y↓ - pi↑ - R↑): more aggressive on inflation and GDP and less so on inflation
# trade off betwen GDP and inflation will probably dampen interest rate voltility so you can allow yourself to smooth less

# mark-up shocks (Y↓ - pi↑ - R↑): less aggressive on inflation and GDP but more smoothing
# low effectiveness wrt inflation, high costs, inflation less sticky



# try with EA parameters from estimation
estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)


prob = OptimizationProblem(f, initial_values, regularisation / 100, ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0));
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 4-element view(::Vector{Symbol},...)
# →   Standard_deviation_and_∂standard_deviation∂parameter ∈ 11-element view(::Vector{Symbol},...)
# And data, 4×11 view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 20, 19, 21, 22]) with eltype Float64:
#               (:z_ea)     (:z_eb)      (:z_eg)      (:z_em)     (:z_ew)    (:z_eqs)      (:z_epinf)   (:crr)    (:crpi)     (:cry)       (:crdy)
#   (:ygap)      0.0173547   0.151379     0.00335788   0.303146    4.07402    0.0062702     1.15872    -60.0597    0.0553865  -0.0208848   -0.116337
#   (:pinfobs)   0.0112278   2.48401e-5   9.97486e-5   0.134175    8.97266    0.000271719   3.75081     90.5474   -0.0832394   0.0312395    0.105122
#   (:drobs)     0.289815    3.7398       0.0452899    0.0150356   0.731132   0.0148536     0.297607     4.20058  -0.0045197   0.00175464   0.121104
#   (:robs)      0.216192    1.82174      0.0424333    0.115266    7.89551    0.0742737     2.57712     80.8386   -0.0743082   0.0273497    0.14874

Smets_Wouters_2007.parameter_values[indexin([:crr, :crpi, :cry, :crdy],Smets_Wouters_2007.parameters)]

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[2] for i in coeffs], label = "", ylabel = "(1 - crr) * crpi"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[3] for i in coeffs], label = "", ylabel = "(1 - crr) * cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_EA_$nm.png")
    # display(p)
end





solopt = solve(prob, Optimization.LBFGS(), maxiters = 10000)
soloptjl = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

sol_mlsl = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)


f_zyg = OptimizationFunction(calculate_cb_loss, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# 32.749
@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

sol_zyg = solve(prob_zyg, NLopt.LN_PRAXIS(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_SBPLX(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_NELDERMEAD(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_SLSQP(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_MMA(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_CCSAQ(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_TNEWTON(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxiters = 1000)
sol_zyg.u

f_for = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)


ForwardDiff.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

Zygote.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

sol = Optim.optimize(x->calculate_cb_loss(x,[]), 
                    lbs, 
                    ubs, 
                    initial_values, 
                    # LBFGS(),
                    # NelderMead(),
                    # Optim.Fminbox(NelderMead()), 
                    # Optim.Fminbox(LBFGS()), 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                           show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ))#,ad = AutoZgote())

pars = Optim.minimizer(sol)


get_statistics(Smets_Wouters_2007,   
                    sol.u,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = true)

## Central bank loss function: Loss = θʸ * var(Ŷ) + θᵖⁱ * var(π̂) + θᴿ * var(ΔR)
loss_function_weights = [1, .3, .4]

lbs = [eps(),eps(),eps()]
ubs = [1e2,1e2,1-eps()]
initial_values = [1.5, 0.125, 0.75]

function calculate_cb_loss(parameter_inputs; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

calculate_cb_loss(initial_values, verbose = true)


@time sol = Optim.optimize(calculate_cb_loss, 
                    lbs, 
                    ubs, 
                    initial_values, 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    # LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                        #    show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ));

pars = Optim.minimizer(sol)


out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                initial_values,
                parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                variance = [:Ŷ,:π̂,:ΔR],
                algorithm = :first_order,
                verbose = true)
out[:variance]
dd = Dict{Symbol,Array{<:Real}}()
dd[:variance] = out[1]

init_params = copy(Gali_2015_chapter_3_nonlinear.parameter_values)

function calculate_cb_loss_Opt(parameter_inputs,p; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

f_zyg = OptimizationFunction(calculate_cb_loss_Opt, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

f_for = OptimizationFunction(calculate_cb_loss_Opt, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)

# Import a solver package and solve the optimization problem
# sol = solve(prob, NLopt.LN_PRAXIS());
# sol.u
@time sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

@benchmark sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

using ForwardDiff
ForwardDiff.gradient(calculate_cb_loss,initial_values)
Zygote.gradient(calculate_cb_loss,initial_values)[1]
# SS(Gali_2015_chapter_3_nonlinear, parameters = :std_std_a => .00001)

SS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std³ = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std² = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
std¹ = get_std(Gali_2015_chapter_3_nonlinear)

std³([:π,:Y,:R],:)
std²(:π,:)
std¹(:π,:)

plot_solution(Gali_2015_chapter_3_nonlinear, :ν, algorithm = [:pruned_second_order, :pruned_third_order])

mean² = get_mean(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
mean¹ = get_mean(Gali_2015_chapter_3_nonlinear)

mean²(:π,:)
mean¹(:π,:)

get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => 1.5, :σ_σ_a => 2.0), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)

get_parameters(Gali_2015_chapter_3_nonlinear, values = true)

n = 5
res = zeros(3,n^2)

SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_π
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :σ_π => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_π", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Pi.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_Y
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_Y => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_Y", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Y.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.8,n) # std_std_a
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_a => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_a", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_tfp.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_std_z
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_z => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_z", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_z.png")




SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_θ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_θ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_θ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_calvo.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_ϕᵖⁱ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ϕᵖⁱ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ϕᵖⁱ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_infl_reaction.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_ā
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ā => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (60,65),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ā", 
                    zlabel = "std(Inflation)")

savefig("tfp_std_dev.png")
