using Revise
using MacroModelling
import StatsPlots
using Random
# TODO: 
# - fix color handling for many colors (check how its done wiht auto)
# - write plot_model_estimates! and revisit plot_solution + ! version of it
# - x axis should be Int not floats
# - write model estimates func in get_functions
# - write the plots! funcs for all other alias funcs

# DONE:
# - implement switch to not show shock values | use the shock argument
# - see how palette comes in in the plots.jl codes
# - for model estimate/shock decomp remove zero entries



using CSV, DataFrames
include("../models/FS2000.jl")
using Dates

function quarter_labels(start::Date, n::Int)
    quarters = start:Month(3):(start + Month(3*(n-1)))
    return ["$(year(d))Q$(((month(d)-1) ÷ 3) + 1)" for d in quarters]
end

labels = quarter_labels(Date(1950, 1, 1), 192)

# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = labels)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat,1))
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)
plot_model_estimates(FS2000, data, presample_periods = 150)
plot_model_estimates(FS2000, data, presample_periods = 3, shock_decomposition = true, 
# transparency = 1.0,
plots_per_page = 4,
save_plots = true)

include("../models/GNSS_2010.jl")

ECB_palette = [
    "#003299",  # blue
    "#ffb400",  # yellow
    "#ff4b00",  # orange
    "#65b800",  # green
    "#00b1ea",  # light blue
    "#007816",  # dark green
    "#8139c6",  # purple
    "#5c5c5c"   # gray
]

plot_fevd(Smets_Wouters_2007, 
periods = 10,
plot_attributes = Dict(:xformatter => x -> string(Int(ceil(x))),:palette => ECB_palette)
)

include("models/RBC_CME_calibration_equations.jl")
algorithm = :first_order

vars = [:all, :all_excluding_obc, :all_excluding_auxiliary_and_obc, m.var[1], m.var[1:2], Tuple(m.timings.var), reshape(m.timings.var,1,length(m.timings.var)), string(m.var[1]), string.(m.var[1:2]), Tuple(string.(m.timings.var)), reshape(string.(m.timings.var),1,length(m.timings.var))]

init_state = get_irf(m, algorithm = algorithm, shocks = :none, levels = !(algorithm in [:pruned_second_order, :pruned_third_order]), variables = :all, periods = 1) |> vec

init_states = [[0.0], init_state, algorithm  == :pruned_second_order ? [zero(init_state), init_state] : algorithm == :pruned_third_order ? [zero(init_state), init_state, zero(init_state)] : init_state .* 1.01]

old_params = copy(m.parameter_values)

# options to itereate over
filters = [:inversion, :kalman]

sylvester_algorithms = (algorithm == :first_order ? [:doubling] : [[:doubling, :bicgstab], [:bartels_stewart, :doubling], :bicgstab, :dqgmres, (:gmres, :gmres)])

qme_algorithms = [:schur, :doubling]

lyapunov_algorithms = [:doubling, :bartels_stewart, :bicgstab, :gmres]

params = [old_params, 
            (m.parameters[1] => old_params[1] * exp(rand()*1e-4)), 
            Tuple(m.parameters[1:2] .=> old_params[1:2] .* 1.0001), 
            m.parameters .=> old_params, 
            (string(m.parameters[1]) => old_params[1] * 1.0001), 
            Tuple(string.(m.parameters[1:2]) .=> old_params[1:2] .* exp.(rand(2)*1e-4)), 
            old_params]

import MacroModelling: clear_solution_caches!


# @testset "plot_model_estimates" begin
    sol = get_solution(m)
    
    if length(m.exo) > 3
        n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
        var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]
    else
        var_idxs = [1]
    end

    Random.seed!(41823)

    simulation = simulate(m, algorithm = algorithm)

    data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)
    data = data_in_levels .- m.solution.non_stochastic_steady_state[var_idxs]

    
    
    if !(algorithm in [:second_order, :third_order])
        # plotlyjs_backend()

        # plot_shock_decomposition(m, data, 
        #                             algorithm = algorithm, 
        #                             data_in_levels = false)

        # gr_backend()

        plot_shock_decomposition(m, data, 
                                    algorithm = algorithm, 
                                # smooth = false,
                                    data_in_levels = false)
    end

    plot_model_estimates(m, data, 
                                algorithm = algorithm, 
                                data_in_levels = false)

    plot_model_estimates(m, data, 
                                # plot_attributes = Dict(:palette => :Accent),
                                shock_decomposition = true,
                                algorithm = algorithm, 
                                data_in_levels = false)

                                
    aa = get_estimated_variables(m, data, 
                                algorithm = algorithm, 
                                data_in_levels = false)

    aa = get_estimated_shocks(m, data, 
                                algorithm = algorithm, 
                                data_in_levels = false)

    for quadratic_matrix_equation_algorithm in qme_algorithms
        for lyapunov_algorithm in lyapunov_algorithms
            for sylvester_algorithm in sylvester_algorithms
                for tol in [MacroModelling.Tolerances(), MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                    clear_solution_caches!(m, algorithm)

                    plot_model_estimates(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false, 
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)

                    clear_solution_caches!(m, algorithm)
                
                    plot_model_estimates(m, data_in_levels, 
                                            algorithm = algorithm, 
                                            data_in_levels = true,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)
                end
            end
        end
    end

    for shock_decomposition in (algorithm in [:second_order, :third_order] ? [false] : [true, false])
        for filter in (algorithm == :first_order ? filters : [:inversion])
            for smooth in [true, false]
                for presample_periods in [0, 3]
                    clear_solution_caches!(m, algorithm)

                    plot_model_estimates(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false, 
                                            filter = filter,
                                            smooth = smooth,
                                            presample_periods = presample_periods,
                                            shock_decomposition = shock_decomposition)

                    clear_solution_caches!(m, algorithm)
                
                    plot_model_estimates(m, data_in_levels, 
                                            algorithm = algorithm, 
                                            data_in_levels = true,
                                            filter = filter,
                                            smooth = smooth,
                                            presample_periods = presample_periods,
                                            shock_decomposition = shock_decomposition)
                end
            end
        end
    end

    for parameters in params
            plot_model_estimates(m, data, 
                                    parameters = parameters,
                                    algorithm = algorithm, 
                                    data_in_levels = false)
    end

    for variables in vars
        plot_model_estimates(m, data, 
                                variables = variables,
                                algorithm = algorithm, 
                                data_in_levels = false)
    end

    for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2])]
        plot_model_estimates(m, data, 
                                shocks = shocks,
                                algorithm = algorithm, 
                                data_in_levels = false)
    end 

    for plots_per_page in [4,6]
        for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
            for max_elements_per_legend_row in [3,5]
                for extra_legend_space in [0.0, 0.5]
                    plot_model_estimates(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false,
                                            plot_attributes = plot_attributes,
                                            max_elements_per_legend_row = max_elements_per_legend_row,
                                            extra_legend_space = extra_legend_space,
                                            plots_per_page = plots_per_page,)
                end
            end
        end
    end

    # for backend in (Sys.iswindows() ? [:gr] : [:gr, :plotlyjs])
    #     if backend == :gr
    #         gr_backend()
    #     else
    #         plotlyjs_backend()
    #     end
        for show_plots in [true, false] # (Sys.islinux() ? backend == :plotlyjs ? [false] : [true, false] : [true, false])
            for save_plots in [true, false]
                for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                    for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) # (save_plots ? backend == :gr ? (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) : [:html,:json,:pdf,:png,:svg] : [:pdf])
                        plot_model_estimates(m, data, 
                                                algorithm = algorithm, 
                                                data_in_levels = false,
                                                show_plots = show_plots,
                                                save_plots = save_plots,
                                                save_plots_path = save_plots_path,
                                                save_plots_format = save_plots_format)
                    end
                end
            end
        end
    # end
# end

# @testset "plot_solution" begin
    
                        plot_solution(m, states[1], 
                                # plot_attributes = Dict(:palette => :Accent),
                                        algorithm = algos[end])


    states  = vcat(get_state_variables(m), m.timings.past_not_future_and_mixed)
    
    if algorithm == :first_order
        algos = [:first_order]
    elseif algorithm in [:second_order, :pruned_second_order]
        algos = [[:first_order], [:first_order, :second_order], [:first_order, :pruned_second_order], [:first_order, :second_order, :pruned_second_order]]
    elseif algorithm in [:third_order, :pruned_third_order]
        algos = [[:first_order], [:first_order, :second_order], [:first_order, :third_order], [:second_order, :third_order], [:third_order, :pruned_third_order], [:first_order, :second_order, :third_order], [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order]]
    end
    
    for variables in vars
        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)
            
                        plot_solution(m, states[1], 
                                        algorithm = algos[end],
                                        variables = variables,
                                        tol = tol,
                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        sylvester_algorithm = sylvester_algorithm)
                    end
                end
            end
        end
    end

    for plots_per_page in [1,4]
        for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
            plot_solution(m, states[1], algorithm = algos[end],
                            plot_attributes = plot_attributes,
                            plots_per_page = plots_per_page)
        end
    end

    
    # for backend in (Sys.iswindows() ? [:gr] : [:gr, :plotlyjs])
    #     if backend == :gr
    #         gr_backend()
    #     else
    #         plotlyjs_backend()
    #     end
        for show_plots in [true, false] # (Sys.islinux() ? backend == :plotlyjs ? [false] : [true, false] : [true, false])
            for save_plots in [true, false]
                for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                    for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) # (save_plots ? backend == :gr ? (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) : [:html,:json,:pdf,:png,:svg] : [:pdf])
                        plot_solution(m, states[1], algorithm = algos[end],
                                        show_plots = show_plots,
                                        save_plots = save_plots,
                                        save_plots_path = save_plots_path,
                                        save_plots_format = save_plots_format)
                    end
                end
            end
        end
    # end

    for parameters in params
        plot_solution(m, states[1], algorithm = algos[end],
                        parameters = parameters)
    end

    for σ in [0.5, 5]
        for ignore_obc in [true, false]
            for state in states[[1,end]]
                for algo in algos
                    plot_solution(m, state,
                                    σ = σ,
                                    algorithm = algo,
                                    ignore_obc = ignore_obc)
                end
            end
        end
    end

    # plotlyjs_backend()

    # plot_solution(m, states[1], algorithm = algos[end])

    # gr_backend()
# end


# @testset "plot_irf" begin
    

    # plotlyjs_backend()

    plot_IRF(m, algorithm = algorithm)

    # gr_backend()

    plot_irfs(m, algorithm = algorithm)

    plot_simulations(m, algorithm = algorithm)

    plot_simulation(m, algorithm = algorithm)

    plot_girf(m, algorithm = algorithm)

    for ignore_obc in [true,false]
        for generalised_irf in (algorithm == :first_order ? [false] : [true,false])
            for negative_shock in [true,false]
                for shock_size in [.1,1]
                    for periods in [1,10]
                        plot_irf(m, algorithm = algorithm, 
                                    ignore_obc = ignore_obc,
                                    periods = periods,
                                    generalised_irf = generalised_irf,
                                    negative_shock = negative_shock,
                                    shock_size = shock_size)
                    end
                end
            end
        end
    end

    

    shock_mat = randn(m.timings.nExo,3)

    shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

    shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

    for parameters in params
        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)
                                    
                        plot_irf(m, algorithm = algorithm, 
                                    parameters = parameters,
                                    tol = tol,
                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                    lyapunov_algorithm = lyapunov_algorithm,
                                    sylvester_algorithm = sylvester_algorithm)
                    end
                end
            end
        end
    end

    for initial_state in init_states
        clear_solution_caches!(m, algorithm)
                    
        plot_irf(m, algorithm = algorithm, initial_state = initial_state)
    end

    for variables in vars
        clear_solution_caches!(m, algorithm)
                    
        plot_irf(m, algorithm = algorithm, variables = variables)
    end

    for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
        clear_solution_caches!(m, algorithm)
                    
        plot_irf(m, algorithm = algorithm, shocks = shocks)
    end

    for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red), Dict(:palette => :Set1)]
        for plots_per_page in [4,6]
            plot_irf(m, algorithm = algorithm,
                        plot_attributes = plot_attributes,
                        plots_per_page = plots_per_page)
        end
    end

    # for backend in (Sys.iswindows() ? [:gr] : [:gr, :plotlyjs])
    #     if backend == :gr
    #         gr_backend()
    #     else
    #         plotlyjs_backend()
    #     end
        for show_plots in [true, false] # (Sys.islinux() ? backend == :plotlyjs ? [false] : [true, false] : [true, false])
            for save_plots in [true, false]
                for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                    for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) # (save_plots ? backend == :gr ? (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) : [:html,:json,:pdf,:png,:svg] : [:pdf])
                        plot_irf(m, algorithm = algorithm,
                                    show_plots = show_plots,
                                    save_plots = save_plots,
                                    save_plots_path = save_plots_path,
                                    save_plots_format = save_plots_format)
                    end
                end
            end
        end
    # end
# end


# @testset "plot_conditional_variance_decomposition" begin
    # plotlyjs_backend()
    
    plot_fevd(m)

    # gr_backend()

    plot_forecast_error_variance_decomposition(m)

    for periods in [10,40]
        for variables in vars
            plot_conditional_variance_decomposition(m, periods = periods, variables = variables)
        end
    end

    

    for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
        for quadratic_matrix_equation_algorithm in qme_algorithms
            for lyapunov_algorithm in lyapunov_algorithms
                clear_solution_caches!(m, algorithm)
                    
                plot_conditional_variance_decomposition(m, tol = tol,
                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                        lyapunov_algorithm = lyapunov_algorithm)
            end
        end
    end
    
    # for backend in (Sys.iswindows() ? [:gr] : [:gr, :plotlyjs])
    #     if backend == :gr
    #         gr_backend()
    #     else
    #         plotlyjs_backend()
    #     end
        for show_plots in [true, false] # (Sys.islinux() ? backend == :plotlyjs ? [false] : [true, false] : [true, false])
            for save_plots in [true, false]
                for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                    for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) # (save_plots ? backend == :gr ? (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) : [:html,:json,:pdf,:png,:svg] : [:pdf])
                        for plots_per_page in [4,6]
                            for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
                                for max_elements_per_legend_row in [3,5]
                                    for extra_legend_space in [0.0, 0.5]
                                        plot_conditional_variance_decomposition(m,
                                                                                plot_attributes = plot_attributes,
                                                                                max_elements_per_legend_row = max_elements_per_legend_row,
                                                                                extra_legend_space = extra_legend_space,
                                                                                show_plots = show_plots,
                                                                                save_plots = save_plots,
                                                                                plots_per_page = plots_per_page,
                                                                                save_plots_path = save_plots_path,
                                                                                save_plots_format = save_plots_format)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    # end
# end


# test conditional forecasting

new_sub_irfs_all  = get_irf(m, algorithm = algorithm, verbose = false, variables = :all, shocks = :all)
varnames = axiskeys(new_sub_irfs_all,1)
shocknames = axiskeys(new_sub_irfs_all,3)
sol = get_solution(m)
# var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[[1,end]]
n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]


stst  = get_irf(m, variables = :all, algorithm = algorithm, shocks = :none, periods = 1, levels = true) |> vec

conditions = []

cndtns = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,1),2)
cndtns[var_idxs[1],1] = .01
cndtns[var_idxs[2],2] = .02

push!(conditions, cndtns)

cndtns = spzeros(size(new_sub_irfs_all,1),2)
cndtns[var_idxs[1],1] = .01
cndtns[var_idxs[2],2] = .02

push!(conditions, cndtns)

cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
cndtns[1,1] = .01
cndtns[2,2] = .02

push!(conditions, cndtns)

cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
cndtns[1,1] = .01
cndtns[2,2] = .02

push!(conditions, cndtns)

conditions_lvl = []

cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
cndtns_lvl[1,1] = .01 + stst[var_idxs[1]]
cndtns_lvl[2,2] = .02 + stst[var_idxs[2]]

push!(conditions_lvl, cndtns_lvl)

cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
cndtns_lvl[1,1] = .01 + stst[var_idxs[1]]
cndtns_lvl[2,2] = .02 + stst[var_idxs[2]]

push!(conditions_lvl, cndtns_lvl)


shocks = []

push!(shocks, nothing)

if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
    shcks = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,3),1)
    shcks[1,1] = .1

    push!(shocks, shcks)

    shcks = spzeros(size(new_sub_irfs_all,3),1)
    shcks[1,1] = .1
    
    push!(shocks, shcks)

    shcks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = [shocknames[1]], Periods = [1])
    shcks[1,1] = .1

    push!(shocks, shcks)

    shcks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = string.([shocknames[1]]), Periods = [1])
    shcks[1,1] = .1

    push!(shocks, shcks)
end

# for backend in (Sys.iswindows() ? [:gr] : [:gr, :plotlyjs])
#     if backend == :gr
#         gr_backend()
#     else
#         plotlyjs_backend()
#     end
    for show_plots in [true, false] # (Sys.islinux() ? backend == :plotlyjs ? [false] : [true, false] : [true, false])
        for save_plots in [true, false]
            for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) # (save_plots ? backend == :gr ? (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf]) : [:html,:json,:pdf,:png,:svg] : [:pdf])
                    for plots_per_page in [1,4]
                        for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
                            plot_conditional_forecast(m, conditions[1],
                                                        conditions_in_levels = false,
                                                        initial_state = [0.0],
                                                        algorithm = algorithm, 
                                                        shocks = shocks[1],
                                                        plot_attributes = plot_attributes,
                                                        show_plots = show_plots,
                                                        save_plots = save_plots,
                                                        plots_per_page = plots_per_page,
                                                        save_plots_path = save_plots_path,
                                                        save_plots_format = save_plots_format)
                        end
                    end
                end
            end
        end
    end
# end



for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
    for quadratic_matrix_equation_algorithm in qme_algorithms
        for lyapunov_algorithm in lyapunov_algorithms
            for sylvester_algorithm in sylvester_algorithms
                clear_solution_caches!(m, algorithm)
            
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            algorithm = algorithm, 
                                            shocks = shocks[end],
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)
            end
        end
    end
end

for periods in [0,10]
    for levels in [true, false]
        clear_solution_caches!(m, algorithm)
    
        plot_conditional_forecast(m, conditions[end],
                                    conditions_in_levels = false,
                                    algorithm = algorithm, 
                                    periods = periods,
                                    # levels = levels,
                                    shocks = shocks[end])

        
        clear_solution_caches!(m, algorithm)
    
        plot_conditional_forecast(m, conditions_lvl[end],
                                    algorithm = algorithm, 
                                    periods = periods,
                                    # levels = levels,
                                    shocks = shocks[end])

    end
end

for variables in vars
    plot_conditional_forecast!(m, conditions[end],
                                conditions_in_levels = false,
                                algorithm = algorithm, 
                                plot_attributes = Dict(:palette => :Set2),
                                variables = variables)
end

for initial_state in init_states
    plot_conditional_forecast(m, conditions[end],
                                conditions_in_levels = false,
                                initial_state = initial_state,
                                algorithm = algorithm)
end

for shcks in shocks[2:end]
    plot_conditional_forecast!(m, conditions[end],
                                conditions_in_levels = false,
                                algorithm = algorithm, 
                                shocks = shcks)
end

for parameters in params
    plot_conditional_forecast(m, conditions[end],
                                parameters = parameters,
                                conditions_in_levels = false,
                                algorithm = algorithm)
end

for cndtns in conditions
    plot_conditional_forecast(m, cndtns,
                                conditions_in_levels = false,
                                algorithm = algorithm)
end

for cndtns in conditions
    plot_conditional_forecast!(m, cndtns,
                                conditions_in_levels = false,
                                algorithm = algorithm)
end
cond = copy(conditions[2])
cond.nzval .+= init_states[2][[2,5]]

plot_conditional_forecast(RBC_CME,
                        conditions2, 
                        # shocks = shocks, 
                        # plot_type = :stack,
                        # save_plots = true,
                        conditions_in_levels = false)

plot_conditional_forecast!(m, cond,
                            conditions_in_levels = true,
                            algorithm = algorithm)

plot_conditional_forecast!(m, conditions[2],
                            conditions_in_levels = false,
                            algorithm = algorithm)




# TODO:
# fix other twinx situations (simplify it), especially bar plot in decomposition can have dual axis with the yticks trick
# put plots back in Docs 
# redo plots in docs

@model RBC_CME begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

@parameters RBC_CME begin
    alpha = .157
    beta = .999
    delta = .0226
    Pibar = 1.0008
    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# c is conditioned to deviate by 0.01 in period 1 and y is conditioned to deviate by 0.02 in period 3
conditions = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,3),Variables = [:c,:y], Periods = 1:3)
conditions[1,1] = .01
conditions[2,3] = .02

# in period 2 second shock (eps_z) is conditioned to take a value of 0.05
shocks = Matrix{Union{Nothing,Float64}}(undef,2,1)
shocks[1,1] = .05

plot_conditional_forecast(RBC_CME, conditions, 
                        shocks = shocks, 
                        conditions_in_levels = false)

plot_conditional_forecast!(RBC_CME, conditions, 
                        # shocks = shocks, 
                        # plot_type = :stack,
                        # save_plots = true,
                        conditions_in_levels = false)

conditions2 = Matrix{Union{Nothing,Float64}}(undef,7,2)
conditions2[4,1] = .01
# conditions2[6,2] = .02

conditions2 = KeyedArray(Matrix{Union{Nothing,Float64}}(undef,2,3),Variables = [:c,:y], Periods = 1:3)
conditions2[2,1] = .01
conditions2[1,3] = .02

plot_conditional_forecast(RBC_CME,
                        conditions2, 
                        # shocks = shocks, 
                        # plot_type = :stack,
                        # save_plots = true,
                        conditions_in_levels = false)


plot_conditional_forecast(RBC_CME,
                        conditions2, 
                        shocks = shocks, 
                        algorithm = :pruned_second_order,
                        plot_type = :stack,
                        # save_plots = true,
                        conditions_in_levels = false)


include("../models/GNSS_2010.jl")

model = GNSS_2010
get_shocks(model)

shcks = :e_y
vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = 1.2,
            # plot_type = :stack,
            shocks = shcks, variables = vars,
            save_plots = true)

plot_irf!(model, 
negative_shock = true,
            shock_size = 1.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = 0.2,
            algorithm = :pruned_second_order,
            # plot_type = :stack,
            shocks = shcks, variables = vars)

include("../models/Gali_2015_chapter_3_nonlinear.jl")

get_shocks(Gali_2015_chapter_3_nonlinear)

plot_irf!(Gali_2015_chapter_3_nonlinear, 
            # shock_size = 1.2,
            plot_type = :stack,
            shocks = :eps_a, variables = [:C,:Y])


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, algorithm = :pruned_second_order, shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = 1.2,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = 1.2,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = -1,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, 
            shock_size = -1,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            periods = 5,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

get_shocks(model)

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = :e_j, 
            # plot_type = :stack,
            variables = vars)

plot_irf!(model, shocks = [:e_j, :e_me], 
            plot_type = :stack,
            variables = vars)

plot_irf!(model, 
            # plot_type = :stack,
            variables = vars)



vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

get_shocks(model)

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :pruned_second_order, 
            shock_size = -1,
            plot_type = :stack,
            shocks = shcks, variables = vars)

plot_irf!(model, algorithm = :second_order, 
            shock_size = -1,
            # plot_type = :stack,
            shocks = shcks, variables = vars)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars[2:end], shock_size = -1)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

for a in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
    plot_irf!(model, shocks = shcks, variables = vars, algorithm = a)
end


vars = [:C, :K, :Y, :r_k, :w_p]

plot_irf(model, shocks = shcks, variables = vars)

for a in [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
    plot_irf!(model, shocks = shcks, variables = vars, algorithm = a)
end



vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars)

plot_irf!(model, shocks = shcks, 
            plot_type = :stack,
            variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, shock_size = 2, variables = vars, negative_shock = true)

plot_irf!(model, shocks = :e_j, shock_size = -2, variables = vars, negative_shock = true, algorithm = :second_order)


vars = [:C, :K, :Y, :r_k, :w_p, :rr_e, :pie, :q_h, :l_p]

plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, generalised_irf = true, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order)

plot_irf!(model, shocks = shcks, variables = vars, generalised_irf = true, algorithm = :pruned_third_order)




include("../models/Gali_2015_chapter_3_obc.jl")

model = Gali_2015_chapter_3_obc
get_shocks(model)
get_variables(model)[1:10]
shcks = :eps_z
vars = [:A, :C, :MC, :M_real, :N, :Pi, :Pi_star, :Q, :R, :S]

plot_irf(model, shocks = shcks, variables = vars, periods = 10)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, periods = 10, ignore_obc = true)

plot_irf!(model, shocks = shcks, 
            # plot_type = :stack,
            variables = vars, periods = 10, shock_size = 2, ignore_obc = false)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, shock_size = 2, ignore_obc = true)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, shock_size = 4, ignore_obc = false)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, shock_size = 4, ignore_obc = true)



plot_irf(model, shocks = shcks, variables = vars, periods = 10)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 2, ignore_obc = false)

plot_irf!(model, shocks = shcks, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 2, ignore_obc = true)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 4, ignore_obc = false)

plot_irf!(model, shocks = :eps_a, variables = vars, periods = 10, algorithm = :pruned_second_order, shock_size = 4, ignore_obc = true)


plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order, quadratic_matrix_equation_algorithm = :doubling)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_second_order, sylvester_algorithm = :doubling)

plot_irf(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order)

plot_irf!(model, shocks = shcks, variables = vars, algorithm = :pruned_third_order, quadratic_matrix_equation_algorithm = :doubling)


get_parameters(model, values = true)

plot_irf(model, shocks = shcks, variables = vars, parameters = :α => .25)

plot_irf!(model, shocks = shcks, variables = vars, parameters = :α => .2)

SS(model, derivatives = false, parameters = :α => .25)(:R)
SS(model, derivatives = false, parameters = :α => .2)(:R)


# DONE: handle initial state and tol

init_state = get_irf(model, shocks = :none, variables = :all,
       periods = 1, levels = true)

init_state[1] += 1

plot_irf(model, shocks = shcks, variables = vars, ignore_obc = true,
initial_state = vec(init_state))


plot_irf!(model, shocks = :none, variables = vars, ignore_obc = true,
       initial_state = vec(init_state), 
            plot_type = :stack,
    #    algorithm = :second_order
       )
       
init_state_2 = get_irf(model, shocks = :none, variables = :all, periods = 1, levels = true)

init_state_2[1] += 2

init_state[1] += 2

plot_irf!(model, shocks = :none, variables = vars, ignore_obc = true,initial_state = vec(init_state))

       
# init_state_2 = get_irf(model, shocks = :none, variables = :all, periods = 1, levels = true)

init_state[1] += .2

plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true,
       algorithm = :second_order,
initial_state = vec(init_state)
)


init_state_2 = get_irf(model, shocks = :none, variables = :all, algorithm = :pruned_second_order, periods = 1, levels = false)

plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true,
       algorithm = :pruned_second_order,
initial_state = vec(init_state_2)
)


plot_irf(model, shocks = shcks, variables = vars)

# plot_irf!(model, shocks = shcks, variables = vars, ignore_obc = true)

plot_irf!(model, shocks = shcks, variables = vars, tol = Tolerances(NSSS_acceptance_tol = 1e-8))

plot_irf!(model, shocks = shcks, variables = vars, quadratic_matrix_equation_algorithm = :doubling)


@model RBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end;

@parameters RBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end;

plot_irf(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2], algorithm = :second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.95, :ρ => 0.2], algorithm = :pruned_second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.955, :ρ => 0.2], algorithm = :second_order)

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.957, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.012, :β => 0.97, :ρ => 0.5])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.01, :β => 0.97, :ρ => 0.55])

MacroModelling.plot_irf!(RBC, parameters = [:std_z => 0.021, :β => 0.97, :ρ => 0.55])


include("models/SW07_nonlinear.jl")

hcat(SS(SW07_nonlinear, derivatives = false, parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])[30:end]
,SS(SW07_nonlinear, derivatives = false, parameters = :calfa => 0.15)[30:end])

get_shocks(SW07_nonlinear)
shock_series = KeyedArray(zeros(2,12), Shocks = [:eb, :ew], Periods = 1:12)
shock_series[1,2] = 1
shock_series[2,12] = -1
plot_irf(SW07_nonlinear, shocks = :ew, 
                        # negative_shock = true,
                        # generalised_irf = false,
                        # algorithm = :pruned_second_order,
                        # variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        # :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])


plot_irf!(SW07_nonlinear, shocks = shock_series, 
                        # negative_shock = true,
                        # generalised_irf = false,
                        # algorithm = :pruned_second_order,
                        # variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        # :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])

plot_irf!(SW07_nonlinear, shocks = :ew, 
                        # generalised_irf = true,
                        algorithm = :pruned_second_order,
                        # shock_size = 2,
                        # quadratic_matrix_equation_algorithm = :doubling,
                        # tol = MacroModelling.Tolerances(NSSS_acceptance_tol = 1e-10),
                        # negative_shock = true,
                        variables = [:robs,:ygap,:pinf,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:c,:k],
                        # variables = [:ygap],
                        parameters = [:ctrend => .365, :curvw => 10, :calfa => 0.18003])

for s in setdiff(get_shocks(SW07_nonlinear),["ew"])
    MacroModelling.plot_irf!(SW07_nonlinear, shocks = s,
                            # variables = [:robs,:ygap,:pinf,
                            # :gamw1,:gamw2,:gamw3,
                            # :inve,:c,:k],
                            # variables = [:ygap],
                            # plot_type = :stack,
                            parameters = [:ctrend => .35, :curvw => 10, :calfa => 0.18003])
end

# DONE: handle case where one plots had one shock, the other has multiple ones
# DONE: when difference is along one dimension dont use label but legend only
MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :epinf,
                        variables = [:gam1,:gam2,:gam3,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k])

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = [:ew],
                        # plots_per_page = 9,
                        variables = [:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :calfa => 0.15)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = [:epinf,:ew],
                        variables = [:gam1,:gam2,:gam3,
                        # :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :calfa => 0.15)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        variables = [:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :curvw => 9)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        variables = [#:gam1,:gam2,:gam3,
                        :gamw1,:gamw2,:gamw3,
                        :inve,:kp,:k],
                        parameters = :cgy => .45)

MacroModelling.plot_irf!(SW07_nonlinear, 
                        shocks = :ew,
                        # plots_per_page = 4,
                        # variables = [:zcap,:gam1],
                        # variables = [:dy,:robs,:y,
                        # :xi,:ygap,
                        # :wnew,:xi,:ygap,
                        # :k,:kp,:r],
                        parameters = :ctrend => .5)

get_parameters(SW07_nonlinear, values = true)

diffdict = MacroModelling.compare_args_and_kwargs(MacroModelling.irf_active_plot_container)

using StatsPlots, DataFrames
using Plots

diffdict[:parameters]
mapreduce((x, y) -> x ∪ y, diffdict[:parameters])

df = diffdict[:parameters]|>DataFrame
param_nms = diffdict[:parameters]|>keys|>collect|>sort

plot_vector = Pair{String,Any}[]
for param in param_nms
    push!(plot_vector, String(param) => diffdict[:parameters][param])
end

pushfirst!(plot_vector, "Plot index" => 1:length(diffdict[:parameters][param_nms[1]]))


function plot_df(plot_vector::Vector{Pair{String,Any}})
    # Determine dimensions from plot_vector
    ncols = length(plot_vector)
    nrows = length(plot_vector[1].second)
        
    bg_matrix = ones(nrows + 1, ncols)
    bg_matrix[1, :] .= 0.35 # Header row
    for i in 3:2:nrows+1
        bg_matrix[i, :] .= 0.85
    end
 
    # draw the "cells"
    df_plot = heatmap(bg_matrix;
                c = cgrad([:lightgrey, :white]),      # Color gradient for background
                yflip = true,  
                tick=:none,
                legend=false,
                framestyle = :none, # Keep the outer box 
                cbar=false)
 
    # overlay the header and numeric values
    for j in 1:ncols
        annotate!(df_plot, j, 1, text(plot_vector[j].first, :center, 8)) # Header
        for i in 1:nrows
            annotate!(df_plot, j, i+1, text(string(plot_vector[j].second[i]), :center, 8))
        end
    end
    return df_plot
end

plot_df(plot_vector)
