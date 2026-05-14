import Zygote, FiniteDifferences, ForwardDiff, Mooncake, DifferentiationInterface, ADTypes
import MatrixEquations
import LinearAlgebra as ℒ
import StatsPlots
using Random
Random.seed!(1234)

# Diagnostic wrapper: prints achieved atol/rtol when isapprox fails
function check_isapprox(a, b; kwargs...)
    result = isapprox(a, b; kwargs...)
    if !result
        d = a .- b
        frobenius_diff = ℒ.norm(d)
        maxnorm = max(ℒ.norm(a), ℒ.norm(b))
        eff_rtol = maxnorm > 0 ? frobenius_diff / maxnorm : Inf
        max_abs = maximum(abs.(d))
        safe_denom = max.(abs.(a), abs.(b), eps())
        max_rel = maximum(abs.(d) ./ safe_denom)
        has_nan = any(isnan, a) || any(isnan, b)
        has_inf = any(isinf, a) || any(isinf, b)
        printstyled("  ⚠ APPROX FAIL: eff_rtol=$(eff_rtol), max_elem_abs=$(max_abs), max_elem_rel=$(max_rel), has_nan=$(has_nan), has_inf=$(has_inf), size=$(size(a))\n", color=:yellow)
    end
    return result
end

function functionality_test(m, m2; algorithm = :first_order, plots = true)
    rndnmbr = rand(max(length(m.parameter_values),2))
    old_params = copy(m.parameter_values)
    old_params2 = copy(m2.parameter_values)
    
    # options to itereate over
    filters = [:inversion, :kalman]

    sylvester_algorithms = (algorithm == :first_order ? [:doubling] : [[:doubling, :bicgstab], [:bartels_stewart, :doubling], :bicgstab, :dqgmres, (:gmres, :gmres)])

    qme_algorithms = [:schur, :doubling]

    lyapunov_algorithms = [:doubling, :bartels_stewart, :bicgstab, :gmres]

    params = [old_params, 
                (m.constants.post_complete_parameters.parameters[1] => old_params[1] * exp(rndnmbr[1]*1e-4)), 
                Tuple(m.constants.post_complete_parameters.parameters[1:2] .=> old_params[1:2] .* 1.0001), 
                m.constants.post_complete_parameters.parameters .=> old_params, 
                (string(m.constants.post_complete_parameters.parameters[1]) => old_params[1] * 1.0001), 
                Tuple(string.(m.constants.post_complete_parameters.parameters[1:2]) .=> old_params[1:2] .* exp.(-rndnmbr[1:2]*1e-4)), 
                old_params]
                
    
    params2 = [old_params2, 
                (m2.constants.post_complete_parameters.parameters[1] => old_params2[1] * exp(rndnmbr[1]*1e-4)), 
                Tuple(m2.constants.post_complete_parameters.parameters[1:2] .=> old_params2[1:2] .* 1.0001), 
                m2.constants.post_complete_parameters.parameters .=> old_params2, 
                (string(m2.constants.post_complete_parameters.parameters[1]) => old_params2[1] * 1.0001), 
                Tuple(string.(m2.constants.post_complete_parameters.parameters[1:2]) .=> old_params2[1:2] .* exp.(-rndnmbr[1:2]*1e-4)), 
                old_params2]

    param_derivs = [:all, 
                    m.constants.post_complete_parameters.parameters[1], 
                    m.constants.post_complete_parameters.parameters[1:3], 
                    Tuple(m.constants.post_complete_parameters.parameters[1:3]), 
                    reshape(m.constants.post_complete_parameters.parameters[1:3],3,1), 
                    string.(m.constants.post_complete_parameters.parameters[1]), 
                    string.(m.constants.post_complete_parameters.parameters[1:2]), 
                    string.(Tuple(m.constants.post_complete_parameters.parameters[1:3])), 
                    string.(reshape(m.constants.post_complete_parameters.parameters[1:3],3,1))]

    vars = [:all, :all_excluding_obc, :all_excluding_auxiliary_and_obc, m.constants.post_model_macro.var[1], m.constants.post_model_macro.var[1:2], Tuple(m.constants.post_model_macro.var), reshape(m.constants.post_model_macro.var,1,length(m.constants.post_model_macro.var)), string(m.constants.post_model_macro.var[1]), string.(m.constants.post_model_macro.var[1:2]), Tuple(string.(m.constants.post_model_macro.var)), reshape(string.(m.constants.post_model_macro.var),1,length(m.constants.post_model_macro.var))]

    rename_dicts = [
        Dict((m.constants.post_model_macro.var) .=> (replace.(String.(m.constants.post_model_macro.var), "_" => " ", "◖" => " {", "◗" => "}"))), 
        Dict((m.constants.post_model_macro.var) .=> Symbol.(replace.(String.(m.constants.post_model_macro.var), "_" => " ", "◖" => " {", "◗" => "}"))), 
        Dict(String.(m.constants.post_model_macro.var) .=> (replace.(String.(m.constants.post_model_macro.var), "_" => " ", "◖" => " {", "◗" => "}"))), 
        Dict{Symbol,String}()
    ]

    init_state = get_irf(m, algorithm = algorithm, shocks = :none, levels = !(algorithm in [:pruned_second_order, :pruned_third_order]), variables = :all, periods = 1) |> vec

    init_states = [[0.0], init_state, algorithm  == :pruned_second_order ? [zero(init_state), init_state] : algorithm == :pruned_third_order ? [zero(init_state), init_state, zero(init_state)] : init_state .* 1.01]

    if plots
        @testset "plot_model_estimates" begin
            sol2 = get_solution(m2) # TODO: investigate why this creates world age problems in tests
            
            if length(m2.constants.post_model_macro.exo) > 3
                n_shocks_influence_var = vec(sum(abs.(sol2[end-length(m2.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
                var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m2.equations.obc_violation) > 0 ? 2 : end]]
            else
                var_idxs = [1]
            end

            Random.seed!(41823)

            simulation = simulate(m2, algorithm = algorithm)

            last_stable_col = -5
            
            for i in eachcol(simulation[:,:,1])
                last_stable_col += 1
                if any(isnan,i) break end
            end

            simulation = simulation[:,1:last_stable_col,:]

            data_in_levels2 = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m2.constants.post_model_macro.var[var_idxs]) : m2.constants.post_model_macro.var[var_idxs],:,:simulate)
            data2 = data_in_levels2 .- m2.caches.non_stochastic_steady_state[var_idxs]



            sol = get_solution(m)
            
            if length(m.constants.post_model_macro.exo) > 3
                n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
                var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.equations.obc_violation) > 0 ? 2 : end]]
            else
                var_idxs = [1]
            end

            Random.seed!(41823)

            simulation = simulate(m, algorithm = algorithm)

            last_stable_col = -5
            
            for i in eachcol(simulation[:,:,1])
                last_stable_col += 1
                if any(isnan,i) break end
            end

            simulation = simulation[:,1:last_stable_col,:]

            data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.constants.post_model_macro.var[var_idxs]) : m.constants.post_model_macro.var[var_idxs],:,:simulate)
            data = data_in_levels .- m.caches.non_stochastic_steady_state[var_idxs]

            
            if !(algorithm in [:second_order, :third_order])
                # plotlyjs_backend()

                # plot_shock_decomposition(m, data, 
                #                             algorithm = algorithm, 
                #                             data_in_levels = false)

                # gr_backend()

                plot_shock_decomposition(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false)
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

            
            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)

            i = 1

            for (model, dat) in zip([m, m2], [data, data2])
                for filter in (algorithm == :first_order ? filters : [:inversion])
                    for smooth in [true, false]
                        for presample_periods in [0, 3]
                            if i % 4 == 0
                                plot_model_estimates(m, data_in_levels, 
                                                        algorithm = algorithm, 
                                                        data_in_levels = true)
                            end

                            i += 1
                            
                            clear_solution_caches!(model, algorithm)

                            plot_model_estimates!(model, dat, 
                                                    algorithm = algorithm, 
                                                    data_in_levels = false, 
                                                    filter = filter,
                                                    smooth = smooth,
                                                    presample_periods = presample_periods)
                        end
                    end
                end
            end


            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for tol in [MacroModelling.Tolerances(), MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
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

            
            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)

            i = 1
            
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for tol in [MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14)), MacroModelling.Tolerances()]
                            if i % 4 == 0
                                plot_model_estimates(m, data_in_levels, 
                                                        algorithm = algorithm, 
                                                        data_in_levels = true)
                            end

                            i += 1
                            
                            clear_solution_caches!(m, algorithm)

                            plot_model_estimates!(m, data, 
                                                    algorithm = algorithm, 
                                                    data_in_levels = false, 
                                                    tol = tol,
                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                    sylvester_algorithm = sylvester_algorithm)
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


            
            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)

            i = 1

            for parameters in params
                if i % 4 == 0
                    plot_model_estimates(m, data_in_levels, 
                                            algorithm = algorithm, 
                                            data_in_levels = true)
                end

                i += 1

                plot_model_estimates!(m, data, 
                                        parameters = parameters,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end

 

            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)
                                  
            i = 1

            for shocks in [:all, :all_excluding_obc, :none, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2])]
                if i % 4 == 0
                    plot_model_estimates(m, data_in_levels, 
                                            algorithm = algorithm, 
                                            data_in_levels = true)
                end

                i += 1

                plot_model_estimates!(m, data, 
                                        label = shocks isa String ? shocks : shocks isa Symbol ? string(shocks) : join(string.(collect(shocks)), " "),
                                        shocks = shocks,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end

            for shocks in [:all, :all_excluding_obc, :none, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2])]
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
                                                    plots_per_page = plots_per_page)
                        end
                    end
                end
            end

            for plots_per_page in [4,6]
                for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
                    for label in [:dil, "data in levels", 0, 0.01]
                        plot_model_estimates(m, data, 
                                                algorithm = algorithm,
                                                parameters = params[1], 
                                                label = "baseline",
                                                data_in_levels = false)
                                                
                        plot_model_estimates!(m, data_in_levels, 
                                                algorithm = algorithm, 
                                                data_in_levels = true,
                                                label = label,
                                                parameters = params[2],
                                                plot_attributes = plot_attributes,
                                                plots_per_page = plots_per_page)
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

                                plot_model_estimates!(m, data_in_levels, 
                                                        algorithm = algorithm, 
                                                        data_in_levels = true,
                                                        show_plots = show_plots,
                                                        save_plots = save_plots,
                                                        save_plots_path = save_plots_path,
                                                        save_plots_format = save_plots_format)
                            end
                        end
                    end
                end
            # end

            for variables in vars
                plot_model_estimates(m, data, 
                                        variables = variables,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end


            plot_model_estimates(m, data_in_levels, 
                                    parameters = params[1],
                                    algorithm = algorithm, 
                                    data_in_levels = true)
            
            i = 1
            for rename_dict in rename_dicts
                for variables in vars
                    if i % 4 == 0
                        plot_model_estimates(m, data_in_levels,
                                                parameters = params[1],
                                                algorithm = algorithm, 
                                                data_in_levels = true)
                    end

                    i += 1

                    plot_model_estimates!(m, data, 
                                            variables = variables,
                                            label = string(variables),
                                            rename_dictionary = rename_dict,
                                            algorithm = algorithm, 
                                            data_in_levels = false)
                end
            end

            # Test forecast_periods argument
            for forecast_periods in [0, 6, 12, 24]
                plot_model_estimates(m, data, 
                                        algorithm = algorithm, 
                                        data_in_levels = false,
                                        forecast_periods = forecast_periods)
            end

            # Test forecast_periods with plot_model_estimates!
            plot_model_estimates(m, data, 
                                    algorithm = algorithm, 
                                    data_in_levels = false,
                                    forecast_periods = 12)
            
            for forecast_periods in [0, 8, 18]
                plot_model_estimates!(m, data_in_levels, 
                                        algorithm = algorithm, 
                                        data_in_levels = true,
                                        forecast_periods = forecast_periods)
            end

            # Test forecast_periods with different filters
            for filter in (algorithm == :first_order ? filters : [:inversion])
                for forecast_periods in [0, 12]
                    clear_solution_caches!(m, algorithm)
                    
                    plot_model_estimates(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false,
                                            filter = filter,
                                            forecast_periods = forecast_periods)
                end
            end

            # Test forecast_periods with shock_decomposition
            if !(algorithm in [:second_order, :third_order])
                for forecast_periods in [0, 12]
                    clear_solution_caches!(m, algorithm)
                    
                    plot_model_estimates(m, data, 
                                            algorithm = algorithm, 
                                            data_in_levels = false,
                                            shock_decomposition = true,
                                            forecast_periods = forecast_periods)
                end
            end
        end

        @testset "plot_solution" begin
            states  = vcat(get_state_variables(m), m.constants.post_model_macro.past_not_future_and_mixed)
            states2 = vcat(get_state_variables(m2), m2.constants.post_model_macro.past_not_future_and_mixed)

            if algorithm == :first_order
                algos = [:first_order]
            elseif algorithm in [:second_order, :pruned_second_order]
                algos = [:first_order, :second_order, :pruned_second_order]
            elseif algorithm in [:third_order, :pruned_third_order]
                algos = [:first_order, :second_order, :pruned_second_order, :third_order, :pruned_third_order]
            end
            
            for variables in vars
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                clear_solution_caches!(m, algorithm)
                    
                                # Test single algorithm
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
                            # Test single algorithm
                            plot_solution(m, state,
                                            σ = σ,
                                            algorithm = algo,
                                            ignore_obc = ignore_obc)
                        end
                    end
                end
            end


            plot_solution(m2, states2[end])

            # i = 1

            # Test plot_solution! for combining multiple algorithms
            for ignore_obc in [true, false]
                for (model, stt) in [(m, states), (m2, states2)]
                    for state in stt[[1,end]]
                        for σ in [0.5, 5]
                            # if i % 3 == 0
                            #     plot_solution(m, states[2])
                            # end

                            # i += 1
                            
                            plot_solution!(model, state, σ = σ, ignore_obc = ignore_obc)
                        end
                    end
                end
            end

             
            plot_solution(m2, states2[1])

            i = 1

            # Test plot_solution! for combining multiple algorithms
            for (model, state, pars) in [(m, states[1], params), (m2, states2[1], params2)]
                for parameters in pars
                    for algo in algos
                        if i % 10 == 0
                            plot_solution(m, states[1])
                        end

                        i += 1
                        
                        plot_solution!(model, state, algorithm = algo, parameters = parameters)
                    end
                end
            end


            plot_solution(m2, states2[1])
            
            i = 1
            for rename_dict in rename_dicts
                for variables in vars
                    if i % 4 == 0
                        plot_solution(m2, states2[1])
                    end

                    i += 1
                    
                    plot_solution!(m, states[1],
                                    variables = variables,
                                    rename_dictionary = rename_dict)
                end
            end

            # plotlyjs_backend()

            # plot_solution(m, states[1], algorithm = algos[end])

            # gr_backend()
        end


        @testset "plot_irf" begin
            # plotlyjs_backend()

            plot_IRF(m, algorithm = algorithm)

            # gr_backend()

            plot_irfs(m, algorithm = algorithm)

            if algorithm != :first_order
                plot_girf!(m, algorithm = algorithm)
            end
            
            plot_simulations(m, algorithm = algorithm)

            plot_irf!(m, algorithm = algorithm)

            plot_simulation(m, algorithm = algorithm)

            plot_irfs!(m, algorithm = algorithm)

            plot_girf(m, algorithm = algorithm)

            plot_simulation!(m, algorithm = algorithm)

            for ignore_obc in [true,false]
                for generalised_irf in (algorithm == :first_order ? [false] : [true, false])
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


            plot_irf(m, algorithm = algorithm)

            i = 1

            for ignore_obc in [true,false]
                for generalised_irf in (algorithm == :first_order ? [false] : [true,false])
                    for negative_shock in [true,false]
                        for shock_size in [.1,1]
                            for periods in [1,10]
                                if i % 10 == 0
                                    plot_irf(m, algorithm = algorithm)
                                end

                                i += 1

                                plot_irf!(m, algorithm = algorithm, 
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
            

            plot_irf(m, algorithm = algorithm)

            i = 1

            for model in [m, m2]
                for generalised_irf in (algorithm == :first_order ? [false] : [true,false])
                    for negative_shock in [true,false]
                        for shock_size in [.1,1]
                            for periods in [1,10]
                                if i % 10 == 0
                                    plot_irf(m, algorithm = algorithm)
                                end

                                i += 1

                                plot_irf!(model, algorithm = algorithm, 
                                            periods = periods,
                                            generalised_irf = generalised_irf,
                                            negative_shock = negative_shock,
                                            shock_size = shock_size)
                            end
                        end
                    end
                end
            end
            

            plot_irf(m, algorithm = algorithm)
            
            for negative_shock in [true,false]
                for shock_size in [.1,1]
                    for plot_type in [:compare, :stack]
                        plot_irf!(m, algorithm = algorithm, 
                                    plot_type = plot_type,
                                    negative_shock = negative_shock,
                                    shock_size = shock_size)
                    end
                end
            end


            shock_mat = randn(m.constants.post_model_macro.nExo,3)

            shock_mat2 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = m.constants.post_model_macro.exo, Periods = 1:10)

            shock_mat3 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = string.(m.constants.post_model_macro.exo), Periods = 1:10)

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        # for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                clear_solution_caches!(m, algorithm)
                                            
                                plot_irf(m, algorithm = algorithm, 
                                            parameters = parameters,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            # lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)
                            end
                        # end
                    end
                end
            end
    

            plot_irf(m, algorithm = algorithm)

            i  = 1

            for parameters in params
                for tol in [MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14)), MacroModelling.Tolerances()]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        # for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                if i % 10 == 0
                                    plot_irf(m, algorithm = algorithm)
                                end
                                
                                i += 1

                                clear_solution_caches!(m, algorithm)
                                            
                                plot_irf!(m, algorithm = algorithm, 
                                            parameters = parameters,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            # lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)
                            end
                        # end
                    end
                end
            end


            plot_irf(m, algorithm = algorithm,
                        parameters = params[1])

            i = 1

            for initial_state in init_states
                if i % 10 == 0
                    plot_irf(m, algorithm = algorithm)
                end
                
                i += 1

                clear_solution_caches!(m, algorithm)
                            
                plot_irf!(m, algorithm = algorithm, initial_state = initial_state,
                        parameters = params[2])
            end

            for initial_state in init_states
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, initial_state = initial_state)
            end


            for variables in vars
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, variables = variables)
            end

            
            plot_irf(m, parameters = params[2], algorithm = algorithm)
            
            i = 1
            for rename_dict in rename_dicts
                for variables in vars
                    if i % 4 == 0
                        plot_irf(m, parameters = params[2], algorithm = algorithm)
                    end

                    i += 1
                    
                    plot_irf!(m,
                                variables = variables,
                                parameters = params[1],
                                label = string(variables),
                                rename_dictionary = rename_dict,
                                algorithm = algorithm)
                end
            end


            for shocks in [:all, :all_excluding_obc, :none, :simulate, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, shocks = shocks)
            end
            
            plot_irf(m, algorithm = algorithm)
            
            i = 1

            for shocks in [:none, :all, :all_excluding_obc, :simulate, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                if i % 4 == 0
                    plot_irf(m, algorithm = algorithm)
                end

                i += 1
                
                clear_solution_caches!(m, algorithm)
                            
                plot_irf!(m, algorithm = algorithm, shocks = shocks)
            end


            for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
                for plots_per_page in [4,6]
                    for label in [:dil, "data in levels", 0, 0.01]
                        plot_irf(m, algorithm = algorithm,
                                    label = "baseline",
                                    parameters = params[1],
                                    plot_attributes = plot_attributes,
                                    plots_per_page = plots_per_page)

                        plot_irf!(m, algorithm = algorithm,
                                    parameters = params[2],
                                    label = label,
                                    plot_attributes = plot_attributes,
                                    plots_per_page = plots_per_page)
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
                                plot_irf(m, algorithm = algorithm,
                                            parameters = params[1],
                                            show_plots = show_plots,
                                            save_plots = save_plots,
                                            save_plots_path = save_plots_path,
                                            save_plots_format = save_plots_format)
                                            
                                plot_irf!(m, algorithm = algorithm,
                                            parameters = params[2],
                                            show_plots = show_plots,
                                            save_plots = save_plots,
                                            save_plots_path = save_plots_path,
                                            save_plots_format = save_plots_format)
                            end
                        end
                    end
                end
            # end
        end


        @testset "plot_conditional_variance_decomposition" begin
            # plotlyjs_backend()
            
            plot_fevd(m)

            # gr_backend()

            plot_forecast_error_variance_decomposition(m)

            for periods in [10,40]
                for variables in vars
                    for rename_dict in rename_dicts
                        plot_conditional_variance_decomposition(m, 
                                                                periods = periods, 
                                                                variables = variables, 
                                                                rename_dictionary = rename_dict)
                    end
                end
            end

            

            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    # for lyapunov_algorithm in lyapunov_algorithms
                       clear_solution_caches!(m, algorithm)
                            
                        plot_conditional_variance_decomposition(m, tol = tol,
                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                # lyapunov_algorithm = lyapunov_algorithm
                                                                )
                    # end
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
        end

        @testset "plot_conditional_forecast" begin
            # test conditional forecasting
            new_sub_irfs_all  = get_irf(m2, algorithm = algorithm, verbose = false, variables = :all, shocks = :all)
            varnames = axiskeys(new_sub_irfs_all,1)
            shocknames = axiskeys(new_sub_irfs_all,3)
            sol = get_solution(m2)
            # var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[[1,end]]
            n_shocks_influence_var = vec(sum(abs.(sol[end-length(m2.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
            var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m2.equations.obc_violation) > 0 ? 2 : end]]


            stst  = get_irf(m2, variables = :all, algorithm = algorithm, shocks = :none, periods = 1, levels = true) |> vec

            conditions2 = []

            cndtns = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,1),2)
            cndtns[var_idxs[1],1] = .01
            cndtns[var_idxs[2],2] = .02

            push!(conditions2, cndtns)

            cndtns = spzeros(size(new_sub_irfs_all,1),2)
            cndtns[var_idxs[1],1] = .011
            cndtns[var_idxs[2],2] = .024

            push!(conditions2, cndtns)

            cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
            cndtns[1,1] = .014
            cndtns[2,2] = .0207

            push!(conditions2, cndtns)

            cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
            cndtns[1,1] = .014
            cndtns[2,2] = .025

            push!(conditions2, cndtns)

            conditions_lvl2 = []

            cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
            cndtns_lvl[1,1] = .017 + stst[var_idxs[1]]
            cndtns_lvl[2,2] = .02 + stst[var_idxs[2]]

            push!(conditions_lvl2, cndtns_lvl)

            cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
            cndtns_lvl[1,1] = .01 + stst[var_idxs[1]]
            cndtns_lvl[2,2] = .027 + stst[var_idxs[2]]
        
            push!(conditions_lvl2, cndtns_lvl)



            # test conditional forecasting
            new_sub_irfs_all  = get_irf(m, algorithm = algorithm, verbose = false, variables = :all, shocks = :all)
            varnames = axiskeys(new_sub_irfs_all,1)
            shocknames = axiskeys(new_sub_irfs_all,3)
            sol = get_solution(m)
            # var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[[1,end]]
            n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
            var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.equations.obc_violation) > 0 ? 2 : end]]


            stst  = get_irf(m, variables = :all, algorithm = algorithm, shocks = :none, periods = 1, levels = true) |> vec

            conditions = []

            cndtns = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,1),2)
            cndtns[var_idxs[1],1] = .01
            cndtns[var_idxs[2],2] = .02

            push!(conditions, cndtns)

            cndtns = spzeros(size(new_sub_irfs_all,1),2)
            cndtns[var_idxs[1],1] = .011
            cndtns[var_idxs[2],2] = .024

            push!(conditions, cndtns)

            cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
            cndtns[1,1] = .014
            cndtns[2,2] = .0207

            push!(conditions, cndtns)

            cndtns = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
            cndtns[1,1] = .014
            cndtns[2,2] = .025

            push!(conditions, cndtns)

            conditions_lvl = []

            cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs], Periods = 1:2)
            cndtns_lvl[1,1] = .017 + stst[var_idxs[1]]
            cndtns_lvl[2,2] = .02 + stst[var_idxs[2]]

            push!(conditions_lvl, cndtns_lvl)

            cndtns_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs]), Periods = 1:2)
            cndtns_lvl[1,1] = .01 + stst[var_idxs[1]]
            cndtns_lvl[2,2] = .027 + stst[var_idxs[2]]
        
            push!(conditions_lvl, cndtns_lvl)


            shocks = []

            push!(shocks, nothing)

            if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
                shcks = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,3),1)
                shcks[1,1] = .13

                push!(shocks, shcks)

                shcks = spzeros(size(new_sub_irfs_all,3),1)
                shcks[1,1] = .18
                
                push!(shocks, shcks)

                shcks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = [shocknames[1]], Periods = [1])
                shcks[1,1] = .12

                push!(shocks, shcks)

                shcks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = string.([shocknames[1]]), Periods = [1])
                shcks[1,1] = .19

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

                                        plot_conditional_forecast!(m, conditions[1],
                                                                    conditions_in_levels = false,
                                                                    initial_state = [0.0],
                                                                    algorithm = algorithm, 
                                                                    shocks = shocks[end],
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

            
            for tol in [MacroModelling.Tolerances(), MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    # for lyapunov_algorithm in lyapunov_algorithms
                        for sylvester_algorithm in sylvester_algorithms
                            clear_solution_caches!(m, algorithm)
                        
                            plot_conditional_forecast(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[end],
                                                        tol = tol,
                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                        # lyapunov_algorithm = lyapunov_algorithm,
                                                        sylvester_algorithm = sylvester_algorithm)

                            plot_conditional_forecast!(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[1],
                                                        tol = tol,
                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                        # lyapunov_algorithm = lyapunov_algorithm,
                                                        sylvester_algorithm = sylvester_algorithm)
                        end
                    # end
                end
            end

            plot_conditional_forecast(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[1])

            i = 1

            for tol in [MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14)), MacroModelling.Tolerances()]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    # for lyapunov_algorithm in lyapunov_algorithms
                        for sylvester_algorithm in sylvester_algorithms
                            if i % 4 == 0
                                plot_conditional_forecast(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[1])
                            end

                            i += 1

                            clear_solution_caches!(m, algorithm)
                        
                            plot_conditional_forecast!(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[end],
                                                        tol = tol,
                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                        # lyapunov_algorithm = lyapunov_algorithm,
                                                        sylvester_algorithm = sylvester_algorithm)
                        end
                    # end
                end
            end

            for periods in [0,10]
                # for levels in [true, false]
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

                # end
            end


            plot_conditional_forecast(m, conditions_lvl[end],
                                        algorithm = algorithm,
                                        shocks = shocks[end])
            
            for periods in [0,10]
                for (model, cond) in zip([m, m2], [conditions, conditions2])
                    clear_solution_caches!(model, algorithm)
                
                    plot_conditional_forecast!(model, cond[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm, 
                                                periods = periods)
                end
            end


            for variables in vars
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            algorithm = algorithm, 
                                            variables = variables)
            end
            

            plot_conditional_forecast(m2, conditions2[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm)

            i = 1

            for rename_dict in rename_dicts
                for variables in vars
                    if i % 4 == 0
                        plot_conditional_forecast(m2, conditions2[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm)
                    end

                    i += 1

                    plot_conditional_forecast!(m, conditions[end],
                                                conditions_in_levels = false,
                                                initial_state = init_states[end], 
                                                rename_dictionary = rename_dict,
                                                variables = variables,
                                                algorithm = algorithm)
                end
            end

            for initial_state in init_states
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            initial_state = initial_state,
                                            algorithm = algorithm)
            end

            plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        parameters = params[1],
                                        algorithm = algorithm)

            i = 1

            for initial_state in init_states
                if i % 4 == 0
                    plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        parameters = params[1],
                                        algorithm = algorithm)
                end

                i += 1

                plot_conditional_forecast!(m, conditions[end],
                                            conditions_in_levels = false,
                                            parameters = params[2],
                                            initial_state = initial_state,
                                            algorithm = algorithm)
            end


            for shcks in shocks
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            algorithm = algorithm, 
                                            shocks = shcks)
            end


            plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm, 
                                        shocks = shocks[end])

            i = 1

            for shcks in shocks
                if i % 4 == 0
                    plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm, 
                                        shocks = shocks[end])
                end

                i += 1

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


            plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm, 
                                        parameters = params[2])

            i = 1

            for parameters in params
                if i % 4 == 0
                    plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm, 
                                        parameters = params[2])
                end

                i += 1

                plot_conditional_forecast!(m, conditions[end],
                                            parameters = parameters,
                                            conditions_in_levels = false,
                                            algorithm = algorithm)
            end

            for cndtns in conditions
                plot_conditional_forecast(m, cndtns,
                                            conditions_in_levels = false,
                                            algorithm = algorithm)
            end

            plot_conditional_forecast(m, conditions[end],
                                    conditions_in_levels = false,
                                    algorithm = algorithm, 
                                    shocks = shocks[end])

            i = 1

            for cndtns in conditions
                if i % 4 == 0
                    plot_conditional_forecast(m, conditions[end],
                                    conditions_in_levels = false,
                                    algorithm = algorithm, 
                                    shocks = shocks[end])
                end

                i += 1

                plot_conditional_forecast!(m, cndtns,
                                            conditions_in_levels = false,
                                            algorithm = algorithm)
            end
            

            plot_conditional_forecast(m, conditions[end],
                                    conditions_in_levels = false,
                                    algorithm = algorithm, 
                                    shocks = shocks[end])

            i = 1

            for cndtns in conditions
                for plot_type in [:compare, :stack]
                    if i % 4 == 0
                        plot_conditional_forecast(m, conditions[end],
                                    conditions_in_levels = false,
                                    algorithm = algorithm, 
                                    shocks = shocks[end])
                    end

                    i += 1

                    plot_conditional_forecast!(m, cndtns,
                                                conditions_in_levels = false,
                                                plot_type = plot_type,
                                                algorithm = algorithm)
                end
            end
            
            # plotlyjs_backend()

            # plot_conditional_forecast(m, conditions[end],
            #                                 conditions_in_levels = false,
            #                                 algorithm = algorithm)

            # gr_backend()
        end
    end

    @testset "filter, smooth, loglikelihood" begin
        sol = get_solution(m)
        
        if length(m.constants.post_model_macro.exo) > 3
            n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
            var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.equations.obc_violation) > 0 ? 2 : end]]
        elseif length(m.constants.post_model_macro.var) == 17
            var_idxs = [5]
        else
            var_idxs = [1]
        end

        Random.seed!(418023)

        simulation = simulate(m, algorithm = algorithm)

        last_stable_col = -5
        
        for i in eachcol(simulation[:,:,1])
            last_stable_col += 1
            if any(isnan,i) break end
        end

        simulation = simulation[:,1:last_stable_col,:]

        data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.constants.post_model_macro.var[var_idxs]) : m.constants.post_model_macro.var[var_idxs],:,:simulate)
        data = data_in_levels .- m.caches.non_stochastic_steady_state[var_idxs]


        if !(algorithm ∈ [:second_order, :third_order])
            for filter in (algorithm == :first_order ? filters : [:inversion])
                for smooth in [true, false]
                    for verbose in [false] # [true, false]
                        for quadratic_matrix_equation_algorithm in qme_algorithms
                            for lyapunov_algorithm in lyapunov_algorithms
                                for sylvester_algorithm in sylvester_algorithms
                                    clear_solution_caches!(m, algorithm)

                                    estim1 = get_shock_decomposition(m, data, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = false, 
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)

                                    clear_solution_caches!(m, algorithm)
                                
                                    estim2 = get_shock_decomposition(m, data_in_levels, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = true,
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)
                                    @test check_isapprox(estim1, estim2, rtol = 1e-8)

                                    clear_solution_caches!(m, algorithm)

                                    estim1 = get_estimated_shocks(m, data, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = false, 
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)

                                    clear_solution_caches!(m, algorithm)
                                
                                    estim2 = get_estimated_shocks(m, data_in_levels, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = true,
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)
                                    @test check_isapprox(estim1, estim2, rtol = 1e-8)

                                    for levels in [true, false]
                                        clear_solution_caches!(m, algorithm)
                                    
                                        estim1 = get_estimated_variables(m, data, 
                                                                        algorithm = algorithm, 
                                                                        data_in_levels = false, 
                                                                        levels = levels,
                                                                        filter = filter,
                                                                        smooth = smooth,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm,
                                                                        verbose = verbose)

                                        clear_solution_caches!(m, algorithm)
                                                                    
                                        estim2 = get_estimated_variables(m, data_in_levels, 
                                                                        algorithm = algorithm, 
                                                                        data_in_levels = true, 
                                                                        levels = levels,
                                                                        filter = filter,
                                                                        smooth = smooth,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm,
                                                                        verbose = verbose)
                                        @test check_isapprox(estim1, estim2, rtol = 1e-8)

                                        
                                        clear_solution_caches!(m, algorithm)
                                    
                                        estim1 = get_model_estimates(m, data, 
                                                                        algorithm = algorithm, 
                                                                        data_in_levels = false, 
                                                                        levels = levels,
                                                                        filter = filter,
                                                                        smooth = smooth,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm,
                                                                        verbose = verbose)

                                        clear_solution_caches!(m, algorithm)
                                                                    
                                        estim2 = get_model_estimates(m, data_in_levels, 
                                                                        algorithm = algorithm, 
                                                                        data_in_levels = true, 
                                                                        levels = levels,
                                                                        filter = filter,
                                                                        smooth = smooth,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm,
                                                                        verbose = verbose)
                                        @test check_isapprox(estim1, estim2, rtol = 1e-8)
                                    end
                                end
                            end
                        end
                    end
                end
            end

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    get_shock_decomposition(m, data, 
                                            parameters = parameters,
                                            algorithm = algorithm, 
                                            tol = tol,
                                            data_in_levels = false, 
                                            verbose = false)
                    get_shock_decomposition(m, data_in_levels, 
                                            parameters = parameters,
                                            algorithm = algorithm, 
                                            data_in_levels = true,
                                            verbose = false)


                    get_estimated_shocks(m, data, 
                                    parameters = parameters,
                                    algorithm = algorithm, 
                                    tol = tol,
                                    data_in_levels = false, 
                                    verbose = false)
                    get_estimated_shocks(m, data_in_levels, 
                                    parameters = parameters,
                                    algorithm = algorithm, 
                                    tol = tol,
                                    data_in_levels = true,
                                    verbose = false)

                    get_model_estimates(m, data, 
                                    parameters = parameters,
                                    algorithm = algorithm, 
                                    tol = tol,
                                    data_in_levels = false, 
                                    verbose = false)
                    get_model_estimates(m, data_in_levels, 
                                    parameters = parameters,
                                    algorithm = algorithm, 
                                    tol = tol,
                                    data_in_levels = true,
                                    verbose = false)
                    

                    get_estimated_variables(m, data, 
                                            parameters = parameters,
                                            algorithm = algorithm, 
                                            tol = tol,
                                            data_in_levels = false, 
                                            verbose = false)
                    get_estimated_variables(m, data_in_levels, 
                                            parameters = parameters,
                                            algorithm = algorithm, 
                                            tol = tol,
                                            data_in_levels = true,
                                            verbose = false)
                end
            end

            if algorithm in (:pruned_second_order, :pruned_third_order)
                clear_solution_caches!(m, algorithm)

                sd_default = get_shock_decomposition(m, data,
                                                    algorithm = algorithm,
                                                    data_in_levels = false,
                                                    verbose = false)
                sd_mc = get_shock_decomposition(m, data,
                                                algorithm = algorithm,
                                                data_in_levels = false,
                                                marginal_contribution = true,
                                                verbose = false)

                @test :Nonlinearities ∈ axiskeys(sd_default, :Shocks)
                @test :Nonlinearities ∉ axiskeys(sd_mc, :Shocks)
                @test :Initial_values ∈ axiskeys(sd_mc, :Shocks)
                @test size(sd_mc, :Shocks) == size(sd_default, :Shocks) - 1
                init_mc = sd_mc[:, :Initial_values, :]
                shock_keys_mc = filter(!=(:Initial_values), collect(axiskeys(sd_mc, :Shocks)))
                shock_sum_mc = dropdims(sum(collect(sd_mc[:, shock_keys_mc, :]), dims = 2), dims = 2)
                # In marginal-contribution mode the zero-shock / initial-values
                # path stays separate; only the incremental response is
                # reallocated across the shock columns.
                sum_default = dropdims(sum(collect(sd_default), dims = 2), dims = 2)
                sum_mc      = dropdims(sum(collect(sd_mc),      dims = 2), dims = 2)
                @test isapprox(shock_sum_mc .+ Array(init_mc), sum_default, atol = 1e-8)
                @test isapprox(sum_default, sum_mc, atol = 1e-8)
            end

            # First-order with marginal_contribution = true is silently ignored
            # (with an @info notice) and returns the standard first-order
            # decomposition.
            if algorithm == :first_order
                clear_solution_caches!(m, algorithm)
                sd_fo_default = get_shock_decomposition(m, data,
                                                        algorithm = algorithm,
                                                        data_in_levels = false,
                                                        verbose = false)
                sd_fo_mc      = get_shock_decomposition(m, data,
                                                        algorithm = algorithm,
                                                        data_in_levels = false,
                                                        marginal_contribution = true,
                                                        verbose = false)
                @test axiskeys(sd_fo_mc, :Shocks) == axiskeys(sd_fo_default, :Shocks)
                @test isapprox(collect(sd_fo_mc), collect(sd_fo_default), rtol = 1e-10)
            end
        end

        

        if algorithm == :first_order
            for smooth in [true, false]
                for verbose in [false] # [true, false]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms

                            clear_solution_caches!(m, algorithm)
                        
                            estim1 = get_estimated_variable_standard_deviations(m, data, 
                                                                                data_in_levels = false, 
                                                                                smooth = smooth,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                verbose = verbose)

                            clear_solution_caches!(m, algorithm)
                        
                            estim2 = get_estimated_variable_standard_deviations(m, data_in_levels, 
                                                                                data_in_levels = true,
                                                                                smooth = smooth,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                verbose = verbose)
                            @test check_isapprox(estim1,estim2)
                        end
                    end
                end
            end

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    get_estimated_variable_standard_deviations(m, data, 
                                                                parameters = parameters,
                                                                data_in_levels = false, 
                                                                verbose = false)
                    get_estimated_variable_standard_deviations(m, data_in_levels, 
                                                                parameters = parameters,
                                                                data_in_levels = true,
                                                                verbose = false)
                end
            end
        end

        

        for filter in (algorithm == :first_order ? filters : [:inversion])
            for presample_periods in [0, 3]
                for initial_covariance in [:diagonal, :theoretical]
                    for verbose in [false] # [true, false]
                        for parameter_values in [old_params, old_params .* exp.(-rndnmbr[1:length(old_params)]*1e-4)]
                            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                                llh = get_loglikelihood(m, data_in_levels, parameter_values,
                                                        algorithm = algorithm,
                                                        filter = filter,
                                                        presample_periods = presample_periods,
                                                        initial_covariance = initial_covariance,
                                                        tol = tol,
                                                        verbose = verbose)

                                    clear_solution_caches!(m, algorithm)

                                    moon_grad_llh = DifferentiationInterface.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                    algorithm = algorithm,
                                                                                                    filter = filter,
                                                                                                    presample_periods = presample_periods,
                                                                                                    initial_covariance = initial_covariance,
                                                                                                    tol = tol,
                                                                                                    verbose = verbose), ADTypes.AutoMooncake(config = nothing), parameter_values)

                                    zyg_grad_llh = Zygote.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                    algorithm = algorithm,
                                                                                                    filter = filter,
                                                                                                    presample_periods = presample_periods,
                                                                                                    initial_covariance = initial_covariance,
                                                                                                    tol = tol,
                                                                                                    verbose = verbose), parameter_values)[1]

                                    if algorithm == :first_order && filter == :kalman
                                        for i in 1:100
                                            local fin_grad_llh = FiniteDifferences.grad(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3), 
                                                                                    x -> begin 
                                                                                            clear_solution_caches!(m, algorithm)
        
                                                                                            get_loglikelihood(m, data_in_levels, x,
                                                                                                            algorithm = algorithm,
                                                                                                            filter = filter,
                                                                                                            presample_periods = presample_periods,
                                                                                                            initial_covariance = initial_covariance,
                                                                                                            tol = tol,
                                                                                                            verbose = verbose)
                                                                                            end, parameter_values)
                                            if isfinite(ℒ.norm(fin_grad_llh[1]))
                                                @test check_isapprox(fin_grad_llh[1], moon_grad_llh, rtol = 1e-4, atol = 1e-6)
                                                @test check_isapprox(fin_grad_llh[1], zyg_grad_llh, rtol = 1e-4, atol = 1e-6)
                                                @test check_isapprox(fin_grad_llh[1], moon_grad_llh, rtol = 1e-4, atol = 1e-6)
                                                @test check_isapprox(fin_grad_llh[1], zyg_grad_llh, rtol = 1e-4, atol = 1e-6)
                                                break
                                            end
                                        end
                                    end
                                                                  
                                for quadratic_matrix_equation_algorithm in qme_algorithms
                                    for lyapunov_algorithm in lyapunov_algorithms
                                        for sylvester_algorithm in sylvester_algorithms
                                            
                                            clear_solution_caches!(m, algorithm)
                                        
                                            LLH = get_loglikelihood(m, data_in_levels, parameter_values,
                                                                    algorithm = algorithm,
                                                                    filter = filter,
                                                                    presample_periods = presample_periods,
                                                                    initial_covariance = initial_covariance,
                                                                    tol = tol,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)
                                            @test check_isapprox(llh, LLH, rtol = 1e-8)

                                                clear_solution_caches!(m, algorithm)
                                        
                                                MOON_grad_llh = DifferentiationInterface.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                                algorithm = algorithm,
                                                                                                                filter = filter,
                                                                                                                presample_periods = presample_periods,
                                                                                                                initial_covariance = initial_covariance,
                                                                                                                tol = tol,
                                                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                                                verbose = verbose), ADTypes.AutoMooncake(config = nothing), parameter_values)

                                                ZYG_grad_llh = Zygote.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                                algorithm = algorithm,
                                                                                                                filter = filter,
                                                                                                                presample_periods = presample_periods,
                                                                                                                initial_covariance = initial_covariance,
                                                                                                                tol = tol,
                                                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                                                verbose = verbose), parameter_values)[1]
                
                                                @test check_isapprox(MOON_grad_llh, moon_grad_llh, rtol = 1e-6)
                                                @test check_isapprox(ZYG_grad_llh, zyg_grad_llh, rtol = 1e-6)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "get_conditional_forecast" begin
        # test conditional forecasting
        new_sub_irfs_all  = get_irf(m, algorithm = algorithm, verbose = false, variables = :all, shocks = :all)
        varnames = axiskeys(new_sub_irfs_all,1)
        shocknames = axiskeys(new_sub_irfs_all,3)
        sol = get_solution(m)
        # var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[[1,end]]
        n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.constants.post_model_macro.exo)+1:end,:]) .> eps(),dims = 1))
        var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.equations.obc_violation) > 0 ? 2 : end]]


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

        cond_fcst = get_conditional_forecast(m, conditions[1],
                                            conditions_in_levels = false,
                                            initial_state = [0.0],
                                            algorithm = algorithm, 
                                            shocks = shocks[1])

        

        for periods in [0,10]
            for variables in vars
                for levels in [true, false]
                    for verbose in [false] # [true, false]
                        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                # for lyapunov_algorithm in lyapunov_algorithms
                                    for sylvester_algorithm in sylvester_algorithms
                                        
                                        clear_solution_caches!(m, algorithm)
                                    
                                        cond_fcst = get_conditional_forecast(m, conditions[end],
                                                                            conditions_in_levels = false,
                                                                            algorithm = algorithm, 
                                                                            variables = variables,
                                                                            periods = periods,
                                                                            levels = levels,
                                                                            shocks = shocks[end],
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            # lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm,
                                                                            verbose = verbose)

                                        
                                        clear_solution_caches!(m, algorithm)
                                    
                                        cond_fcst_lvl = get_conditional_forecast(m, conditions_lvl[end],
                                                                                algorithm = algorithm, 
                                                                                variables = variables,
                                                                                periods = periods,
                                                                                levels = levels,
                                                                                shocks = shocks[end],
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                # lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                verbose = verbose)

                                        @test check_isapprox(cond_fcst, cond_fcst_lvl)

                                        clear_solution_caches!(m, algorithm)
                                    
                                        cond_fcst = get_conditional_forecast(m, conditions[end-1],
                                                                                conditions_in_levels = false,
                                                                                algorithm = algorithm, 
                                                                                variables = variables,
                                                                                periods = periods,
                                                                                levels = levels,
                                                                                shocks = shocks[end],
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                # lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                verbose = verbose)

                                        clear_solution_caches!(m, algorithm)
                                    
                                        cond_fcst_lvl = get_conditional_forecast(m, conditions_lvl[end-1],
                                                                                algorithm = algorithm, 
                                                                                variables = variables,
                                                                                periods = periods,
                                                                                levels = levels,
                                                                                shocks = shocks[end],
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                # lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                verbose = verbose)
                                                                                
                                        @test check_isapprox(cond_fcst, cond_fcst_lvl)
                                    end
                                # end
                            end
                        end
                    end
                end
            end
        end

        for cndtns in conditions
            cond_fcst = get_conditional_forecast(m, cndtns,
                                                conditions_in_levels = false,
                                                algorithm = algorithm)
        end

        for variables in vars
            cond_fcst = get_conditional_forecast(m, conditions[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm, 
                                                variables = variables)
        end
        
        for initial_state in init_states
            cond_fcst = get_conditional_forecast(m, conditions[end],
                                                conditions_in_levels = false,
                                                initial_state = initial_state,
                                                algorithm = algorithm)
        end

        for shcks in shocks
            cond_fcst = get_conditional_forecast(m, conditions[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm, 
                                                shocks = shcks)
        end

        for parameters in params
            cond_fcst = get_conditional_forecast(m, conditions[end],
                                                parameters = parameters,
                                                conditions_in_levels = false,
                                                algorithm = algorithm)
        end
    end

    @testset "(auto) correlation, (conditional) variance decomposition" begin
        if algorithm in [:first_order, :pruned_second_order, :pruned_third_order]
            corrl = get_correlation(m, algorithm = algorithm)

            get_corr(m, algorithm = algorithm)

            corr(m, algorithm = algorithm)

            autocorr_ = get_autocorrelation(m, algorithm = algorithm)

            get_autocorr(m, algorithm = algorithm)

            autocorr(m, algorithm = algorithm)

            if algorithm == :first_order
                var_decomp = get_variance_decomposition(m)

                get_var_decomp(m)

                cond_var_decomp = get_conditional_variance_decomposition(m)

                get_fevd(m)

                get_forecast_error_variance_decomposition(m)

                fevd(m)
            end

            if algorithm == :pruned_second_order || algorithm == :pruned_third_order
                clear_solution_caches!(m, algorithm)

                var_decomp_higher = get_variance_decomposition(m, algorithm = algorithm)

                nE = length(m.constants.post_model_macro.exo)

                @test size(var_decomp_higher, 2) == nE + 1
                @test axiskeys(var_decomp_higher, 2)[end] == :Cross_shock_interaction
                @test all(isapprox.(sum(collect(var_decomp_higher), dims = 2), 1, atol = 1e-6))

                clear_solution_caches!(m, algorithm)

                var_decomp_mc = get_variance_decomposition(m, algorithm = algorithm, marginal_contribution = true)

                @test size(var_decomp_mc, 2) == nE
                @test :Cross_shock_interaction ∉ axiskeys(var_decomp_mc, 2)
                # Rows whose total variance is non-trivial must satisfy Shapley
                # efficiency (per-shock shares sum to one). Rows with negligible
                # variance are reported as exact zeros.
                row_sums_mc = vec(sum(collect(var_decomp_mc), dims = 2))
                row_sums_raw = vec(sum(collect(var_decomp_higher), dims = 2))
                for v in eachindex(row_sums_mc)
                    if isapprox(row_sums_raw[v], 1, atol = 1e-6)
                        @test isapprox(row_sums_mc[v], 1, atol = 1e-6)
                    else
                        @test isapprox(row_sums_mc[v], 0, atol = 1e-10)
                    end
                end

                # First-order with marginal_contribution = true is silently
                # ignored (with an @info notice) and returns the standard
                # first-order shares (additive across shocks, no interaction
                # column).
                vd_first_default = get_variance_decomposition(m)
                vd_first_mc      = get_variance_decomposition(m, marginal_contribution = true)
                @test axiskeys(vd_first_mc, 2) == axiskeys(vd_first_default, 2)
                @test isapprox(collect(vd_first_mc), collect(vd_first_default), rtol = 1e-10)
            end

            
            
            for parameters in params
                clear_solution_caches!(m, algorithm)
                                
                get_correlation(m, algorithm = algorithm, parameters = parameters, verbose = false)

                for autocorrelation_periods in [1:5, 1:3]
                    clear_solution_caches!(m, algorithm)
                        
                    get_autocorrelation(m, 
                                        algorithm = algorithm, 
                                        autocorrelation_periods = autocorrelation_periods, 
                                        parameters = parameters, 
                                        verbose = false)
                end

                if algorithm == :first_order
                    clear_solution_caches!(m, algorithm)
                                    
                    get_variance_decomposition(m, parameters = parameters, verbose = false)

                    for periods in [[1,Inf,10], [3,Inf], 1:3]
                        clear_solution_caches!(m, algorithm)
                        
                        get_conditional_variance_decomposition(m, periods = periods, parameters = parameters, verbose = false)
                    end
                end
            end

            

            for verbose in [false] # [true, false]
                for tol in [MacroModelling.Tolerances(), MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            
                            if algorithm == :first_order
                                clear_solution_caches!(m, algorithm)

                                VAR_DECOMP = get_variance_decomposition(m,
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        verbose = verbose)
                                                                        
                                @test check_isapprox(var_decomp, VAR_DECOMP, rtol = 1e-8, nans = true)

                                clear_solution_caches!(m, algorithm)
                                                                        
                                COND_VAR_DECOMP = get_conditional_variance_decomposition(m,
                                                                                        tol = tol,
                                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                                        verbose = verbose)

                                @test check_isapprox(cond_var_decomp, COND_VAR_DECOMP, rtol = 1e-8, nans = true)

                            end

                            for sylvester_algorithm in sylvester_algorithms
                                clear_solution_caches!(m, algorithm)
                                
                                CORRL = get_correlation(m,
                                                algorithm = algorithm,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm,
                                                verbose = verbose)

                                @test check_isapprox(corrl, CORRL, rtol = 1e-5, nans = true)

                                clear_solution_caches!(m, algorithm)
                                
                                AUTOCORR = get_autocorrelation(m,
                                                                algorithm = algorithm,
                                                                tol = tol,
                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                sylvester_algorithm = sylvester_algorithm,
                                                                verbose = verbose)

                                @test check_isapprox(autocorr_, AUTOCORR, rtol = 1e-8, nans = true)
                            end
                        end
                    end
                end
            end
        end
    end



    @testset "get_solution" begin
        sol = get_solution(m, algorithm = algorithm)

        get_first_order_solution(m)

        get_perturbation_solution(m)

        if algorithm in [:second_order, :pruned_second_order,:third_order, :pruned_third_order]
            get_second_order_solution(m)

            if algorithm in [:third_order, :pruned_third_order]
                get_third_order_solution(m)
            end
        end

        for parameters in params          
            get_solution(m, algorithm = algorithm, parameters = parameters, verbose = false)
        end

        

        for verbose in [false] # [true, false]
            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)
                        
                        SOL = get_solution(m,
                                            algorithm = algorithm,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm,
                                            verbose = verbose)
                        @test check_isapprox(sol, SOL)#, rtol = eps(Float32))
                    end
                end
            end
        end
    end

    @testset "get_solution with parameter input" begin
        for parameter_values in [old_params, old_params .* exp.(-rndnmbr[1:length(old_params)]*1e-4)]
            get_first_order_solution(m, parameter_values)

            get_perturbation_solution(m, parameter_values)
            
            if algorithm in [:second_order, :pruned_second_order,:third_order, :pruned_third_order]
                get_second_order_solution(m, parameter_values)

                if algorithm in [:third_order, :pruned_third_order]
                    get_third_order_solution(m, parameter_values)
                end
            end

            sol = get_solution(m, parameter_values, algorithm = algorithm)

            # Helper to extract element i in flattened order: 1→SS, 2→sol_mats[1], 3→sol_mats[2], ...
            _sol_el(s, i) = i == 1 ? s[1] : s[2][i-1]

            deriv_sol = nothing
            deriv_sol_zyg = nothing
                clear_solution_caches!(m, algorithm)

                deriv_sol = []
                for i in 1:length(sol[2])
                    push!(deriv_sol, ForwardDiff.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm), i), parameter_values))
                end

                clear_solution_caches!(m, algorithm)

                deriv_sol_fin = []
                for i in 1:length(sol[2])
                    push!(deriv_sol_fin, FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                            x -> begin 
                                                                clear_solution_caches!(m, algorithm)
                                                                
                                                                _sol_el(get_solution(m, x, algorithm = algorithm), i)
                                                            end, parameter_values)[1])
                end

                clear_solution_caches!(m, algorithm)

                deriv_sol_moon = []
                for i in 1:length(sol[2])
                    push!(deriv_sol_moon, DifferentiationInterface.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm), i), ADTypes.AutoMooncake(config = nothing), parameter_values))
                end

                clear_solution_caches!(m, algorithm)

                deriv_sol_zyg = []
                for i in 1:length(sol[2])
                    push!(deriv_sol_zyg, Zygote.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm), i), parameter_values)[1])
                end

                @test check_isapprox(deriv_sol_moon, deriv_sol_fin, rtol = 1e-5)
                @test check_isapprox(deriv_sol_zyg, deriv_sol_fin, rtol = 1e-5)
                
                @test check_isapprox(deriv_sol, deriv_sol_fin, rtol = 1e-5)

            for tol in [MacroModelling.Tolerances(second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))), MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)

                        SOL = get_solution(m, parameter_values, algorithm = algorithm, tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)

                        @test check_isapprox(vcat([sol[1]], sol[2]), vcat([SOL[1]], SOL[2]), rtol = 1e-8)

                            clear_solution_caches!(m, algorithm)

                            DERIV_SOL = []
                            for i in 1:length(sol[2])
                                push!(DERIV_SOL, ForwardDiff.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm, 
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                sylvester_algorithm = sylvester_algorithm), i), parameter_values))
                            end

                            @test check_isapprox(deriv_sol, DERIV_SOL, rtol = 1e-8)

                            clear_solution_caches!(m, algorithm)

                            DERIV_SOL_moon = []
                            for i in 1:length(sol[2])
                                push!(DERIV_SOL_moon, DifferentiationInterface.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm, 
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                sylvester_algorithm = sylvester_algorithm), i), ADTypes.AutoMooncake(config = nothing), parameter_values))
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV_SOL_zyg = []
                            for i in 1:length(sol[2])
                                push!(DERIV_SOL_zyg, Zygote.jacobian(x -> _sol_el(get_solution(m, x, algorithm = algorithm, 
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                sylvester_algorithm = sylvester_algorithm), i), parameter_values)[1])
                            end

                            @test check_isapprox(DERIV_SOL_moon, DERIV_SOL, rtol = 1e-8)
                            @test check_isapprox(DERIV_SOL_zyg, DERIV_SOL, rtol = 1e-8)
                    end
                end
            end
        end
    end

    @testset "solve counters" begin
        m.counters = MacroModelling.SolveCounters()
        clear_solution_caches!(m, algorithm)

        get_solution(m, m.parameter_values, algorithm = algorithm)

        counts = get_solution_counts(m)

        @test counts.ss_solves_total == 0
        @test counts.ss_solves_failed == 0
        @test counts.ss_solves_total_estimation == 1
        @test counts.ss_solves_failed_estimation == 0

        @test counts.first_order_solves_total == 0
        @test counts.first_order_solves_failed == 0
        @test counts.first_order_solves_total_estimation == 1
        @test counts.first_order_solves_failed_estimation == 0

        if algorithm in [:pruned_second_order, :second_order, :pruned_third_order, :third_order]
            @test counts.second_order_solves_total_estimation == 1
            @test counts.second_order_solves_failed_estimation == 0
        else
            @test counts.second_order_solves_total_estimation == 0
            @test counts.second_order_solves_failed_estimation == 0
        end

        if algorithm in [:pruned_third_order, :third_order]
            @test counts.third_order_solves_total_estimation == 1
            @test counts.third_order_solves_failed_estimation == 0
        else
            @test counts.third_order_solves_total_estimation == 0
            @test counts.third_order_solves_failed_estimation == 0
        end

        @test counts.second_order_solves_total == 0
        @test counts.second_order_solves_failed == 0
        @test counts.third_order_solves_total == 0
        @test counts.third_order_solves_failed == 0
    end


    @testset "get_irf with parameter input" begin
        if algorithm == :first_order
            for parameter_values in [old_params, old_params .* exp.(-rndnmbr[1:length(old_params)]*1e-4)]
                for levels in [true,false]
                    for negative_shock in [true,false]
                        for periods in [1,10]
                            get_irf(m, parameter_values,
                                    levels = levels,
                                    periods = periods,
                                    negative_shock = negative_shock)

                            get_IRF(m, parameter_values,
                                    levels = levels,
                                    periods = periods,
                                    negative_shock = negative_shock)

                            get_irfs(m, parameter_values,
                                    levels = levels,
                                    periods = periods,
                                    negative_shock = negative_shock)
                        end
                    end
                end

                shock_mat = randn(m.constants.post_model_macro.nExo,3)

                shock_mat2 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = m.constants.post_model_macro.exo, Periods = 1:10)

                shock_mat3 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = string.(m.constants.post_model_macro.exo), Periods = 1:10)

                for initial_state in init_states
                    clear_solution_caches!(m, algorithm)
                                
                    irf_ = get_irf(m, parameter_values, initial_state = initial_state)
                    
                    clear_solution_caches!(m, algorithm)
                             
                    deriv_for = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state)[:,1,1], parameter_values)

                    for i in 1:100
                        local deriv_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-4), 
                                                                    x -> begin 
                                                                        clear_solution_caches!(m, algorithm)
    
                                                                        get_irf(m, x, initial_state = initial_state)[:,1,1]
                                                                    end, parameter_values)
                        if isfinite(ℒ.norm(deriv_fin[1]))
                            @test check_isapprox(deriv_for, deriv_fin[1], rtol = 1e-5)
                            break
                        end
                    end

                    clear_solution_caches!(m, algorithm)

                    deriv_moon = DifferentiationInterface.jacobian(x -> get_irf(m, x, initial_state = initial_state)[:,1,1], ADTypes.AutoMooncake(config = nothing), parameter_values)
                    deriv_zyg = Zygote.jacobian(x -> get_irf(m, x, initial_state = initial_state)[:,1,1], parameter_values)[1]

                    for i in 1:100
                        local deriv_fin_zyg = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-4), 
                                                                    x -> begin 
                                                                        clear_solution_caches!(m, algorithm)
    
                                                                        get_irf(m, x, initial_state = initial_state)[:,1,1]
                                                                    end, parameter_values)
                        if isfinite(ℒ.norm(deriv_fin_zyg[1]))
                                @test check_isapprox(deriv_moon, deriv_fin_zyg[1], rtol = 1e-5)
                            @test check_isapprox(deriv_zyg, deriv_fin_zyg[1], rtol = 1e-5)
                            break
                        end
                    end

                    # Last period derivative tests (ForwardDiff)
                    clear_solution_caches!(m, algorithm)

                    deriv_for_last = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state)[:,end,1], parameter_values)

                    for i in 1:100
                        local deriv_fin_last = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-4), 
                                                                    x -> begin 
                                                                        clear_solution_caches!(m, algorithm)
    
                                                                        get_irf(m, x, initial_state = initial_state)[:,end,1]
                                                                    end, parameter_values)
                        if isfinite(ℒ.norm(deriv_fin_last[1]))
                            @test check_isapprox(deriv_for_last, deriv_fin_last[1], rtol = 1e-5)
                            break
                        end
                    end

                    # Last period derivative tests (Mooncake)
                    clear_solution_caches!(m, algorithm)

                    deriv_moon_last = DifferentiationInterface.jacobian(x -> get_irf(m, x, initial_state = initial_state)[:,end,1], ADTypes.AutoMooncake(config = nothing), parameter_values)
                    deriv_zyg_last = Zygote.jacobian(x -> get_irf(m, x, initial_state = initial_state)[:,end,1], parameter_values)[1]

                    for i in 1:100
                        local deriv_fin_zyg_last = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-4), 
                                                                    x -> begin 
                                                                        clear_solution_caches!(m, algorithm)
    
                                                                        get_irf(m, x, initial_state = initial_state)[:,end,1]
                                                                    end, parameter_values)
                        if isfinite(ℒ.norm(deriv_fin_zyg_last[1]))
                                @test check_isapprox(deriv_moon_last, deriv_fin_zyg_last[1], rtol = 1e-5)
                            @test check_isapprox(deriv_zyg_last, deriv_fin_zyg_last[1], rtol = 1e-5)
                            break
                        end
                    end

                    for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                        for quadratic_matrix_equation_algorithm in qme_algorithms
                            clear_solution_caches!(m, algorithm)
                                        
                            IRF_ = get_irf(m, 
                                            parameter_values, 
                                            initial_state = initial_state,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)
                            @test check_isapprox(irf_, IRF_, rtol = 1e-8)

                            DERIV_for = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state, tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)[:,1,1], parameter_values)

                            @test check_isapprox(deriv_for, DERIV_for, rtol = 1e-8)
                        end
                    end
                    for variables in vars
                        for shocks in [:all, :all_excluding_obc, :none, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                            clear_solution_caches!(m, algorithm)
                                        
                            get_irf(m, parameter_values, variables = variables, initial_state = initial_state, shocks = shocks)
                        end
                    end
                end
            end
        end
    end

    
    @testset "get_statistics" begin
        for parameter_values in [old_params, old_params .* exp.(-rndnmbr[1:length(old_params)]*1e-4)]
            for non_stochastic_steady_state in (Symbol[], vars...)
                for mean in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? (Symbol[], vars[1]) : Symbol[])
                    for standard_deviation in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? (Symbol[], vars[1]) : Symbol[])
                        for variance in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? (Symbol[], vars[1]) : Symbol[])
                            for covariance in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? (Symbol[], vars[1]) : Symbol[])
                                for autocorrelation in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? (Symbol[], vars[1]) : Symbol[])
                                    if !(!(non_stochastic_steady_state == Symbol[]) || !(standard_deviation == Symbol[]) || !(mean == Symbol[]) || !(variance == Symbol[]) || !(covariance == Symbol[]) || !(autocorrelation == Symbol[]))
                                        non_stochastic_steady_state = vars[1]
                                    end
                                    
                                    get_statistics(m, parameter_values, algorithm = algorithm,
                                                    non_stochastic_steady_state = non_stochastic_steady_state,
                                                    mean = mean,
                                                    standard_deviation = standard_deviation,
                                                    variance = variance,
                                                    covariance = covariance,
                                                    correlation = covariance,
                                                    autocorrelation = autocorrelation
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
        
        

        for parameter_values in [old_params, old_params .* exp.(-rndnmbr[1:length(old_params)]*1e-4)]
            clear_solution_caches!(m, algorithm)

            stats = get_statistics(m, parameter_values, algorithm = algorithm,
                                    # tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                    non_stochastic_steady_state = :all,
                                    mean = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    standard_deviation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    variance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    covariance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    correlation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    autocorrelation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]))

            for tol in [MacroModelling.Tolerances(second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            clear_solution_caches!(m, algorithm)
                            
                            STATS = get_statistics(m, parameter_values, algorithm = algorithm,
                                                non_stochastic_steady_state = :all,
                                                mean = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                standard_deviation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                variance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                covariance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                correlation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                autocorrelation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                # println("mean: $(ℒ.norm(stats[:mean] - STATS[:mean]) / max(ℒ.norm(stats[:mean]), ℒ.norm(STATS[:mean])))")
                                # println("variance: $(ℒ.norm(stats[:variance] - STATS[:variance]) / max(ℒ.norm(stats[:variance]), ℒ.norm(STATS[:variance])))")
                                # println("standard_deviation: $(ℒ.norm(stats[:standard_deviation] - STATS[:standard_deviation]) / max(ℒ.norm(stats[:standard_deviation]), ℒ.norm(STATS[:standard_deviation])))")
                                # println("covariance: $(ℒ.norm(stats[:covariance] - STATS[:covariance]) / max(ℒ.norm(stats[:covariance]), ℒ.norm(STATS[:covariance])))")
                                # println("autocorrelation (qme: $quadratic_matrix_equation_algorithm, sylv: $sylvester_algorithm, lyap: $lyapunov_algorithm, tol: $tol): $(ℒ.norm(stats[:autocorrelation] - STATS[:autocorrelation]) / max(ℒ.norm(stats[:autocorrelation]), ℒ.norm(STATS[:autocorrelation])))")
                                @test check_isapprox(stats[:non_stochastic_steady_state], STATS[:non_stochastic_steady_state], rtol = 1e-8)
                                @test check_isapprox(stats[:mean], STATS[:mean], rtol = 1e-8)
                                @test check_isapprox(stats[:standard_deviation], STATS[:standard_deviation], rtol = 1e-8)
                                @test check_isapprox(stats[:variance], STATS[:variance], rtol = 1e-8)
                                @test check_isapprox(stats[:covariance], STATS[:covariance], rtol = 1e-8, atol = 1e-8)
                                @test check_isapprox(stats[:correlation], STATS[:correlation], rtol = 1e-8, atol = 1e-8, nans = true)
                                @test check_isapprox(stats[:autocorrelation], STATS[:autocorrelation], rtol = 1e-8, atol = 1e-8, nans = true)
                            else
                                @test check_isapprox(stats[:non_stochastic_steady_state], STATS[:non_stochastic_steady_state], rtol = 1e-8)
                            end
                        end
                    end
                end
            end
        end


            clear_solution_caches!(m, algorithm)

            deriv1 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)

            deriv1_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], ADTypes.AutoMooncake(config = nothing), old_params)
            deriv1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)[1]
                 
            for i in 1:100        
                local deriv1_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                    x -> begin 
                                                        clear_solution_caches!(m, algorithm)
        
                                                        get_statistics(m, x, 
                                                                        algorithm = algorithm, 
                                                                        non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state]
                                                    end, old_params)
                if isfinite(ℒ.norm(deriv1_fin[1]))
                    # ℒ.norm(deriv1 - deriv1_fin[1]) / max(ℒ.norm(deriv1), ℒ.norm(deriv1_fin[1]))
                    # ℒ.norm(deriv1 - deriv1_zyg) / max(ℒ.norm(deriv1), ℒ.norm(deriv1_zyg))
            
                    @test check_isapprox(deriv1_moon, deriv1_fin[1], rtol = 1e-5)
                    @test check_isapprox(deriv1_zyg, deriv1_fin[1], rtol = 1e-5)
            
                    @test check_isapprox(deriv1, deriv1_fin[1], rtol = 1e-5)
                    break
                end
            end
        
                        
            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            clear_solution_caches!(m, algorithm)

            deriv2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            mean = :all_excluding_obc)[:mean], old_params)
            
            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv2_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                mean = :all_excluding_obc)[:mean], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                mean = :all_excluding_obc)[:mean], old_params)[1]
            end

            for i in 1:100
                local deriv2_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                        x -> begin 
                                                            clear_solution_caches!(m, algorithm)
    
                                                            get_statistics(m, x, 
                                                                            algorithm = algorithm, 
                                                                            mean = :all_excluding_obc)[:mean]
                                                        end, old_params)
                              
                if isfinite(ℒ.norm(deriv2_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv2_moon, deriv2_fin[1], rtol = 1e-5)
                        @test check_isapprox(deriv2_zyg, deriv2_fin[1], rtol = 1e-5)
                    end
                    
                    @test check_isapprox(deriv2, deriv2_fin[1], rtol = 1e-5)
                    break
                end
            end                            

            clear_solution_caches!(m, algorithm)

            deriv3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
            
            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv3_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)[1]
            end                    

            for i in 1:100        
                local deriv3_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                                                        x -> begin 
                                                            clear_solution_caches!(m, algorithm)

                                                            get_statistics(m, x, algorithm = algorithm, standard_deviation = :all_excluding_obc)[:standard_deviation]
                                                        end, old_params)
                              
                if isfinite(ℒ.norm(deriv3_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv3_moon, deriv3_fin[1], rtol = 1e-5, atol = 1e-8)
                        @test check_isapprox(deriv3_zyg, deriv3_fin[1], rtol = 1e-5, atol = 1e-8)
                    end
                    
                    @test check_isapprox(deriv3, deriv3_fin[1], rtol = 1e-5, atol = 1e-8)
                    break
                end
            end
            
            clear_solution_caches!(m, algorithm)

            deriv4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            variance = :all_excluding_obc)[:variance], old_params)

            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv4_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                variance = :all_excluding_obc)[:variance], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                variance = :all_excluding_obc)[:variance], old_params)[1]
            end

            for i in 1:100
                local deriv4_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                                                            x -> begin 
                                                                clear_solution_caches!(m, algorithm)
                                                                
                                                                get_statistics(m, x, algorithm = algorithm, variance = :all_excluding_obc)[:variance]
                                                            end, old_params)
                if isfinite(ℒ.norm(deriv4_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv4_moon, deriv4_fin[1], rtol = 1e-5, atol = 1e-8)
                        @test check_isapprox(deriv4_zyg, deriv4_fin[1], rtol = 1e-5, atol = 1e-8)
                    end
                    @test check_isapprox(deriv4, deriv4_fin[1], rtol = 1e-5, atol = 1e-8)
                    break
                end
            end

            clear_solution_caches!(m, algorithm)

            deriv5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                                            covariance = :all_excluding_obc)[:covariance], old_params)

            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv5_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                                                covariance = :all_excluding_obc)[:covariance], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                                                covariance = :all_excluding_obc)[:covariance], old_params)[1]
            end         

            for i in 1:100        
                local deriv5_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                                                                x -> begin 
                                                                    clear_solution_caches!(m, algorithm)
                                                                    
                                                                    get_statistics(m, x, algorithm = algorithm, 
                                                                                    tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                                                                    covariance = :all_excluding_obc)[:covariance]
                                                                end, old_params)
                if isfinite(ℒ.norm(deriv5_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv5_moon, deriv5_fin[1], rtol = 1e-4, atol = 1e-8)
                        @test check_isapprox(deriv5_zyg, deriv5_fin[1], rtol = 1e-4, atol = 1e-8)
                    end

                    # println(ℒ.norm(deriv5 - deriv5_fin[1]) / max(ℒ.norm(deriv5), ℒ.norm(deriv5_fin[1])))                      
                    @test check_isapprox(deriv5, deriv5_fin[1], rtol = 1e-4, atol = 1e-8)
                    break
                end
            end

            clear_solution_caches!(m, algorithm)

            deriv6 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            autocorrelation = :all_excluding_obc)[:autocorrelation], old_params)

            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv6_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                autocorrelation = :all_excluding_obc)[:autocorrelation], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv6_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                autocorrelation = :all_excluding_obc)[:autocorrelation], old_params)[1]
            end

            for i in 1:100
                local deriv6_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                                                            x -> begin 
                                                                clear_solution_caches!(m, algorithm)
                                                                
                                                                get_statistics(m, x, algorithm = algorithm, autocorrelation = :all_excluding_obc)[:autocorrelation]
                                                            end, old_params)
                if isfinite(ℒ.norm(deriv6_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv6_moon, deriv6_fin[1], rtol = 1e-4)
                        @test check_isapprox(deriv6_zyg, deriv6_fin[1], rtol = 1e-4)
                    end
                    @test check_isapprox(deriv6, deriv6_fin[1], rtol = 1e-4)
                    break
                end
            end

            clear_solution_caches!(m, algorithm)

            # Restrict the correlation jacobian comparison to non-degenerate
            # variables. Degenerate-variance entries produce NaN/0-over-0
            # correlations whose FD jacobian is dominated by perturbation
            # noise (huge magnitude), while AD computes the analytic value
            # cleanly. Comparing only over non-degenerate entries keeps the
            # AD-vs-FD check meaningful without silently masking real bugs.
            corr_target_vars_jac = let
                _all_vars_jac = m.constants.post_model_macro.var
                _sd_jac = get_statistics(m, old_params, algorithm = algorithm,
                                         standard_deviation = _all_vars_jac)[:standard_deviation]
                _all_vars_jac[findall(>(1e-6), _sd_jac)]
            end

            deriv7 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                            correlation = corr_target_vars_jac)[:correlation], old_params)

            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                deriv7_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                correlation = corr_target_vars_jac)[:correlation], ADTypes.AutoMooncake(config = nothing), old_params)
                deriv7_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                correlation = corr_target_vars_jac)[:correlation], old_params)[1]
            end

            for i in 1:100
                local deriv7_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                                                            x -> begin
                                                                clear_solution_caches!(m, algorithm)

                                                                get_statistics(m, x, algorithm = algorithm, correlation = corr_target_vars_jac)[:correlation]
                                                            end, old_params)
                if isfinite(ℒ.norm(deriv7_fin[1]))
                    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                        @test check_isapprox(deriv7_moon, deriv7_fin[1], rtol = 1e-4, atol = 1e-8, nans = true)
                        @test check_isapprox(deriv7_zyg, deriv7_fin[1], rtol = 1e-4, atol = 1e-8, nans = true)
                    end
                    @test check_isapprox(deriv7, deriv7_fin[1], rtol = 1e-4, atol = 1e-8, nans = true)
                    break
                end
            end

            if algorithm == :pruned_third_order
                var_obj = x -> begin
                    clear_solution_caches!(m, algorithm)
                    get_statistics(m, x, algorithm = algorithm, variance = :all_excluding_obc)[:variance] |> sum
                end

                autocorr_obj = x -> begin
                    clear_solution_caches!(m, algorithm)
                    get_statistics(m, x, algorithm = algorithm, autocorrelation = :all_excluding_obc)[:autocorrelation] |> sum
                end

                var_grad_moon = DifferentiationInterface.gradient(var_obj, ADTypes.AutoMooncake(config = nothing), old_params)
                var_grad_zyg = Zygote.gradient(var_obj, old_params)[1]
                var_grad_fin = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3), var_obj, old_params)[1]
                @test all(isfinite, var_grad_moon)
                @test all(isfinite, var_grad_zyg)
                @test all(isfinite, var_grad_fin)
                @test ℒ.norm(var_grad_moon - var_grad_fin) / max(ℒ.norm(var_grad_fin), eps()) < 1e-4
                @test ℒ.norm(var_grad_zyg - var_grad_fin) / max(ℒ.norm(var_grad_fin), eps()) < 1e-4

                autocorr_grad_moon = DifferentiationInterface.gradient(autocorr_obj, ADTypes.AutoMooncake(config = nothing), old_params)
                autocorr_grad_zyg = Zygote.gradient(autocorr_obj, old_params)[1]
                autocorr_grad_fin = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3), autocorr_obj, old_params)[1]
                @test all(isfinite, autocorr_grad_moon)
                @test all(isfinite, autocorr_grad_zyg)
                @test all(isfinite, autocorr_grad_fin)
                @test ℒ.norm(autocorr_grad_moon - autocorr_grad_fin) / max(ℒ.norm(autocorr_grad_fin), eps()) < 1e-4
                @test ℒ.norm(autocorr_grad_zyg - autocorr_grad_fin) / max(ℒ.norm(autocorr_grad_fin), eps()) < 1e-4

                corr_obj = x -> begin
                    clear_solution_caches!(m, algorithm)
                    get_statistics(m, x, algorithm = algorithm, correlation = corr_target_vars_jac)[:correlation] |> sum
                end

                corr_grad_moon = DifferentiationInterface.gradient(corr_obj, ADTypes.AutoMooncake(config = nothing), old_params)
                corr_grad_zyg = Zygote.gradient(corr_obj, old_params)[1]
                corr_grad_fin = FiniteDifferences.grad(FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3), corr_obj, old_params)[1]
                @test all(isfinite, corr_grad_moon)
                @test all(isfinite, corr_grad_zyg)
                @test all(isfinite, corr_grad_fin)
                @test ℒ.norm(corr_grad_moon - corr_grad_fin) / max(ℒ.norm(corr_grad_fin), eps()) < 1e-4
                @test ℒ.norm(corr_grad_zyg - corr_grad_fin) / max(ℒ.norm(corr_grad_fin), eps()) < 1e-4
            end
            end
        

        

            for tol in [MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            clear_solution_caches!(m, algorithm)

                            DERIV1 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)
                            @test check_isapprox(deriv1, DERIV1, rtol = 1e-8)
                            
                            DERIV1_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], ADTypes.AutoMooncake(config = nothing), old_params)
                            DERIV1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)[1]
                            @test check_isapprox(DERIV1_moon, DERIV1, rtol = 1e-8)
                            @test check_isapprox(DERIV1_zyg, DERIV1, rtol = 1e-8)
                        

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)

                            DERIV2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            mean = :all_excluding_obc)[:mean], old_params)
                            @test check_isapprox(deriv2, DERIV2, rtol = 1e-8)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)
    
                                DERIV2_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                mean = :all_excluding_obc)[:mean], ADTypes.AutoMooncake(config = nothing), old_params)
                                DERIV2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                mean = :all_excluding_obc)[:mean], old_params)[1]
                                @test check_isapprox(DERIV2_moon, DERIV2, rtol = 1e-8)
                                @test check_isapprox(DERIV2_zyg, DERIV2, rtol = 1e-8)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
                            @test check_isapprox(deriv3, DERIV3, rtol = 1e-8)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)
    
                                DERIV3_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], ADTypes.AutoMooncake(config = nothing), old_params)
                                DERIV3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)[1]
                                @test check_isapprox(DERIV3_moon, DERIV3, rtol = 1e-6)
                                @test check_isapprox(DERIV3_zyg, DERIV3, rtol = 1e-6)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            variance = :all_excluding_obc)[:variance], old_params)
                            @test check_isapprox(deriv4, DERIV4, rtol = 1e-8)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)
    
                                DERIV4_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                variance = :all_excluding_obc)[:variance], ADTypes.AutoMooncake(config = nothing), old_params)
                                DERIV4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                variance = :all_excluding_obc)[:variance], old_params)[1]
                                @test check_isapprox(DERIV4_moon, DERIV4, rtol = 1e-8)
                                @test check_isapprox(DERIV4_zyg, DERIV4, rtol = 1e-8)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            covariance = :all_excluding_obc)[:covariance], old_params)
                            # println(ℒ.norm(deriv5 - DERIV5) / max(ℒ.norm(deriv5), ℒ.norm(DERIV5)))                      
							@test check_isapprox(deriv5, DERIV5, rtol = 1e-4)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)
    
                                DERIV5_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                covariance = :all_excluding_obc)[:covariance], ADTypes.AutoMooncake(config = nothing), old_params)
                                DERIV5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                covariance = :all_excluding_obc)[:covariance], old_params)[1]
                                @test check_isapprox(DERIV5_moon, DERIV5, rtol = 1e-4)
                                @test check_isapprox(DERIV5_zyg, DERIV5, rtol = 1e-4)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV6 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            autocorrelation = :all_excluding_obc)[:autocorrelation], old_params)
                            @test check_isapprox(deriv6, DERIV6, rtol = 1e-4)

                            if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                                clear_solution_caches!(m, algorithm)
    
                                DERIV6_moon = DifferentiationInterface.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                autocorrelation = :all_excluding_obc)[:autocorrelation], ADTypes.AutoMooncake(config = nothing), old_params)
                                DERIV6_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                autocorrelation = :all_excluding_obc)[:autocorrelation], old_params)[1]
                                @test check_isapprox(DERIV6_moon, DERIV6, rtol = 1e-4)
                                @test check_isapprox(DERIV6_zyg, DERIV6, rtol = 1e-4)
                            end
                            end
                        end
                    end
                end
            end
    end


    @testset "get_statistics - grouped covariance" begin
        # Test grouped covariance functionality
        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            # Test with 2 groups
            stats_grouped = get_statistics(m, old_params, 
                                          algorithm = algorithm,
                                          covariance = [m.constants.post_model_macro.var[2:3], m.constants.post_model_macro.var[4:5]])
            
            @test haskey(stats_grouped, :covariance)
            @test stats_grouped[:covariance] isa Matrix
            @test size(stats_grouped[:covariance]) == (4, 4)
            
            # Compare with non-grouped version for validation
            stats_non_grouped_1 = get_statistics(m, old_params,
                                                algorithm = algorithm,
                                                covariance = m.constants.post_model_macro.var[2:3])
            
            stats_non_grouped_2 = get_statistics(m, old_params,
                                                algorithm = algorithm,
                                                covariance = m.constants.post_model_macro.var[4:5])
            
            # Check that within-group covariances match
            @test check_isapprox(stats_grouped[:covariance][1:2, 1:2], stats_non_grouped_1[:covariance], rtol = 1e-6, nans = true)
            @test check_isapprox(stats_grouped[:covariance][3:4, 3:4], stats_non_grouped_2[:covariance], rtol = 1e-6, nans = true)
            
            # Check that cross-group covariances are zero
            @test all(stats_grouped[:covariance][1:2, 3:4] .== 0)
            @test all(stats_grouped[:covariance][3:4, 1:2] .== 0)
            
            # Test with different group sizes
            stats_varied = get_statistics(m, old_params,
                                         algorithm = algorithm,
                                         covariance = [[m.constants.post_model_macro.var[2]], m.constants.post_model_macro.var[3:5]])
            
            @test stats_varied[:covariance] isa Matrix
            @test size(stats_varied[:covariance]) == (4, 4)
            # First group is 1x1, second group is 3x3
            @test stats_varied[:covariance][1, 1] != 0  # within first group
            @test all(stats_varied[:covariance][1, 2:4] .== 0)  # cross-group
            @test all(stats_varied[:covariance][2:4, 1] .== 0)  # cross-group
        end
    end


    @testset "get_statistics - correlation" begin
        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            # Pick the first 4 model variables that are non-degenerate (positive variance,
            # well above sqrt(eps)). Some models (e.g. Smets_Wouters_2007) have
            # near-constant variables in their leading positions which would produce
            # NaN/Inf-like correlation entries and break the cov/(sd*sd') cross-check.
            _all_vars = m.constants.post_model_macro.var
            _all_sd = let s = get_statistics(m, old_params, algorithm = algorithm,
                                              standard_deviation = _all_vars)
                s[:standard_deviation]
            end
            _nondeg_idx = findall(>(1e-6), _all_sd)
            vars_corr = _all_vars[_nondeg_idx]

            # Flat input: full correlation matrix among requested variables
            stats_corr = get_statistics(m, old_params, algorithm = algorithm,
                                        correlation = vars_corr)
            @test haskey(stats_corr, :correlation)
            @test stats_corr[:correlation] isa AbstractMatrix
            @test size(stats_corr[:correlation]) == (length(vars_corr), length(vars_corr))
            # Diagonal must be 1 (or NaN for degenerate variables)
            for i in 1:length(vars_corr)
                @test check_isapprox(stats_corr[:correlation][i, i], 1.0, rtol = 1e-6, nans = true)
            end
            # Symmetric
            @test check_isapprox(stats_corr[:correlation], stats_corr[:correlation]', rtol = 1e-6, nans = true)
            # All entries in [-1, 1] (or NaN)
            @test all(x -> isnan(x) || (-1 - 1e-6 <= x <= 1 + 1e-6), stats_corr[:correlation])

            # Cross-check correlation = covariance / (std * std')
            stats_combo = get_statistics(m, old_params, algorithm = algorithm,
                                         standard_deviation = vars_corr,
                                         covariance = vars_corr,
                                         correlation = vars_corr)
            cov_full = stats_combo[:covariance] + stats_combo[:covariance]' - ℒ.Diagonal(stats_combo[:covariance])
            sd = stats_combo[:standard_deviation]
            expected_corr = cov_full ./ (sd * sd')
            @test check_isapprox(stats_combo[:correlation], expected_corr, rtol = 1e-6, atol = 1e-8, nans = true)

            # Grouped correlation: cross-group entries are zero, within-group preserved
            if length(vars_corr) >= 4
                stats_grouped_corr = get_statistics(m, old_params, algorithm = algorithm,
                                                    correlation = [vars_corr[1:2], vars_corr[3:4]])
                @test stats_grouped_corr[:correlation] isa Matrix
                @test size(stats_grouped_corr[:correlation]) == (4, 4)
                # Within-group blocks match unrestricted correlation
                stats_block1 = get_statistics(m, old_params, algorithm = algorithm,
                                              correlation = vars_corr[1:2])
                stats_block2 = get_statistics(m, old_params, algorithm = algorithm,
                                              correlation = vars_corr[3:4])
                @test check_isapprox(stats_grouped_corr[:correlation][1:2, 1:2], stats_block1[:correlation], rtol = 1e-6, nans = true)
                @test check_isapprox(stats_grouped_corr[:correlation][3:4, 3:4], stats_block2[:correlation], rtol = 1e-6, nans = true)
                # Cross-group entries are zero
                @test all(stats_grouped_corr[:correlation][1:2, 3:4] .== 0)
                @test all(stats_grouped_corr[:correlation][3:4, 1:2] .== 0)
            end
        end
    end


    @testset "get_moments" begin
        for non_stochastic_steady_state in [true, false]
            for mean in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? [true, false] : [false])
                for standard_deviation in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? [true, false] : [false])
                    for variance in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? [true, false] : [false])
                        for covariance in (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? [true, false] : [false])
                            for derivatives in [true, false]
                                get_moments(m,
                                            algorithm = algorithm,
                                            non_stochastic_steady_state = non_stochastic_steady_state,
                                            mean = mean,
                                            standard_deviation = standard_deviation,
                                            variance = variance,
                                            covariance = covariance,
                                            derivatives = derivatives)
                            end
                        end
                    end
                end
            end
        end

        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            get_variance(m, algorithm = algorithm)
            
            get_var(m, algorithm = algorithm)

            var(m, algorithm = algorithm)


            get_standard_deviation(m, algorithm = algorithm)

            get_std(m, algorithm = algorithm)

            get_stdev(m, algorithm = algorithm)

            stdev(m, algorithm = algorithm)

            std(m, algorithm = algorithm)


            get_covariance(m, algorithm = algorithm)

            get_cov(m, algorithm = algorithm)

            cov(m, algorithm = algorithm)

            get_correlation(m, algorithm = algorithm)

            get_corr(m, algorithm = algorithm)

            corr(m, algorithm = algorithm)

            
            get_mean(m, algorithm = algorithm)
        end
            

        for parameter_derivatives in param_derivs
                get_moments(m,
                            algorithm = algorithm,
                            non_stochastic_steady_state = true,
                            mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            correlation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            parameter_derivatives = parameter_derivatives,
                            derivatives = true)
        end
        
        for variables in vars
                get_moments(m,
                            algorithm = algorithm,
                            variables = variables,
                            non_stochastic_steady_state = true,
                            mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            correlation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            derivatives = true)
        end

        

        for parameters in params
            # derivatives=false: sweep all solver combos to verify numerical consistency
            clear_solution_caches!(m, algorithm)
        
            moms = get_moments(m,
                                algorithm = algorithm,
                                parameters = parameters,
                                non_stochastic_steady_state = true,
                                mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                derivatives = false)
                            
            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            clear_solution_caches!(m, algorithm)
                            
                            MOMS = get_moments(m,
                                                algorithm = algorithm,
                                                parameters = parameters,
                                                non_stochastic_steady_state = true,
                                                mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                                standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                                variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                                covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                                derivatives = false,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm)

                            @test check_isapprox([v for (k,v) in moms], [v for (k,v) in MOMS], rtol = 1e-8)
                        end
                    end
                end
            end

            # derivatives=true: only test one representative solver combo (derivatives don't depend on solver choice)
                clear_solution_caches!(m, algorithm)

                moms_d = get_moments(m,
                                    algorithm = algorithm,
                                    parameters = parameters,
                                    non_stochastic_steady_state = true,
                                    mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    derivatives = true)

                clear_solution_caches!(m, algorithm)

                MOMS_d = get_moments(m,
                                    algorithm = algorithm,
                                    parameters = parameters,
                                    non_stochastic_steady_state = true,
                                    mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    derivatives = true,
                                    tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14)),
                                    quadratic_matrix_equation_algorithm = :doubling,
                                    lyapunov_algorithm = :doubling,
                                    sylvester_algorithm = :doubling)

                @test check_isapprox([v for (k,v) in moms_d], [v for (k,v) in MOMS_d], rtol = 1e-8)
        end

        # FD parity for get_moments derivative columns (rrule-based VJP Jacobians)
        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            # NSSS derivatives
            clear_solution_caches!(m, algorithm)
            mom_nsss = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = true, standard_deviation = false, derivatives = true)
            nsss_jac = collect(mom_nsss[:non_stochastic_steady_state])[:, 2:end]

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        collect(get_moments(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, non_stochastic_steady_state = true, standard_deviation = false, derivatives = false)[:non_stochastic_steady_state])
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(nsss_jac, fd[1], rtol = 1e-5)
                    break
                end
            end
            m.parameter_values .= old_params

            # Variance derivatives
            clear_solution_caches!(m, algorithm)
            mom_var = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, variance = true, derivatives = true)
            var_jac = collect(mom_var[:variance])[:, 2:end]

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        collect(get_moments(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, variance = true, derivatives = false)[:variance])
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(var_jac, fd[1], rtol = 1e-4)
                    break
                end
            end
            m.parameter_values .= old_params

            # Standard deviation derivatives
            clear_solution_caches!(m, algorithm)
            mom_std = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = true, variance = false, derivatives = true)
            std_jac = collect(mom_std[:standard_deviation])[:, 2:end]

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        collect(get_moments(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = true, variance = false, derivatives = false)[:standard_deviation])
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(std_jac, fd[1], rtol = 1e-4)
                    break
                end
            end
            m.parameter_values .= old_params

            # Covariance derivatives
            clear_solution_caches!(m, algorithm)
            mom_cov = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, covariance = true,
                                  tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                  derivatives = true)
            cov_ka = collect(mom_cov[:covariance])
            n_cv = size(cov_ka, 1)
            cov_jac = reshape(cov_ka[:, :, 2:end], n_cv * n_cv, :)

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        vec(collect(get_moments(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, covariance = true,
                            tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                            derivatives = false)[:covariance]))
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(cov_jac, fd[1], rtol = 1e-4, nans = true)
                    break
                end
            end
            m.parameter_values .= old_params

            # Mean derivatives (for algorithms that support it)
            if algorithm ∈ [:pruned_second_order, :pruned_third_order]
                clear_solution_caches!(m, algorithm)
                mom_mean = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, mean = true, derivatives = true)
                mean_jac = collect(mom_mean[:mean])[:, 2:end]

                for i in 1:100
                    local fd = FiniteDifferences.jacobian(
                        FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3),
                        x -> begin
                            clear_solution_caches!(m, algorithm)
                            collect(get_moments(m,
                                parameters = m.constants.post_complete_parameters.parameters .=> x,
                                algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, mean = true, derivatives = false)[:mean])
                        end, old_params)
                    if isfinite(ℒ.norm(fd[1]))
                        @test check_isapprox(mean_jac, fd[1], rtol = 1e-4)
                        break
                    end
                end
                m.parameter_values .= old_params
            end

            # Correlation derivatives
            clear_solution_caches!(m, algorithm)
            mom_corr = get_moments(m, algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, correlation = true,
                                  tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                                  derivatives = true)
            corr_ka = collect(mom_corr[:correlation])
            n_cr = size(corr_ka, 1)
            corr_jac = reshape(corr_ka[:, :, 2:end], n_cr * n_cr, :)

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.central_fdm(length(m.constants.post_complete_parameters.parameters) > 20 ? 5 : 4, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        vec(collect(get_moments(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, non_stochastic_steady_state = false, standard_deviation = false, correlation = true,
                            tol = MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14), second_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14)), third_order = MacroModelling.HigherOrderTolerances(sylvester = MacroModelling.SolverTolerances(acceptance_tol = 1e-14), lyapunov = MacroModelling.SolverTolerances(acceptance_tol = 1e-14))),
                            derivatives = false)[:correlation]))
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(corr_jac, fd[1], rtol = 1e-4, nans = true)
                    break
                end
            end
            m.parameter_values .= old_params
        end
    end


    @testset "get_irf" begin
        m.parameter_values .= old_params
        clear_solution_caches!(m, algorithm)
        Random.seed!(123)

        for ignore_obc in [true,false]
            for generalised_irf in (algorithm == :first_order ? [false] : [true,false])
                for negative_shock in [true,false]
                    for shock_size in [.1,1]
                        get_irf(m, 
                                algorithm = algorithm, 
                                ignore_obc = ignore_obc,
                                generalised_irf = generalised_irf,
                                negative_shock = negative_shock,
                                shock_size = shock_size)
                    end
                end
            end
        end
        
        simulate(m, algorithm = algorithm)

        get_simulation(m, algorithm = algorithm)

        get_simulations(m, algorithm = algorithm)

        get_girf(m, algorithm = algorithm)

        for periods in [1,10]
            for levels in [true,false]
                get_irf(m, 
                        algorithm = algorithm, 
                        levels = levels,
                        periods = periods)

                get_irfs(m, 
                        algorithm = algorithm, 
                        levels = levels,
                        periods = periods)

                get_IRF(m, 
                        algorithm = algorithm, 
                        levels = levels,
                        periods = periods)
            end
        end

        shock_mat = randn(m.constants.post_model_macro.nExo,3)

        shock_mat2 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = m.constants.post_model_macro.exo, Periods = 1:10)

        shock_mat3 = KeyedArray(randn(m.constants.post_model_macro.nExo,10),Shocks = string.(m.constants.post_model_macro.exo), Periods = 1:10)

        for parameters in params
            for initial_state in init_states
                clear_solution_caches!(m, algorithm)
                
                irf_ = get_irf(m, 
                                algorithm = algorithm, 
                                parameters = parameters, 
                                ignore_obc = true, 
                                initial_state = initial_state)
                
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        # for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                clear_solution_caches!(m, algorithm)
                                            
                                IRF_ = get_irf(m, 
                                                algorithm = algorithm, 
                                                ignore_obc = true,
                                                parameters = parameters,
                                                initial_state = initial_state,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                # lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm)
                                @test check_isapprox(irf_, IRF_, rtol = 1e-6)
                            end
                        # end
                    end
                end
                
                for variables in vars
                    clear_solution_caches!(m, algorithm)
                                
                    get_irf(m, algorithm = algorithm, 
                            parameters = parameters, 
                            ignore_obc = true, 
                            variables = variables, 
                            initial_state = initial_state)
                end

                for shocks in [:all, :all_excluding_obc, :none, :simulate, m.constants.post_model_macro.exo[1], m.constants.post_model_macro.exo[1:2], reshape(m.constants.post_model_macro.exo,1,length(m.constants.post_model_macro.exo)), Tuple(m.constants.post_model_macro.exo), Tuple(string.(m.constants.post_model_macro.exo)), string(m.constants.post_model_macro.exo[1]), reshape(string.(m.constants.post_model_macro.exo),1,length(m.constants.post_model_macro.exo)), string.(m.constants.post_model_macro.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                    clear_solution_caches!(m, algorithm)
                                
                    get_irf(m, algorithm = algorithm, 
                            parameters = parameters, 
                            ignore_obc = true, 
                            initial_state = initial_state, 
                            shocks = shocks)
                end
            end
        end
    end

    @testset "get_non_stochastic_steady_state_residuals" begin
        stst = SS(m, derivatives = false)
        
        

        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
            for parameters in params 
                clear_solution_caches!(m, algorithm)

                res = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false, parameters = parameters)

                for values in [Dict(axiskeys(stst)[1] .=> collect(stst)), Dict(string.(axiskeys(stst)[1]) .=> collect(stst)), collect(stst)]   
                    clear_solution_caches!(m, algorithm)
                    
                    RES = get_non_stochastic_steady_state_residuals(m, values, tol = tol, verbose = false, parameters = parameters)

                    @test check_isapprox(res, RES, rtol = 1e-8, atol = 1e-8, nans = true)
                end
            end

            clear_solution_caches!(m, algorithm)

            res1 = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false)

            clear_solution_caches!(m, algorithm)

            res2 = get_non_stochastic_steady_state_residuals(m, stst[1:3], tol = tol, verbose = false)

            @test check_isapprox(res1, res2, rtol = 1e-8, atol = 1e-8, nans = true)

            get_residuals(m, stst)

            check_residuals(m, stst)
        end
    end

    @testset "get_steady_state" begin
        clear_solution_caches!(m, algorithm)
        get_non_stochastic_steady_state(m)
        
        clear_solution_caches!(m, algorithm)
        SS(m)

        clear_solution_caches!(m, algorithm)
        steady_state(m)

        clear_solution_caches!(m, algorithm)
        get_SS(m)

        clear_solution_caches!(m, algorithm)
        get_ss(m)

        clear_solution_caches!(m, algorithm)
        ss(m)

        if !(algorithm == :first_order)
            clear_solution_caches!(m, algorithm)
            get_stochastic_steady_state(m)

            clear_solution_caches!(m, algorithm)
            get_SSS(m)

            clear_solution_caches!(m, algorithm)
            SSS(m)

            clear_solution_caches!(m, algorithm)
            sss(m)
        end 

        

        for derivatives in [true, false]
            for stochastic in (algorithm == :first_order ? [false] : [true, false])
                for return_variables_only in [true, false]
                    for verbose in [false]
                        for silent in [true, false]
                            clear_solution_caches!(m, algorithm)
            
                            NSSS = get_steady_state(m, 
                                                    verbose = verbose, 
                                                    silent = silent, 
                                                    return_variables_only = return_variables_only, 
                                                    algorithm = algorithm, 
                                                    stochastic = stochastic, 
                                                    derivatives = derivatives)
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                for sylvester_algorithm in sylvester_algorithms
                                    clear_solution_caches!(m, algorithm)
                    
                                    nsss = get_steady_state(m, 
                                                            verbose = verbose, 
                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm, 
                                                            sylvester_algorithm = sylvester_algorithm, 
                                                            silent = silent, 
                                                            return_variables_only = return_variables_only, 
                                                            algorithm = algorithm, 
                                                            stochastic = stochastic, 
                                                            derivatives = derivatives)
                                    @test check_isapprox(NSSS, nsss, rtol = 1e-8)
                                end
                            end
                        end
                    end
                end
            end
        end

        for parameter_derivatives in param_derivs
            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(nsss = MacroModelling.NsssTolerances(xtol = 1e-14))]
                    clear_solution_caches!(m, algorithm)
    
                    nsss = get_steady_state(m, 
                                            parameters = parameters, 
                                            algorithm = algorithm, 
                                            parameter_derivatives = parameter_derivatives,
                                            tol = tol,
                                            verbose = false)
                end
            end
        end

            # FD parity for get_steady_state derivative columns (rrule-based VJP Jacobians)
            # NSSS derivatives
            clear_solution_caches!(m, algorithm)
            nsss_d = get_steady_state(m, algorithm = algorithm, stochastic = false, derivatives = true, return_variables_only = true)
            nsss_jac = collect(nsss_d)[:, 2:end]

            for i in 1:100
                local fd = FiniteDifferences.jacobian(
                    FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3),
                    x -> begin
                        clear_solution_caches!(m, algorithm)
                        collect(get_steady_state(m,
                            parameters = m.constants.post_complete_parameters.parameters .=> x,
                            algorithm = algorithm, stochastic = false, derivatives = false, return_variables_only = true))
                    end, old_params)
                if isfinite(ℒ.norm(fd[1]))
                    @test check_isapprox(nsss_jac, fd[1], rtol = 1e-5)
                    break
                end
            end
            m.parameter_values .= old_params

            # Stochastic SS derivatives (non-first-order only)
            if algorithm != :first_order
                clear_solution_caches!(m, algorithm)
                sss_d = get_steady_state(m, algorithm = algorithm, stochastic = true, derivatives = true, return_variables_only = true)
                sss_jac = collect(sss_d)[:, 2:end]

                for i in 1:100
                    local fd = FiniteDifferences.jacobian(
                        FiniteDifferences.forward_fdm(3, 1, max_range = 1e-3),
                        x -> begin
                            clear_solution_caches!(m, algorithm)
                            collect(get_steady_state(m,
                                parameters = m.constants.post_complete_parameters.parameters .=> x,
                                algorithm = algorithm, stochastic = true, derivatives = false, return_variables_only = true))
                        end, old_params)
                    if isfinite(ℒ.norm(fd[1]))
                        @test check_isapprox(sss_jac, fd[1], rtol = 1e-4)
                        break
                    end
                end
                m.parameter_values .= old_params
            end
    end

    GC.gc()
    # Inspect Model
    get_equations(m) 
    get_steady_state_equations(m) 
    get_dynamic_equations(m) 
    get_calibration_equations(m) 
    get_parameters(m) 
    get_parameters(m, values = true) 
    get_calibrated_parameters(m) 
    get_calibrated_parameters(m, values = true) 
    get_parameters_in_equations(m) 
    get_parameters_defined_by_parameters(m) 
    get_parameters_defining_parameters(m) 
    get_calibration_equation_parameters(m) 
    get_variables(m) 
    get_nonnegativity_auxiliary_variables(m) 
    get_dynamic_auxiliary_variables(m) 
    get_shocks(m) 
    get_state_variables(m) 
    get_jump_variables(m)

    GC.gc()

    if algorithm == :first_order
        lvl_irfs  = get_irf(m, old_params, verbose = true, levels = true, variables = :all)
        new_sub_lvl_irfs  = get_irf(m, old_params, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]), levels = true, variables = :all)
        @test check_isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))
    end

end