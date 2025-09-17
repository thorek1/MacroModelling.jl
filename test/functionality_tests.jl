function functionality_test(m; algorithm = :first_order, plots = true)
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
                
    param_derivs = [:all, 
                    m.parameters[1], 
                    m.parameters[1:3], 
                    Tuple(m.parameters[1:3]), 
                    reshape(m.parameters[1:3],3,1), 
                    string.(m.parameters[1]), 
                    string.(m.parameters[1:2]), 
                    string.(Tuple(m.parameters[1:3])), 
                    string.(reshape(m.parameters[1:3],3,1))]

    vars = [:all, :all_excluding_obc, :all_excluding_auxiliary_and_obc, m.var[1], m.var[1:2], Tuple(m.timings.var), reshape(m.timings.var,1,length(m.timings.var)), string(m.var[1]), string.(m.var[1:2]), Tuple(string.(m.timings.var)), reshape(string.(m.timings.var),1,length(m.timings.var))]

    init_state = get_irf(m, algorithm = algorithm, shocks = :none, levels = !(algorithm in [:pruned_second_order, :pruned_third_order]), variables = :all, periods = 1) |> vec

    init_states = [[0.0], init_state, algorithm  == :pruned_second_order ? [zero(init_state), init_state] : algorithm == :pruned_third_order ? [zero(init_state), init_state, zero(init_state)] : init_state .* 1.01]

    if plots
        @testset "plot_model_estimates" begin
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
                                            data_in_levels = false)
            end

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

            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)

            i = 1
            
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for tol in [MacroModelling.Tolerances(NSSS_xtol = 1e-14), MacroModelling.Tolerances()]
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

            # for shock_decomposition in (algorithm in [:second_order, :third_order] ? [false] : [true, false])
                for filter in (algorithm == :first_order ? filters : [:inversion])
                    for smooth in [true, false]
                        for presample_periods in [0, 3]
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
                                                    filter = filter,
                                                    smooth = smooth,
                                                    presample_periods = presample_periods)
                        end
                    end
                end
            # end


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

            for variables in vars
                plot_model_estimates(m, data, 
                                        variables = variables,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end


            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)
                                    
            for variables in vars
                plot_model_estimates!(m, data, 
                                        variables = variables,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end


            plot_model_estimates(m, data_in_levels, 
                                    algorithm = algorithm, 
                                    data_in_levels = true)
                                    
            for shocks in [:all, :all_excluding_obc, :none, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2])]
                plot_model_estimates!(m, data, 
                                        shocks = shocks,
                                        algorithm = algorithm, 
                                        data_in_levels = false)
            end

            for shocks in [:all, :all_excluding_obc, :none, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2])]
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
        end
        
        @testset "plot_solution" begin
            

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
        end


        @testset "plot_irf" begin
            

            # plotlyjs_backend()

            plot_IRF(m, algorithm = algorithm)

            # gr_backend()

            plot_irfs(m, algorithm = algorithm)

            plot_girf!(m, algorithm = algorithm)

            plot_simulations(m, algorithm = algorithm)

            plot_irf!(m, algorithm = algorithm)

            plot_simulation(m, algorithm = algorithm)

            plot_irfs!(m, algorithm = algorithm)

            plot_girf(m, algorithm = algorithm)

            plot_simulation!(m, algorithm = algorithm)

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
    

            plot_irf(m, algorithm = algorithm)

            i  = 1

            for parameters in params
                for tol in [MacroModelling.Tolerances(NSSS_xtol = 1e-14), MacroModelling.Tolerances()]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
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
                                            lyapunov_algorithm = lyapunov_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)
                            end
                        end
                    end
                end
            end


            plot_irf(m, algorithm = algorithm)

            i = 1

            for initial_state in sort(init_states, rev = true)
                if i % 10 == 0
                    plot_irf(m, algorithm = algorithm)
                end
                
                i += 1

                clear_solution_caches!(m, algorithm)
                            
                plot_irf!(m, algorithm = algorithm, initial_state = initial_state)
            end

            for initial_state in init_states
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, initial_state = initial_state)
            end


            for variables in vars
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, variables = variables)
            end

            
            plot_irf(m, algorithm = algorithm)
            
            i = 1

            for shocks in [:none, :all, :all_excluding_obc, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                if i % 4 == 0
                    plot_irf(m, algorithm = algorithm)
                end

                i += 1
                
                clear_solution_caches!(m, algorithm)
                            
                plot_irf!(m, algorithm = algorithm, shocks = shocks)
            end

            for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                clear_solution_caches!(m, algorithm)
                            
                plot_irf(m, algorithm = algorithm, shocks = shocks)
            end
            
            for plot_attributes in [Dict(), Dict(:plot_titlefontcolor => :red)]
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
        end


        @testset "plot_conditional_variance_decomposition" begin
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
        end

        @testset "plot_conditional_forecast" begin
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

            plot_conditional_forecast(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[end])

            i = 1

            for tol in [MacroModelling.Tolerances(NSSS_xtol = 1e-14), MacroModelling.Tolerances()]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for lyapunov_algorithm in lyapunov_algorithms
                        for sylvester_algorithm in sylvester_algorithms
                            if i % 4 == 0
                                plot_conditional_forecast(m, conditions[end],
                                                        conditions_in_levels = false,
                                                        algorithm = algorithm, 
                                                        shocks = shocks[end])
                            end

                            i += 1

                            clear_solution_caches!(m, algorithm)
                        
                            plot_conditional_forecast!(m, conditions[end],
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
                # for levels in [true, false]
                    clear_solution_caches!(m, algorithm)
                
                    plot_conditional_forecast!(m, conditions[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm, 
                                                periods = periods,
                                                # levels = levels,
                                                shocks = shocks[end])
                # end
            end


            for variables in vars
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            algorithm = algorithm, 
                                            variables = variables)
            end
            
            for initial_state in init_states
                plot_conditional_forecast(m, conditions[end],
                                            conditions_in_levels = false,
                                            initial_state = initial_state,
                                            algorithm = algorithm)
            end

            plot_conditional_forecast(m, conditions[end],
                                        conditions_in_levels = false,
                                        algorithm = algorithm)

            for initial_state in sort(init_states, rev = true)
                plot_conditional_forecast!(m, conditions[end],
                                            conditions_in_levels = false,
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

            for shcks in shocks
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

            for parameters in params
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

            for cndtns in conditions
                plot_conditional_forecast!(m, cndtns,
                                            conditions_in_levels = false,
                                            algorithm = algorithm)
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
        
        if length(m.exo) > 3
            n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
            var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]
        elseif length(m.var) == 17
            var_idxs = [5]
        else
            var_idxs = [1]
        end

        Random.seed!(418023)

        simulation = simulate(m, algorithm = algorithm)

        data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)
        data = data_in_levels .- m.solution.non_stochastic_steady_state[var_idxs]


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
                                    @test isapprox(estim1, estim2, rtol = 1e-8)

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
                                    @test isapprox(estim1, estim2, rtol = 1e-8)

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
                                        @test isapprox(estim1, estim2, rtol = 1e-8)

                                        
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
                                        @test isapprox(estim1, estim2, rtol = 1e-8)
                                    end
                                end
                            end
                        end
                    end
                end
            end

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
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
                            @test isapprox(estim1,estim2)
                        end
                    end
                end
            end

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
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
                        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
                            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                                llh = get_loglikelihood(m, data_in_levels, parameter_values,
                                                        algorithm = algorithm,
                                                        filter = filter,
                                                        presample_periods = presample_periods,
                                                        initial_covariance = initial_covariance,
                                                        tol = tol,
                                                        verbose = verbose)

                                clear_solution_caches!(m, algorithm)
                        
                                zyg_grad_llh = Zygote.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                algorithm = algorithm,
                                                                                                filter = filter,
                                                                                                presample_periods = presample_periods,
                                                                                                initial_covariance = initial_covariance,
                                                                                                tol = tol,
                                                                                                verbose = verbose), parameter_values)

                                if algorithm == :first_order && filter == :kalman
                                    for i in 1:100
                                        local fin_grad_llh = FiniteDifferences.grad(FiniteDifferences.central_fdm(length(m.parameters) > 20 ? 3 : 4, 1, max_range = 1e-3), 
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
                                            @test isapprox(fin_grad_llh[1], zyg_grad_llh[1], rtol = 1e-5)
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
                                            @test isapprox(llh, LLH, rtol = 1e-8)

                                            clear_solution_caches!(m, algorithm)
                                    
                                            ZYG_grad_llh = Zygote.gradient(x -> get_loglikelihood(m, data_in_levels, x,
                                                                                                            algorithm = algorithm,
                                                                                                            filter = filter,
                                                                                                            presample_periods = presample_periods,
                                                                                                            initial_covariance = initial_covariance,
                                                                                                            tol = tol,
                                                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                                                            sylvester_algorithm = sylvester_algorithm,
                                                                                                            verbose = verbose), parameter_values)
            
                                            @test isapprox(ZYG_grad_llh[1], zyg_grad_llh[1], rtol = 1e-6)    
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

        cond_fcst = get_conditional_forecast(m, conditions[1],
                                            conditions_in_levels = false,
                                            initial_state = [0.0],
                                            algorithm = algorithm, 
                                            shocks = shocks[1])

        

        for periods in [0,10]
            for variables in vars
                for levels in [true, false]
                    for verbose in [false] # [true, false]
                        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                for lyapunov_algorithm in lyapunov_algorithms
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
                                                                            lyapunov_algorithm = lyapunov_algorithm,
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
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                verbose = verbose)

                                        @test isapprox(cond_fcst, cond_fcst_lvl)

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
                                                                                lyapunov_algorithm = lyapunov_algorithm,
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
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm,
                                                                                verbose = verbose)
                                                                                
                                        @test isapprox(cond_fcst, cond_fcst_lvl)
                                    end
                                end
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
                for tol in [MacroModelling.Tolerances(), MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            
                            if algorithm == :first_order
                                clear_solution_caches!(m, algorithm)

                                VAR_DECOMP = get_variance_decomposition(m,
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        verbose = verbose)
                                                                        
                                @test isapprox(var_decomp, VAR_DECOMP, rtol = 1e-8)

                                clear_solution_caches!(m, algorithm)
                                                                        
                                COND_VAR_DECOMP = get_conditional_variance_decomposition(m,
                                                                                        tol = tol,
                                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                                        verbose = verbose)

                                @test isapprox(cond_var_decomp, COND_VAR_DECOMP, rtol = 1e-8)

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

                                @test isapprox(corrl, CORRL, rtol = 1e-5)

                                clear_solution_caches!(m, algorithm)
                                
                                AUTOCORR = get_autocorrelation(m,
                                                                algorithm = algorithm,
                                                                tol = tol,
                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                sylvester_algorithm = sylvester_algorithm,
                                                                verbose = verbose)

                                @test isapprox(autocorr_, AUTOCORR, rtol = 1e-8)
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
            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)
                        
                        SOL = get_solution(m,
                                            algorithm = algorithm,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm,
                                            verbose = verbose)
                        @test isapprox(sol, SOL)#, rtol = eps(Float32))
                    end
                end
            end
        end
    end

    @testset "get_solution with parameter input" begin
        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
            get_first_order_solution(m, parameter_values)

            get_perturbation_solution(m, parameter_values)
            
            if algorithm in [:second_order, :pruned_second_order,:third_order, :pruned_third_order]
                get_second_order_solution(m, parameter_values)

                if algorithm in [:third_order, :pruned_third_order]
                    get_third_order_solution(m, parameter_values)
                end
            end

            sol = get_solution(m, parameter_values, algorithm = algorithm)

            clear_solution_caches!(m, algorithm)

            deriv_sol = []
            for i in 1:length(sol)-2
                push!(deriv_sol, ForwardDiff.jacobian(x->get_solution(m, x, algorithm = algorithm)[i], parameter_values))
            end

            clear_solution_caches!(m, algorithm)

            deriv_sol_fin = []
            for i in 1:length(sol)-2
                push!(deriv_sol_fin, FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                        x -> begin 
                                                            clear_solution_caches!(m, algorithm)
                                                            
                                                            get_solution(m, x, algorithm = algorithm)[i]
                                                        end, parameter_values)[1])
            end

            clear_solution_caches!(m, algorithm)

            deriv_sol_zyg = []
            for i in 1:length(sol)-2
                push!(deriv_sol_zyg, Zygote.jacobian(x->get_solution(m, x, algorithm = algorithm)[i], parameter_values)[1])
            end

            @test isapprox(deriv_sol_zyg, deriv_sol_fin, rtol = 1e-5)
            
            @test isapprox(deriv_sol, deriv_sol_fin, rtol = 1e-5)

            for tol in [MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14), MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14, NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        clear_solution_caches!(m, algorithm)

                        SOL = get_solution(m, parameter_values, algorithm = algorithm, tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)

                        @test isapprox([s for s in sol[1:end-1]], [S for S in SOL[1:end-1]], rtol = 1e-8)

                        clear_solution_caches!(m, algorithm)

                        DERIV_SOL = []
                        for i in 1:length(sol)-2
                            push!(DERIV_SOL, ForwardDiff.jacobian(x->get_solution(m, x, algorithm = algorithm, 
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)[i], parameter_values))
                        end

                        @test isapprox(deriv_sol, DERIV_SOL, rtol = 1e-8)

                        clear_solution_caches!(m, algorithm)

                        DERIV_SOL_zyg = []
                        for i in 1:length(sol)-2
                            push!(DERIV_SOL_zyg, Zygote.jacobian(x->get_solution(m, x, algorithm = algorithm, 
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)[i], parameter_values)[1])
                        end

                        @test isapprox(deriv_sol_zyg, DERIV_SOL_zyg, rtol = 1e-8)
                    end
                end
            end
        end
    end


    @testset "get_irf with parameter input" begin
        if algorithm == :first_order
            for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
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

                shock_mat = randn(m.timings.nExo,3)

                shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

                shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

                for initial_state in init_states
                    clear_solution_caches!(m, algorithm)
                                
                    irf_ = get_irf(m, parameter_values, initial_state = initial_state)
                    
                    clear_solution_caches!(m, algorithm)
                             
                    deriv_for = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state)[:,1,1], parameter_values)

                    for i in 1:100
                        local deriv_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.parameters) > 20 ? 3 : 4, 1, max_range = 1e-4), 
                                                                    x -> begin 
                                                                        clear_solution_caches!(m, algorithm)
    
                                                                        get_irf(m, x, initial_state = initial_state)[:,1,1]
                                                                    end, parameter_values)
                        if isfinite(ℒ.norm(deriv_fin[1]))
                            @test isapprox(deriv_for, deriv_fin[1], rtol = 1e-5)
                            break
                        end
                    end

                    for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                        for quadratic_matrix_equation_algorithm in qme_algorithms
                            clear_solution_caches!(m, algorithm)
                                        
                            IRF_ = get_irf(m, 
                                            parameter_values, 
                                            initial_state = initial_state,
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)
                            @test isapprox(irf_, IRF_, rtol = 1e-8)

                            DERIV_for = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state, tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm)[:,1,1], parameter_values)

                            @test isapprox(deriv_for, DERIV_for, rtol = 1e-8)
                        end
                    end
                    for variables in vars
                        for shocks in [:all, :all_excluding_obc, :none, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                            clear_solution_caches!(m, algorithm)
                                        
                            get_irf(m, parameter_values, variables = variables, initial_state = initial_state, shocks = shocks)
                        end
                    end
                end
            end
        end
    end

    
    @testset "get_statistics" begin
        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
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
                                                    autocorrelation = autocorrelation
                                    )
                                end
                            end
                        end
                    end
                end
            end
        end
        
        

        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
            clear_solution_caches!(m, algorithm)

            stats = get_statistics(m, parameter_values, algorithm = algorithm,
                                    # tol = MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14, NSSS_xtol = 1e-14),
                                    non_stochastic_steady_state = :all,
                                    mean = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    standard_deviation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    variance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    covariance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    autocorrelation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]))

            for tol in [MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14),MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14,NSSS_xtol = 1e-14)]
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
                                @test isapprox(stats[:non_stochastic_steady_state], STATS[:non_stochastic_steady_state], rtol = 1e-8)
                                @test isapprox(stats[:mean], STATS[:mean], rtol = 1e-8)
                                @test isapprox(stats[:standard_deviation], STATS[:standard_deviation], rtol = 1e-8)
                                @test isapprox(stats[:variance], STATS[:variance], rtol = 1e-8)
                                @test isapprox(stats[:covariance], STATS[:covariance], rtol = 1e-8)
                                @test isapprox(stats[:autocorrelation], STATS[:autocorrelation], rtol = 1e-8)
                            else
                                @test isapprox(stats[:non_stochastic_steady_state], STATS[:non_stochastic_steady_state], rtol = 1e-8)
                            end
                        end
                    end
                end
            end
        end


        clear_solution_caches!(m, algorithm)

        deriv1 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)

        deriv1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)
                 
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
                # ℒ.norm(deriv1 - deriv1_zyg[1]) / max(ℒ.norm(deriv1), ℒ.norm(deriv1_zyg[1]))
        
                @test isapprox(deriv1_zyg[1], deriv1_fin[1], rtol = 1e-5)
        
                @test isapprox(deriv1, deriv1_fin[1], rtol = 1e-5)
                break
            end
        end
        
                        
        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            clear_solution_caches!(m, algorithm)

            deriv2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            mean = :all_excluding_obc)[:mean], old_params)
            
            if algorithm == :first_order
                deriv2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                mean = :all_excluding_obc)[:mean], old_params)
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
                    if algorithm == :first_order
                        @test isapprox(deriv2_zyg[1], deriv2_fin[1], rtol = 1e-5)
                    end
                    
                    @test isapprox(deriv2, deriv2_fin[1], rtol = 1e-5)
                    break
                end
            end                            

            clear_solution_caches!(m, algorithm)

            deriv3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
            
            if algorithm == :first_order
                deriv3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
            end                    

            for i in 1:100        
                local deriv3_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                        x -> begin 
                                                            clear_solution_caches!(m, algorithm)

                                                            get_statistics(m, x, algorithm = algorithm, standard_deviation = :all_excluding_obc)[:standard_deviation]
                                                        end, old_params)
                              
                if isfinite(ℒ.norm(deriv3_fin[1]))
                    if algorithm == :first_order
                        @test isapprox(deriv3_zyg[1], deriv3_fin[1], rtol = 1e-5)
                    end
                    
                    @test isapprox(deriv3, deriv3_fin[1], rtol = 1e-5)
                    break
                end
            end
            
            clear_solution_caches!(m, algorithm)

            deriv4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            variance = :all_excluding_obc)[:variance], old_params)

            if algorithm == :first_order
                deriv4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                variance = :all_excluding_obc)[:variance], old_params)
            end

            for i in 1:100
                local deriv4_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(length(m.parameters) > 20 ? 3 : 4, 1, max_range = 1e-3),
                                                            x -> begin 
                                                                clear_solution_caches!(m, algorithm)
                                                                
                                                                get_statistics(m, x, algorithm = algorithm, variance = :all_excluding_obc)[:variance]
                                                            end, old_params)
                if isfinite(ℒ.norm(deriv4_fin[1]))
                    if algorithm == :first_order
                        @test isapprox(deriv4_zyg[1], deriv4_fin[1], rtol = 1e-5)
                    end
                    @test isapprox(deriv4, deriv4_fin[1], rtol = 1e-5)
                    break
                end
            end

            clear_solution_caches!(m, algorithm)

            deriv5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                            sylvester_acceptance_tol = 1e-14),
                                                            covariance = :all_excluding_obc)[:covariance], old_params)

            if algorithm == :first_order_
                deriv5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                                sylvester_acceptance_tol = 1e-14),
                                                                covariance = :all_excluding_obc)[:covariance], old_params)
            end         

            for i in 1:100        
                local deriv5_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                                x -> begin 
                                                                    clear_solution_caches!(m, algorithm)
                                                                    
                                                                    get_statistics(m, x, algorithm = algorithm, 
                                                                                    tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                                                    sylvester_acceptance_tol = 1e-14),
                                                                                    covariance = :all_excluding_obc)[:covariance]
                                                                end, old_params)
                if isfinite(ℒ.norm(deriv5_fin[1]))
                    if algorithm == :first_order_
                        @test isapprox(deriv5_zyg[1], deriv5_fin[1], rtol = 1e-4)
                    end

                    # println(ℒ.norm(deriv5 - deriv5_fin[1]) / max(ℒ.norm(deriv5), ℒ.norm(deriv5_fin[1])))                      
                    @test isapprox(deriv5, deriv5_fin[1], rtol = 1e-4)
                    break
                end
            end
        end
        

        

        for tol in [MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14)]
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
                        @test isapprox(deriv1, DERIV1, rtol = 1e-8)
                        
                        DERIV1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm, 
                                                                        non_stochastic_steady_state = :all_excluding_obc)[:non_stochastic_steady_state], old_params)
                        @test isapprox(deriv1_zyg[1], DERIV1_zyg[1], rtol = 1e-8)
                        

                        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                            clear_solution_caches!(m, algorithm)

                            DERIV2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            mean = :all_excluding_obc)[:mean], old_params)
                            @test isapprox(deriv2, DERIV2, rtol = 1e-8)

                            if algorithm == :first_order
                                clear_solution_caches!(m, algorithm)
    
                                DERIV2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                mean = :all_excluding_obc)[:mean], old_params)
                                @test isapprox(deriv2_zyg[1], DERIV2_zyg[1], rtol = 1e-8)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
                            @test isapprox(deriv3, DERIV3, rtol = 1e-8)

                            if algorithm == :first_order
                                clear_solution_caches!(m, algorithm)
    
                                DERIV3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                standard_deviation = :all_excluding_obc)[:standard_deviation], old_params)
                                @test isapprox(deriv3_zyg[1], DERIV3_zyg[1], rtol = 1e-8)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            variance = :all_excluding_obc)[:variance], old_params)
                            @test isapprox(deriv4, DERIV4, rtol = 1e-8)

                            if algorithm == :first_order
                                clear_solution_caches!(m, algorithm)
    
                                DERIV4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                variance = :all_excluding_obc)[:variance], old_params)
                                @test isapprox(deriv4_zyg[1], DERIV4_zyg[1], rtol = 1e-8)
                            end

                            clear_solution_caches!(m, algorithm)

                            DERIV5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            covariance = :all_excluding_obc)[:covariance], old_params)
                            # println(ℒ.norm(deriv5 - DERIV5) / max(ℒ.norm(deriv5), ℒ.norm(DERIV5)))                      
							@test isapprox(deriv5, DERIV5, rtol = 1e-4)

                            if algorithm == :first_order_
                                clear_solution_caches!(m, algorithm)
    
                                DERIV5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                covariance = :all_excluding_obc)[:covariance], old_params)
                                @test isapprox(deriv5_zyg[1], DERIV5_zyg[1], rtol = 1e-4)
                            end
                        end
                    end
                end
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
                        derivatives = true)
        end

        

        for parameters in params
            for derivatives in [true, false]
                clear_solution_caches!(m, algorithm)
            
                moms = get_moments(m,
                                    algorithm = algorithm,
                                    parameters = parameters,
                                    non_stochastic_steady_state = true,
                                    mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                                    derivatives = derivatives)
                            
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
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
                                                    derivatives = derivatives,
                                                    tol = tol,
                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                    sylvester_algorithm = sylvester_algorithm)

                                @test isapprox([v for (k,v) in moms], [v for (k,v) in MOMS], rtol = 1e-8)
                            end
                        end
                    end
                end
            end
        end
    end


    @testset "get_irf" begin
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

        shock_mat = randn(m.timings.nExo,3)

        shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

        shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

        for parameters in params
            for initial_state in init_states
                clear_solution_caches!(m, algorithm)
                
                irf_ = get_irf(m, 
                                algorithm = algorithm, 
                                parameters = parameters, 
                                ignore_obc = true, 
                                initial_state = initial_state)
                
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                clear_solution_caches!(m, algorithm)
                                            
                                IRF_ = get_irf(m, 
                                                algorithm = algorithm, 
                                                ignore_obc = true,
                                                parameters = parameters,
                                                initial_state = initial_state,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm)
                                @test isapprox(irf_, IRF_, rtol = 1e-6)
                            end
                        end
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

                for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
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
        
        

        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for parameters in params 
                clear_solution_caches!(m, algorithm)

                res = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false, parameters = parameters)

                for values in [Dict(axiskeys(stst)[1] .=> collect(stst)), Dict(string.(axiskeys(stst)[1]) .=> collect(stst)), collect(stst)]   
                    clear_solution_caches!(m, algorithm)
                    
                    RES = get_non_stochastic_steady_state_residuals(m, values, tol = tol, verbose = false, parameters = parameters)

                    @test isapprox(res, RES, rtol = 1e-8)
                end
            end

            clear_solution_caches!(m, algorithm)

            res1 = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false)

            clear_solution_caches!(m, algorithm)

            res2 = get_non_stochastic_steady_state_residuals(m, stst[1:3], tol = tol, verbose = false)

            @test isapprox(res1, res2, rtol = 1e-8)

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
                                    @test isapprox(NSSS, nsss, rtol = 1e-8)
                                end
                            end
                        end
                    end
                end
            end
        end

        for parameter_derivatives in param_derivs
            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
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
        @test isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))
    end

end