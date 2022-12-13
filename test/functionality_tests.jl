
function functionality_test(m; second_order = true, third_order = true, plots = true)
    m_orig = deepcopy(m)
    # figure out dependencies for defined parameters
    # check why i cant do getSS stochastic immediately. why do i need to calc second order first with other call

    # Check different inputs for get_steady_state
    nsss = get_steady_state(m)
    nsss_no_derivs = get_steady_state(m, derivatives = false)
    params = setdiff(m.par, m.parameters_as_function_of_parameters)
    nsss_select_par_deriv1 = get_steady_state(m, parameter_derivatives = params[1])
    nsss_select_par_deriv2 = get_steady_state(m, parameter_derivatives = params[1:2])
    nsss_select_par_deriv3 = get_steady_state(m, parameter_derivatives = Tuple(params[1:3]))
    nsss_select_par_deriv4 = get_steady_state(m, parameter_derivatives = reshape(params[1:3],3,1))


    old_par_vals = copy(m.parameter_values)
    new_nsss1 = get_steady_state(m, parameters = m.parameter_values * 1.01)
    new_nsss2 = get_steady_state(m, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
    new_nsss3 = get_steady_state(m, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
    new_nsss4 = get_steady_state(m, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
    old_nsss = get_steady_state(m, parameters = old_par_vals)


    sols = get_solution(m)
    new_sols1 = get_solution(m, parameters = m.parameter_values * 1.01)
    new_sols2 = get_solution(m, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
    new_sols3 = get_solution(m, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
    new_sols4 = get_solution(m, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
    old_sols = get_solution(m, parameters = old_par_vals)


    # Check different inputs for get_moments
    moms = get_moments(m)
    moms_var = get_moments(m, variance = true)
    moms_covar = get_moments(m, covariance = true)
    moms_no_nsss = get_moments(m, non_stochastic_steady_state = false)
    moms_no_nsss = get_moments(m, standard_deviation = false)
    moms_no_derivs = get_moments(m, derivatives = false)

    params = setdiff(m.par, m.parameters_as_function_of_parameters)
    moms_select_par_deriv1 = get_moments(m, parameter_derivatives = params[1])
    moms_select_par_deriv2 = get_moments(m, parameter_derivatives = params[1:2])
    moms_select_par_deriv3 = get_moments(m, parameter_derivatives = Tuple(params[1:3]))
    moms_select_par_deriv4 = get_moments(m, parameter_derivatives = reshape(params[1:3],3,1))



    new_moms1 = get_moments(m, parameters = m.parameter_values * 1.01)
    new_moms2 = get_moments(m, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
    new_moms3 = get_moments(m, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
    new_moms4 = get_moments(m, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
    old_moms  = get_moments(m, parameters = old_par_vals)


    # irfs
    irfs = get_irf(m)
    irfs_10 = get_irf(m, periods = 10)
    irfs_100 = get_irf(m, periods = 100)
    new_irfs1 = get_irf(m, parameters = m.parameter_values * 1.01)
    new_irfs2 = get_irf(m, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
    new_irfs3 = get_irf(m, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
    new_irfs4 = get_irf(m, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
    lvl_irfs  = get_irf(m, parameters = old_par_vals, levels = true)
    lvlv_init_irfs  = get_irf(m, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
    lvlv_init_neg_irfs  = get_irf(m, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
    lvlv_init_neg_gen_irfs  = get_irf(m, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

    new_sub_irfs  = get_irf(m, shocks = m.exo[1])
    new_sub_irfs  = get_irf(m, shocks = m.exo)
    new_sub_irfs  = get_irf(m, shocks = Tuple(m.exo))
    new_sub_irfs  = get_irf(m, shocks = reshape(m.exo,1,length(m.exo)))
    new_sub_irfs  = get_irf(m, shocks = :all)
    new_sub_irfs  = get_irf(m, shocks = :simulate)
    new_sub_irfs  = get_irf(m, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
    new_sub_lvl_irfs  = get_irf(m, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])), levels = true)
    @test isapprox(collect(new_sub_lvl_irfs(:,1,:)), collect(lvl_irfs(:,6,m.exo[1])),rtol = eps(Float32))



    new_sub_irfs  = get_irf(m, variables = m.timings.var[1])
    new_sub_irfs  = get_irf(m, variables = m.timings.var[end-1:end])
    new_sub_irfs  = get_irf(m, variables = m.timings.var)
    new_sub_irfs  = get_irf(m, variables = Tuple(m.timings.var))
    new_sub_irfs  = get_irf(m, variables = reshape(m.timings.var,1,length(m.timings.var)))
    new_sub_irfs  = get_irf(m, variables = :all)
    sims = simulate(m)

    if plots
        # plots
        plot(m)
        plot(m, periods = 10)
        plot(m, periods = 100)
        plot(m, parameters = m.parameter_values * 1.01)
        plot(m, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
        plot(m, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
        plot(m, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
        plot(m, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
        plot(m, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        plot(m, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        plot(m, shocks = m.exo[1])
        plot(m, shocks = m.exo)
        plot(m, shocks = Tuple(m.exo))
        plot(m, shocks = reshape(m.exo,1,length(m.exo)))
        plot(m, shocks = :all)
        plot(m, shocks = :simulate)
        plot(m, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))

        plot(m, variables = m.timings.var[1])
        plot(m, variables = m.timings.var[end-1:end])
        plot(m, variables = m.timings.var)
        plot(m, variables = Tuple(m.timings.var))
        plot(m, variables = reshape(m.timings.var,1,length(m.timings.var)))
        plot(m, variables = :all)

        plot(m, plots_per_page = 4)
        plot(m,show_plots = false)
        plot(m,show_plots = false, save_plots = true)
        plot(m,show_plots = false, save_plots = true, save_plots_format = :png)
        plot(m,show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
    end

    if second_order
        # second order
        sss = get_SS(m,stochastic = true)
        new_sss1 = get_SS(m,stochastic = true, parameters = m.parameter_values * 1.01)
        new_sss2 = get_SS(m,stochastic = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
        new_sss3 = get_SS(m,stochastic = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
        new_sss4 = get_SS(m,stochastic = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
        m = deepcopy(m_orig)
        old_sss  = get_SS(m,stochastic = true, parameters = old_par_vals)


        # irfs
        second_irfs = get_irf(m, algorithm = :second_order)
        second_irfs_10 = get_irf(m, algorithm = :second_order, periods = 10)
        second_irfs_100 = get_irf(m, algorithm = :second_order, periods = 100)
        second_new_irfs1 = get_irf(m, algorithm = :second_order, parameters = m.parameter_values * 1.01)
        second_new_irfs2 = get_irf(m, algorithm = :second_order, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
        second_new_irfs3 = get_irf(m, algorithm = :second_order, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
        second_new_irfs4 = get_irf(m, algorithm = :second_order, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
        second_lvl_irfs  = get_irf(m, algorithm = :second_order, parameters = old_par_vals, levels = true)
        second_lvlv_init_irfs  = get_irf(m, algorithm = :second_order, parameters = old_par_vals, levels = true, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])))
        m = deepcopy(m_orig)
        second_lvlv_init_neg_irfs  = get_irf(m, algorithm = :second_order, parameters = old_par_vals, levels = true, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        second_lvlv_init_neg_gen_irfs  = get_irf(m, algorithm = :second_order, parameters = old_par_vals, levels = true, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = m.exo[1])
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = m.exo)
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = Tuple(m.exo))
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = reshape(m.exo,1,length(m.exo)))
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = :all)
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = :simulate)
        m = deepcopy(m_orig)
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, shocks = :none, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])))
        second_new_sub_lvl_irfs  = get_irf(m, algorithm = :second_order, shocks = :none, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])), levels = true)
        @test isapprox(collect(second_new_sub_lvl_irfs(:,1,:)), collect(second_lvl_irfs(:,6,m.exo[1])),rtol = eps(Float32))



        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = m.timings.var[1])
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = m.timings.var[end-1:end])
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = m.timings.var)
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = Tuple(m.timings.var))
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = reshape(m.timings.var,1,length(m.timings.var)))
        second_new_sub_irfs  = get_irf(m, algorithm = :second_order, variables = :all)
        second_sims = simulate(m, algorithm = :second_order)

        if plots
            # plots
            plot(m, algorithm = :second_order)
            plot(m, algorithm = :second_order, periods = 10)
            plot(m, algorithm = :second_order, periods = 100)
            plot(m, algorithm = :second_order, parameters = m.parameter_values * 1.01)
            plot(m, algorithm = :second_order, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
            plot(m, algorithm = :second_order, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
            plot(m, algorithm = :second_order, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
            m = deepcopy(m_orig)
            plot(m, algorithm = :second_order, parameters = old_par_vals, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])))
            plot(m, algorithm = :second_order, parameters = old_par_vals, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])), negative_shock = true)
            plot(m, algorithm = :second_order, parameters = old_par_vals, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

            plot(m, algorithm = :second_order, shocks = m.exo[1])
            plot(m, algorithm = :second_order, shocks = m.exo)
            plot(m, algorithm = :second_order, shocks = Tuple(m.exo))
            plot(m, algorithm = :second_order, shocks = reshape(m.exo,1,length(m.exo)))
            plot(m, algorithm = :second_order, shocks = :all)
            plot(m, algorithm = :second_order, shocks = :simulate)
            plot(m, algorithm = :second_order, shocks = :none, initial_state = collect(second_lvl_irfs(:,5,m.exo[1])))

            plot(m, algorithm = :second_order, variables = m.timings.var[1])
            plot(m, algorithm = :second_order, variables = m.timings.var[end-1:end])
            plot(m, algorithm = :second_order, variables = m.timings.var)
            plot(m, algorithm = :second_order, variables = Tuple(m.timings.var))
            m = deepcopy(m_orig)
            plot(m, algorithm = :second_order, variables = reshape(m.timings.var,1,length(m.timings.var)))
            plot(m, algorithm = :second_order, variables = :all)

            plot(m, algorithm = :second_order, plots_per_page = 4)
            plot(m, algorithm = :second_order,show_plots = false)
            plot(m, algorithm = :second_order,show_plots = false, save_plots = true)
            plot(m, algorithm = :second_order,show_plots = false, save_plots = true, save_plots_format = :png)
            plot(m, algorithm = :second_order,show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
        end
    end

    if third_order
        # # third order

        # irfs
        third_irfs = get_irf(m, algorithm = :third_order)
        third_irfs_10 = get_irf(m, algorithm = :third_order, periods = 10)
        third_irfs_100 = get_irf(m, algorithm = :third_order, periods = 100)
        third_new_irfs1 = get_irf(m, algorithm = :third_order, parameters = m.parameter_values * 1.01)
        third_new_irfs2 = get_irf(m, algorithm = :third_order, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
        third_new_irfs3 = get_irf(m, algorithm = :third_order, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
        third_new_irfs4 = get_irf(m, algorithm = :third_order, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
        third_lvl_irfs  = get_irf(m, algorithm = :third_order, parameters = old_par_vals, levels = true)
        third_lvlv_init_irfs  = get_irf(m, algorithm = :third_order, parameters = old_par_vals, levels = true, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])))
        m = deepcopy(m_orig)
        third_lvlv_init_neg_irfs  = get_irf(m, algorithm = :third_order, parameters = old_par_vals, levels = true, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        third_lvlv_init_neg_gen_irfs  = get_irf(m, algorithm = :third_order, parameters = old_par_vals, levels = true, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = m.exo[1])
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = m.exo)
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = Tuple(m.exo))
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = reshape(m.exo,1,length(m.exo)))
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = :all)
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = :simulate)
        m = deepcopy(m_orig)
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, shocks = :none, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])))
        third_new_sub_lvl_irfs  = get_irf(m, algorithm = :third_order, shocks = :none, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])), levels = true)
        @test isapprox(collect(third_new_sub_lvl_irfs(:,1,:)), collect(third_lvl_irfs(:,6,m.exo[1])),rtol = eps(Float32))



        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = m.timings.var[1])
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = m.timings.var[end-1:end])
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = m.timings.var)
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = Tuple(m.timings.var))
        m = deepcopy(m_orig)
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = reshape(m.timings.var,1,length(m.timings.var)))
        third_new_sub_irfs  = get_irf(m, algorithm = :third_order, variables = :all)
        third_sims = simulate(m, algorithm = :third_order)

        if plots
            # plots
            plot(m, algorithm = :third_order)
            plot(m, algorithm = :third_order, periods = 10)
            plot(m, algorithm = :third_order, periods = 100)
            plot(m, algorithm = :third_order, parameters = m.parameter_values * 1.01)
            plot(m, algorithm = :third_order, parameters = (m.parameters[1] => m.parameter_values[1] * 1.01))
            plot(m, algorithm = :third_order, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.01))
            plot(m, algorithm = :third_order, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.01))
            m = deepcopy(m_orig)
            plot(m, algorithm = :third_order, parameters = old_par_vals, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])))
            plot(m, algorithm = :third_order, parameters = old_par_vals, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])), negative_shock = true)
            plot(m, algorithm = :third_order, parameters = old_par_vals, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

            plot(m, algorithm = :third_order, shocks = m.exo[1])
            plot(m, algorithm = :third_order, shocks = m.exo)
            plot(m, algorithm = :third_order, shocks = Tuple(m.exo))
            plot(m, algorithm = :third_order, shocks = reshape(m.exo,1,length(m.exo)))
            plot(m, algorithm = :third_order, shocks = :all)
            plot(m, algorithm = :third_order, shocks = :simulate)
            plot(m, algorithm = :third_order, shocks = :none, initial_state = collect(third_lvl_irfs(:,5,m.exo[1])))

            plot(m, algorithm = :third_order, variables = m.timings.var[1])
            plot(m, algorithm = :third_order, variables = m.timings.var[end-1:end])
            plot(m, algorithm = :third_order, variables = m.timings.var)
            m = deepcopy(m_orig)
            plot(m, algorithm = :third_order, variables = Tuple(m.timings.var))
            plot(m, algorithm = :third_order, variables = reshape(m.timings.var,1,length(m.timings.var)))
            plot(m, algorithm = :third_order, variables = :all)

            plot(m, algorithm = :third_order, plots_per_page = 4)
            plot(m, algorithm = :third_order,show_plots = false)
            plot(m, algorithm = :third_order,show_plots = false, save_plots = true)
            plot(m, algorithm = :third_order,show_plots = false, save_plots = true, save_plots_format = :png)
            plot(m, algorithm = :third_order,show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
        end
    end
end