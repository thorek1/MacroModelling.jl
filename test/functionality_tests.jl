function functionality_test(m; algorithm = :first_order, plots = true, verbose = true)
    m_orig = deepcopy(m)
    # figure out dependencies for defined parameters

    # Check different inputs for get_steady_state
    nsss = get_steady_state(m, verbose = true)
    nsss_no_derivs = get_steady_state(m, verbose = true, derivatives = false)
    nsss_select_par_deriv1 = get_steady_state(m, verbose = true, parameter_derivatives = m.parameters[1])
    nsss_select_par_deriv2 = get_steady_state(m, verbose = true, parameter_derivatives = m.parameters[1:2])
    nsss_select_par_deriv3 = get_steady_state(m, verbose = true, parameter_derivatives = Tuple(m.parameters[1:3]))
    nsss_select_par_deriv4 = get_steady_state(m, verbose = true, parameter_derivatives = reshape(m.parameters[1:3],3,1))

    old_par_vals = copy(m.parameter_values)
    new_nsss1 = get_steady_state(m, verbose = true, parameters = m.parameter_values * 1.0001)
    new_nsss2 = get_steady_state(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
    new_nsss3 = get_steady_state(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
    new_nsss4 = get_steady_state(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
    old_nsss = get_steady_state(m, verbose = true, parameters = old_par_vals)
    nsss = get_non_stochastic_steady_state(m)
    nsss = get_SS(m)

    if algorithm == :first_order
        sols_nv = get_solution(m)
        sols = get_solution(m, verbose = true)
        new_sols1 = get_solution(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_sols2 = get_solution(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_sols3 = get_solution(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_sols4 = get_solution(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        old_sols = get_solution(m, verbose = true, parameters = old_par_vals)

        auto_corr_nv = get_autocorrelation(m)
        auto_corrr = get_autocorrelation(m, verbose = true)
        new_auto_corr = get_autocorrelation(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_auto_corr1 = get_autocorrelation(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_auto_corr2 = get_autocorrelation(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_auto_corr3 = get_autocorrelation(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_auto_corr3 = get_autocorr(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_auto_corr3 = autocorr(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        old_auto_corr = get_autocorrelation(m, verbose = true, parameters = old_par_vals)

        corr_nv = get_correlation(m)
        corrr = get_correlation(m, verbose = true)
        new_corr = get_correlation(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_corr1 = get_correlation(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_corr2 = get_correlation(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_corr3 = get_correlation(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_corr3 = get_corr(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_corr3 = corr(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        old_corr = get_correlation(m, verbose = true, parameters = old_par_vals)

        var_decomp_nv = get_variance_decomposition(m)
        var_decomp = get_variance_decomposition(m, verbose = true)
        new_var_decomp = get_variance_decomposition(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_var_decomp1 = get_variance_decomposition(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_var_decomp2 = get_variance_decomposition(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_var_decomp3 = get_variance_decomposition(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_var_decomp3 = get_var_decomp(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        old_var_decomp = get_variance_decomposition(m, verbose = true, parameters = old_par_vals)


        # Check different inputs for get_moments
        moms_nv = get_moments(m)
        moms = get_moments(m, verbose = true)
        moms_var = get_moments(m, verbose = true, variance = true)
        moms_covar = get_moments(m, verbose = true, covariance = true)
        moms_no_nsss = get_moments(m, verbose = true, non_stochastic_steady_state = false)
        moms_no_nsss = get_moments(m, verbose = true, standard_deviation = false)
        moms_no_nsss = get_moments(m, verbose = true, standard_deviation = false, variance = true)
        moms_no_derivs = get_moments(m, verbose = true, derivatives = false)
        moms_no_derivs_var = get_moments(m, verbose = true, derivatives = false, variance = true)

        moms_select_par_deriv1 = get_moments(m, verbose = true, parameter_derivatives = m.parameters[1])
        moms_select_par_deriv2 = get_moments(m, verbose = true, parameter_derivatives = m.parameters[1:2])
        moms_select_par_deriv3 = get_moments(m, verbose = true, parameter_derivatives = Tuple(m.parameters[1:3]))
        moms_select_par_deriv4 = get_moments(m, verbose = true, parameter_derivatives = reshape(m.parameters[1:3],3,1))

        new_moms1 = get_moments(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_moms2 = get_moments(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_moms3 = get_moments(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_moments(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        old_moms  = get_moments(m, verbose = true, parameters = old_par_vals)


        new_moms4 = get_standard_deviation(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_variance(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_covariance(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = get_std(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_var(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_cov(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = std(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = var(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = cov(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.000))
    end

    if algorithm ∈ [:second_order, :third_order]
        SSS = get_stochastic_steady_state(m)
    end
    
    # irfs
    irfs_nv = get_irf(m, algorithm = algorithm)
    irfs = get_irf(m, verbose = true, algorithm = algorithm)
    irfs_10 = get_irf(m, verbose = true, algorithm = algorithm, periods = 10)
    irfs_100 = get_irf(m, verbose = true, algorithm = algorithm, periods = 100)
    new_irfs1 = get_irf(m, verbose = true, algorithm = algorithm, parameters = m.parameter_values * 1.0001)
    new_irfs2 = get_irf(m, verbose = true, algorithm = algorithm, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
    new_irfs3 = get_irf(m, verbose = true, algorithm = algorithm, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
    new_irfs4 = get_irf(m, verbose = true, algorithm = algorithm, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
    lvl_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_par_vals, levels = true)
    lvlv_init_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
    lvlv_init_neg_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
    lvlv_init_neg_gen_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_par_vals, levels = true, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)
    init_neg_gen_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = m.exo[1])
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = m.exo)
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = Tuple(m.exo))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = reshape(m.exo,1,length(m.exo)))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = :all)
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = :simulate)
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
    new_sub_lvl_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])), levels = true)
    @test isapprox(collect(new_sub_lvl_irfs(:,1,:)), collect(lvl_irfs(:,6,m.exo[1])),rtol = eps(Float32))



    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = m.timings.var[1])
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = m.timings.var[end-1:end])
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = m.timings.var)
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = Tuple(m.timings.var))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = reshape(m.timings.var,1,length(m.timings.var)))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, variables = :all)
    sims = simulate(m, algorithm = algorithm)

    if plots
        # plots
        plot(m, verbose = true, algorithm = algorithm, show_plots = true)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 10)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 100)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo[1])
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(m.exo))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(m.exo,1,length(m.exo)))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :all)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :simulate)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))

        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[1])
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = :all)

        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, plots_per_page = 6)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png)
        plot(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
    end
end