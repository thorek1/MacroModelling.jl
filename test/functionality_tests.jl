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

        cond_var_decomp_nv = get_conditional_variance_decomposition(m)
        cond_var_decomp = get_conditional_variance_decomposition(m, verbose = true)
        cond_var_decomp1 = get_conditional_variance_decomposition(m, verbose = true, periods = [1])
        cond_var_decomp2 = get_conditional_variance_decomposition(m, verbose = true, periods = [10])
        cond_var_decomp3 = get_conditional_variance_decomposition(m, verbose = true, periods = [1,10])
        cond_var_decomp5 = get_conditional_variance_decomposition(m, verbose = true, periods = [1,Inf])
        cond_var_decomp6 = get_conditional_variance_decomposition(m, verbose = true, periods = [Inf,2])
        new_cond_var_decomp = get_conditional_variance_decomposition(m, verbose = true, parameters = m.parameter_values * 1.0001)
        new_cond_var_decomp1 = get_conditional_variance_decomposition(m, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_cond_var_decomp2 = get_conditional_variance_decomposition(m, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_cond_var_decomp3 = get_conditional_variance_decomposition(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_cond_var_decomp3 = fevd(m, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        old_cond_var_decomp = get_conditional_variance_decomposition(m, verbose = true, parameters = old_par_vals)


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


        irfs_nv = get_irf(m, m.parameter_values)
        irfs = get_irf(m, m.parameter_values, verbose = true)
        irfs_10 = get_irf(m, m.parameter_values, verbose = true, periods = 10)
        irfs_100 = get_irf(m, m.parameter_values, verbose = true, periods = 100)
        new_irfs1 = get_irf(m, m.parameter_values * 1.0001, verbose = true)
        lvl_irfs  = get_irf(m, old_par_vals, verbose = true, levels = true)
        lvlv_init_irfs  = get_irf(m, old_par_vals, verbose = true, levels = true, initial_state = collect(lvl_irfs[:,5,1]))
        lvlv_init_neg_irfs = get_irf(m, old_par_vals, verbose = true, levels = true, initial_state = collect(lvl_irfs[:,5,1]), negative_shock = true)

        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = m.exo[1])
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = m.exo)
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = Tuple(m.exo))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = reshape(m.exo,1,length(m.exo)))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = :all)
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = randn(m.timings.nExo,10))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]))
        new_sub_lvl_irfs  = get_irf(m, old_par_vals, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]), levels = true)
        @test isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))

        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = m.timings.var[1])
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = m.timings.var[end-1:end])
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = m.timings.var)
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = Tuple(m.timings.var))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = :all)

        new_sub_irfs  = get_irf(m, old_par_vals, verbose = true, variables = :all_including_auxilliary)

        # test conditional forecasting
        new_sub_irfs_all  = get_irf(m, verbose = true, variables = :all_including_auxilliary)
        varnames = axiskeys(new_sub_irfs_all,1)
        shocknames = axiskeys(new_sub_irfs_all,3)
        sol = get_solution(m)
        var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[1:2]

        conditions = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,1),2)
        conditions[var_idxs[1],1] = .01
        conditions[var_idxs[2],2] = .02
        
        cond_fcst = get_conditional_forecast(m, conditions)

        if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[1:2]] .!= 0, dims = 1)) .> 0)
            shocks = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,3),1)
            shocks[1,1] = .1
            cond_fcst = get_conditional_forecast(m, conditions, shocks = shocks)
        end

        conditions = spzeros(size(new_sub_irfs_all,1),2)
        conditions[var_idxs[1],1] = .01
        conditions[var_idxs[2],2] = .02
        
        cond_fcst = get_conditional_forecast(m, conditions)

        if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[1:2]] .!= 0, dims = 1)) .> 0)
            shocks = spzeros(size(new_sub_irfs_all,3),1)
            shocks[1,1] = .1
            cond_fcst = get_conditional_forecast(m, conditions, shocks = shocks)
        end

        conditions = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs[1:2]], Periods = 1:2)
        conditions[var_idxs[1],1] = .01
        conditions[var_idxs[2],2] = .02

        cond_fcst = get_conditional_forecast(m, conditions)

        if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[1:2]] .!= 0, dims = 1)) .> 0)
            shocks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = [shocknames[1]], Periods = [1])
            shocks[1,1] = .1
            cond_fcst = get_conditional_forecast(m, conditions, shocks = shocks)
        end

        cond_fcst = get_conditional_forecast(m, conditions)
        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, verbose = true)
        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), verbose = true)
        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, parameters = old_par_vals, variables = :all, verbose = true)
        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), variables = varnames[1], verbose = true)
        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, parameters = old_par_vals, variables = varnames[1], verbose = true)


        if plots
            plot_conditional_forecast(m, conditions, save_plots = false, show_plots = true)
            plot_conditional_forecast(m, conditions, save_plots = true, show_plots = false, periods = 10, verbose = true)
            plot_conditional_forecast(m, conditions, save_plots = true, show_plots = false, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), verbose = true)
            plot_conditional_forecast(m, conditions, save_plots = true, show_plots = false, periods = 10, parameters = old_par_vals, variables = :all, verbose = true)
            plot_conditional_forecast(m, conditions, save_plots = true, show_plots = false, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), variables = varnames[1], verbose = true)
            plot_conditional_forecast(m, conditions, save_plots = true, show_plots = false, periods = 10, parameters = old_par_vals, variables = varnames[1], verbose = true)
        end

        NSSS = get_SS(m,derivatives = false)
        full_SS = sort(union(m.var,m.aux,m.exo_present))
        full_SS[indexin(m.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  m.aux)
        reference_steady_state = [s ∈ m.exo_present ? 0 : NSSS(s) for s in full_SS]

        conditions_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs[1:2]], Periods = 1:2)
        conditions_lvl[var_idxs[1],1] = .01 + reference_steady_state[var_idxs[1]]
        conditions_lvl[var_idxs[2],2] = .02 + reference_steady_state[var_idxs[2]]

        cond_fcst = get_conditional_forecast(m, conditions_lvl, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), variables = varnames[1], conditions_in_levels = true, verbose = true)

        cond_fcst = get_conditional_forecast(m, conditions, periods = 10, parameters = old_par_vals, variables = varnames[1], levels = true, verbose = true)

        # Test filtering and smoothing
        sol = get_solution(m)

        if length(m.exo) > 1
            var_idxs = findall(vec(sum(sol[end-length(m.exo)+1:end,:] .!= 0,dims = 1)) .> 0)[1:2]
        else
            var_idxs = [1]
        end
        
        simulation = simulate(m)

        data = simulation(m.var[var_idxs],:,:simulate)
        data_in_levels = data .+ m.solution.non_stochastic_steady_state[var_idxs]

        estim_vars1 = get_estimated_variables(m, data, data_in_levels = false, verbose = true)
        estim_vars2 = get_estimated_variables(m, data_in_levels, verbose = true)
        @test isapprox(estim_vars1,estim_vars2)

        estim_vars1 = get_estimated_variables(m, data, data_in_levels = false, smooth = false, verbose = true)
        estim_vars2 = get_estimated_variables(m, data_in_levels, smooth = false, verbose = true)
        @test isapprox(estim_vars1,estim_vars2)

        estim_vars1 = get_estimated_variables(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        estim_vars2 = get_estimated_variables(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = old_par_vals)


        estim_shocks1 = get_estimated_shocks(m, data, data_in_levels = false, verbose = true)
        estim_shocks2 = get_estimated_shocks(m, data_in_levels, verbose = true)
        @test isapprox(estim_shocks1,estim_shocks2)

        estim_shocks1 = get_estimated_shocks(m, data, data_in_levels = false, smooth = false, verbose = true)
        estim_shocks2 = get_estimated_shocks(m, data_in_levels, smooth = false, verbose = true)
        @test isapprox(estim_shocks1,estim_shocks2)

        estim_shocks1 = get_estimated_shocks(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        estim_shocks2 = get_estimated_shocks(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = old_par_vals)


        estim_stds1 = get_estimated_variable_standard_deviations(m, data, data_in_levels = false, verbose = true)
        estim_stds2 = get_estimated_variable_standard_deviations(m, data_in_levels, verbose = true)
        @test isapprox(estim_stds1,estim_stds2)

        estim_stds1 = get_estimated_variable_standard_deviations(m, data, data_in_levels = false, smooth = false, verbose = true)
        estim_stds2 = get_estimated_variable_standard_deviations(m, data_in_levels, smooth = false, verbose = true)
        @test isapprox(estim_stds1,estim_stds2)

        estim_stds1 = get_estimated_variable_standard_deviations(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        estim_stds2 = get_estimated_variable_standard_deviations(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = old_par_vals)
    

        estim_decomp1 = get_shock_decomposition(m, data, data_in_levels = false, verbose = true)
        estim_decomp2 = get_shock_decomposition(m, data_in_levels, verbose = true)
        @test isapprox(estim_decomp1,estim_decomp2)

        estim_decomp1 = get_shock_decomposition(m, data, data_in_levels = false, smooth = false, verbose = true)
        estim_decomp2 = get_shock_decomposition(m, data_in_levels, smooth = false, verbose = true)
        @test isapprox(estim_decomp1,estim_decomp2)

        estim_decomp1 = get_shock_decomposition(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        estim_decomp2 = get_shock_decomposition(m, data, data_in_levels = false, smooth = false, verbose = true, parameters = old_par_vals)
    end

    if algorithm ∈ [:second_order, :third_order]
        SSS = get_stochastic_steady_state(m, algorithm = algorithm)
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
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = randn(m.timings.nExo,10))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
    new_sub_irfs  = get_irf(m, verbose = true, algorithm = algorithm, generalised_irf = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
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
        plot_irf(m, algorithm = algorithm, show_plots = true)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = true)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 10)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 100)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_par_vals, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo[1])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(m.exo))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(m.exo,1,length(m.exo)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :all)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = randn(m.timings.nExo,10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :simulate)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[1])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = :all)

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, plots_per_page = 6)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)

        if algorithm == :first_order
            plot_fevd(m)
            plot_fevd(m, verbose = true)
            plot_fevd(m, verbose = true, show_plots = true)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, periods = 10)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, periods = 100)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[1])
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = :all)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)


            # Test filtering and smoothing
            sol = get_solution(m)

            if length(m.exo) > 1
                var_idxs = findall(vec(sum(sol[end-length(m.exo)+1:end,:] .!= 0,dims = 1)) .> 0)[1:2]
            else
                var_idxs = [1]
            end
            
            simulation = simulate(m)

            data = simulation(m.var[var_idxs],:,:simulate)
            data_in_levels = data .+ m.solution.non_stochastic_steady_state[var_idxs]

            plot_model_estimates(m, data, data_in_levels = false)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true)
            plot_model_estimates(m, data, data_in_levels = false, shock_decomposition = true, verbose = true)
            plot_model_estimates(m, data, data_in_levels = false, smooth = false, shock_decomposition = true, verbose = true)
            plot_model_estimates(m, data_in_levels, verbose = true)
            plot_model_estimates(m, data_in_levels, smooth = false, verbose = true)
            plot_model_estimates(m, data_in_levels, smooth = false, shock_decomposition = true, verbose = true)

            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[1])
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = :all)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = :all)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = m.timings.exo)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = m.timings.exo[1])
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = Tuple(m.timings.exo))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)

            plot_shock_decomposition(m, data, data_in_levels = false)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true)
            plot_shock_decomposition(m, data, data_in_levels = false, shock_decomposition = true, verbose = true)
            plot_shock_decomposition(m, data, data_in_levels = false, smooth = false, shock_decomposition = true, verbose = true)
            plot_shock_decomposition(m, data_in_levels, verbose = true)
            plot_shock_decomposition(m, data_in_levels, smooth = false, verbose = true)
            plot_shock_decomposition(m, data_in_levels, smooth = false, shock_decomposition = true, verbose = true)

            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[1])
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = :all)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = :all)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = m.timings.exo)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = m.timings.exo[1])
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = Tuple(m.timings.exo))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
        end


        states = m.timings.past_not_future_and_mixed
        plot_solution(m, states[1])
        plot_solution(m, states[end], verbose = true)
        plot_solution(m, states[end], algorithm = algorithm, verbose = true)
        plot_solution(m, states[end], algorithm = [:first_order, algorithm], verbose = true)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, σ = 10, verbose = true, show_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, σ = .1, verbose = true, show_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[1])
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = m.timings.var)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = :all)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
    end
end