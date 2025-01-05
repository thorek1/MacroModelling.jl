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

    vars = [:all, :all_excluding_obc, :all_excluding_auxilliary_and_obc, m.var[1], m.var[1:2], Tuple(m.timings.var), reshape(m.timings.var,1,length(m.timings.var)), string(m.var[1]), string.(m.var[1:2]), Tuple(string.(m.timings.var)), reshape(string.(m.timings.var),1,length(m.timings.var))]

    init_state = get_irf(m, algorithm = algorithm, shocks = :none, levels = !(algorithm in [:pruned_second_order, :pruned_third_order]), variables = :all, periods = 1) |> vec

    init_states = [[0.0], init_state, algorithm  == :pruned_second_order ? [zero(init_state), init_state] : algorithm == :pruned_third_order ? [zero(init_state), init_state, zero(init_state)] : init_state .* 1.01]

    
    @testset "filter, smooth, loglikelihood" begin
        sol = get_solution(m)
        
        if length(m.exo) > 3
            n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
            var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]
        else
            var_idxs = [1]
        end

        Random.seed!(123)

        simulation = simulate(m, algorithm = algorithm)

        data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)
        data = data_in_levels .- m.solution.non_stochastic_steady_state[var_idxs]

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        if !(algorithm ∈ [:second_order, :third_order])
            for filter in filters
                for smooth in [true, false]
                    for verbose in [false] # [true, false]
                        for quadratic_matrix_equation_algorithm in qme_algorithms
                            for lyapunov_algorithm in lyapunov_algorithms
                                for sylvester_algorithm in sylvester_algorithms
                                    # Clear solution caches
                                    pop!(m.NSSS_solver_cache)
                                    m.solution.perturbation.qme_solution = zeros(0,0)
                                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                                    m.solution.perturbation.third_order_solution = spzeros(0,0)

                                    estim1 = get_shock_decomposition(m, data, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = false, 
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)

                                    # Clear solution caches
                                    pop!(m.NSSS_solver_cache)
                                    m.solution.perturbation.qme_solution = zeros(0,0)
                                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
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

                                    # Clear solution caches
                                    pop!(m.NSSS_solver_cache)
                                    m.solution.perturbation.qme_solution = zeros(0,0)
                                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                                    m.solution.perturbation.third_order_solution = spzeros(0,0)

                                    estim1 = get_estimated_shocks(m, data, 
                                                                    algorithm = algorithm, 
                                                                    data_in_levels = false, 
                                                                    filter = filter,
                                                                    smooth = smooth,
                                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                                    sylvester_algorithm = sylvester_algorithm,
                                                                    verbose = verbose)

                                    # Clear solution caches
                                    pop!(m.NSSS_solver_cache)
                                    m.solution.perturbation.qme_solution = zeros(0,0)
                                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
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
                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
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

                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                                                    
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        if algorithm == :first_order
            for smooth in [true, false]
                for verbose in [false] # [true, false]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
                            estim1 = get_estimated_variable_standard_deviations(m, data, 
                                                                                data_in_levels = false, 
                                                                                smooth = smooth,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                verbose = verbose)

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for filter in filters
            for presample_periods in [0, 10]
                for initial_covariance in [:diagonal, :theoretical]
                    for verbose in [false] # [true, false]
                        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
                            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                                llh = get_loglikelihood(m, data, parameter_values,
                                                        algorithm = algorithm,
                                                        filter = filter,
                                                        presample_periods = presample_periods,
                                                        initial_covariance = initial_covariance,
                                                        tol = tol,
                                                        verbose = verbose)
                                for quadratic_matrix_equation_algorithm in qme_algorithms
                                    for lyapunov_algorithm in lyapunov_algorithms
                                        for sylvester_algorithm in sylvester_algorithms
                                            
                                            # Clear solution caches
                                            pop!(m.NSSS_solver_cache)
                                            m.solution.perturbation.qme_solution = zeros(0,0)
                                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                                        
                                            LLH = get_loglikelihood(m, data, parameter_values,
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for periods in [0,10]
            for variables in vars
                for levels in [true, false]
                    for verbose in [false] # [true, false]
                        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                for lyapunov_algorithm in lyapunov_algorithms
                                    for sylvester_algorithm in sylvester_algorithms
                                        
                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
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

                                        
                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
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

                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
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

                                        # Clear solution caches
                                        pop!(m.NSSS_solver_cache)
                                        m.solution.perturbation.qme_solution = zeros(0,0)
                                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
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

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end
            
            for parameters in params
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
                get_correlation(m, algorithm = algorithm, parameters = parameters, verbose = false)

                for autocorrelation_periods in [1:5, 1:3]
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
                    get_autocorrelation(m, 
                                        algorithm = algorithm, 
                                        autocorrelation_periods = autocorrelation_periods, 
                                        parameters = parameters, 
                                        verbose = false)
                end

                if algorithm == :first_order
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
                    get_variance_decomposition(m, parameters = parameters, verbose = false)

                    for periods in [[1,Inf,10], [3,Inf], 1:3]
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
                        get_conditional_variance_decomposition(m, periods = periods, parameters = parameters, verbose = false)
                    end
                end
            end

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

            for verbose in [false] # [true, false]
                for tol in [MacroModelling.Tolerances(qme_acceptance_tol = 1e-14),MacroModelling.Tolerances(NSSS_xtol = 1e-20, qme_acceptance_tol = 1e-14)]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            
                            if algorithm == :first_order
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)

                                VAR_DECOMP = get_variance_decomposition(m,
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        verbose = verbose)
                                                                        
                                @test isapprox(var_decomp, VAR_DECOMP, rtol = 1e-8)

                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                                                        
                                COND_VAR_DECOMP = get_conditional_variance_decomposition(m,
                                                                                        tol = tol,
                                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                                        verbose = verbose)

                                @test isapprox(cond_var_decomp, COND_VAR_DECOMP, rtol = 1e-8)

                            end

                            for sylvester_algorithm in sylvester_algorithms
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
                                CORRL = get_correlation(m,
                                                algorithm = algorithm,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm,
                                                verbose = verbose)

                                @test isapprox(corrl, CORRL, rtol = 1e-5)

                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for verbose in [false] # [true, false]
            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.outdated_NSSS = true
                        push!(m.solution.outdated_algorithms, algorithm)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
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

                while length(m.NSSS_solver_cache) > 2
                    pop!(m.NSSS_solver_cache)
                end

                shock_mat = randn(m.timings.nExo,3)

                shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

                shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

                for initial_state in init_states
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
                    irf_ = get_irf(m, parameter_values, initial_state = initial_state)
                    
                    deriv_for = ForwardDiff.jacobian(x->get_irf(m, x, initial_state = initial_state)[:,1,1], parameter_values)

                    deriv_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3), x->get_irf(m, x, initial_state = initial_state)[:,1,1], parameter_values)

                    @test isapprox(deriv_for, deriv_fin[1], rtol = 1e-6)

                    for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                        for quadratic_matrix_equation_algorithm in qme_algorithms
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                                        
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
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                                        
                            get_irf(m, parameter_values, variables = variables, initial_state = initial_state, shocks = shocks)
                        end
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

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv_sol = []
            for i in 1:length(sol)-2
                push!(deriv_sol, ForwardDiff.jacobian(x->get_solution(m, x, algorithm = algorithm)[i], parameter_values))
            end

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv_sol_fin = []
            for i in 1:length(sol)-2
                push!(deriv_sol_fin, FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                    x->get_solution(m, x, algorithm = algorithm)[i], parameter_values)[1])
            end

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv_sol_zyg = []
            for i in 1:length(sol)-2
                push!(deriv_sol_zyg, Zygote.jacobian(x->get_solution(m, x, algorithm = algorithm)[i], parameter_values)[1])
            end

            @test isapprox(deriv_sol_zyg, deriv_sol_fin, rtol = 1e-6)
            
            @test isapprox(deriv_sol, deriv_sol_fin, rtol = 1e-6)

            for tol in [MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14), MacroModelling.Tolerances(lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14, NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.outdated_NSSS = true
                        push!(m.solution.outdated_algorithms, algorithm)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)

                        SOL = get_solution(m, parameter_values, algorithm = algorithm, tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)

                        @test isapprox([s for s in sol[1:end-1]], [S for S in SOL[1:end-1]], rtol = 1e-8)

                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.outdated_NSSS = true
                        push!(m.solution.outdated_algorithms, algorithm)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)

                        DERIV_SOL = []
                        for i in 1:length(sol)-2
                            push!(DERIV_SOL, ForwardDiff.jacobian(x->get_solution(m, x, algorithm = algorithm, 
                                            tol = tol,
                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                            sylvester_algorithm = sylvester_algorithm)[i], parameter_values))
                        end

                        @test isapprox(deriv_sol, DERIV_SOL, rtol = 1e-8)

                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.outdated_NSSS = true
                        push!(m.solution.outdated_algorithms, algorithm)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)

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
        
        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for parameter_values in [old_params, old_params .* exp.(rand(length(old_params))*1e-4)]
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

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
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.outdated_NSSS = true
                            push!(m.solution.outdated_algorithms, algorithm)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
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


        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        m.solution.outdated_NSSS = true
        push!(m.solution.outdated_algorithms, algorithm)
        m.solution.perturbation.qme_solution = zeros(0,0)
        m.solution.perturbation.second_order_solution = spzeros(0,0)
        m.solution.perturbation.third_order_solution = spzeros(0,0)

        deriv1 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)

        deriv1_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)

        deriv1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)
                 
        @test isapprox(deriv1_zyg[1], deriv1_fin[1], rtol = 1e-6)

        @test isapprox(deriv1, deriv1_fin[1], rtol = 1e-6)

        # ℒ.norm(deriv1 - deriv1_fin[1]) / max(ℒ.norm(deriv1), ℒ.norm(deriv1_fin[1]))
        # ℒ.norm(deriv1 - deriv1_zyg[1]) / max(ℒ.norm(deriv1), ℒ.norm(deriv1_zyg[1]))

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end
                        
        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            mean = m.var)[:mean], old_params)
            

            deriv2_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            mean = m.var)[:mean], old_params)
                          
            if algorithm == :first_order
                deriv2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                mean = m.var)[:mean], old_params)
                                                                
                @test isapprox(deriv2_zyg[1], deriv2_fin[1], rtol = 1e-6)
            end
            
            @test isapprox(deriv2, deriv2_fin[1], rtol = 1e-6)

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end
                                          

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = m.var)[:standard_deviation], old_params)
            
            deriv3_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = m.var)[:standard_deviation], old_params)
                        
            if algorithm == :first_order
                deriv3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                standard_deviation = m.var)[:standard_deviation], old_params)
                                                                
                @test isapprox(deriv3_zyg[1], deriv3_fin[1], rtol = 1e-6)
            end
            
            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end
                                            
            @test isapprox(deriv3, deriv3_fin[1], rtol = 1e-6)
            
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            variance = m.var)[:variance], old_params)

            deriv4_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            variance = m.var)[:variance], old_params)
                    
            if algorithm == :first_order
                deriv4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                variance = m.var)[:variance], old_params)
                                                                
                @test isapprox(deriv4_zyg[1], deriv4_fin[1], rtol = 1e-6)
            end

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end
                                                                    
            @test isapprox(deriv4, deriv4_fin[1], rtol = 1e-6)
            
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                            sylvester_acceptance_tol = 1e-14),
                                                            covariance = m.var)[:covariance], old_params)

            deriv5_fin = FiniteDifferences.jacobian(FiniteDifferences.forward_fdm(3,1, max_range = 1e-3),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                            sylvester_acceptance_tol = 1e-14),
                                                            covariance = m.var)[:covariance], old_params)
   
            if algorithm == :first_order_
                deriv5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                tol = MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, 
                                                                sylvester_acceptance_tol = 1e-14),
                                                                covariance = m.var)[:covariance], old_params)
                                                                
                @test isapprox(deriv5_zyg[1], deriv5_fin[1], rtol = 1e-4)
            end
            # println(ℒ.norm(deriv5 - deriv5_fin[1]) / max(ℒ.norm(deriv5), ℒ.norm(deriv5_fin[1])))                      
            @test isapprox(deriv5, deriv5_fin[1], rtol = 1e-4)
        end
        

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for tol in [MacroModelling.Tolerances(NSSS_xtol = 1e-14, lyapunov_acceptance_tol = 1e-14, sylvester_acceptance_tol = 1e-14)]
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for sylvester_algorithm in sylvester_algorithms
                    for lyapunov_algorithm in lyapunov_algorithms
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.outdated_NSSS = true
                        push!(m.solution.outdated_algorithms, algorithm)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)

                        DERIV1 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm, 
                                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)
                        @test isapprox(deriv1, DERIV1, rtol = 1e-8)
                        
                        DERIV1_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                        tol = tol,
                                                                        quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                        lyapunov_algorithm = lyapunov_algorithm,
                                                                        sylvester_algorithm = sylvester_algorithm, 
                                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)
                        @test isapprox(deriv1_zyg[1], DERIV1_zyg[1], rtol = 1e-8)
                        

                        if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.outdated_NSSS = true
                            push!(m.solution.outdated_algorithms, algorithm)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            DERIV2 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            mean = m.var)[:mean], old_params)
                            @test isapprox(deriv2, DERIV2, rtol = 1e-8)

                            if algorithm == :first_order
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.outdated_NSSS = true
                                push!(m.solution.outdated_algorithms, algorithm)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
    
                                DERIV2_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                mean = m.var)[:mean], old_params)
                                @test isapprox(deriv2_zyg[1], DERIV2_zyg[1], rtol = 1e-8)
                            end

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.outdated_NSSS = true
                            push!(m.solution.outdated_algorithms, algorithm)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            DERIV3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            standard_deviation = m.var)[:standard_deviation], old_params)
                            @test isapprox(deriv3, DERIV3, rtol = 1e-8)

                            if algorithm == :first_order
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.outdated_NSSS = true
                                push!(m.solution.outdated_algorithms, algorithm)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
    
                                DERIV3_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                standard_deviation = m.var)[:standard_deviation], old_params)
                                @test isapprox(deriv3_zyg[1], DERIV3_zyg[1], rtol = 1e-8)
                            end

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.outdated_NSSS = true
                            push!(m.solution.outdated_algorithms, algorithm)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            DERIV4 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            variance = m.var)[:variance], old_params)
                            @test isapprox(deriv4, DERIV4, rtol = 1e-8)

                            if algorithm == :first_order
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.outdated_NSSS = true
                                push!(m.solution.outdated_algorithms, algorithm)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
    
                                DERIV4_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                variance = m.var)[:variance], old_params)
                                @test isapprox(deriv4_zyg[1], DERIV4_zyg[1], rtol = 1e-8)
                            end

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.outdated_NSSS = true
                            push!(m.solution.outdated_algorithms, algorithm)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            DERIV5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm,
                                                                            tol = tol,
                                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                            lyapunov_algorithm = lyapunov_algorithm,
                                                                            sylvester_algorithm = sylvester_algorithm, 
                                                                            covariance = m.var)[:covariance], old_params)
                            @test isapprox(deriv5, DERIV5, rtol = 1e-8)

                            if algorithm == :first_order_
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.outdated_NSSS = true
                                push!(m.solution.outdated_algorithms, algorithm)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
    
                                DERIV5_zyg = Zygote.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                                                tol = tol,
                                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                                lyapunov_algorithm = lyapunov_algorithm,
                                                                                sylvester_algorithm = sylvester_algorithm, 
                                                                                covariance = m.var)[:covariance], old_params)
                                @test isapprox(deriv5_zyg[1], DERIV5_zyg[1], rtol = 1e-8)
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for parameters in params
            for derivatives in [true, false]
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.outdated_NSSS = true
                push!(m.solution.outdated_algorithms, algorithm)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
            
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
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.outdated_NSSS = true
                                push!(m.solution.outdated_algorithms, algorithm)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                
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

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        shock_mat = randn(m.timings.nExo,3)

        shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

        shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

        for parameters in params
            for initial_state in init_states
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
                irf_ = get_irf(m, algorithm = algorithm, parameters = parameters, initial_state = initial_state)
                
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                            
                                IRF_ = get_irf(m, 
                                                algorithm = algorithm, 
                                                parameters = parameters,
                                                initial_state = initial_state,
                                                tol = tol,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                lyapunov_algorithm = lyapunov_algorithm,
                                                sylvester_algorithm = sylvester_algorithm)
                                @test isapprox(irf_, IRF_, rtol = 1e-8)
                            end
                        end
                    end
                end
                for variables in vars
                    for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                                    
                        get_irf(m, algorithm = algorithm, parameters = parameters, variables = variables, initial_state = initial_state, shocks = shocks)
                    end
                end
            end
        end
    end

    @testset "get_non_stochastic_steady_state_residuals" begin
        stst = SS(m, derivatives = false)
        
        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for parameters in params 
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)

                res = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false, parameters = parameters)

                for values in [Dict(axiskeys(stst)[1] .=> collect(stst)), Dict(string.(axiskeys(stst)[1]) .=> collect(stst)), collect(stst)]   
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                    
                    RES = get_non_stochastic_steady_state_residuals(m, values, tol = tol, verbose = false, parameters = parameters)

                    @test isapprox(res, RES, rtol = 1e-8)
                end
            end

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            res1 = get_non_stochastic_steady_state_residuals(m, stst, tol = tol, verbose = false)

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            res2 = get_non_stochastic_steady_state_residuals(m, stst[1:3], tol = tol, verbose = false)

            @test isapprox(res1, res2, rtol = 1e-8)

            get_residuals(m, stst)

            check_residuals(m, stst)
        end
    end

    @testset "get_steady_state" begin
        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        get_non_stochastic_steady_state(m)
        
        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        SS(m)

        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        steady_state(m)

        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        get_SS(m)

        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        get_ss(m)

        # Clear solution caches
        pop!(m.NSSS_solver_cache)
        ss(m)

        if !(algorithm == :first_order)
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            get_stochastic_steady_state(m)

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            get_SSS(m)

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            SSS(m)

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            sss(m)
        end 

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for derivatives in [true, false]
            for stochastic in (algorithm == :first_order ? [false] : [true, false])
                for return_variables_only in [true, false]
                    for verbose in [false]
                        for silent in [true, false]
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
            
                            NSSS = get_steady_state(m, 
                                                    verbose = verbose, 
                                                    silent = silent, 
                                                    return_variables_only = return_variables_only, 
                                                    algorithm = algorithm, 
                                                    stochastic = stochastic, 
                                                    derivatives = derivatives)
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                for sylvester_algorithm in sylvester_algorithms
                                    # Clear solution caches
                                    pop!(m.NSSS_solver_cache)
                                    m.solution.perturbation.qme_solution = zeros(0,0)
                                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                    
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
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
    
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
    get_nonnegativity_auxilliary_variables(m) 
    get_dynamic_auxilliary_variables(m) 
    get_shocks(m) 
    get_state_variables(m) 
    get_jump_variables(m)

    GC.gc()

    if algorithm == :first_order
        lvl_irfs  = get_irf(m, old_params, verbose = true, levels = true, variables = :all)
        new_sub_lvl_irfs  = get_irf(m, old_params, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]), levels = true, variables = :all)
        @test isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))
    end




    if plots
        @testset "plot_solution" begin
            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

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
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                    
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
                for plot_attributes in [Dict(), Dict(:plottitle => "Title")]
                    plot_solution(m, states[1], algorithm = algos[end],
                                    plot_attributes = plot_attributes,
                                    plots_per_page = plots_per_page)
                end
            end

            for show_plots in [true, false]
                for save_plots in [true, false]
                    for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                        for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf])
                            plot_solution(m, states[1], algorithm = algos[end],
                                            show_plots = show_plots,
                                            save_plots = save_plots,
                                            save_plots_path = save_plots_path,
                                            save_plots_format = save_plots_format)
                        end
                    end
                end
            end

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

            plotlyjs_backend()

            plot_solution(m, states[1], algorithm = algos[end])

            gr_backend()
        end


        @testset "plot_irf" begin
            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

            plotlyjs_backend()

            plot_IRF(m, algorithm = algorithm)

            gr_backend()

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

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

            shock_mat = randn(m.timings.nExo,3)

            shock_mat2 = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10)

            shock_mat3 = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10)

            for parameters in params
                for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                    for quadratic_matrix_equation_algorithm in qme_algorithms
                        for lyapunov_algorithm in lyapunov_algorithms
                            for sylvester_algorithm in sylvester_algorithms
                                # Clear solution caches
                                pop!(m.NSSS_solver_cache)
                                m.solution.perturbation.qme_solution = zeros(0,0)
                                m.solution.perturbation.second_order_solution = spzeros(0,0)
                                m.solution.perturbation.third_order_solution = spzeros(0,0)
                                            
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
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
                plot_irf(m, algorithm = algorithm, initial_state = initial_state)
            end

            for variables in vars
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
                plot_irf(m, algorithm = algorithm, variables = variables)
            end

            for shocks in [:all, :all_excluding_obc, :none, :simulate, m.timings.exo[1], m.timings.exo[1:2], reshape(m.exo,1,length(m.exo)), Tuple(m.exo), Tuple(string.(m.exo)), string(m.timings.exo[1]), reshape(string.(m.exo),1,length(m.exo)), string.(m.timings.exo[1:2]), shock_mat, shock_mat2, shock_mat3]
                # Clear solution caches
                pop!(m.NSSS_solver_cache)
                m.solution.perturbation.qme_solution = zeros(0,0)
                m.solution.perturbation.second_order_solution = spzeros(0,0)
                m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
                plot_irf(m, algorithm = algorithm, shocks = shocks)
            end

            for plot_attributes in [Dict(), Dict(:plottitle => "Title")]
                for plots_per_page in [4,6]
                    plot_irf(m, algorithm = algorithm,
                                plot_attributes = plot_attributes,
                                plots_per_page = plots_per_page)
                end
            end

            for show_plots in [true, false]
                for save_plots in [true, false]
                    for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                        for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf])
                            plot_irf(m, algorithm = algorithm,
                                        show_plots = show_plots,
                                        save_plots = save_plots,
                                        save_plots_path = save_plots_path,
                                        save_plots_format = save_plots_format)
                        end
                    end
                end
            end
        end


        @testset "plot_conditional_variance_decomposition" begin
            plotlyjs_backend()

            plot_fevd(m)

            gr_backend()

            plot_forecast_error_variance_decomposition(m)

            for periods in [10,40]
                for variables in vars
                    plot_conditional_variance_decomposition(m, periods = periods, variables = variables)
                end
            end

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for lyapunov_algorithm in lyapunov_algorithms
                        # Clear solution caches
                        pop!(m.NSSS_solver_cache)
                        m.solution.perturbation.qme_solution = zeros(0,0)
                        m.solution.perturbation.second_order_solution = spzeros(0,0)
                        m.solution.perturbation.third_order_solution = spzeros(0,0)
                            
                        plot_conditional_variance_decomposition(m, tol = tol,
                                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                                lyapunov_algorithm = lyapunov_algorithm)
                    end
                end
            end
            
            for show_plots in [true, false]
                for save_plots in [true, false]
                    for plots_per_page in [4,6]
                        for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                            for plot_attributes in [Dict(), Dict(:plottitle => "Title")]
                                for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf])
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

            for show_plots in [true, false]
                for save_plots in [true, false]
                    for plots_per_page in [1,4]
                        for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                            for plot_attributes in [Dict(), Dict(:plottitle => "Title")]
                                for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf])
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

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end

            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for lyapunov_algorithm in lyapunov_algorithms
                        for sylvester_algorithm in sylvester_algorithms
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
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
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                
                    plot_conditional_forecast(m, conditions[end],
                                                conditions_in_levels = false,
                                                algorithm = algorithm, 
                                                periods = periods,
                                                levels = levels,
                                                shocks = shocks[end])

                    
                    # Clear solution caches
                    pop!(m.NSSS_solver_cache)
                    m.solution.perturbation.qme_solution = zeros(0,0)
                    m.solution.perturbation.second_order_solution = spzeros(0,0)
                    m.solution.perturbation.third_order_solution = spzeros(0,0)
                
                    plot_conditional_forecast(m, conditions_lvl[end],
                                                algorithm = algorithm, 
                                                periods = periods,
                                                levels = levels,
                                                shocks = shocks[end])

                end
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

            for shcks in shocks
                plot_conditional_forecast(m, conditions[end],
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

            plotlyjs_backend()

            plot_conditional_forecast(m, conditions[end], conditions_in_levels = false)

            gr_backend()

        end
        @testset "plot_model_estimates" begin
            sol = get_solution(m)
            
            if length(m.exo) > 3
                n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
                var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,length(m.obc_violation_equations) > 0 ? 2 : end]]
            else
                var_idxs = [1]
            end

            Random.seed!(123)

            simulation = simulate(m, algorithm = algorithm)

            data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)
            data = data_in_levels .- m.solution.non_stochastic_steady_state[var_idxs]

            while length(m.NSSS_solver_cache) > 2
                pop!(m.NSSS_solver_cache)
            end
            
            plot_shock_decomposition(m, data, 
                                        algorithm = algorithm, 
                                        data_in_levels = false)

            for quadratic_matrix_equation_algorithm in qme_algorithms
                for lyapunov_algorithm in lyapunov_algorithms
                    for sylvester_algorithm in sylvester_algorithms
                        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            plot_model_estimates(m, data, 
                                                    algorithm = algorithm, 
                                                    data_in_levels = false, 
                                                    tol = tol,
                                                    quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                    lyapunov_algorithm = lyapunov_algorithm,
                                                    sylvester_algorithm = sylvester_algorithm)

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
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
                for filter in filters
                    for smooth in [true, false]
                        for presample_periods in [0, 10]
                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)

                            plot_model_estimates(m, data, 
                                                    algorithm = algorithm, 
                                                    data_in_levels = false, 
                                                    filter = filter,
                                                    smooth = smooth,
                                                    presample_periods = presample_periods,
                                                    shock_decomposition = shock_decomposition)

                            # Clear solution caches
                            pop!(m.NSSS_solver_cache)
                            m.solution.perturbation.qme_solution = zeros(0,0)
                            m.solution.perturbation.second_order_solution = spzeros(0,0)
                            m.solution.perturbation.third_order_solution = spzeros(0,0)
                        
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
                for plot_attributes in [Dict(), Dict(:plottitle => "Title")]
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

            for show_plots in [true, false]
                for save_plots in [true, false]
                    for save_plots_path in (save_plots ? [pwd(), "../"] : [pwd()])
                        for save_plots_format in (save_plots ? [:pdf,:png,:ps,:svg] : [:pdf])
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
        end
    end
end