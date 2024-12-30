function functionality_test(m; algorithm = :first_order, plots = true)
    old_params = copy(m.parameter_values)
    
    # options to itereate over
    filters = [:inversion, :kalman]

    sylvester_alogorithms = (algorithm == :first_order ? [:doubling] : [[:doubling, :bicgstab], [:bartels_stewart, :doubling], :bicgstab, :dqgmres, (:gmres, :gmres)])

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
                                for sylvester_algorithm in sylvester_alogorithms
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
                                    @test isapprox(estim1,estim2)

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
                                    @test isapprox(estim1,estim2)

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
                                        @test isapprox(estim1,estim2)
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
                                        for sylvester_algorithm in sylvester_alogorithms
                                            
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
                                            @test isapprox(llh,LLH)
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
                                    for sylvester_algorithm in sylvester_alogorithms
                                        
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
            for variables in vars
                for initial_state in init_states
                    cond_fcst = get_conditional_forecast(m, cndtns,
                                                        conditions_in_levels = false,
                                                        initial_state = initial_state,
                                                        algorithm = algorithm, 
                                                        variables = variables,
                                                        verbose = false)
                end
            end
            
            for parameters in params
                for shcks in shocks
                            cond_fcst = get_conditional_forecast(m, cndtns,
                                                                parameters = parameters,
                                                                conditions_in_levels = false,
                                                                algorithm = algorithm, 
                                                                shocks = shcks,
                                                                verbose = false)
                end
            end
        end
    end

    @testset "(auto) correlation, (conditional) variance decomposition" begin
        if algorithm in [:first_order, :pruned_second_order, :pruned_third_order]
            corrl = get_correlation(m, algorithm = algorithm)

            autocorr = get_autocorrelation(m, algorithm = algorithm)

            if algorithm == :first_order
                var_decomp = get_variance_decomposition(m)

                cond_var_decomp = get_conditional_variance_decomposition(m)
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
                                                                        
                                @test isapprox(var_decomp, VAR_DECOMP) #, rtol = eps(Float32))

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

                                @test isapprox(cond_var_decomp, COND_VAR_DECOMP) #, rtol = eps(Float32))

                            end

                            for sylvester_algorithm in sylvester_alogorithms
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

                                @test isapprox(autocorr, AUTOCORR)#, rtol = eps(Float32))
                            end
                        end
                    end
                end
            end
        end
    end



    @testset "solution" begin
        sol = get_solution(m, algorithm = algorithm)

        for parameters in params          
            get_solution(m, algorithm = algorithm, parameters = parameters, verbose = false)
        end

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for verbose in [false] # [true, false]
            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_alogorithms
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

    # get_irf
    # get_solution
    # get_statistics

    # plot_model_estimates
    # plot_irf
    # plot_conditional_variance_decomposition
    # plot_solution
    # plot_conditional_forecast


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
                                    non_stochastic_steady_state = :all,
                                    mean = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    standard_deviation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    variance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    covariance = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]),
                                    autocorrelation = (algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order] ? :all : Symbol[]))

            for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
                for quadratic_matrix_equation_algorithm in qme_algorithms
                    for sylvester_algorithm in sylvester_alogorithms
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

                            @test isapprox([v for (k,v) in stats], [v for (k,v) in STATS], rtol = 1e-10)
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

        deriv1_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),
                                                x->get_statistics(m, x, algorithm = algorithm, 
                                                        non_stochastic_steady_state = m.var)[:non_stochastic_steady_state], old_params)
                                                        
        @test isapprox(deriv1, deriv1_fin[1], rtol = 1e-6)
                        
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

            deriv2_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            mean = m.var)[:mean], old_params)
                                                                    
            @test isapprox(deriv2, deriv2_fin[1], rtol = 1e-6)

            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv3 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = m.var)[:standard_deviation], old_params)
            
            deriv3_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            standard_deviation = m.var)[:standard_deviation], old_params)
                                                                    
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

            deriv4_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            variance = m.var)[:variance], old_params)
                                                                    
            @test isapprox(deriv4, deriv4_fin[1], rtol = 1e-6)
            
            # Clear solution caches
            pop!(m.NSSS_solver_cache)
            m.solution.outdated_NSSS = true
            push!(m.solution.outdated_algorithms, algorithm)
            m.solution.perturbation.qme_solution = zeros(0,0)
            m.solution.perturbation.second_order_solution = spzeros(0,0)
            m.solution.perturbation.third_order_solution = spzeros(0,0)

            deriv5 = ForwardDiff.jacobian(x->get_statistics(m, x, algorithm = algorithm, 
                                                            covariance = m.var)[:covariance], old_params)

            deriv5_fin = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(4,1),
                                                            x->get_statistics(m, x, algorithm = algorithm, 
                                                            covariance = m.var)[:covariance], old_params)
                                                                    
            @test isapprox(deriv5, deriv5_fin[1], rtol = 1e-6)
        end

        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for quadratic_matrix_equation_algorithm in qme_algorithms
                for sylvester_algorithm in sylvester_alogorithms
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
                        @test isapprox(deriv1, DERIV1)
                        
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
                            @test isapprox(deriv2, DERIV2)

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
                            @test isapprox(deriv3, DERIV3)

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
                            @test isapprox(deriv4, DERIV4)

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
                            @test isapprox(deriv5, DERIV5, rtol = 1e-10)
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

        for parameter_derivatives in param_derivs
            for variables in vars
                get_moments(m,
                            algorithm = algorithm,
                            variables = variables,
                            non_stochastic_steady_state = true,
                            mean = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            standard_deviation = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            variance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            covariance = algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order],
                            parameter_derivatives = parameter_derivatives,
                            derivatives = true)
            end
        end

        while length(m.NSSS_solver_cache) > 2
            pop!(m.NSSS_solver_cache)
        end

        for derivatives in [true, false]
            for parameters in params
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
                        for sylvester_algorithm in sylvester_alogorithms
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

                                @test isapprox([v for (k,v) in moms], [v for (k,v) in MOMS], rtol = 1e-10)
                            end
                        end
                    end
                end
            end
        end
    end


    @testset "get_irf" begin
        for ignore_obc in [true,false]
            for levels in [true,false]
                for generalised_irf in [true,false]
                    for negative_shock in [true,false]
                        for shock_size in [.1,1]
                            for periods in [1,10]
                                get_irf(m, 
                                        algorithm = algorithm, 
                                        ignore_obc = ignore_obc,
                                        levels = levels,
                                        periods = periods,
                                        generalised_irf = generalised_irf,
                                        negative_shock = negative_shock,
                                        shock_size = shock_size)
                            end
                        end
                    end
                end
            end
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
                            for sylvester_algorithm in sylvester_alogorithms
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
                                @test isapprox(irf_, IRF_)
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
        steady_state = SS(m, derivatives = false)
        
        for tol in [MacroModelling.Tolerances(),MacroModelling.Tolerances(NSSS_xtol = 1e-14)]
            for parameters in params 

                res = get_non_stochastic_steady_state_residuals(m, steady_state, tol = tol, verbose = false, parameters = parameters)

                for values in [Dict(axiskeys(steady_state)[1] .=> collect(steady_state)), Dict(string.(axiskeys(steady_state)[1]) .=> collect(steady_state)), collect(steady_state)]   
                    RES = get_non_stochastic_steady_state_residuals(m, values, tol = tol, verbose = false, parameters = parameters)

                    @test isapprox(res,RES)
                end
            end

            res1 = get_non_stochastic_steady_state_residuals(m, steady_state, tol = tol, verbose = false)

            res2 = get_non_stochastic_steady_state_residuals(m, steady_state[1:3], tol = tol, verbose = false)

            @test isapprox(res1,res2)
        end
    end

    @testset "get_steady_state" begin
        for derivatives in [true, false]
            for stochastic in (algorithm == :first_order ? [false] : [true, false])
                for return_variables_only in [true, false]
                    for verbose in [false]
                        for silent in [true, false]
                            NSSS = get_steady_state(m, 
                                                    verbose = verbose, 
                                                    silent = silent, 
                                                    return_variables_only = return_variables_only, 
                                                    algorithm = algorithm, 
                                                    stochastic = stochastic, 
                                                    derivatives = derivatives)
                            for quadratic_matrix_equation_algorithm in qme_algorithms
                                for sylvester_algorithm in sylvester_alogorithms
                                    nsss = get_steady_state(m, 
                                                            verbose = verbose, 
                                                            quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm, 
                                                            sylvester_algorithm = sylvester_algorithm, 
                                                            silent = silent, 
                                                            return_variables_only = return_variables_only, 
                                                            algorithm = algorithm, 
                                                            stochastic = stochastic, 
                                                            derivatives = derivatives)
                                    @test isapprox(NSSS,nsss)
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


    nsss = get_steady_state(m, verbose = true)

    NSSS = get_SS(m, derivatives = false)

    @test maximum(collect(check_residuals(m, NSSS))) < 1e-12
    @test maximum(collect(check_residuals(m, collect(NSSS)))) < 1e-12
    @test maximum(collect(check_residuals(m, Dict(axiskeys(NSSS, 1) .=> collect(NSSS))))) < 1e-12

    if algorithm ∈ [:pruned_second_order,:second_order]
        sols_nv = get_second_order_solution(m)
    elseif algorithm ∈ [:pruned_third_order,:third_order]
        sols_nv = get_third_order_solution(m)
    end

    if algorithm ∈ [:first_order, :pruned_second_order, :pruned_third_order]
        # Check different inputs for get_moments
        moms_nv = get_moments(m, algorithm = algorithm)
        moms = get_moments(m, algorithm = algorithm, verbose = true)
        moms_var = get_moments(m, algorithm = algorithm, verbose = true, variance = true)
        moms_covar = get_moments(m, algorithm = algorithm, verbose = true, covariance = true)
        moms_no_nsss = get_moments(m, algorithm = algorithm, verbose = true, non_stochastic_steady_state = false)
        moms_no_nsss = get_moments(m, algorithm = algorithm, verbose = true, standard_deviation = false)
        moms_no_nsss = get_moments(m, algorithm = algorithm, verbose = true, standard_deviation = false, variance = true)
        moms_no_derivs = get_moments(m, algorithm = algorithm, verbose = true, derivatives = false)
        moms_no_derivs_var = get_moments(m, algorithm = algorithm, verbose = true, derivatives = false, variance = true)
        moms_no_derivs_var = get_moments(m, algorithm = algorithm, verbose = true, derivatives = false, variance = true, variables = m.var[2:4])

        moms_select_par_deriv1 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = m.parameters[1])
        moms_select_par_deriv2 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = m.parameters[1:2])
        moms_select_par_deriv3 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = Tuple(m.parameters[1:3]))
        moms_select_par_deriv4 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = reshape(m.parameters[1:3],3,1))
        moms_select_par_deriv4 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = reshape(m.parameters[1:3],3,1), variables = m.var[2:4])

        moms_select_par_deriv1 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = string.(m.parameters[1]))
        moms_select_par_deriv2 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = string.(m.parameters[1:2]))
        moms_select_par_deriv3 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = Tuple(string.(m.parameters[1:3])))
        moms_select_par_deriv4 = get_moments(m, algorithm = algorithm, verbose = true, parameter_derivatives = reshape(string.(m.parameters[1:3]),3,1))

        new_moms1 = get_moments(m, algorithm = algorithm, verbose = true, parameters = m.parameter_values * 1.0001)
        new_moms2 = get_moments(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001))
        new_moms3 = get_moments(m, algorithm = algorithm, verbose = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_moments(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        old_moms  = get_moments(m, algorithm = algorithm, verbose = true, parameters = old_params)

        new_moms2 = get_moments(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1]) => m.parameter_values[1] * 1.0001))
        new_moms3 = get_moments(m, algorithm = algorithm, verbose = true, parameters = Tuple(string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_moments(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001))
        old_moms  = get_moments(m, algorithm = algorithm, verbose = true, parameters = old_params)


        new_moms4 = get_standard_deviation(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_variance(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_covariance(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = get_std(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_var(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_cov(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = std(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = var(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = cov(m, algorithm = algorithm, verbose = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.000))

        new_moms4 = get_standard_deviation(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_variance(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_covariance(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = get_std(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = get_var(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = get_cov(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0002))
        new_moms4 = std(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001))
        new_moms4 = var(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001))
        new_moms4 = cov(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.000))
        new_moms4 = get_mean(m, algorithm = algorithm, verbose = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.000))
    end

    GC.gc()

    if algorithm == :first_order
        irfs_nv = get_irf(m, m.parameter_values)
        irfs = get_irf(m, m.parameter_values, verbose = true)
        irfs_10 = get_irf(m, m.parameter_values, verbose = true, periods = 10)
        irfs_100 = get_irf(m, m.parameter_values, verbose = true, periods = 100)
        new_irfs1 = get_irf(m, m.parameter_values * 1.0001, verbose = true)
        lvl_irfs  = get_irf(m, old_params, verbose = true, levels = true, variables = :all)
        lvlv_init_irfs  = get_irf(m, old_params, verbose = true, levels = true, initial_state = collect(lvl_irfs[:,5,1]))
        lvlv_init_neg_irfs = get_irf(m, old_params, verbose = true, levels = true, initial_state = collect(lvl_irfs[:,5,1]), negative_shock = true)

        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = m.exo[1])
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = m.exo)
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = Tuple(m.exo))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = reshape(m.exo,1,length(m.exo)))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = :all)

        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = string.(m.exo[1]))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = string.(m.exo))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = Tuple(string.(m.exo)))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = reshape(string.(m.exo),1,length(m.exo)))
        # new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = string.(:all))

        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = randn(m.timings.nExo,10))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = KeyedArray(randn(1,10),Shocks = string.([m.timings.exo[1]]), Periods = 1:10))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]))
        new_sub_lvl_irfs  = get_irf(m, old_params, verbose = true, shocks = :none, initial_state = collect(lvl_irfs[:,5,1]), levels = true, variables = :all)
        # new_sub_irfs  = get_irf(m, old_params, verbose = true, shocks = string.(:none), initial_state = collect(lvl_irfs[:,5,1]))
        # new_sub_lvl_irfs  = get_irf(m, old_params, verbose = true, shocks = string.(:none), initial_state = collect(lvl_irfs[:,5,1]), levels = true)
        @test isapprox(collect(new_sub_lvl_irfs[:,1,:]), collect(lvl_irfs[:,6,1]),rtol = eps(Float32))

        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = m.timings.var[1])
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = m.timings.var[end-1:end])
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = m.timings.var)
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = Tuple(m.timings.var))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = :all)

        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = :all_excluding_obc)


        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = string.(m.timings.var[1]))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = string.(m.timings.var[end-1:end]))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = string.(m.timings.var))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = Tuple(string.(m.timings.var)))
        new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
        # new_sub_irfs  = get_irf(m, old_params, verbose = true, variables = string.(:all))
    end

    if algorithm ∈ [:second_order, :pruned_second_order, :third_order, :pruned_third_order]
        SSS = get_stochastic_steady_state(m, algorithm = algorithm)
    end
    


    # test conditional forecasting
    new_sub_irfs_all  = get_irf(m, algorithm = algorithm, verbose = true, variables = :all, shocks = :all)
    varnames = axiskeys(new_sub_irfs_all,1)
    shocknames = axiskeys(new_sub_irfs_all,3)
    sol = get_solution(m)
    # var_idxs = findall(vec(sum(sol[end-length(shocknames)+1:end,:] .!= 0,dims = 1)) .> 0)[[1,end]]
    n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
    var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[[1,end]]

    conditions = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,1),2)
    conditions[var_idxs[1],1] = .01
    conditions[var_idxs[2],2] = .02

    cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false)

    if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
        shocks = Matrix{Union{Nothing, Float64}}(undef,size(new_sub_irfs_all,3),1)
        shocks[1,1] = .1
        cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, shocks = shocks)
    end

    conditions = spzeros(size(new_sub_irfs_all,1),2)
    conditions[var_idxs[1],1] = .01
    conditions[var_idxs[2],2] = .02

    cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false)

    if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
        shocks = spzeros(size(new_sub_irfs_all,3),1)
        shocks[1,1] = .1
        cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, shocks = shocks)
    end

    conditions = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs[[1, end]]]), Periods = 1:2)
    conditions[1,1] = .01
    conditions[2,2] = .02

    cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false)

    conditions = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs[[1, end]]], Periods = 1:2)
    conditions[1,1] = .01
    conditions[2,2] = .02

    cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false)

    if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
        shocks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = [shocknames[1]], Periods = [1])
        shocks[1,1] = .1
        cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, shocks = shocks)
    end

    if all(vec(sum(sol[end-length(shocknames)+1:end,var_idxs[[1, end]]] .!= 0, dims = 1)) .> 0)
        shocks = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,1,1), Shocks = string.([shocknames[1]]), Periods = [1])
        shocks[1,1] = .1
        cond_fcst = get_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, shocks = shocks)
    end

    if plots
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = false, show_plots = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001), verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = old_params, variables = :all, verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), variables = varnames[1], verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001), variables = string.(varnames[1]), verbose = true)
        plot_conditional_forecast(m, conditions, algorithm = algorithm, conditions_in_levels = false, save_plots = true, show_plots = false, periods = 10, parameters = old_params, variables = varnames[1], verbose = true)
    end

    NSSS = get_SS(m,derivatives = false)
    full_SS = sort(union(m.var,m.aux,m.exo_present))
    full_SS[indexin(m.aux,full_SS)] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾|ᴸ⁽[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),  m.aux)
    reference_steady_state = [s ∈ m.exo_present ? 0 : NSSS(axiskeys(NSSS,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol(s) : s) for s in full_SS]

    conditions_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = varnames[var_idxs[[1, end]]], Periods = 1:2)
    conditions_lvl[1,1] = .01 + reference_steady_state[var_idxs[1]]
    conditions_lvl[2,2] = .02 + reference_steady_state[var_idxs[2]]

    cond_fcst = get_conditional_forecast(m, conditions_lvl, algorithm = algorithm, periods = 10, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), variables = varnames[1], verbose = true)

    conditions_lvl = KeyedArray(Matrix{Union{Nothing, Float64}}(undef,2,2), Variables = string.(varnames[var_idxs[[1, end]]]), Periods = 1:2)
    conditions_lvl[1,1] = .01 + reference_steady_state[var_idxs[1]]
    conditions_lvl[2,2] = .02 + reference_steady_state[var_idxs[2]]

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

    lvl_irfs  = get_irf(m, verbose = true, algorithm = algorithm, parameters = old_params, levels = true, variables = :all)

    lvl_irfs = axiskeys(lvl_irfs,3) isa Vector{String} ? rekey(lvl_irfs,3 => axiskeys(lvl_irfs,3) .|> Meta.parse .|> MacroModelling.replace_indices) : lvl_irfs

    if plots
        if length(m.obc_violation_equations) > 0
            plot_irf(m, algorithm = algorithm, show_plots = true, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = true, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 10, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, periods = 100, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = m.parameter_values * 1.0001, ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1] => m.parameter_values[1] * 1.0001), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = Tuple(m.parameters[1:2] .=> m.parameter_values[1:2] * 1.0001), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (m.parameters[1:2] .=> m.parameter_values[1:2] / 1.0001), ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (string.(m.parameters[1]) => m.parameter_values[1] * 1.0001), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = Tuple(string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001), ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true, ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo[1], ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(m.exo), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(m.exo,1,length(m.exo)), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :all, ignore_obc = true)


            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(m.exo[1]), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(m.exo), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(string.(m.exo)), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(string.(m.exo),1,length(m.exo)), ignore_obc = true)
            # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:all))
            
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = randn(m.timings.nExo,10), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(1,10),Shocks = string.([m.timings.exo[1]]), Periods = 1:10), ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :simulate, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])), ignore_obc = true)

            # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:simulate))
            # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:none), initial_state = collect(lvl_irfs(:,5,m.exo[1])))

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[1], ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end], ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(m.timings.var), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = :all, ignore_obc = true)

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)), ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)), ignore_obc = true)
            # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(:all))

            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, plots_per_page = 6, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png, ignore_obc = true)
            plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4, ignore_obc = true)
        end

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

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (string.(m.parameters[1]) => m.parameter_values[1] * 1.0001))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = Tuple(string.(m.parameters[1:2]) .=> m.parameter_values[1:2] * 1.0001))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = (string.(m.parameters[1:2]) .=> m.parameter_values[1:2] / 1.0001))

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, parameters = old_params, initial_state = collect(lvl_irfs(:,5,m.exo[1])), negative_shock = true, generalised_irf = true)

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo[1])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = m.exo)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(m.exo))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(m.exo,1,length(m.exo)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :all)


        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(m.exo[1]))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(m.exo))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = Tuple(string.(m.exo)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = reshape(string.(m.exo),1,length(m.exo)))
        # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:all))
        
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = randn(m.timings.nExo,10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = m.timings.exo, Periods = 1:10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(1,10),Shocks = [m.timings.exo[1]], Periods = 1:10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(m.timings.nExo,10),Shocks = string.(m.timings.exo), Periods = 1:10))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = KeyedArray(randn(1,10),Shocks = string.([m.timings.exo[1]]), Periods = 1:10))

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :simulate)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = :none, initial_state = collect(lvl_irfs(:,5,m.exo[1])))

        # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:simulate))
        # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, shocks = string.(:none), initial_state = collect(lvl_irfs(:,5,m.exo[1])))

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[1])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var[end-1:end])
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = m.timings.var)
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(m.timings.var))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(m.timings.var,1,length(m.timings.var)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = :all)

        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(m.timings.var))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)))
        plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
        # plot_irf(m, verbose = true, algorithm = algorithm, show_plots = false, save_plots = true, variables = string.(:all))

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

            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)))
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
            # plot_fevd(m, verbose = true, show_plots = false, save_plots = true, variables = string.(:all))

            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
            plot_fevd(m, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)


            # Test filtering and smoothing
            sol = get_solution(m)

            if length(m.exo) > 1
                # var_idxs = findall(vec(sum(sol[end-length(m.exo)+1:end,:] .!= 0,dims = 1)) .> 0)[1:2]
                n_shocks_influence_var = vec(sum(abs.(sol[end-length(m.exo)+1:end,:]) .> eps(),dims = 1))
                var_idxs = findall(n_shocks_influence_var .== maximum(n_shocks_influence_var))[1:2]
            else
                var_idxs = [1]
            end
            
            simulation = simulate(m)

            data_in_levels = simulation(axiskeys(simulation,1) isa Vector{String} ? MacroModelling.replace_indices_in_symbol.(m.var[var_idxs]) : m.var[var_idxs],:,:simulate)
            # data_in_levels = simulation(m.var[var_idxs],:,:simulate)
            data = data_in_levels .- m.solution.non_stochastic_steady_state[var_idxs]
    
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

            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
            # plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(:all))
            # plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(:all))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(m.timings.exo))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(m.timings.exo[1]))
            plot_model_estimates(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = Tuple(string.(m.timings.exo)))

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

            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
            # plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, variables = string.(:all))
            # plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(:all))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(m.timings.exo))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = string.(m.timings.exo[1]))
            plot_shock_decomposition(m, data, data_in_levels = false, verbose = true, show_plots = false, save_plots = true, shocks = Tuple(string.(m.timings.exo)))

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

        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[1]))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var[end-1:end]))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = string.(m.timings.var))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = Tuple(string.(m.timings.var)))
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = reshape(string.(m.timings.var),1,length(m.timings.var)))
        # plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, variables = string.(:all))

        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, plots_per_page = 6)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png)
        plot_solution(m, states[1], algorithm = algorithm, verbose = true, show_plots = false, save_plots = true, save_plots_format = :png, plots_per_page = 4)
    end
end