# Tests for purely backward looking models (nFuture_not_past_and_mixed = 0)
# These models have no forward-looking variables and include VARs and nonlinear growth models

@testset verbose = true "Backward looking models" begin
    
    @testset "Linear backward looking model (VAR)" begin
        # Two-variable VAR model
        @model VAR2 begin
            y[0] = a11 * y[-1] + a12 * x[-1] + sigma_y * eps_y[x]
            x[0] = a21 * y[-1] + a22 * x[-1] + sigma_x * eps_x[x]
        end

        @parameters VAR2 begin
            a11 = 0.5
            a12 = 0.3
            a21 = 0.2
            a22 = 0.4
            sigma_y = 0.1
            sigma_x = 0.1
        end

        # Verify it's a backward looking model
        @test VAR2.timings.nFuture_not_past_and_mixed == 0
        
        # Test steady state (should be zero for this linear model)
        ss = get_steady_state(VAR2)
        @test isapprox(ss(:y, :Steady_state), 0.0, atol=1e-10)
        @test isapprox(ss(:x, :Steady_state), 0.0, atol=1e-10)
        
        # Test solution
        sol = get_solution(VAR2)
        @test !isnothing(sol)
        
        # Test IRFs - default algorithm should be :newton for backward looking models
        irf_default = get_irf(VAR2)
        @test size(irf_default, 1) == 2  # 2 variables
        @test size(irf_default, 3) == 2  # 2 shocks
        
        # Test IRFs with first_order algorithm (explicitly specified)
        irf_first = get_irf(VAR2, algorithm = :first_order)
        @test size(irf_first, 1) == 2  # 2 variables
        @test size(irf_first, 3) == 2  # 2 shocks
        
        # Test simulation
        sim = simulate(VAR2)
        @test size(sim, 1) == 2  # 2 variables
        
        # For linear models, first_order and newton should give approximately the same results
        @test isapprox(collect(irf_first), collect(irf_default), rtol=1e-6)
        
        VAR2 = nothing
    end

    @testset "Nonlinear backward looking model (Solow growth)" begin
        # Solow growth model - purely backward looking nonlinear model
        @model SolowGrowth begin
            k[0] = (1 - delta) * k[-1] + s * k[-1]^alpha * exp(z[-1])
            z[0] = rho * z[-1] + sigma * eps_z[x]
        end

        @parameters SolowGrowth begin
            alpha = 0.3
            delta = 0.1
            s = 0.2
            rho = 0.9
            sigma = 0.01
        end

        # Verify it's a backward looking model
        @test SolowGrowth.timings.nFuture_not_past_and_mixed == 0
        
        # Test steady state exists and is positive for capital
        ss = get_steady_state(SolowGrowth)
        k_ss = ss(:k, :Steady_state)
        z_ss = ss(:z, :Steady_state)
        
        @test k_ss > 0  # Capital should be positive at steady state
        @test isapprox(z_ss, 0.0, atol=1e-10)  # z steady state should be 0
        
        # Verify steady state: k_ss = (s/delta)^(1/(1-alpha))
        k_ss_analytical = (0.2/0.1)^(1/(1-0.3))
        @test isapprox(k_ss, k_ss_analytical, rtol=1e-6)
        
        # Test solution
        sol = get_solution(SolowGrowth)
        @test !isnothing(sol)
        
        # Test IRFs - default algorithm for backward looking is :newton
        irf = get_irf(SolowGrowth)
        @test size(irf, 1) == 2  # 2 variables (k, z)
        @test size(irf, 3) == 1  # 1 shock
        
        # Test simulation
        sim = simulate(SolowGrowth)
        @test size(sim, 1) == 2  # 2 variables
        
        # Test first_order algorithm for comparison
        irf_first = get_irf(SolowGrowth, algorithm = :first_order)
        @test size(irf_first, 1) == 2  # 2 variables (k, z)
        @test size(irf_first, 3) == 1  # 1 shock
        
        # Test simulation with newton
        sim_newton = simulate(SolowGrowth, algorithm = :newton)
        @test size(sim_newton, 1) == 2  # 2 variables
        
        # Test conditional forecasting with newton
        # Condition z to be 0.02 in period 1
        conditions = Matrix{Union{Nothing,Float64}}(nothing, 2, 2)
        k_idx = findfirst(x -> x == :k, SolowGrowth.var)
        z_idx = findfirst(x -> x == :z, SolowGrowth.var)
        conditions[z_idx, 1] = 0.02  # Condition z in period 1
        conditions[z_idx, 2] = 0.01  # Condition z in period 2
        
        cf_newton = get_conditional_forecast(SolowGrowth, conditions, algorithm = :newton)
        
        # Check that conditions are met
        @test isapprox(cf_newton(:z, 1), 0.02, rtol=1e-4)
        @test isapprox(cf_newton(:z, 2), 0.01, rtol=1e-4)
        
        SolowGrowth = nothing
    end
    
    @testset "Backward looking model with initial_state in levels" begin
        # Test with Solow growth model from custom initial capital
        @model SolowGrowth2 begin
            k[0] = (1 - delta) * k[-1] + s * k[-1]^alpha * exp(z[-1])
            z[0] = rho * z[-1] + sigma * eps_z[x]
        end

        @parameters SolowGrowth2 begin
            alpha = 0.3
            delta = 0.1
            s = 0.2
            rho = 0.9
            sigma = 0.01
        end
        
        # Verify it's a backward looking model
        @test SolowGrowth2.timings.nFuture_not_past_and_mixed == 0
        
        # Start from initial capital below steady state
        k_ss = (0.2/0.1)^(1/(1-0.3))
        initial_k = k_ss * 0.5  # Start at 50% of steady state capital
        initial_state_solow = [initial_k, 0.0]  # [k, z]
        
        # For backward looking models with newton algorithm, initial_state is in levels
        # shocks = :none returns result with 3rd dimension as [:none]
        result = get_irf(SolowGrowth2, 
                        algorithm = :newton, 
                        initial_state = initial_state_solow, 
                        shocks = :none,
                        levels = true,  # Request levels output
                        periods = 10)
        
        @test size(result, 1) == 2  # 2 variables
        @test size(result, 2) == 10  # 10 periods
        
        # Capital should increase over time (converging toward steady state)
        # Access with 3 indices since result is 3D (Vars, Periods, Shocks)
        @test result(:k, 1, :none) > initial_k  # First period k should increase
        @test result(:k, 10, :none) > result(:k, 1, :none)  # Later periods should be higher
        
        SolowGrowth2 = nothing
    end
    
    @testset "Model with unstable eigenvalue but valid SS at zero" begin
        # Model y[0] = (1 + g) * y[-1] + a * z[-1] with g > 0 
        # Has eigenvalue > 1 (unstable) but valid SS at y = 0
        # Note: y depends on z[-1], not z[0], so y response to eps_z shock
        # appears only in period 2+ (delayed by one period)
        @model UnstableButValidSS begin
            y[0] = (1 + g) * y[-1] + a * z[-1]
            z[0] = rho * z[-1] + sigma * eps_z[x]
        end

        @parameters UnstableButValidSS begin
            g = 0.02      # 2% growth rate - eigenvalue > 1
            a = 0.5
            rho = 0.9
            sigma = 0.01
        end
        
        # Verify it's a backward looking model  
        @test UnstableButValidSS.timings.nFuture_not_past_and_mixed == 0
        
        # This model HAS a valid SS at y=0, z=0
        ss = get_steady_state(UnstableButValidSS)
        @test isapprox(ss(:y, :Steady_state), 0.0, atol=1e-10)
        @test isapprox(ss(:z, :Steady_state), 0.0, atol=1e-10)
        
        # Since SS is valid, newton works WITHOUT requiring initial_state
        # (starts from SS=0 and computes IRFs in deviations)
        # Default algorithm for backward looking models is now :newton
        irf_newton = get_irf(UnstableButValidSS)
        @test size(irf_newton, 1) == 2  # 2 variables
        @test size(irf_newton, 3) == 1  # 1 shock
        
        # IRF from SS should show the effect of a shock
        # y depends on z[-1], so y responds to z shock in period 2 (delayed by one period)
        @test irf_newton(:y, 2, :eps_z) != 0.0  # y responds to z shock in period 2
        @test irf_newton(:y, 1, :eps_z) == 0.0  # y has no immediate response (depends on z[-1])
        
        # Can also provide initial_state in levels for simulation from non-SS point
        initial_y = 100.0  # Start at y = 100
        initial_state_levels = [initial_y, 0.0]  # [y, z]
        
        result = get_irf(UnstableButValidSS,
                        algorithm = :newton,
                        initial_state = initial_state_levels,
                        shocks = :none,
                        levels = true,  # Request levels output
                        periods = 10)
        
        @test size(result, 1) == 2  # 2 variables
        @test size(result, 2) == 10  # 10 periods
        
        # y should grow each period at rate g (with z = 0)
        # With levels = true, result is in levels
        y_values = result(:y, :, :none)
        @test y_values[1] ≈ initial_y * (1 + 0.02) rtol=1e-6  # y_1 = (1+g) * y_0
        @test y_values[2] > y_values[1]  # Continues growing
        
        # Test deviations_from = :baseline
        # Get IRF with shocks in deviations from baseline mode
        irf_baseline = get_irf(UnstableButValidSS,
                        algorithm = :newton,
                        initial_state = initial_state_levels,
                        deviations_from = :baseline,
                        periods = 10)
        
        # Get IRF in levels mode
        irf_lev = get_irf(UnstableButValidSS,
                        algorithm = :newton,
                        initial_state = initial_state_levels,
                        levels = true,
                        periods = 10)
        
        # Get baseline path (no shocks) in levels
        baseline_lev = get_irf(UnstableButValidSS,
                        algorithm = :newton,
                        initial_state = initial_state_levels,
                        shocks = :none,
                        levels = true,
                        periods = 10)
        
        # Identity: irf_baseline + baseline_lev ≈ irf_lev
        # (deviation from baseline + baseline in levels = simulation in levels)
        @test isapprox(collect(irf_baseline(:y, :, :eps_z)) .+ collect(baseline_lev(:y, :, :none)), 
                       collect(irf_lev(:y, :, :eps_z)), rtol=1e-10)
        @test isapprox(collect(irf_baseline(:z, :, :eps_z)) .+ collect(baseline_lev(:z, :, :none)), 
                       collect(irf_lev(:z, :, :eps_z)), rtol=1e-10)
        
        # Test conditional forecasting with non-trivial starting point
        # Note: y[1] = (1+g)*y[-1] + a*z[-1] depends only on predetermined values,
        # so we can only condition z (which has shock eps_z) in period 1.
        # For period 2, y depends on z[1] which is affected by eps_z[1], so both can be conditioned.
        conditions = Matrix{Union{Nothing,Float64}}(nothing, 2, 2)
        z_idx = findfirst(x -> x == :z, UnstableButValidSS.var)
        y_idx = findfirst(x -> x == :y, UnstableButValidSS.var)
        conditions[z_idx, 1] = 0.05  # Condition z in period 1 (affected by eps_z shock)
        conditions[z_idx, 2] = 0.04  # Condition z in period 2
        
        cf_result = get_conditional_forecast(UnstableButValidSS, 
                        conditions, 
                        algorithm = :newton, 
                        initial_state = initial_state_levels,
                        periods = 5)
        
        # Check that conditions are satisfied
        @test isapprox(cf_result(:z, 1), 0.05, rtol=1e-4)
        @test isapprox(cf_result(:z, 2), 0.04, rtol=1e-4)
        
        UnstableButValidSS = nothing
    end
    
    @testset "Unit root model (AR(1) with rho=1, has valid SS at zero)" begin
        # Unit root model using AR(1) formulation with rho=1
        # y[0] = rho_y * y[-1] + sigma * eps_y[x] with rho_y = 1.0
        # This has a valid steady state at y = 0 but has a unit root
        # Written this way (instead of y[0] = y[-1] + e[x]) so SymPy can solve SS
        @model UnitRootAR1 begin
            y[0] = rho_y * y[-1] + sigma * eps_y[x]
            z[0] = rho_z * z[-1] + sigma_z * eps_z[x]
        end

        @parameters UnitRootAR1 begin
            rho_y = 1.0  # Unit root - permanent shocks
            rho_z = 0.5  # Stationary
            sigma = 0.1
            sigma_z = 0.1
        end
        
        # Verify it's a backward looking model  
        @test UnitRootAR1.timings.nFuture_not_past_and_mixed == 0
        
        # Should find steady state at zero
        ss = get_steady_state(UnitRootAR1)
        @test isapprox(ss(:y, :Steady_state), 0.0, atol=1e-10)
        @test isapprox(ss(:z, :Steady_state), 0.0, atol=1e-10)
        
        # For unit root models with valid SS, newton algorithm should work WITHOUT requiring initial_state
        # Deviations are computed against the no-shock reference path
        # Default algorithm for backward looking is :newton
        irf_newton = get_irf(UnitRootAR1)
        @test size(irf_newton, 1) == 2  # 2 variables
        @test size(irf_newton, 3) == 2  # 2 shocks
        
        # The IRF for y to eps_y shock should show permanent effect (unit root)
        # First period deviation should be equal to sigma (shock size = 1)
        @test isapprox(irf_newton(:y, 1, :eps_y), 0.1, rtol=1e-6)  # sigma * 1
        # Effect should persist (unit root with rho=1)
        @test isapprox(irf_newton(:y, 10, :eps_y), 0.1, rtol=1e-6)  # Still 0.1
        
        # For z with rho = 0.5, effect should decay
        @test isapprox(irf_newton(:z, 1, :eps_z), 0.1, rtol=1e-6)  # sigma_z * 1
        @test irf_newton(:z, 10, :eps_z) < irf_newton(:z, 1, :eps_z)  # Decays
        
        # First order and newton should give same results for linear model
        irf_first = get_irf(UnitRootAR1, algorithm = :first_order)
        @test isapprox(collect(irf_first), collect(irf_newton), rtol=1e-6)
        
        UnitRootAR1 = nothing
    end
    
    @testset "Conditional forecasting with newton algorithm" begin
        # Test conditional forecasting for backward looking models with newton
        @model VAR_cond begin
            y[0] = a11 * y[-1] + a12 * x[-1] + sigma_y * eps_y[x]
            x[0] = a21 * y[-1] + a22 * x[-1] + sigma_x * eps_x[x]
        end

        @parameters VAR_cond begin
            a11 = 0.5
            a12 = 0.3
            a21 = 0.2
            a22 = 0.4
            sigma_y = 0.1
            sigma_x = 0.1
        end
        
        # Verify it's a backward looking model
        @test VAR_cond.timings.nFuture_not_past_and_mixed == 0
        
        # Test conditional forecast with newton - condition y to be 0.05 in period 1
        # Use Matrix directly instead of KeyedArray to avoid indexing issues
        conditions = Matrix{Union{Nothing,Float64}}(nothing, 2, 2)
        y_idx = findfirst(x -> x == :y, VAR_cond.var)
        x_idx = findfirst(x -> x == :x, VAR_cond.var)
        conditions[y_idx, 1] = 0.05  # Condition y in period 1
        conditions[x_idx, 2] = 0.02  # Condition x in period 2
        
        # Use newton algorithm (default for backward looking) for conditional forecast
        cf_newton = get_conditional_forecast(VAR_cond, conditions)
        
        # Check that conditions are met
        @test isapprox(cf_newton(:y, 1), 0.05, rtol=1e-6)
        @test isapprox(cf_newton(:x, 2), 0.02, rtol=1e-6)
        
        # Compare with first_order conditional forecast - should give similar results for linear model
        cf_first = get_conditional_forecast(VAR_cond, conditions, algorithm = :first_order)
        @test isapprox(cf_first(:y, 1), 0.05, rtol=1e-6)
        @test isapprox(cf_first(:x, 2), 0.02, rtol=1e-6)
        
        # Shocks should be similar for linear models
        @test isapprox(cf_newton(:eps_y₍ₓ₎, 1), cf_first(:eps_y₍ₓ₎, 1), rtol=1e-4)
        
        VAR_cond = nothing
    end
    
    GC.gc()
end
