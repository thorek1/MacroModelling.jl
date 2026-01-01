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
        
        # Test IRFs with first_order algorithm
        irf_first = get_irf(VAR2, algorithm = :first_order)
        @test size(irf_first, 1) == 2  # 2 variables
        @test size(irf_first, 3) == 2  # 2 shocks
        
        # Test simulation
        sim = simulate(VAR2)
        @test size(sim, 1) == 2  # 2 variables
        
        # Test newton algorithm - should produce same results as first_order for linear model
        irf_newton = get_irf(VAR2, algorithm = :newton)
        @test size(irf_newton, 1) == 2  # 2 variables
        @test size(irf_newton, 3) == 2  # 2 shocks
        
        # For linear models, first_order and newton should give approximately the same results
        @test isapprox(collect(irf_first), collect(irf_newton), rtol=1e-6)
        
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
        
        # Test IRFs with first_order algorithm
        irf = get_irf(SolowGrowth)
        @test size(irf, 1) == 2  # 2 variables (k, z)
        @test size(irf, 3) == 1  # 1 shock
        
        # Test simulation
        sim = simulate(SolowGrowth)
        @test size(sim, 1) == 2  # 2 variables
        
        # Test newton algorithm for nonlinear backward looking model
        irf_newton = get_irf(SolowGrowth, algorithm = :newton)
        @test size(irf_newton, 1) == 2  # 2 variables (k, z)
        @test size(irf_newton, 3) == 1  # 1 shock
        
        # Test simulation with newton
        sim_newton = simulate(SolowGrowth, algorithm = :newton)
        @test size(sim_newton, 1) == 2  # 2 variables
        
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
        result = get_irf(SolowGrowth2, 
                        algorithm = :newton, 
                        initial_state = initial_state_solow, 
                        shocks = :none,
                        periods = 10)
        
        @test size(result, 1) == 2  # 2 variables
        @test size(result, 2) == 10  # 10 periods
        
        # Capital should increase over time (converging toward steady state)
        @test result(:k, 1) > initial_k  # First period k should increase
        @test result(:k, 10) > result(:k, 1)  # Later periods should be higher
        
        SolowGrowth2 = nothing
    end
    
    @testset "Model with unstable eigenvalue but valid SS at zero" begin
        # Model y[0] = (1 + g) * y[-1] with g > 0 
        # Has eigenvalue > 1 (unstable) but valid SS at y = 0
        # From SS, iterating stays at SS. From any other point, it explodes.
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
        irf_newton = get_irf(UnstableButValidSS, algorithm = :newton)
        @test size(irf_newton, 1) == 2  # 2 variables
        @test size(irf_newton, 3) == 1  # 1 shock
        
        # IRF from SS should show the effect of a shock
        # y responds to z shock through the 'a' parameter
        @test irf_newton(:y, 1, :eps_z) != 0.0  # y responds to z shock
        
        # Can also provide initial_state in levels for simulation from non-SS point
        initial_y = 100.0  # Start at y = 100
        initial_state_levels = [initial_y, 0.0]  # [y, z]
        
        result = get_irf(UnstableButValidSS,
                        algorithm = :newton,
                        initial_state = initial_state_levels,
                        shocks = :none,
                        periods = 10)
        
        @test size(result, 1) == 2  # 2 variables
        @test size(result, 2) == 10  # 10 periods
        
        # y should grow each period at rate g (with z = 0)
        # Since initial_state is in levels and model has valid SS,
        # result is in levels (initial_state_levels provided)
        y_values = result(:y, :, :none)
        @test y_values[1] ≈ initial_y * (1 + 0.02) rtol=1e-6  # y_1 = (1+g) * y_0
        @test y_values[2] > y_values[1]  # Continues growing
        
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
        irf_newton = get_irf(UnitRootAR1, algorithm = :newton)
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
    
    GC.gc()
end
