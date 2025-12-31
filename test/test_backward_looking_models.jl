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
        
        # Test IRFs
        irf = get_irf(VAR2)
        @test size(irf, 1) == 2  # 2 variables
        @test size(irf, 3) == 2  # 2 shocks
        
        # Test simulation
        sim = simulate(VAR2)
        @test size(sim, 1) == 2  # 2 variables
        
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
        
        # Test IRFs
        irf = get_irf(SolowGrowth)
        @test size(irf, 1) == 2  # 2 variables (k, z)
        @test size(irf, 3) == 1  # 1 shock
        
        # Test simulation
        sim = simulate(SolowGrowth)
        @test size(sim, 1) == 2  # 2 variables
        
        SolowGrowth = nothing
    end
    
    GC.gc()
end
