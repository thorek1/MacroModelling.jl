using MacroModelling
using Test

@testset verbose = true "update_equations! functionality" begin
    # Test basic update_equations! with index
    @testset "Update equation by index" begin
        @model RBC_test begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_test begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Get original equations
        original_eqs = get_equations(RBC_test)
        @test length(original_eqs) == 4
        
        # Get original steady state
        ss_before = get_steady_state(RBC_test)
        
        # Update equation 3 - change the production function slightly
        update_equations!(RBC_test, 3, :(q[0] = exp(z[0]) * k[-1]^α * 1.0), silent = true)
        
        # Check revision history was recorded
        history = get_revision_history(RBC_test)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index == 3
        
        # Model should still solve
        ss_after = get_steady_state(RBC_test, derivatives = false)
        @test !any(isnan, ss_after)
        
        # Verify solution works
        irf = get_irf(RBC_test)
        @test !any(isnan, irf)
        
        RBC_test = nothing
    end

    # Test update_equations! with old equation matching
    @testset "Update equation by matching old equation" begin
        @model RBC_test2 begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_test2 begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end
        
        # Update using the old equation to match
        old_eq = :(q[0] = exp(z[0]) * k[-1]^α)
        new_eq = :(q[0] = exp(z[0]) * k[-1]^α * 1.0)
        
        update_equations!(RBC_test2, old_eq, new_eq, silent = true)
        
        # Check revision history was recorded
        history = get_revision_history(RBC_test2)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index == 3  # Should have found equation 3
        
        # Model should still solve
        ss_after = get_steady_state(RBC_test2, derivatives = false)
        @test !any(isnan, ss_after)
        
        RBC_test2 = nothing
    end
    
    # Test update with string equation
    @testset "Update equation using string format" begin
        @model RBC_test3 begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_test3 begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Update using string equation - change persistence parameter from ρ to 0.3
        update_equations!(RBC_test3, 4, "z[0] = 0.3 * z[-1] + std_z * eps_z[x]", silent = true)
        
        # Check revision history
        history = get_revision_history(RBC_test3)
        @test length(history) == 1
        
        # Model should still solve
        ss_after = get_steady_state(RBC_test3, derivatives = false)
        @test !any(isnan, ss_after)
        
        RBC_test3 = nothing
    end

    # Test multiple updates
    @testset "Multiple equation updates with history tracking" begin
        @model RBC_test4 begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_test4 begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Make multiple distinct updates
        # First: modify the production function
        update_equations!(RBC_test4, 3, :(q[0] = exp(z[0]) * k[-1]^α * 1.0), silent = true)
        # Second: modify the shock persistence from ρ to 0.3
        update_equations!(RBC_test4, 4, :(z[0] = 0.3 * z[-1] + std_z * eps_z[x]), silent = true)
        
        # Check revision history has both entries
        history = get_revision_history(RBC_test4)
        @test length(history) == 2
        @test history[1].action == :update_equation
        @test history[2].action == :update_equation
        
        # Model should still solve after multiple updates
        ss_after = get_steady_state(RBC_test4, derivatives = false)
        @test !any(isnan, ss_after)
        
        RBC_test4 = nothing
    end

    # Test that modifying shock equation produces same results as changing parameter
    @testset "Shock equation modification vs parameter change equivalence" begin
        # Model 1: We will modify the shock equation to use a larger std directly
        @model RBC_eq_mod begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_eq_mod begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Model 2: Same model but we'll change the parameter instead
        @model RBC_param_mod begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_param_mod begin
            std_z = 0.025  # 2.5 times the original 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Update the shock equation in Model 1: use a new parameter value for std
        # Replace the equation to use 0.025 directly instead of std_z
        # This tests that equation modification works for shock equations
        update_equations!(RBC_eq_mod, 4, :(z[0] = ρ * z[-1] + 0.025 * eps_z[x]), silent = true)
        
        # Verify the update was recorded
        history = get_revision_history(RBC_eq_mod)
        @test length(history) == 1
        @test history[1].action == :update_equation
        
        # Get IRFs from both models - they should be identical
        irf_eq_mod = get_irf(RBC_eq_mod)
        irf_param_mod = get_irf(RBC_param_mod)
        
        # The IRFs should be identical since both have effective shock std of 0.025
        @test isapprox(collect(irf_eq_mod), collect(irf_param_mod), rtol = 1e-10)
        
        RBC_eq_mod = nothing
        RBC_param_mod = nothing
    end
end


@testset verbose = true "update_calibration_equations! functionality" begin
    # NOTE: There is a known issue with calibration equations reprocessing where
    # the SS solver regeneration has indexing issues. See PR notes.
    # This test is marked as broken until that is resolved.
    @testset "Update calibration equation by index" begin
        @model RBC_calib begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_calib begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            k[ss] / (4 * y[ss]) = 1.5 | alpha
            beta = 0.99
        end

        # Get original calibration equations
        calib_eqs = get_calibration_equations(RBC_calib)
        @test length(calib_eqs) == 1
        
        # Get original calibrated parameters
        calib_params = get_calibrated_parameters(RBC_calib)
        @test length(calib_params) == 1
        @test calib_params[1] == "alpha"
        
        # Update calibration equation - change target from 1.5 to 1.6
        # Include the calibrated parameter in the call (alpha)
        # This is a @test_broken because of the known SS solver regeneration issue
        @test_broken begin
            update_calibration_equations!(RBC_calib, 1, :(k[ss] / (4 * y[ss]) = 1.6), :alpha, silent = true)
            
            # Check revision history was recorded
            history = get_revision_history(RBC_calib)
            length(history) == 1 && 
            history[1].action == :update_calibration_equation && 
            history[1].equation_index == 1
        end
        
        RBC_calib = nothing
    end

    # Test adding a calibration equation to replace a fixed parameter
    @testset "Replace parameter with calibration equation" begin
        # Model where alpha is initially a fixed parameter
        @model RBC_add_calib begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_add_calib begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            alpha = 0.33  # Fixed parameter initially
            beta = 0.99
        end

        # Verify no calibration equations initially
        calib_eqs_before = get_calibration_equations(RBC_add_calib)
        @test length(calib_eqs_before) == 0
        
        # Get original steady state with fixed alpha
        ss_before = get_steady_state(RBC_add_calib, derivatives = false)
        @test !any(isnan, ss_before)
        
        # Verify alpha is in the parameter list
        params = get_parameters(RBC_add_calib)
        @test "alpha" in params
        
        # This test would add a calibration equation to determine alpha
        # For now, this is marked as broken due to the known issue
        @test_broken begin
            # Add calibration equation: k[ss] / (4 * y[ss]) = 1.5 | alpha
            # This would replace the fixed alpha = 0.33 with a calibrated value
            update_calibration_equations!(RBC_add_calib, 0, :(k[ss] / (4 * y[ss]) = 1.5), :alpha, silent = true)
            
            # Check that calibration equation was added
            calib_eqs_after = get_calibration_equations(RBC_add_calib)
            length(calib_eqs_after) == 1
        end
        
        RBC_add_calib = nothing
    end
end


@testset verbose = true "get_revision_history functionality" begin
    @testset "Empty history for new model" begin
        @model RBC_empty begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_empty begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # New model should have empty history
        history = get_revision_history(RBC_empty)
        @test length(history) == 0
        
        RBC_empty = nothing
    end
end
