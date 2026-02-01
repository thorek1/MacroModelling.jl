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
        
        # Get original steady state and IRF before any update
        ss_before = get_steady_state(RBC_test, derivatives = false)
        irf_before = get_irf(RBC_test)
        
        # Update equation 3 - change the production function by multiplying by 1.0 (mathematically equivalent)
        update_equations!(RBC_test, 3, :(q[0] = exp(z[0]) * k[-1]^α * 1.0), silent = true)
        
        # Check revision history was recorded
        history = get_revision_history(RBC_test)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index == 3
        
        # Model should still solve and produce same results (since * 1.0 is identity)
        ss_after = get_steady_state(RBC_test, derivatives = false)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-10)
        
        # IRF should be the same as before (equation is mathematically equivalent)
        irf_after = get_irf(RBC_test)
        @test isapprox(collect(irf_before), collect(irf_after), rtol = 1e-10)
        
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

    # Test update by matching old equation in string form
    @testset "Update equation by matching old equation string" begin
        @model RBC_test5 begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_test5 begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        old_eq = "q[0] = exp(z[0]) * k[-1]^α"
        new_eq = "q[0] = exp(z[0]) * k[-1]^α * 1.0"

        update_equations!(RBC_test5, old_eq, new_eq, silent = true)

        history = get_revision_history(RBC_test5)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index == 3

        ss_after = get_steady_state(RBC_test5, derivatives = false)
        @test !any(isnan, ss_after)

        RBC_test5 = nothing
    end

    # Test update with Taylor rule exchange
    @testset "Update Taylor rule equation" begin
        include(joinpath(@__DIR__, "..", "models", "Gali_2015_chapter_3_nonlinear.jl"))
        model = getfield(Main, :Gali_2015_chapter_3_nonlinear)

        old_rule = :(R[0] = 1 / β * Pi[0] ^ ϕᵖⁱ * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))
        new_rule = :(R[0] = 1 / β * Pi[0] ^ 1.7 * (Y[0] / Y[ss]) ^ ϕʸ * exp(nu[0]))

        push!(model.constants.post_complete_parameters.missing_parameters, :__skip__)

        update_equations!(model, old_rule, new_rule, silent = true)

        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index !== nothing

        updated_eqs = get_equations(model)
        @test any(eq -> occursin("1.7", eq), updated_eqs)

        model = nothing
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
        
        # Update calibration equation - accept `| alpha` syntax
        @test update_calibration_equations!(
            RBC_calib,
            1,
            :(k[ss] / (4 * y[ss]) = 1.6 | alpha),
            silent = true,
        ) === nothing
        
        RBC_calib = nothing
    end

    @testset "Switch calibrated parameter to another model parameter" begin
        # Test scenario 1: switching from alpha to delta (both are used in model equations)
        @model RBC_switch begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_switch begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            k[ss] / (4 * y[ss]) = 1.5 | alpha  # alpha is calibrated
            beta = 0.99
        end

        # Initially alpha is calibrated
        calib_params_before = get_calibrated_parameters(RBC_switch)
        @test "alpha" in calib_params_before
        @test !("delta" in calib_params_before)
        
        # Get original alpha value from steady state
        ss_before = get_steady_state(RBC_switch, derivatives = false)
        
        # Switch calibration from alpha to delta
        # This means: alpha will become a fixed parameter (with its SS value)
        # and delta will become calibrated
        update_calibration_equations!(
            RBC_switch,
            1,
            :(k[ss] / (4 * y[ss]) = 1.5 | delta),
            silent = true,
        )
        
        # Now delta should be calibrated, not alpha
        calib_params_after = get_calibrated_parameters(RBC_switch)
        @test !("alpha" in calib_params_after)
        @test "delta" in calib_params_after
        
        # Model should still solve
        ss_after = get_steady_state(RBC_switch, derivatives = false)
        @test !any(isnan, ss_after)
        
        RBC_switch = nothing
    end

    @testset "Error when calibrating parameter not in model equations" begin
        # Test scenario 2: trying to calibrate a parameter that isn't used in any equation
        @model RBC_error begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_error begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            k[ss] / (4 * y[ss]) = 1.5 | alpha
            beta = 0.99
        end

        # Try to switch to a parameter that doesn't exist in model equations
        # "unused_param" is not in the model, so this should error
        @test_throws ErrorException update_calibration_equations!(
            RBC_error,
            1,
            :(k[ss] / (4 * y[ss]) = 1.5 | unused_param),
            silent = true,
        )
        
        RBC_error = nothing
    end
end


@testset verbose = true "add_calibration_equation! functionality" begin
    @testset "Add calibration equation to model without any" begin
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
        
        # Verify alpha is NOT calibrated initially
        calib_params_before = get_calibrated_parameters(RBC_add_calib)
        @test !("alpha" in calib_params_before)
        
        # Get original steady state with fixed alpha
        ss_before = get_steady_state(RBC_add_calib, derivatives = false)
        @test !any(isnan, ss_before)
        
        # Add a calibration equation - alpha will become calibrated
        add_calibration_equation!(
            RBC_add_calib,
            :(k[ss] / (4 * y[ss]) = 1.5 | alpha),
            silent = true,
        )
        
        # Verify calibration equation was added
        calib_eqs_after = get_calibration_equations(RBC_add_calib)
        @test length(calib_eqs_after) == 1
        
        # Verify alpha is now calibrated
        calib_params_after = get_calibrated_parameters(RBC_add_calib)
        @test "alpha" in calib_params_after
        
        # Model should still solve
        ss_after = get_steady_state(RBC_add_calib, derivatives = false)
        @test !any(isnan, ss_after)
        
        RBC_add_calib = nothing
    end

    @testset "Add calibration equation - error if parameter not in model" begin
        @model RBC_add_error begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_add_error begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            alpha = 0.33
            beta = 0.99
        end

        # Try to add calibration for a parameter not in model equations
        @test_throws ErrorException add_calibration_equation!(
            RBC_add_error,
            :(k[ss] / (4 * y[ss]) = 1.5 | unused_param),
            silent = true,
        )
        
        RBC_add_error = nothing
    end

    @testset "Add calibration equation - error if parameter already calibrated" begin
        @model RBC_already_calib begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_already_calib begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            k[ss] / (4 * y[ss]) = 1.5 | alpha  # alpha is already calibrated
            beta = 0.99
        end

        # Try to add another calibration for alpha - should error
        @test_throws ErrorException add_calibration_equation!(
            RBC_already_calib,
            :(c[ss] / y[ss] = 0.8 | alpha),
            silent = true,
        )
        
        RBC_already_calib = nothing
    end

    @testset "Add calibration equation - error if no | parameter syntax" begin
        @model RBC_no_pipe begin
            y[0] = A[0] * k[-1]^alpha
            1/c[0] = beta * 1/c[1] * (alpha * A[1] * k[0]^(alpha - 1) + (1 - delta))
            y[0] = c[0] + k[0] - (1 - delta) * k[-1]
            A[0] = 1 - rhoz + rhoz * A[-1] + std_eps * eps_z[x]
        end

        @parameters RBC_no_pipe begin
            std_eps = 0.01
            rhoz = 0.9
            delta = 0.025
            alpha = 0.33
            beta = 0.99
        end

        # Try to add a calibration equation without | parameter syntax
        @test_throws ErrorException add_calibration_equation!(
            RBC_no_pipe,
            :(k[ss] / (4 * y[ss]) = 1.5),  # Missing | alpha
            silent = true,
        )
        
        RBC_no_pipe = nothing
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
