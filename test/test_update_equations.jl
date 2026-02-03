
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
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        old_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
        new_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])

        update_equations!(model, old_rule, new_rule, silent = true)

        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index !== nothing

        updated_eqs = get_equations(model)
        # Verify the updated Taylor rule no longer contains the crdy growth term
        taylor_rule_eq = updated_eqs[history[1].equation_index]
        @test !occursin("crdy", string(taylor_rule_eq))
        # Verify the equation still contains the expected components
        @test occursin("ms", string(taylor_rule_eq))
        @test occursin("crr", string(taylor_rule_eq))

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


@testset verbose = true "add_equation! functionality" begin
    @testset "Add equation - basic functionality" begin
        @model RBC_add begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_add begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Initial state
        @test length(get_equations(RBC_add)) == 4
        @test length(get_variables(RBC_add)) == 4

        # Add an investment equation
        add_equation!(RBC_add, :(i[0] = k[0] - (1 - δ) * k[-1]), silent = true)

        # Check model was updated
        @test length(get_equations(RBC_add)) == 5
        @test length(get_variables(RBC_add)) == 5
        @test "i" in get_variables(RBC_add)

        # Check revision history
        history = get_revision_history(RBC_add)
        @test length(history) == 1
        @test history[1].action == :add_equation
        @test history[1].equation_index == 5
        @test history[1].old_equation === nothing

        # Model should still solve
        ss = get_steady_state(RBC_add, derivatives = false)
        @test !any(isnan, ss)

        RBC_add = nothing
    end

    @testset "Add equation - string format" begin
        @model RBC_add_str begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_add_str begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Add equation using string
        add_equation!(RBC_add_str, "i[0] = k[0] - (1 - δ) * k[-1]", silent = true)

        @test length(get_equations(RBC_add_str)) == 5
        @test "i" in get_variables(RBC_add_str)

        RBC_add_str = nothing
    end

    @testset "Add equation - multiple additions" begin
        @model RBC_multi_add begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_multi_add begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Add two equations
        add_equation!(RBC_multi_add, :(i[0] = k[0] - (1 - δ) * k[-1]), silent = true)
        add_equation!(RBC_multi_add, :(y[0] = q[0]), silent = true)

        @test length(get_equations(RBC_multi_add)) == 6
        @test length(get_variables(RBC_multi_add)) == 6
        @test "i" in get_variables(RBC_multi_add)
        @test "y" in get_variables(RBC_multi_add)

        # Check revision history
        history = get_revision_history(RBC_multi_add)
        @test length(history) == 2
        @test history[1].action == :add_equation
        @test history[2].action == :add_equation

        RBC_multi_add = nothing
    end
end


@testset verbose = true "remove_equation! functionality" begin
    @testset "Remove equation by index" begin
        @model RBC_remove begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
            i[0] = k[0] - (1 - δ) * k[-1]  # Extra equation to remove
        end

        @parameters RBC_remove begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Initial state
        @test length(get_equations(RBC_remove)) == 5
        @test length(get_variables(RBC_remove)) == 5
        @test "i" in get_variables(RBC_remove)

        # Remove equation 5 (the investment equation)
        remove_equation!(RBC_remove, 5, silent = true)

        # Check model was updated
        @test length(get_equations(RBC_remove)) == 4
        @test length(get_variables(RBC_remove)) == 4
        @test !("i" in get_variables(RBC_remove))

        # Check revision history
        history = get_revision_history(RBC_remove)
        @test length(history) == 1
        @test history[1].action == :remove_equation
        @test history[1].equation_index == 5
        @test history[1].new_equation === nothing

        # Model should still solve
        ss = get_steady_state(RBC_remove, derivatives = false)
        @test !any(isnan, ss)

        RBC_remove = nothing
    end

    @testset "Remove equation by matching" begin
        @model RBC_remove_match begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
            i[0] = k[0] - (1 - δ) * k[-1]
        end

        @parameters RBC_remove_match begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Remove by matching the equation
        remove_equation!(RBC_remove_match, :(i[0] = k[0] - (1 - δ) * k[-1]), silent = true)

        @test length(get_equations(RBC_remove_match)) == 4
        @test !("i" in get_variables(RBC_remove_match))

        RBC_remove_match = nothing
    end

    @testset "Remove equation - error cases" begin
        @model RBC_remove_err begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_remove_err begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Error: index out of bounds
        @test_throws AssertionError remove_equation!(RBC_remove_err, 5, silent = true)
        @test_throws AssertionError remove_equation!(RBC_remove_err, 0, silent = true)

        # Error: equation not found
        @test_throws AssertionError remove_equation!(
            RBC_remove_err,
            :(nonexistent[0] = 1),
            silent = true,
        )

        RBC_remove_err = nothing
    end

    @testset "Remove equation - cannot remove to less than 2 equations" begin
        @model RBC_two_eqs begin
            x[0] = ρ * x[-1] + std_x * eps_x[x]
            y[0] = x[0]
        end

        @parameters RBC_two_eqs begin
            std_x = 0.01
            ρ = 0.9
        end

        @test length(get_equations(RBC_two_eqs)) == 2

        # Trying to remove an equation when it would leave < 2 equations throws an error
        # (either AssertionError from our check, or from process_model_equations)
        @test_throws Exception remove_equation!(RBC_two_eqs, 2, silent = true)

        RBC_two_eqs = nothing
    end

    @testset "Add then remove equation - round trip" begin
        @model RBC_roundtrip begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_roundtrip begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Get original state
        ss_before = get_steady_state(RBC_roundtrip, derivatives = false)
        n_eqs_before = length(get_equations(RBC_roundtrip))

        # Add then remove
        add_equation!(RBC_roundtrip, :(i[0] = k[0] - (1 - δ) * k[-1]), silent = true)
        @test length(get_equations(RBC_roundtrip)) == n_eqs_before + 1

        remove_equation!(RBC_roundtrip, n_eqs_before + 1, silent = true)
        @test length(get_equations(RBC_roundtrip)) == n_eqs_before

        # Steady state should be the same
        ss_after = get_steady_state(RBC_roundtrip, derivatives = false)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-10)

        RBC_roundtrip = nothing
    end
end


@testset verbose = true "remove_calibration_equation! functionality" begin
    @testset "Remove calibration equation by index" begin
        @model RBC_remove_calib begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_remove_calib begin
            std_z = 0.01
            ρ = 0.2
            α = 0.5
            β = 0.95
            k[ss] / q[ss] = 10 | δ  # Calibration equation
        end

        # Initial state - δ is calibrated
        @test "δ" in get_calibrated_parameters(RBC_remove_calib)
        @test length(get_calibration_equations(RBC_remove_calib)) == 1

        # Remove the calibration equation with explicit new value
        remove_calibration_equation!(RBC_remove_calib, 1, new_value = 0.025, silent = true)

        # Check δ is now a fixed parameter
        @test !("δ" in get_calibrated_parameters(RBC_remove_calib))
        @test "δ" in get_parameters(RBC_remove_calib)
        @test length(get_calibration_equations(RBC_remove_calib)) == 0

        # Check revision history
        history = get_revision_history(RBC_remove_calib)
        @test length(history) == 1
        @test history[1].action == :remove_calibration_equation
        @test history[1].new_equation === nothing

        # Model should still solve
        ss = get_steady_state(RBC_remove_calib, derivatives = false)
        @test !any(isnan, ss)

        RBC_remove_calib = nothing
    end

    @testset "Remove calibration equation - use current value" begin
        @model RBC_remove_calib_auto begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_remove_calib_auto begin
            std_z = 0.01
            ρ = 0.2
            α = 0.5
            β = 0.95
            k[ss] / q[ss] = 10 | δ
        end

        # Remove without specifying new_value - should use calibrated value
        remove_calibration_equation!(RBC_remove_calib_auto, 1, silent = true)

        @test !("δ" in get_calibrated_parameters(RBC_remove_calib_auto))
        @test "δ" in get_parameters(RBC_remove_calib_auto)

        # Model should still solve
        ss = get_steady_state(RBC_remove_calib_auto, derivatives = false)
        @test !any(isnan, ss)

        RBC_remove_calib_auto = nothing
    end

    @testset "Remove calibration equation - error cases" begin
        @model RBC_no_calib begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_no_calib begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Error: no calibration equations
        @test_throws AssertionError remove_calibration_equation!(RBC_no_calib, 1, silent = true)

        RBC_no_calib = nothing
    end

    @testset "Add then remove calibration equation - round trip" begin
        @model RBC_calib_roundtrip begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_calib_roundtrip begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02  # Fixed initially
            α = 0.5
            β = 0.95
        end

        # Get original δ value
        original_delta = 0.02

        # Add calibration equation
        add_calibration_equation!(RBC_calib_roundtrip, :(k[ss] / q[ss] = 10 | δ), silent = true)
        @test "δ" in get_calibrated_parameters(RBC_calib_roundtrip)

        # Remove calibration equation with original value
        remove_calibration_equation!(RBC_calib_roundtrip, 1, new_value = original_delta, silent = true)
        @test !("δ" in get_calibrated_parameters(RBC_calib_roundtrip))
        @test "δ" in get_parameters(RBC_calib_roundtrip)

        # Model should solve
        ss = get_steady_state(RBC_calib_roundtrip, derivatives = false)
        @test !any(isnan, ss)

        RBC_calib_roundtrip = nothing
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


@testset verbose = true "SW07 model - update_equations!" begin
    @testset "Update Taylor rule equation - remove output growth term" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Get original equations and steady state
        original_eqs = get_equations(model)
        n_eqs_original = length(original_eqs)
        ss_before = get_steady_state(model, derivatives = false)
        std_before = get_standard_deviation(model, derivatives = false)
        
        # Find the Taylor rule equation (equation with r[0], r[-1], pinf[0], etc.)
        old_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
        new_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])
        
        update_equations!(model, old_rule, new_rule, silent = true)
        
        # Check revision history
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index !== nothing
        
        # Model should still solve
        ss_after = get_steady_state(model, derivatives = false)
        std_after = get_standard_deviation(model, derivatives = false)
        @test !any(isnan, ss_after)
        
        # Number of equations should remain the same
        @test length(get_equations(model)) == n_eqs_original
        
        # Revert back to original Taylor rule
        update_equations!(model, new_rule, old_rule, silent = true)
        
        # Check revision history now has 2 entries
        history = get_revision_history(model)
        @test length(history) == 2
        @test history[2].action == :update_equation
        
        # Model should still solve
        ss_reverted = get_steady_state(model, derivatives = false)
        std_reverted = get_standard_deviation(model, derivatives = false)
        @test !any(isnan, ss_reverted)
        
        # Steady state and standard deviation should match original
        @test isapprox(collect(ss_before), collect(ss_reverted), rtol = 1e-10)
        @test isapprox(collect(std_before), collect(std_reverted), rtol = 1e-10)
        
        model = nothing
    end
    
    @testset "Update shock equation by index" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Find the technology shock equation: a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x]
        eqs = get_equations(model)
        shock_eq_idx = findfirst(eq -> occursin("a[0]") && occursin("crhoa") && occursin("ea[x]"), string.(eqs))
        @test shock_eq_idx !== nothing
        
        # Update to change persistence slightly (mathematically equivalent but different form)
        new_eq = :(a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x] * 1.0)
        update_equations!(model, shock_eq_idx, new_eq, silent = true)
        
        # Check revision history
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].equation_index == shock_eq_idx
        
        # Model should still solve
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Update consumption Euler equation" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # The consumption Euler equation involves xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1]
        old_eq = :(xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1])
        # Multiply by 1.0 (identity transformation)
        new_eq = :(xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1] * 1.0)
        
        update_equations!(model, old_eq, new_eq, silent = true)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Multiple updates on SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Make two distinct updates
        # 1. Update technology shock
        eqs = get_equations(model)
        shock_eq_idx = findfirst(eq -> occursin("a[0]") && occursin("crhoa") && occursin("ea[x]"), string.(eqs))
        update_equations!(model, shock_eq_idx, :(a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x] * 1.0), silent = true)
        
        # 2. Update Euler equation
        old_eq = :(xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1])
        new_eq = :(xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1] * 1.0)
        update_equations!(model, old_eq, new_eq, silent = true)
        
        # Check revision history has both
        history = get_revision_history(model)
        @test length(history) == 2
        @test all(h -> h.action == :update_equation, history)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - update_calibration_equations!" begin
    @testset "Update flexible price markup calibration" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # SW07 has calibration equation: mcflex = mc[ss] | mcflex
        calib_eqs = get_calibration_equations(model)
        @test length(calib_eqs) >= 1
        
        # Update the mcflex calibration (just change RHS symbolically)
        update_calibration_equations!(model, 1, :(mcflex = mc[ss] * 1.0 | mcflex), silent = true)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_calibration_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Switch calibrated parameter in SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Original: mcflex = mc[ss] | mcflex (mcflex is calibrated)
        calib_params_before = get_calibrated_parameters(model)
        @test "mcflex" in calib_params_before
        
        # Try to update to calibrate cpie instead (which appears in model equations)
        # This should work since cpie is used in the model
        update_calibration_equations!(model, 1, :(mcflex = mc[ss] | cpie), silent = true)
        
        calib_params_after = get_calibrated_parameters(model)
        @test "cpie" in calib_params_after
        @test !("mcflex" in calib_params_after)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - add_equation!" begin
    @testset "Add auxiliary variable equation" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        n_eqs_before = length(get_equations(model))
        n_vars_before = length(get_variables(model))
        
        # Add an auxiliary equation: inflation_gap[0] = pinf[0] - cpie
        add_equation!(model, :(inflation_gap[0] = pinf[0] - cpie), silent = true)
        
        @test length(get_equations(model)) == n_eqs_before + 1
        @test length(get_variables(model)) == n_vars_before + 1
        @test "inflation_gap" in get_variables(model)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :add_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Add multiple equations to SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        n_eqs_before = length(get_equations(model))
        
        # Add two auxiliary variables
        add_equation!(model, :(real_wage[0] = w[0] / pinf[0]), silent = true)
        add_equation!(model, :(output_gap_pct[0] = 100 * (y[0] - yflex[0]) / yflex[0]), silent = true)
        
        @test length(get_equations(model)) == n_eqs_before + 2
        @test "real_wage" in get_variables(model)
        @test "output_gap_pct" in get_variables(model)
        
        history = get_revision_history(model)
        @test length(history) == 2
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - remove_equation!" begin
    @testset "Remove observable equation from SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        n_eqs_before = length(get_equations(model))
        n_vars_before = length(get_variables(model))
        
        # Remove the labor observable equation: labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1)
        old_eq = :(labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1))
        remove_equation!(model, old_eq, silent = true)
        
        @test length(get_equations(model)) == n_eqs_before - 1
        @test length(get_variables(model)) == n_vars_before - 1
        @test !("labobs" in get_variables(model))
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :remove_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Remove equation by index in SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Find and remove the wage observable equation
        eqs = get_equations(model)
        dwobs_idx = findfirst(eq -> occursin("dwobs[0]") && occursin("ctrend"), string.(eqs))
        @test dwobs_idx !== nothing
        
        n_eqs_before = length(eqs)
        remove_equation!(model, dwobs_idx, silent = true)
        
        @test length(get_equations(model)) == n_eqs_before - 1
        @test !("dwobs" in get_variables(model))
        
        model = nothing
    end
    
    @testset "Add then remove in SW07 - round trip" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        n_eqs_original = length(get_equations(model))
        ss_before = get_steady_state(model, derivatives = false)
        
        # Add then remove an equation
        add_equation!(model, :(temp_var[0] = y[0] + c[0]), silent = true)
        @test length(get_equations(model)) == n_eqs_original + 1
        
        remove_equation!(model, n_eqs_original + 1, silent = true)
        @test length(get_equations(model)) == n_eqs_original
        
        # Steady state should be approximately the same
        ss_after = get_steady_state(model, derivatives = false)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-8)
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - add_calibration_equation!" begin
    @testset "Add calibration for fixed parameter" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # In SW07, ctou = .025 is a fixed parameter (depreciation rate)
        # Let's add a calibration equation for it
        calib_params_before = get_calibrated_parameters(model)
        @test !("ctou" in calib_params_before)
        
        # Add calibration: capital-to-output ratio determines depreciation
        # k[ss] / y[ss] = 10 | ctou (example calibration)
        add_calibration_equation!(model, :(k[ss] / y[ss] = 10 | ctou), silent = true)
        
        calib_params_after = get_calibrated_parameters(model)
        @test "ctou" in calib_params_after
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :add_calibration_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Error when adding calibration for non-existent parameter" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # Try to add calibration for a parameter not in the model
        @test_throws ErrorException add_calibration_equation!(
            model,
            :(y[ss] = 1.5 | fake_param),
            silent = true,
        )
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - remove_calibration_equation!" begin
    @testset "Remove existing calibration from SW07" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # SW07 has calibration equations - let's remove one
        calib_eqs_before = get_calibration_equations(model)
        n_calib_before = length(calib_eqs_before)
        @test n_calib_before >= 1
        
        calib_params_before = get_calibrated_parameters(model)
        
        # Remove first calibration equation with explicit value
        remove_calibration_equation!(model, 1, new_value = 0.5, silent = true)
        
        calib_eqs_after = get_calibration_equations(model)
        @test length(calib_eqs_after) == n_calib_before - 1
        
        # The previously calibrated parameter should now be fixed
        calib_params_after = get_calibrated_parameters(model)
        @test length(calib_params_after) == length(calib_params_before) - 1
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :remove_calibration_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Remove calibration - use current value" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        calib_eqs_before = get_calibration_equations(model)
        @test length(calib_eqs_before) >= 1
        
        # Remove without specifying new_value - should use the current calibrated value
        remove_calibration_equation!(model, 1, silent = true)
        
        calib_eqs_after = get_calibration_equations(model)
        @test length(calib_eqs_after) == length(calib_eqs_before) - 1
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
end


@testset verbose = true "SW07 model - complex modification scenarios" begin
    @testset "Update Taylor rule and add auxiliary variables" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # 1. Update Taylor rule to remove output growth term
        old_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
        new_rule = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])
        update_equations!(model, old_rule, new_rule, silent = true)
        
        # 2. Add auxiliary variables for analysis
        add_equation!(model, :(interest_rate_gap[0] = r[0] - r[ss]), silent = true)
        add_equation!(model, :(inflation_deviation[0] = pinf[0] - cpie), silent = true)
        
        # 3. Check everything worked
        history = get_revision_history(model)
        @test length(history) == 3
        @test history[1].action == :update_equation
        @test history[2].action == :add_equation
        @test history[3].action == :add_equation
        
        @test "interest_rate_gap" in get_variables(model)
        @test "inflation_deviation" in get_variables(model)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Modify calibration and update multiple equations" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        # 1. Update a calibration equation
        update_calibration_equations!(model, 1, :(mcflex = mc[ss] * 1.0 | mcflex), silent = true)
        
        # 2. Update a shock process
        eqs = get_equations(model)
        shock_idx = findfirst(eq -> occursin("ms[0]") && occursin("crhoms"), string.(eqs))
        if shock_idx !== nothing
            update_equations!(model, shock_idx, :(ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em / 100 * em[x] * 1.0), silent = true)
        end
        
        # 3. Add an equation
        add_equation!(model, :(policy_rate_ann[0] = 400 * (r[0] - 1)), silent = true)
        
        history = get_revision_history(model)
        @test length(history) >= 2
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Complex sequence: add, update, remove" begin
        include("../models/Smets_Wouters_2007.jl")
        model = Smets_Wouters_2007
        
        n_eqs_start = length(get_equations(model))
        ss_start = get_steady_state(model, derivatives = false)
        
        # 1. Add two equations
        add_equation!(model, :(temp1[0] = y[0]), silent = true)
        add_equation!(model, :(temp2[0] = c[0]), silent = true)
        @test length(get_equations(model)) == n_eqs_start + 2
        
        # 2. Update one of them
        update_equations!(model, n_eqs_start + 1, :(temp1[0] = y[0] * 1.0), silent = true)
        
        # 3. Remove both
        remove_equation!(model, n_eqs_start + 2, silent = true)
        remove_equation!(model, n_eqs_start + 1, silent = true)
        @test length(get_equations(model)) == n_eqs_start
        
        # 4. Verify we're back to original state
        ss_end = get_steady_state(model, derivatives = false)
        @test isapprox(collect(ss_start), collect(ss_end), rtol = 1e-8)
        
        # 5. Check history tracked everything
        history = get_revision_history(model)
        @test length(history) == 5  # 2 adds + 1 update + 2 removes
        
        model = nothing
    end
end


@testset verbose = true "Vector/Tuple batch operations" begin
    @testset "Vector update_equations!" begin
        @model RBC_vec_update begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_update begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Update two equations at once
        update_equations!(RBC_vec_update, [
            (3, :(q[0] = exp(z[0]) * k[-1]^α * 1.01)),
            (4, :(z[0] = 0.25 * z[-1] + std_z * eps_z[x]))
        ], silent = true)
        
        # Check revision history has 2 entries
        history = get_revision_history(RBC_vec_update)
        @test length(history) == 2
        @test all(h.action == :update_equation for h in history)
        
        # Model should still solve
        ss = get_steady_state(RBC_vec_update, derivatives = false)
        @test !any(isnan, ss)
        
        RBC_vec_update = nothing
    end

    @testset "Vector update_equations! with Pair syntax" begin
        @model RBC_vec_pair begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_pair begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Use Pair syntax
        update_equations!(RBC_vec_pair, [
            3 => :(q[0] = exp(z[0]) * k[-1]^α * 1.02),
        ], silent = true)
        
        history = get_revision_history(RBC_vec_pair)
        @test length(history) == 1
        
        RBC_vec_pair = nothing
    end

    @testset "Vector add_equation!" begin
        @model RBC_vec_add begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_add begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test length(get_equations(RBC_vec_add)) == 4
        @test length(get_variables(RBC_vec_add)) == 4
        
        # Add two equations at once
        add_equation!(RBC_vec_add, [
            :(i[0] = k[0] - (1 - δ) * k[-1]),
            :(y[0] = q[0])
        ], silent = true)
        
        @test length(get_equations(RBC_vec_add)) == 6
        @test length(get_variables(RBC_vec_add)) == 6
        @test "i" in get_variables(RBC_vec_add)
        @test "y" in get_variables(RBC_vec_add)
        
        # Check revision history
        history = get_revision_history(RBC_vec_add)
        @test length(history) == 2
        @test all(h.action == :add_equation for h in history)
        
        RBC_vec_add = nothing
    end

    @testset "Vector remove_equation!" begin
        @model RBC_vec_remove begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
            i[0] = k[0] - (1 - δ) * k[-1]
            y[0] = q[0]
        end

        @parameters RBC_vec_remove begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test length(get_equations(RBC_vec_remove)) == 6
        
        # Remove two equations at once (descending order)
        remove_equation!(RBC_vec_remove, [6, 5], silent = true)
        
        @test length(get_equations(RBC_vec_remove)) == 4
        @test !("i" in get_variables(RBC_vec_remove))
        @test !("y" in get_variables(RBC_vec_remove))
        
        # Check revision history
        history = get_revision_history(RBC_vec_remove)
        @test length(history) == 2
        @test all(h.action == :remove_equation for h in history)
        
        RBC_vec_remove = nothing
    end

    @testset "Vector add_calibration_equation!" begin
        @model RBC_vec_add_calib begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_add_calib begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        @test length(get_calibration_equations(RBC_vec_add_calib)) == 0
        
        # Add calibration equation using vector
        add_calibration_equation!(RBC_vec_add_calib, [
            :(k[ss] / q[ss] = 10 | δ)
        ], silent = true)
        
        @test length(get_calibration_equations(RBC_vec_add_calib)) == 1
        @test "δ" in get_calibrated_parameters(RBC_vec_add_calib)
        
        RBC_vec_add_calib = nothing
    end

    @testset "Vector update_calibration_equations!" begin
        @model RBC_vec_update_calib begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_update_calib begin
            std_z = 0.01
            ρ = 0.2
            α = 0.5
            β = 0.95
            k[ss] / q[ss] = 10 | δ
        end

        @test "δ" in get_calibrated_parameters(RBC_vec_update_calib)
        
        # Update using vector
        update_calibration_equations!(RBC_vec_update_calib, [
            (1, :(k[ss] / q[ss] = 15 | δ))
        ], silent = true)
        
        # Check revision history
        history = get_revision_history(RBC_vec_update_calib)
        @test length(history) == 1
        @test history[1].action == :update_calibration_equation
        
        RBC_vec_update_calib = nothing
    end

    @testset "Vector remove_calibration_equation!" begin
        @model RBC_vec_remove_calib begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_vec_remove_calib begin
            std_z = 0.01
            ρ = 0.2
            α = 0.5
            β = 0.95
            k[ss] / q[ss] = 10 | δ
        end

        @test length(get_calibration_equations(RBC_vec_remove_calib)) == 1
        @test "δ" in get_calibrated_parameters(RBC_vec_remove_calib)
        
        # Remove using vector
        remove_calibration_equation!(RBC_vec_remove_calib, [1], silent = true)
        
        @test length(get_calibration_equations(RBC_vec_remove_calib)) == 0
        @test !("δ" in get_calibrated_parameters(RBC_vec_remove_calib))
        
        RBC_vec_remove_calib = nothing
    end

    @testset "Calibration equation without [ss] variables" begin
        @model RBC_no_ss begin
            1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
            c[0] + k[0] = (1 - δ) * k[-1] + q[0]
            q[0] = exp(z[0]) * k[-1]^α
            z[0] = ρ * z[-1] + std_z * eps_z[x]
        end

        @parameters RBC_no_ss begin
            std_z = 0.01
            ρ = 0.2
            δ = 0.02
            α = 0.5
            β = 0.95
        end

        # Add calibration equation without [ss] - using parameter relationship
        # This calibrates δ based on ρ (both parameters used in model equations)
        add_calibration_equation!(RBC_no_ss, :(δ = ρ / 10 | δ), silent = true)
        
        @test "δ" in get_calibrated_parameters(RBC_no_ss)
        @test length(get_calibration_equations(RBC_no_ss)) == 1
        
        # Model should still solve
        ss = get_steady_state(RBC_no_ss, derivatives = false)
        @test !any(isnan, ss)
        
        RBC_no_ss = nothing
    end
end
