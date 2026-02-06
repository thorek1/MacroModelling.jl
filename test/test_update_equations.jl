using MacroModelling
using Test

# Load SW07 model once at module level
include("../models/Smets_Wouters_2007.jl")

# Helper to get a fresh copy of the SW07 model for testing
# We need to reload to get a clean state without revision history
function load_sw07()
    # Re-include the model file to get a fresh model instance
    Base.invokelatest(include, "../models/Smets_Wouters_2007.jl")
    return Base.invokelatest(() -> Smets_Wouters_2007)
end

function load_RBC_baseline()
    # Re-include the model file to get a fresh model instance
    Base.invokelatest(include, "../models/RBC_baseline.jl")
    return Base.invokelatest(() -> RBC_baseline)
end

function load_FS2000()
    # Re-include the model file to get a fresh model instance
    Base.invokelatest(include, "../models/FS2000.jl")
    return Base.invokelatest(() -> FS2000)
end

@testset verbose = true "SW07 update_equations! functionality" begin
    
    @testset "Update equation by index - modify Taylor rule (remove output growth term)" begin
        model = load_sw07()
        
        # Get original state
        original_eqs = get_equations(model)
        n_eqs_original = length(original_eqs)
        ss_before = get_steady_state(model, derivatives = false)
        
        # Find the Taylor rule equation (contains r[0] and crpi)
        taylor_idx = findfirst(eq -> occursin("r[0]", eq) && occursin("crpi", eq) && occursin("crr", eq), string.(original_eqs))
        @test taylor_idx !== nothing
        
        # Modify Taylor rule: remove output growth term (crdy term) but keep all parameters
        # Original: r[0] = r[ss]^(1-crr) * r[-1]^crr * (pinf[0]/cpie)^((1-crr)*crpi) * (y[0]/yflex[0])^((1-crr)*cry) * (y[0]/yflex[0]/(y[-1]/yflex[-1]))^crdy * ms[0]
        # New: remove the output growth response term (crdy term)
        new_taylor = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])
        
        update_equations!(model, taylor_idx, new_taylor)
        
        # Check revision history was recorded
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        @test history[1].equation_index == taylor_idx
        @test history[1].old_equation !== nothing
        @test history[1].new_equation == new_taylor
        
        # Model should still solve
        ss_after = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss_after)
        
        # Number of equations should remain the same
        @test length(get_equations(model)) == n_eqs_original
        
        # Steady state should be unchanged (Taylor rule modification doesn't affect SS)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-10)
        
        # IRF should work
        irf_after = get_irf(model)
        @test size(irf_after, 1) > 0
        
        model = nothing
    end
    
    @testset "Update equation by matching - change shock process structure" begin
        model = load_sw07()
        
        # Original technology shock: a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x]
        old_shock = :(a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x])
        # Make technology shock more persistent by changing the mean reversion structure
        # Use different coefficients but keep crhoa parameter
        new_shock = :(a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x-1])
        
        update_equations!(model, old_shock, new_shock)
        
        # Check revision history
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        
        # Model should still solve
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Update equation using string format - modify marginal utility equation" begin
        model = load_sw07()
        
        # Find the marginal utility equation with consumption habit
        eqs = get_equations(model)
        xi_idx = findfirst(eq -> occursin("xi[0]", eq) && occursin("chabb", eq) && occursin("csigma", eq), string.(eqs))
        @test xi_idx !== nothing
        
        # Modify to reduce habit formation strength (multiply chabb by 0.5)
        new_xi_eq = "xi[0] = exp(((csigma - 1) / (1 + csigl)) * ((lab[0] * (curvW + wdot[0])) / (1 + curvW)) ^ (1 + csigl)) * (c[0] - (c[ss] * chabb) / cgamma) ^ -csigma"
        
        update_equations!(model, xi_idx, new_xi_eq)
        
        history = get_revision_history(model)
        @test length(history) == 1
        
        # Model should still solve
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Update equation by matching old equation string" begin
        model = load_sw07()
        
        # Match monetary policy shock equation using string
        old_eq = "ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em / 100 * em[x]"
        # Modify structure while keeping parameters
        new_eq = "ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em / 100 * em[x-2]"
        
        update_equations!(model, old_eq, new_eq)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Multiple equation updates with tuple syntax" begin
        model = load_sw07()
        
        ss_before = get_steady_state(model, derivatives = false)
        eqs = get_equations(model)
        
        # Find shock equation indices
        a_idx = findfirst(eq -> occursin("a[0]", eq) && occursin("crhoa", eq) && occursin("ea[x]", eq), string.(eqs))
        b_idx = findfirst(eq -> occursin("b[0]", eq) && occursin("crhob", eq) && occursin("eb[x]", eq), string.(eqs))
        @test a_idx !== nothing
        @test b_idx !== nothing
        
        # Update both shock processes at once - double the shock standard deviations
        update_equations!(model, [
            (a_idx, :(a[0] = 1 - crhoa + crhoa * a[-2] + z_ea / 100 * ea[x])),
            (b_idx, :(b[0] = 1 - crhob + crhob * b[-2] + z_eb / 100 * SCALE1_eb * eb[x]))
        ])
        
        # Check revision history has 2 entries
        history = get_revision_history(model)
        @test length(history) == 2
        @test all(h.action == :update_equation for h in history)
        
        # Model should solve
        ss_after = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss_after)
        
        # Steady state unchanged (shock size doesn't affect SS)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-10)
        
        model = nothing
    end
    
    @testset "Update and revert - round trip with same parameters" begin
        model = load_sw07()
        
        ss_original = get_steady_state(model, derivatives = false)
        irf_original = get_irf(model)
        
        # Original Taylor rule - keep all parameters in both versions
        old_taylor = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
        # Modified Taylor rule - remove output growth term but keep crdy in a benign way
        new_taylor = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])
        
        # Update
        update_equations!(model, old_taylor, new_taylor)
        @test length(get_revision_history(model)) == 1
        
        # Revert
        update_equations!(model, new_taylor, old_taylor)
        @test length(get_revision_history(model)) == 2
        
        # Should be back to original
        ss_final = get_steady_state(model, derivatives = false, parameters = :crdy => 0.2347)
        irf_final = get_irf(model)
        
        @test isapprox(collect(ss_original), collect(ss_final), rtol = 1e-10)
        @test isapprox(collect(irf_original), collect(irf_final), rtol = 1e-10)
        
        model = nothing
    end
    
    @testset "Error cases" begin
        model = load_sw07()
        
        n_eqs = length(get_equations(model))
        
        # Error: index out of bounds
        @test_throws AssertionError update_equations!(model, n_eqs + 1, :(x[0] = 1))
        @test_throws AssertionError update_equations!(model, 0, :(x[0] = 1))
        
        # Error: equation not found
        @test_throws AssertionError update_equations!(
            model, 
            :(nonexistent_variable[0] = some_other[0] + 1), 
            :(x[0] = 1)
        )
        
        model = nothing
    end
end


@testset verbose = true "SW07 add_equation! functionality" begin
    
    @testset "Add auxiliary output gap variable" begin
        model = load_sw07()
        
        n_eqs_before = length(get_equations(model))
        n_vars_before = length(get_variables(model))
        
        # Add an inflation gap measure (deviation from target)
        add_equation!(model, :(inflation_gap[0] = pinf[0] - cpie))
        
        @test length(get_equations(model)) == n_eqs_before + 1
        @test length(get_variables(model)) == n_vars_before + 1
        @test "inflation_gap" in get_variables(model)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :add_equation
        @test history[1].equation_index == n_eqs_before + 1
        @test history[1].old_equation === nothing
        
        ss = get_steady_state(model, derivatives = false)
        @test !isnan(ss(:inflation_gap))
        
        model = nothing
    end
    
    @testset "Add equation using string format" begin
        model = load_sw07()
        
        n_eqs_before = length(get_equations(model))
        
        # Add real interest rate definition
        add_equation!(model, "real_rate[0] = r[0] / pinf[1]")
        
        @test length(get_equations(model)) == n_eqs_before + 1
        @test "real_rate" in get_variables(model)
        
        model = nothing
    end
    
    @testset "Add multiple equations at once" begin
        model = load_sw07()
        
        n_eqs_before = length(get_equations(model))
        
        # Add several auxiliary variables
        add_equation!(model, [
            :(nominal_gdp[0] = y[0] * pinf[0]),
            :(investment_ratio[0] = inve[0] / y[0]),
            :(consumption_ratio[0] = c[0] / y[0])
        ])
        
        @test length(get_equations(model)) == n_eqs_before + 3
        @test "nominal_gdp" in get_variables(model)
        @test "investment_ratio" in get_variables(model)
        @test "consumption_ratio" in get_variables(model)
        
        history = get_revision_history(model)
        @test length(history) == 3
        @test all(h.action == :add_equation for h in history)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss([:nominal_gdp, :investment_ratio, :consumption_ratio]))
        
        model = nothing
    end
    
    @testset "Add equation with lagged variables" begin
        model = load_sw07()
        
        # Add output growth measure
        add_equation!(model, :(output_growth[0] = y[0] / y[-1] - 1))
        
        @test "output_growth" in get_variables(model)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Add equation with forward-looking variables" begin
        model = load_sw07()
        
        # Add expected inflation measure
        add_equation!(model, :(expected_inflation[0] = pinf[1]))
        
        @test "expected_inflation" in get_variables(model)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
end


@testset verbose = true "SW07 remove_equation! functionality" begin
    
    @testset "Remove observable equation by index" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        n_eqs_before = length(eqs)
        n_vars_before = length(get_variables(model))
        
        # Find labobs equation: labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1)
        labobs_idx = findfirst(eq -> occursin("labobs[0]", eq) && occursin("constelab", eq), string.(eqs))
        @test labobs_idx !== nothing
        
        remove_equation!(model, labobs_idx)
        
        @test length(get_equations(model)) == n_eqs_before - 1
        @test length(get_variables(model)) == n_vars_before - 1
        @test !("labobs" in get_variables(model))
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :remove_equation
        @test history[1].new_equation === nothing
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Remove equation by matching" begin
        model = load_sw07()
        
        n_eqs_before = length(get_equations(model))
        
        # Remove wage growth observable
        remove_equation!(model, :(dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)))
        
        @test length(get_equations(model)) == n_eqs_before - 1
        @test !("dwobs" in get_variables(model))
        
        model = nothing
    end
    
    @testset "Remove multiple equations" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        n_eqs_before = length(eqs)
        
        # Find observable equations to remove
        labobs_idx = findfirst(eq -> occursin("labobs[0]", eq), string.(eqs))
        dwobs_idx = findfirst(eq -> occursin("dwobs[0]", eq), string.(eqs))
        @test labobs_idx !== nothing
        @test dwobs_idx !== nothing
        
        # Remove in descending order
        remove_equation!(model, sort([labobs_idx, dwobs_idx], rev=true))
        
        @test length(get_equations(model)) == n_eqs_before - 2
        @test !("labobs" in get_variables(model))
        @test !("dwobs" in get_variables(model))
        
        history = get_revision_history(model)
        @test length(history) == 2
        
        model = nothing
    end
    
    @testset "Add then remove - round trip" begin
        model = load_sw07()
        
        ss_before = get_steady_state(model, derivatives = false)
        n_eqs_before = length(get_equations(model))
        
        # Add auxiliary variable
        add_equation!(model, :(temp_var[0] = y[0] + c[0]))
        @test length(get_equations(model)) == n_eqs_before + 1
        
        # Remove it
        remove_equation!(model, :(temp_var[0] = y[0] + c[0]))
        @test length(get_equations(model)) == n_eqs_before
        
        # Steady state should match original
        ss_after = get_steady_state(model, derivatives = false)
        @test isapprox(collect(ss_before), collect(ss_after), rtol = 1e-10)
        
        model = nothing
    end
    
    @testset "Error cases" begin
        model = load_sw07()
        
        n_eqs = length(get_equations(model))
        
        # Error: index out of bounds
        @test_throws AssertionError remove_equation!(model, n_eqs + 1)
        @test_throws AssertionError remove_equation!(model, 0)
        
        # Error: equation not found
        @test_throws AssertionError remove_equation!(
            model,
            :(nonexistent[0] = 1)
        )
        
        model = nothing
    end
end


@testset verbose = true "SW07 update_calibration_equations! functionality" begin
    
    @testset "Update calibration equation - change target value" begin
        model = load_RBC_baseline()
        
        # SW07 has calibration: mcflex = mc[ss] | mcflex
        calib_eqs = get_calibration_equations(model)
        @test length(calib_eqs) >= 1
        
        calib_params = get_calibrated_parameters(model)
        @test "ψ" in calib_params
        
        # Modify the calibration target slightly (multiply by 1.01)
        update_calibration_equations!(model, 2, :(l[ss] = 0.5 | ψ))
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :update_calibration_equation
        
        # mcflex should still be calibrated
        @test "ψ" in get_calibrated_parameters(model)
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Update calibration equation using tuple syntax" begin
        model = load_RBC_baseline()
        
        # Update both calibration equations (mcflex and cpie)
        update_calibration_equations!(model, [
            (1, :(l[ss] = 2/3 | ψ)),
            (2, :(ḡ = 1/4 * y[ss]| ḡ))
        ])
        # TODO: accept also: par | calib_eq
        
        history = get_revision_history(model)
        @test length(history) == 2
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Error - calibrate non-existent parameter" begin
        model = load_sw07()
        
        @test_throws ErrorException update_calibration_equations!(
            model,
            1,
            :(mcflex = mc[ss] | nonexistent_param),
            silent = true
        )
        
        model = nothing
    end
end


@testset verbose = true "SW07 add_calibration_equation! functionality" begin
    
    @testset "Add calibration for fixed parameter" begin
        model = load_FS2000()
        
        # ctou = 0.025 is a fixed depreciation rate parameter
        calib_before = get_calibrated_parameters(model)
        @test !("del" in calib_before)
        
        n_calib_before = length(get_calibration_equations(model))
        
        # Add calibration: capital-output ratio determines depreciation
        add_calibration_equation!(model, :(k[ss] / y[ss] = 8.0 | del))
        
        @test length(get_calibration_equations(model)) == n_calib_before + 1
        @test "del" in get_calibrated_parameters(model)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :add_calibration_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Add calibration using vector syntax" begin
        model = load_FS2000()
        
        n_calib_before = length(get_calibration_equations(model))
        
        # Add calibration for calfa (capital share)
        add_calibration_equation!(model, [
            :(k[ss] / y[ss] = 4 | del)
        ])
        
        @test length(get_calibration_equations(model)) == n_calib_before + 1
        @test "del" in get_calibrated_parameters(model)
        
        model = nothing
    end
    
    @testset "Error - add calibration for already calibrated parameter" begin
        model = load_RBC_baseline()
        
        # del is already calibrated
        @test "ψ" in get_calibrated_parameters(model)
        
        @test_throws ErrorException add_calibration_equation!(
            model,
            :(y[ss] = 1.5 | ψ)
        )
        
        model = nothing
    end
    
    @testset "Error - add calibration without | syntax" begin
        model = load_RBC_baseline()
        
        @test_throws ErrorException add_calibration_equation!(
            model,
            :(k[ss] / y[ss] = 8.0)  # Missing | ctou
        )
        
        model = nothing
    end
    
    @testset "Error - calibrate non-existent parameter" begin
        model = load_RBC_baseline()
        
        @test_throws ErrorException add_calibration_equation!(
            model,
            :(y[ss] = 1.5 | fake_param)
        )
        
        model = nothing
    end
end


@testset verbose = true "SW07 remove_calibration_equation! functionality" begin
    
    @testset "Remove calibration with explicit parameters override" begin
        model = load_RBC_baseline()
        
        n_calib_before = length(get_calibration_equations(model))
        calib_params_before = get_calibrated_parameters(model)
        @test "ψ" in calib_params_before
        
        # Remove ψ calibration with explicit value via parameters
        remove_calibration_equation!(model, 2, parameters = :ψ => 1.0)
        
        @test length(get_calibration_equations(model)) == n_calib_before - 1
        @test !("ψ" in get_calibrated_parameters(model))
        @test "ψ" in get_parameters(model)
        
        history = get_revision_history(model)
        @test length(history) == 1
        @test history[1].action == :remove_calibration_equation
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Remove calibration using current value" begin
        model = load_RBC_baseline()
        
        n_calib_before = length(get_calibration_equations(model))
        
        # Remove without specifying parameters - uses current calibrated value
        remove_calibration_equation!(model, 1)
        
        @test length(get_calibration_equations(model)) == n_calib_before - 1
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Remove calibration using vector syntax" begin
        model = load_RBC_baseline()
        
        n_calib_before = length(get_calibration_equations(model))
        @test n_calib_before >= 2  # SW07 has mcflex and cpie calibrations
        
        # Remove first calibration equation
        remove_calibration_equation!(model, [1,2])
        
        @test length(get_calibration_equations(model)) == n_calib_before - 2
        
        model = nothing
    end
    
    @testset "Add then remove calibration - round trip" begin
        model = load_FS2000()
        
        n_calib_before = length(get_calibration_equations(model))
        
        # Add a calibration
        add_calibration_equation!(model, :(k[ss] / y[ss] = 8.0 | del), silent = true)
        @test length(get_calibration_equations(model)) == n_calib_before + 1
        
        ss_calib = get_steady_state(model, derivatives = false)

        # Remove it (last added)
        remove_calibration_equation!(model, n_calib_before + 1)
        @test length(get_calibration_equations(model)) == n_calib_before
        
        # Model should solve
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Error - remove from model with no calibrations" begin
        model = load_sw07()
        
        # Remove all calibration equations first
        while length(get_calibration_equations(model)) > 0
            remove_calibration_equation!(model, 1)
        end
        
        @test length(get_calibration_equations(model)) == 0
        
        # Now trying to remove should error
        @test_throws AssertionError remove_calibration_equation!(model, 1)
        
        model = nothing
    end
end


@testset verbose = true "SW07 get_revision_history functionality" begin
    
    @testset "Empty history for fresh model" begin
        model = load_sw07()
        
        history = get_revision_history(model)
        @test length(history) == 0
        
        model = nothing
    end
    
    @testset "History tracks all operations" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        
        # 1. Update equation
        ms_idx = findfirst(eq -> occursin("ms[0]", eq), string.(eqs))
        update_equations!(model, ms_idx, :(ms[0] = 0.5 + 0.5 * crhoms * ms[-1] + z_em / 100 * em[x]))
        
        # 2. Add equation
        add_equation!(model, :(new_var[0] = y[0]))
        
        # 3. Update calibration
        update_calibration_equations!(model, 1, :(mcflex = mc[ss] * 0.99 | mcflex), silent = true)
        
        # 4. Add calibration
        add_calibration_equation!(model, :(lab[ss] = 1.0 | calfa), silent = true)
        
        # Check history
        history = get_revision_history(model)
        @test length(history) == 4
        @test history[1].action == :update_equation
        @test history[2].action == :add_equation
        @test history[3].action == :update_calibration_equation
        @test history[4].action == :add_calibration_equation
        
        # All entries should have timestamps
        @test all(h -> haskey(h, :timestamp), history)
        
        model = nothing
    end
    
    @testset "History preserves equation content" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        ms_idx = findfirst(eq -> occursin("ms[0]", eq), string.(eqs))
        new_eq = :(ms[0] = 0.5 + 0.5 * crhoms * ms[-1] + z_em / 100 * em[x])
        
        update_equations!(model, ms_idx, new_eq)
        
        history = get_revision_history(model)
        @test length(history) == 1
        
        # Check old and new equations are stored
        @test history[1].old_equation !== nothing
        @test history[1].new_equation == new_eq
        @test history[1].equation_index == ms_idx
        
        model = nothing
    end
end


@testset verbose = true "SW07 complex modification scenarios" begin
    
    @testset "Remove observables and add custom ones" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        
        # Remove standard observables
        labobs_idx = findfirst(eq -> occursin("labobs[0]", eq), string.(eqs))
        remove_equation!(model, labobs_idx)
        
        # Add custom observable (employment rate instead of hours)
        add_equation!(model, :(employment_obs[0] = 100 * log(lab[0] / lab[ss])))
        
        @test !("labobs" in get_variables(model))
        @test "employment_obs" in get_variables(model)
        
        history = get_revision_history(model)
        @test length(history) == 2
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        model = nothing
    end
    
    @testset "Modify multiple shock processes" begin
        model = load_sw07()
        
        eqs = get_equations(model)
        
        # Find and modify technology shock - make more persistent
        a_idx = findfirst(eq -> occursin("a[0]", eq) && occursin("crhoa", eq) && occursin("ea[x]", eq), string.(eqs))
        update_equations!(model, a_idx, :(a[0] = 0.5 * (1 - crhoa) + crhoa * a[-1] + z_ea / 100 * ea[x]))
        
        # Find and modify government spending shock - also make more persistent  
        gy_idx = findfirst(eq -> occursin("gy[0]", eq) && occursin("crhog", eq), string.(eqs))
        update_equations!(model, gy_idx, :(gy[0] - cg = 0.5 * crhog * (gy[-1] - cg) + z_eg / 100 * eg[x] + z_ea / 100 * ea[x] * cgy))
        
        history = get_revision_history(model)
        @test length(history) == 2
        
        ss = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss)
        
        # IRF should work
        irf = get_irf(model)
        @test size(irf, 1) > 0
        
        model = nothing
    end
    
    @testset "Full workflow: update, add, modify calibration" begin
        model = load_sw07()
        
        # Store original state
        n_eqs_original = length(get_equations(model))
        
        # Make several modifications
        eqs = get_equations(model)
        
        # 1. Modify Taylor rule - remove output growth term
        taylor_idx = findfirst(eq -> occursin("r[0]", eq) && occursin("crpi", eq), string.(eqs))
        update_equations!(model, taylor_idx, :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0]))
        
        # 2. Add auxiliary variable
        add_equation!(model, :(policy_stance[0] = r[0] - r[ss]))
        
        # 3. Modify calibration
        update_calibration_equations!(model, 1, :(mcflex = mc[ss] * 1.05 | mcflex), silent = true)
        
        # Verify model works
        ss_modified = get_steady_state(model, derivatives = false)
        @test !any(isnan, ss_modified)
        @test length(get_equations(model)) == n_eqs_original + 1
        
        # Check history has all 3 modifications
        history = get_revision_history(model)
        @test length(history) == 3
        
        model = nothing
    end
end


@testset verbose = true "SW07 dynamics verification" begin
    
    @testset "Taylor rule modification changes IRF" begin
        model = load_sw07()
        
        # Get original IRF
        irf_original = get_irf(model)
        
        # Modify Taylor rule - remove output growth response
        old_taylor = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * (y[0] / yflex[0] / (y[-1] / yflex[-1])) ^ crdy * ms[0])
        new_taylor = :(r[0] = r[ss] ^ (1 - crr) * r[-1] ^ crr * (pinf[0] / cpie) ^ ((1 - crr) * crpi) * (y[0] / yflex[0]) ^ ((1 - crr) * cry) * ms[0])
        update_equations!(model, old_taylor, new_taylor)
        
        irf_modified = get_irf(model)
        
        # IRFs should be different (removed output growth response)
        @test !isapprox(collect(irf_original), collect(irf_modified), rtol = 1e-5)
        
        model = nothing
    end
    
    @testset "Shock size modification changes variance" begin
        model = load_sw07()
        
        std_original = get_standard_deviation(model, derivatives = false)
        
        eqs = get_equations(model)
        a_idx = findfirst(eq -> occursin("a[0]", eq) && occursin("crhoa", eq) && occursin("ea[x]", eq), string.(eqs))
        
        # Double the shock size
        update_equations!(model, a_idx, :(a[0] = 1 - crhoa + crhoa * a[-1] + 2 * z_ea / 100 * ea[x]))
        
        std_modified = get_standard_deviation(model, derivatives = false)
        
        # Standard deviations should change
        @test !isapprox(collect(std_original), collect(std_modified), rtol = 1e-5)
        
        model = nothing
    end
end

