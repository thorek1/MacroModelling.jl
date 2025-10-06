using MacroModelling
using Test

@testset "Partial parameter definition" begin
    # Test 1: Create a model
    @model RBC_partial begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end
    
    # Test 2: Define only some parameters (missing β and δ)
    @test_logs (:info, r"Model set up with undefined parameters.*β.*δ") @parameters RBC_partial silent = true begin
        std_z = 0.01
        ρ = 0.2
        α = 0.5
    end
    
    # Test 3: Verify undefined_parameters field is populated
    @test length(RBC_partial.undefined_parameters) == 2
    @test :β ∈ RBC_partial.undefined_parameters
    @test :δ ∈ RBC_partial.undefined_parameters
    
    # Test 4: Attempt to get IRF should fail with informative error
    @test_logs (:error, r"Cannot compute IRFs.*undefined parameters.*β.*δ") get_irf(RBC_partial)
    
    # Test 5: Attempt to get steady state should fail with informative error
    @test_logs (:error, r"Cannot compute non-stochastic steady state.*undefined parameters.*β.*δ") get_steady_state(RBC_partial)
    
    # Test 6: Now define all parameters
    @test_logs @parameters RBC_partial silent = true begin
        std_z = 0.01
        ρ = 0.2
        α = 0.5
        β = 0.95
        δ = 0.02
    end
    
    # Test 7: Verify undefined_parameters is now empty
    @test length(RBC_partial.undefined_parameters) == 0
    
    # Test 8: Now operations should succeed
    SS = get_steady_state(RBC_partial)
    @test !isnan(SS[1])
    @test length(SS) > 0
    
    # Test 9: IRF should now work
    irf_result = get_irf(RBC_partial, periods = 10)
    @test size(irf_result, 2) == 10
    
    println("✓ All partial parameter definition tests passed")
end

@testset "Provide parameters later via function call" begin
    # Test 1: Create a model and define only some parameters
    @model RBC_later begin
        1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
        c[0] + k[0] = (1 - δ) * k[-1] + q[0]
        q[0] = exp(z[0]) * k[-1]^α
        z[0] = ρ * z[-1] + std_z * eps_z[x]
    end
    
    @test_logs (:info, r"Model set up with undefined parameters") @parameters RBC_later silent = true begin
        std_z = 0.01
        ρ = 0.2
    end
    
    # Test 2: Provide missing parameters via function call
    params_dict = Dict(:α => 0.5, :β => 0.95, :δ => 0.02, :std_z => 0.01, :ρ => 0.2)
    
    # Test 3: IRF with explicit parameters should work even when some are undefined in @parameters
    # Note: This requires updating the undefined_parameters list when parameters are provided
    irf_result = get_irf(RBC_later, parameters = params_dict, periods = 10)
    @test size(irf_result, 2) == 10
    
    println("✓ All 'provide parameters later' tests passed")
end
